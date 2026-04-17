"""
eval.py — Evaluation for nnU-Net on original TIF stacks.
Semantic prediction → connected components → 9 metrics + mAP.

Usage:
    python eval.py                    # evaluate test set
    python eval.py --subset val       # evaluate val set
    python eval.py --no-viz           # skip visualization
"""

from __future__ import annotations
import os, sys, json, gc, time, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tifffile import imread, imwrite as tif_write
from pathlib import Path
from scipy.optimize import linear_sum_assignment

import torch
import config


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG (from config.py + overrides)
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_DIR   = config.in_dataset_dir
RESULTS_DIR   = os.path.join(
    config.nnunet_results,
    config.nnunet_dataset_name,
    f"nnUNetTrainer_Comet__nnUNetPlans__{config.configuration}",
    f"fold_{config.folds[0]}",
)
CHECKPOINT    = os.path.join(RESULTS_DIR, "checkpoint_best.pth")
N_SLICES      = 10
BINARY_THRESH = config.prob_threshold
MIN_VOL       = config.min_instance_volume
CC_CONN       = config.cc_connectivity
TAUS          = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAP_THRESHOLDS = np.arange(0.5, 1.0, 0.05)   # COCO-style 0.50:0.05:0.95
IOU_THRESH    = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Load nnU-Net model
# ═══════════════════════════════════════════════════════════════════════════════
def load_nnunet_model(checkpoint_path: str, device: torch.device):
    """Load nnU-Net model from checkpoint."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    plans = checkpoint["init_args"]["plans"]
    dataset_json = checkpoint["init_args"]["dataset_json"]
    configuration_name = checkpoint["init_args"]["configuration"]

    pm = PlansManager(plans)
    cm = pm.get_configuration(configuration_name)

    num_input_channels = determine_num_input_channels(pm, cm, dataset_json)
    label_manager = pm.get_label_manager(dataset_json)

    network = get_network_from_plans(
        cm.network_arch_class_name,
        cm.network_arch_init_kwargs,
        cm.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        deep_supervision=False,
    ).to(device)

    network.load_state_dict(checkpoint["network_weights"])
    network.eval()

    # Get preprocessing info
    patch_size = cm.patch_size
    intensity_props = plans["foreground_intensity_properties_per_channel"]["0"]

    print(f"✅  Model loaded from {checkpoint_path}")
    print(f"   Patch size: {patch_size}")
    print(f"   Epoch: {checkpoint.get('current_epoch', '?')}")
    print(f"   Best EMA Dice: {checkpoint.get('_best_ema', '?')}")

    return network, patch_size, intensity_props


# ═══════════════════════════════════════════════════════════════════════════════
#  Preprocessing + Inference
# ═══════════════════════════════════════════════════════════════════════════════
def zscore_normalize(vol: np.ndarray, props: dict) -> np.ndarray:
    """Z-score normalization matching nnU-Net's ZScoreNormalization."""
    mean = props["mean"]
    std  = props["std"]
    lo   = props["percentile_00_5"]
    hi   = props["percentile_99_5"]
    vol  = vol.astype(np.float32)
    vol  = np.clip(vol, lo, hi)
    vol  = (vol - mean) / max(std, 1e-8)
    return vol


@torch.inference_mode()
def predict_full_volume(
    model: torch.nn.Module,
    img_3d: np.ndarray,
    patch_size: list[int],
    device: torch.device,
    step_size: float = 0.5,
    mirror_axes: tuple = (0, 1, 2),
) -> np.ndarray:
    """
    Sliding-window inference with Gaussian weighting and test-time augmentation
    (mirroring), matching nnU-Net inference behavior.
    """
    # Gaussian importance map
    def _gaussian_map(shape):
        sigmas = [s / 4 for s in shape]
        coords = [np.arange(s) - s / 2 + 0.5 for s in shape]
        grid = np.meshgrid(*coords, indexing="ij")
        g = np.exp(-sum(c ** 2 / (2 * s ** 2) for c, s in zip(grid, sigmas)))
        g /= g.max()
        g = np.clip(g, 1e-4, None)
        return g.astype(np.float32)

    pz, py, px = patch_size
    oz, oy, ox = img_3d.shape

    # Pad to at least patch_size
    pad_z = max(0, pz - oz)
    pad_y = max(0, py - oy)
    pad_x = max(0, px - ox)
    padded = np.pad(img_3d, [(0, pad_z), (0, pad_y), (0, pad_x)], mode="reflect")
    nz, ny, nx = padded.shape

    # Compute step and grid positions
    sz = max(1, int(pz * step_size))
    sy = max(1, int(py * step_size))
    sx = max(1, int(px * step_size))

    starts_z = list(range(0, nz - pz + 1, sz))
    starts_y = list(range(0, ny - py + 1, sy))
    starts_x = list(range(0, nx - px + 1, sx))
    if starts_z[-1] + pz < nz: starts_z.append(nz - pz)
    if starts_y[-1] + py < ny: starts_y.append(ny - py)
    if starts_x[-1] + px < nx: starts_x.append(nx - px)
    starts_z = sorted(set(starts_z))
    starts_y = sorted(set(starts_y))
    starts_x = sorted(set(starts_x))

    gauss = _gaussian_map(patch_size)
    agg_prob = np.zeros((2, nz, ny, nx), dtype=np.float32)  # 2 classes (bg, fg)
    agg_weight = np.zeros((nz, ny, nx), dtype=np.float32)

    n_patches = len(starts_z) * len(starts_y) * len(starts_x)
    print(f"  Sliding window: {n_patches} patches, step={step_size}")

    done = 0
    for iz in starts_z:
        for iy in starts_y:
            for ix in starts_x:
                patch = padded[iz:iz+pz, iy:iy+py, ix:ix+px]
                x = torch.from_numpy(patch[None, None]).to(device)  # (1, 1, Z, Y, X)

                # Test-time augmentation: mirror axes
                preds = []
                for axes in _mirror_combos(mirror_axes):
                    xf = _flip(x, axes)
                    out = model(xf)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    out = torch.softmax(out, dim=1)
                    out = _flip(out, axes)
                    preds.append(out)

                avg = torch.stack(preds).mean(0)
                avg = avg[0].cpu().numpy()  # (2, pz, py, px)

                agg_prob[:, iz:iz+pz, iy:iy+py, ix:ix+px] += avg * gauss[None]
                agg_weight[iz:iz+pz, iy:iy+py, ix:ix+px] += gauss
                done += 1

    agg_weight = np.maximum(agg_weight, 1e-8)
    prob = agg_prob / agg_weight[None]

    # Return foreground probability, cropped to original size
    return prob[1, :oz, :oy, :ox]


def _mirror_combos(axes):
    """Generate all combinations of mirror axes including no-mirror."""
    combos = [()]
    for a in axes:
        combos = combos + [c + (a,) for c in combos]
    return combos


def _flip(x: torch.Tensor, axes: tuple) -> torch.Tensor:
    """Flip tensor along spatial axes (dims 2,3,4 for 5D tensor)."""
    for a in axes:
        x = torch.flip(x, [a + 2])  # +2 for batch+channel dims
    return x


def binary_to_instances(prob: np.ndarray, thresh: float, min_vol: int) -> np.ndarray:
    """Threshold probability map → connected components → filter small."""
    import cc3d
    binary = (prob > thresh).astype(np.uint8)
    instances = cc3d.connected_components(binary, connectivity=CC_CONN)

    # Filter small objects
    if min_vol > 0:
        ids, counts = np.unique(instances, return_counts=True)
        for oid, cnt in zip(ids, counts):
            if oid == 0:
                continue
            if cnt < min_vol:
                instances[instances == oid] = 0
        # Relabel
        instances = cc3d.connected_components((instances > 0).astype(np.uint8),
                                              connectivity=CC_CONN)
    return instances.astype(np.int32)



def iou_matrix(gt, pred):
    """IoU matrix via bincount. Returns extended tuple for reuse."""
    gt_ids   = np.unique(gt);   gt_ids   = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred); pred_ids = pred_ids[pred_ids != 0]
    ng, np_ = len(gt_ids), len(pred_ids)
    if ng == 0 or np_ == 0:
        return (gt_ids, pred_ids,
                np.zeros((ng, np_), dtype=np.float64),
                np.zeros((ng, np_), dtype=np.int64),
                np.zeros(ng, dtype=np.int64),
                np.zeros(np_, dtype=np.int64))
    gt_rm = np.zeros(int(gt.max()) + 1, dtype=np.int64)
    pr_rm = np.zeros(int(pred.max()) + 1, dtype=np.int64)
    for i, g in enumerate(gt_ids):   gt_rm[g] = i + 1
    for j, p in enumerate(pred_ids): pr_rm[p] = j + 1
    np1 = np.int64(np_ + 1)
    codes = gt_rm[gt.ravel()] * np1 + pr_rm[pred.ravel()]
    tbl = np.bincount(codes, minlength=int((ng + 1) * np1)).reshape(ng + 1, int(np1))
    ov = tbl[1:, 1:].copy()
    gv = tbl[1:, :].sum(1)
    pv = tbl[:, 1:].sum(0)
    un = gv[:, None] + pv[None, :] - ov
    iou = np.zeros_like(ov, dtype=np.float64)
    m = un > 0
    iou[m] = ov[m].astype(np.float64) / un[m].astype(np.float64)
    return gt_ids, pred_ids, iou, ov, gv, pv


def hungarian_match(iou_mat, thresh):
    """Optimal matching via Hungarian algorithm."""
    if iou_mat.size == 0:
        return []
    gt_has = np.any(iou_mat > 0, axis=1)
    pr_has = np.any(iou_mat > 0, axis=0)
    gt_idx = np.where(gt_has)[0]
    pr_idx = np.where(pr_has)[0]
    if len(gt_idx) == 0 or len(pr_idx) == 0:
        return []
    sub = iou_mat[np.ix_(gt_idx, pr_idx)]
    row_ind, col_ind = linear_sum_assignment(-sub)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if sub[r, c] >= thresh:
            matches.append((int(gt_idx[r]), int(pr_idx[c]), float(sub[r, c])))
    return matches


def obj_metrics_at_thresh(iou_mat, n_gt, n_pred, thresh):
    """Object-level F1, Prec, Rec, mean matched IoU at one IoU threshold."""
    if n_gt == 0 and n_pred == 0:
        return dict(f1=1., precision=1., recall=1., mean_iou=1., tp=0, fp=0, fn=0)
    if n_gt == 0:
        return dict(f1=0., precision=0., recall=0., mean_iou=0., tp=0, fp=n_pred, fn=0)
    if n_pred == 0:
        return dict(f1=0., precision=0., recall=0., mean_iou=0., tp=0, fp=0, fn=n_gt)
    matches = hungarian_match(iou_mat, thresh)
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp
    pr = tp / (tp + fp) if tp + fp > 0 else 0.
    rc = tp / (tp + fn) if tp + fn > 0 else 0.
    f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.
    sum_iou = sum(m[2] for m in matches)
    mi = sum_iou / n_gt if n_gt > 0 else 0.
    return dict(f1=f1, precision=pr, recall=rc, mean_iou=mi, tp=tp, fp=fp, fn=fn)


def compute_aji(gt_ids, pred_ids, iou_mat, overlap, gt_vols, pred_vols):
    """AJI (Kumar 2017) from precomputed IoU matrix."""
    ng, np_ = len(gt_ids), len(pred_ids)
    if ng == 0 and np_ == 0: return 1.
    if ng == 0 or np_ == 0:  return 0.
    used = set()
    si, su = 0, 0
    for i in range(ng):
        bj, bv = -1, 0.
        for j in range(np_):
            if j in used: continue
            if iou_mat[i, j] > bv:
                bv = iou_mat[i, j]; bj = j
        if bj >= 0 and bv > 0:
            inter = int(overlap[i, bj])
            union = int(gt_vols[i]) + int(pred_vols[bj]) - inter
            si += inter; su += union; used.add(bj)
        else:
            su += int(gt_vols[i])
    for j in range(np_):
        if j not in used:
            su += int(pred_vols[j])
    return si / su if su > 0 else 0.


def compute_ap_single_thresh(iou_mat, n_gt, n_pred, thresh):
    """COCO-style AP at a single IoU threshold."""
    if n_gt == 0 and n_pred == 0: return 1.
    if n_gt == 0: return 0.
    if n_pred == 0: return 0.
    pred_scores = iou_mat.max(axis=0)
    sorted_j = np.argsort(-pred_scores)
    matched_gt = np.zeros(n_gt, dtype=bool)
    tps = np.zeros(n_pred, dtype=np.float64)
    fps = np.zeros(n_pred, dtype=np.float64)
    for rank, j in enumerate(sorted_j):
        best_i, best_iou = -1, 0.
        for i in range(n_gt):
            if matched_gt[i]: continue
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]; best_i = i
        if best_i >= 0 and best_iou >= thresh:
            tps[rank] = 1
            matched_gt[best_i] = True
        else:
            fps[rank] = 1
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    ap = 0.
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    return ap / 101.


def compute_mAP(iou_mat, n_gt, n_pred):
    """COCO-style mAP averaged over IoU thresholds 0.50:0.05:0.95."""
    return float(np.mean([
        compute_ap_single_thresh(iou_mat, n_gt, n_pred, t)
        for t in MAP_THRESHOLDS
    ]))


def pixel_metrics(gt, pred):
    """Pixel-level Dice, IoU, Precision, Recall."""
    g, p = (gt > 0), (pred > 0)
    tp = int(np.sum(g & p))
    fp = int(np.sum(~g & p))
    fn = int(np.sum(g & ~p))
    pr  = tp / (tp + fp) if tp + fp > 0 else 1.
    rc  = tp / (tp + fn) if tp + fn > 0 else 1.
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.
    return dict(pixel_dice=dice, pixel_iou=iou, pixel_prec=pr, pixel_rec=rc)


def evaluate_volume(gt, pred):
    """Full evaluation of one volume — returns dict with all metrics."""
    gt_ids, pred_ids, iou_mat, overlap, gv, pv = iou_matrix(gt, pred)
    ng, np_ = len(gt_ids), len(pred_ids)

    obj = obj_metrics_at_thresh(iou_mat, ng, np_, IOU_THRESH)
    aji = compute_aji(gt_ids, pred_ids, iou_mat, overlap, gv, pv)
    mAP = compute_mAP(iou_mat, ng, np_)
    pix = pixel_metrics(gt, pred)

    # F1 at all thresholds (for detailed reporting)
    f1_per_tau = {}
    for t in TAUS:
        m = obj_metrics_at_thresh(iou_mat, ng, np_, t)
        f1_per_tau[f"f1_{t:.1f}"] = m["f1"]

    return {
        "n_gt": ng, "n_pred": np_,
        "aji": aji,
        "obj_f1": obj["f1"], "obj_prec": obj["precision"],
        "obj_rec": obj["recall"], "obj_iou": obj["mean_iou"],
        "mAP": mAP,
        "px_dice": pix["pixel_dice"], "px_iou": pix["pixel_iou"],
        "px_prec": pix["pixel_prec"], "px_rec": pix["pixel_rec"],
        **f1_per_tau,
    }


def tp_fp_fn_overlay(gs, ps):
    o = np.zeros((*gs.shape, 3), dtype=np.float32)
    g, p = gs > 0, ps > 0
    o[g & p]  = [0, 1, 0]   # TP green
    o[g & ~p] = [1, 0, 0]   # FN red
    o[~g & p] = [0, 0, 1]   # FP blue
    return o


def _match_instances_3d(gt, pred, thresh=0.5):
    """3D IoU matching via Hungarian → returns sets of matched/unmatched IDs."""
    gt_ids_set = set(np.unique(gt));   gt_ids_set.discard(0)
    pred_ids_set = set(np.unique(pred)); pred_ids_set.discard(0)

    if not gt_ids_set or not pred_ids_set:
        return set(), set(), gt_ids_set, pred_ids_set

    gt_ids, pred_ids, iou_mat, _, _, _ = iou_matrix(gt, pred)
    matches = hungarian_match(iou_mat, thresh)

    matched_gt   = {gt_ids[m[0]]   for m in matches}
    matched_pred = {pred_ids[m[1]] for m in matches}
    fn_ids = gt_ids_set - matched_gt
    fp_ids = pred_ids_set - matched_pred
    return matched_gt, matched_pred, fn_ids, fp_ids


def _per_slice_errors(gt, pred, fn_ids, fp_ids):
    """Per Z-slice error profile."""
    nz = gt.shape[0]
    out = {
        "fp_pixels":  np.zeros(nz, dtype=int),
        "fn_pixels":  np.zeros(nz, dtype=int),
        "fp_objects": np.zeros(nz, dtype=int),
        "fn_objects": np.zeros(nz, dtype=int),
        "n_gt":       np.zeros(nz, dtype=int),
        "n_pred":     np.zeros(nz, dtype=int),
        "dice":       np.zeros(nz, dtype=float),
    }

    for z in range(nz):
        gt_z, pred_z = gt[z], pred[z]
        gt_fg, pred_fg = gt_z > 0, pred_z > 0

        out["fp_pixels"][z] = np.logical_and(~gt_fg, pred_fg).sum()
        out["fn_pixels"][z] = np.logical_and(gt_fg, ~pred_fg).sum()

        gt_labels_z   = set(np.unique(gt_z));   gt_labels_z.discard(0)
        pred_labels_z = set(np.unique(pred_z)); pred_labels_z.discard(0)
        out["n_gt"][z]   = len(gt_labels_z)
        out["n_pred"][z] = len(pred_labels_z)

        out["fn_objects"][z] = len(fn_ids & gt_labels_z)
        out["fp_objects"][z] = len(fp_ids & pred_labels_z)

        tp = np.logical_and(gt_fg, pred_fg).sum()
        vol = gt_fg.sum() + pred_fg.sum()
        out["dice"][z] = (2 * tp / vol) if vol > 0 else 1.0

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="test", choices=["test", "val", "train"])
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--save-dir", default=None, help="Save figures to this dir")
    args = parser.parse_args()

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    model, patch_size, intensity_props = load_nnunet_model(args.checkpoint, device)

    # ── Load data ────────────────────────────────────────────────────────
    with open(os.path.join(DATASET_DIR, "split.json")) as f:
        split = json.load(f)
    file_list = split[args.subset]
    print(f"\n{args.subset}: {len(file_list)} stacks\n")

    # ── File mapping (case_id → original filename) ───────────────────────
    file_map_path = os.path.join(
        config.nnunet_raw, config.nnunet_dataset_name, "file_map.json")
    if os.path.exists(file_map_path):
        with open(file_map_path) as f:
            file_map = json.load(f)
        # Invert: filename → case_id
        inv_map = {v: k for k, v in file_map.items()}
    else:
        inv_map = {}

    mirror_axes = () if args.no_tta else (0, 1, 2)

    # ── Predict + evaluate ───────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    results, preds_all = [], []
    X_raw, Y_raw = [], []

    for i, fname in enumerate(file_list):
        print(f"Stack {i+1}/{len(file_list)}: {fname}")
        t0 = time.time()

        img = imread(os.path.join(DATASET_DIR, "images", fname))
        gt  = imread(os.path.join(DATASET_DIR, "masks", fname))
        if img.ndim == 4: img = img[..., 0]
        if gt.ndim == 4:  gt  = gt[..., 0]
        X_raw.append(img)
        Y_raw.append(gt)

        # Preprocess (same as nnU-Net)
        img_norm = zscore_normalize(img, intensity_props)

        # Predict
        prob = predict_full_volume(
            model, img_norm, patch_size, device,
            step_size=args.step_size,
            mirror_axes=mirror_axes,
        )

        # Post-process: threshold → CC → filter
        pred = binary_to_instances(prob, BINARY_THRESH, MIN_VOL)
        preds_all.append(pred)

        # Metrics (benchmark-consistent)
        r = evaluate_volume(gt, pred)
        r["file"] = fname

        results.append(r)
        dt = time.time() - t0
        print(f"  GT={r['n_gt']}  Pred={r['n_pred']}  "
              f"AJI={100*r['aji']:.2f}%  F1@.5={100*r['obj_f1']:.2f}%  "
              f"mAP={100*r['mAP']:.2f}%  PxIoU={100*r['px_iou']:.2f}%  ({dt:.1f}s)\n")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {args.subset} ({len(file_list)} stacks)")
    print(f"  Model: nnU-Net {config.configuration} | Aug: {config.aug_preset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  TTA: {'OFF' if args.no_tta else 'ON (mirror all axes)'}")
    print(f"{'='*70}")

    keys = [("aji",      "AJI"),
            ("obj_f1",   "Obj F1 @0.5"),
            ("obj_prec", "Obj Prec @0.5"),
            ("obj_rec",  "Obj Rec @0.5"),
            ("obj_iou",  "Obj IoU @0.5"),
            ("mAP",      "mAP (0.50:0.95)"),
            ("px_dice",  "Pixel Dice"),
            ("px_iou",   "Pixel IoU"),
            ("px_prec",  "Pixel Prec"),
            ("px_rec",   "Pixel Rec")]

    print()
    for k, lbl in keys:
        v = [r[k] for r in results]
        print(f"  {lbl:16s}  {100*np.mean(v):6.2f}% ± {100*np.std(v):.2f}%")

    # Per-stack table
    print(f"\n  {'File':40s} {'GT':>4s} {'Pred':>5s} "
          f"{'AJI%':>6s} {'ObjF1':>6s} {'ObjPr':>6s} {'ObjRc':>6s} "
          f"{'ObjIoU':>7s} {'mAP':>6s} {'PxDice':>6s} {'PxIoU':>6s}")
    for r in results:
        print(f"  {r['file']:40s} {r['n_gt']:4d} {r['n_pred']:5d} "
              f"{100*r['aji']:6.2f} {100*r['obj_f1']:6.2f} "
              f"{100*r['obj_prec']:6.2f} {100*r['obj_rec']:6.2f} "
              f"{100*r['obj_iou']:7.2f} {100*r['mAP']:6.2f} "
              f"{100*r['px_dice']:6.2f} {100*r['px_iou']:6.2f}")

    # ── Save results JSON ────────────────────────────────────────────────
    out_json = os.path.join(RESULTS_DIR, f"eval_{args.subset}.json")
    summary = {
        "subset": args.subset,
        "model": "nnU-Net",
        "configuration": config.configuration,
        "aug_preset": config.aug_preset,
        "dataset": config.dataset_id,
        "tta": not args.no_tta,
        "binary_thresh": BINARY_THRESH,
        "min_instance_vol": MIN_VOL,
        "per_stack": results,
        "mean": {k: float(np.mean([r[k] for r in results])) for k, _ in keys},
        "std":  {k: float(np.std([r[k] for r in results]))  for k, _ in keys},
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✅  Results saved: {out_json}")

    # ── Visualization ────────────────────────────────────────────────────
    if not args.no_viz:
        save_dir = args.save_dir or os.path.join(RESULTS_DIR, f"viz_{args.subset}")
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(len(file_list)):
            pred, gt, img = preds_all[idx], Y_raw[idx], X_raw[idx]
            r = results[idx]
            nz = img.shape[0]
            zs = np.linspace(max(0, int(nz * .05)), min(nz - 1, int(nz * .95)),
                             N_SLICES, dtype=int)

            fig, ax = plt.subplots(N_SLICES, 4, figsize=(18, 4 * N_SLICES))
            ax = np.array(ax).reshape(N_SLICES, 4)
            fig.suptitle(
                f"{file_list[idx]}  |  AJI={100*r['aji']:.2f}%  "
                f"F1@0.5={100*r['obj_f1']:.2f}%  "
                f"ObjIoU={100*r['obj_iou']:.2f}%  "
                f"PxIoU={100*r['px_iou']:.2f}%  "
                f"GT={r['n_gt']}  Pred={r['n_pred']}",
                fontsize=12, y=1.0)

            for row, z in enumerate(zs):
                gs, ps = gt[z], pred[z]
                ng  = len(np.unique(gs[gs > 0]))
                np_ = len(np.unique(ps[ps > 0]))

                ax[row, 0].imshow(img[z], cmap="gray")
                ax[row, 0].set_title(f"z={z}/{nz}"); ax[row, 0].axis("off")

                ax[row, 1].imshow(img[z], cmap="gray")
                ax[row, 1].imshow(gs, cmap="viridis", alpha=.7, interpolation="none")
                ax[row, 1].set_title(f"GT (n={ng})"); ax[row, 1].axis("off")

                ax[row, 2].imshow(img[z], cmap="gray")
                ax[row, 2].imshow(ps, cmap="plasma", alpha=.7, interpolation="none")
                ax[row, 2].set_title(f"Pred (n={np_})"); ax[row, 2].axis("off")

                ov = tp_fp_fn_overlay(gs, ps)
                ax[row, 3].imshow(img[z], cmap="gray")
                ax[row, 3].imshow(ov, alpha=.5)
                ax[row, 3].set_title("TP/FP/FN"); ax[row, 3].axis("off")

            plt.tight_layout()
            fig_path = os.path.join(save_dir, f"{Path(file_list[idx]).stem}.png")
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fig_path}")

        print(f"\n  ✅  Figures saved to {save_dir}")

    # ── Z-slice error analysis + save predictions ────────────────────────
    MODEL_NAME = f"nnunet_{config.configuration}_{config.aug_preset}"
    PRED_DIR = os.path.join(DATASET_DIR, "results", MODEL_NAME)
    os.makedirs(PRED_DIR, exist_ok=True)
    print(f"\nResults dir: {PRED_DIR}")

    print(f"\n{'='*65}")
    print(f"  Z-SLICE ERROR ANALYSIS")
    print(f"{'='*65}")

    all_errors = []

    for idx in range(len(file_list)):
        fname = file_list[idx]
        pred = preds_all[idx]
        gt   = Y_raw[idx]
        r    = results[idx]

        print(f"\n── {fname} (GT={r['n_gt']}, Pred={r['n_pred']}) ──")

        # Save prediction TIFF
        pred_tiff_path = os.path.join(PRED_DIR, f"pred_{fname}")
        tif_write(pred_tiff_path, pred.astype(np.uint16))
        print(f"  Saved: {pred_tiff_path}")

        # 3D matching + per-slice errors
        matched_gt, matched_pred, fn_ids, fp_ids = _match_instances_3d(
            gt, pred, thresh=0.5)
        errors = _per_slice_errors(gt, pred, fn_ids, fp_ids)
        all_errors.append(errors)

        print(f"  TP={len(matched_gt)}  FP={len(fp_ids)}  FN={len(fn_ids)}  "
              f"nz={len(errors['dice'])}")

    # ── Aggregated Z-error profile plot ──────────────────────────────────
    max_nz = max(len(e["dice"]) for e in all_errors)
    agg_keys = ["fp_pixels", "fn_pixels", "fp_objects", "fn_objects", "dice"]

    agg = {k: np.zeros(max_nz) for k in agg_keys}
    agg_count = np.zeros(max_nz)

    for errors in all_errors:
        nz = len(errors["dice"])
        for k in agg_keys:
            agg[k][:nz] += errors[k]
        agg_count[:nz] += 1

    for k in agg_keys:
        mask = agg_count > 0
        agg[k][mask] /= agg_count[mask]

    z_ax = np.arange(max_nz)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, max_nz * 0.1)),
                              sharey=True)
    fig.suptitle(f"Z-slice error profile — {MODEL_NAME}\n"
                 f"({len(all_errors)} {args.subset} stacks, mean per-slice)",
                 fontsize=13)

    # 1. Pixel errors
    ax = axes[0]
    ax.plot(agg["fp_pixels"], z_ax, "b-", linewidth=1.5, label="FP pixels")
    ax.plot(agg["fn_pixels"], z_ax, "r-", linewidth=1.5, label="FN pixels")
    ax.fill_betweenx(z_ax, 0, agg["fp_pixels"], color="blue", alpha=0.15)
    ax.fill_betweenx(z_ax, 0, agg["fn_pixels"], color="red", alpha=0.15)
    ax.set_ylabel("Z slice")
    ax.set_xlabel("Mean pixel count")
    ax.set_title("Pixel errors per slice")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # 2. Object errors
    ax = axes[1]
    ax.plot(agg["fp_objects"], z_ax, "b-", linewidth=1.5, label="FP objects")
    ax.plot(agg["fn_objects"], z_ax, "r-", linewidth=1.5, label="FN objects")
    ax.fill_betweenx(z_ax, 0, agg["fp_objects"], color="blue", alpha=0.15)
    ax.fill_betweenx(z_ax, 0, agg["fn_objects"], color="red", alpha=0.15)
    ax.set_xlabel("Mean object count")
    ax.set_title("Object errors per slice")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # 3. Dice
    ax = axes[2]
    ax.plot(agg["dice"], z_ax, "g-", linewidth=1.5)
    ax.fill_betweenx(z_ax, 0, agg["dice"], color="green", alpha=0.15)
    ax.set_xlabel("Mean Dice")
    ax.set_title("Dice per slice")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    plt.tight_layout()
    z_plot_path = os.path.join(PRED_DIR, "z_error_profile.png")
    plt.savefig(z_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {z_plot_path}")
    print(f"All results in: {PRED_DIR}")


if __name__ == "__main__":
    main()