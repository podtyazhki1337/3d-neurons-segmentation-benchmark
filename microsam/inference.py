"""
pred_microsam.py — Inference + evaluation for micro-SAM on 3D neuron stacks.

Runs automatic instance segmentation (2D slice-by-slice → 3D merge)
on test files from split.json, computes full benchmark metrics
(AJI, obj F1/prec/rec/IoU, pixel F1/IoU/prec/rec, mAP@0.1-0.9),
and logs everything to Comet ML.

All parameters come from config_microsam.py (no argparse).
"""

from __future__ import annotations

import os
import sys
import json
import time
import gc

import numpy as np
import matplotlib.pyplot as plt
import tifffile

# ── GPU setup ────────────────────────────────────────────────────────────────
import config_microsam as config

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

import torch

print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

np.random.seed(config.seed)
torch.manual_seed(config.seed)

N_SLICES = 10   # Z-slices for visualization


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics (aligned with benchmark_evaluation — Hungarian + proper AP)
# ══════════════════════════════════════════════════════════════════════════════

from scipy.optimize import linear_sum_assignment

IOU_THRESH = 0.5
MAP_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def iou_matrix(gt, pred):
    """IoU matrix via bincount. Returns gt_ids, pred_ids, iou, overlap, gt_vols, pred_vols."""
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
    gt_has_overlap = np.any(iou_mat > 0, axis=1)
    pr_has_overlap = np.any(iou_mat > 0, axis=0)
    gt_idx = np.where(gt_has_overlap)[0]
    pr_idx = np.where(pr_has_overlap)[0]
    if len(gt_idx) == 0 or len(pr_idx) == 0:
        return []
    sub = iou_mat[np.ix_(gt_idx, pr_idx)]
    cost = -sub
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        iou_val = sub[r, c]
        if iou_val >= thresh:
            matches.append((int(gt_idx[r]), int(pr_idx[c]), float(iou_val)))
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
    """Aggregated Jaccard Index (Kumar 2017)."""
    ng, np_ = len(gt_ids), len(pred_ids)
    if ng == 0 and np_ == 0: return 1.
    if ng == 0 or  np_ == 0: return 0.

    used = set()
    si, su = 0, 0
    for i in range(ng):
        bj, bv = -1, 0.
        for j in range(np_):
            if j in used:
                continue
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
    """Average Precision at a single IoU threshold (101-point interpolation)."""
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
            if matched_gt[i]:
                continue
            if iou_mat[i, j] > best_iou:
                best_iou = iou_mat[i, j]
                best_i = i
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
    """mAP@0.5:0.95 (COCO-style)."""
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
    pr   = tp / (tp + fp) if tp + fp > 0 else 1.
    rc   = tp / (tp + fn) if tp + fn > 0 else 1.
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.
    return dict(pixel_dice=dice, pixel_iou=iou, pixel_prec=pr, pixel_rec=rc)


def tp_fp_fn_overlay(gs, ps):
    o = np.zeros((*gs.shape, 3), dtype=np.float32)
    g, p = gs > 0, ps > 0
    o[g & p]  = [0, 1, 0]   # TP = green
    o[g & ~p] = [1, 0, 0]   # FN = red
    o[~g & p] = [0, 0, 1]   # FP = blue
    return o


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_tif(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 4:
        img = img[..., 0]
    if img.ndim == 2:
        img = img[np.newaxis]
    return img


def normalize_percentile(img, low=1.0, high=99.8):
    """Percentile normalization to [0, 255] uint8 (micro-sam expects this)."""
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(img_norm * 255, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )

    TAUS = config.iou_thresholds

    print("\n" + "=" * 60)
    print(f"  micro-SAM inference — {config.run_name}")
    print(f"  Model:    {config.model_type}")
    print(f"  Checkpoint: {'pretrained' if config.checkpoint_path is None else config.checkpoint_path}")
    print(f"  Mode:     {config.segmentation_mode}")
    print(f"  Dataset:  {config.dataset_id} ({config.modality})")
    print("=" * 60 + "\n")

    # ── Comet ML ─────────────────────────────────────────────────────────
    experiment = None
    try:
        from comet_ml import Experiment
        experiment = Experiment(
            api_key=config.comet_api_key,
            workspace=config.comet_workspace,
            project_name=config.comet_project,
        )
        experiment.add_tag(config.modality)
        experiment.add_tag(config.animal)
        experiment.add_tag(config.model_type)
        experiment.add_tag("pretrained" if config.checkpoint_path is None else "finetuned")
        experiment.add_tag("inference")

        experiment.log_parameters({
            "model_type":         config.model_type,
            "segmentation_mode":  config.segmentation_mode,
            "checkpoint_path":    str(config.checkpoint_path),
            "dataset_id":         config.dataset_id,
            "modality":           config.modality,
            "animal":             config.animal,
            "voxel_size":         str(config.voxel_size),
            "predict_on":         config.predict_on,
            "gap_closing":        config.gap_closing,
            "min_z_extent":       config.min_z_extent,
            "tile_shape":         str(config.tile_shape),
            "halo":               str(config.halo),
            "seed":               config.seed,
            "run_name":           config.run_name,
        })
        try:
            experiment.log_code(file_name="config_microsam.py")
        except Exception:
            pass
    except Exception as e:
        print(f"Comet init failed: {e}")

    # ── Load split ───────────────────────────────────────────────────────
    split_path = os.path.join(config.in_dataset_dir, "split.json")
    with open(split_path) as f:
        split = json.load(f)

    if config.predict_on == "all":
        file_list = split["train"] + split.get("val", []) + split["test"]
    else:
        file_list = split[config.predict_on]

    print(f"Predicting on {len(file_list)} files ({config.predict_on} split)")

    if experiment:
        experiment.log_parameter("n_files", len(file_list))
        experiment.log_asset(split_path, file_name="split.json")

    # ── Load all data ────────────────────────────────────────────────────
    X_raw, Y_raw = [], []
    for fn in file_list:
        X_raw.append(load_tif(os.path.join(config.in_dataset_dir, "images", fn)))
        Y_raw.append(load_tif(os.path.join(config.in_dataset_dir, "masks", fn)))

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading micro-SAM model: {config.model_type} ...")
    t0 = time.time()

    is_tiled = config.tile_shape is not None
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=config.model_type,
        checkpoint=config.checkpoint_path,
        segmentation_mode=config.segmentation_mode,
        is_tiled=is_tiled,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── Output directories ────────────────────────────────────────────────
    # NAS: predictions + vis + error analysis
    RESULTS_DIR = os.path.join(config.in_dataset_dir, "results", config.run_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Local: lightweight copy
    out_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Predict + evaluate ───────────────────────────────────────────────
    results, preds_all = [], []
    gc.collect()

    for i in range(len(X_raw)):
        print(f"\nStack {i+1}/{len(X_raw)}: {file_list[i]}")
        t1 = time.time()

        raw_norm = normalize_percentile(X_raw[i])

        seg_kwargs = {}
        if config.tile_shape is not None:
            seg_kwargs["tile_shape"] = config.tile_shape
            seg_kwargs["halo"] = config.halo

        pred = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=raw_norm,
            ndim=3,
            batch_size=config.embedding_batch_size,
            gap_closing=config.gap_closing,
            min_z_extent=config.min_z_extent,
            **seg_kwargs,
        )

        dt = time.time() - t1
        preds_all.append(pred)
        gt = Y_raw[i]

        # ── All metrics (aligned with benchmark_evaluation) ─────────────────
        gt_ids, pred_ids, iou_mat, overlap, gt_vols, pred_vols = iou_matrix(gt, pred)
        n_gt, n_pred = len(gt_ids), len(pred_ids)

        obj = obj_metrics_at_thresh(iou_mat, n_gt, n_pred, IOU_THRESH)
        aji = compute_aji(gt_ids, pred_ids, iou_mat, overlap, gt_vols, pred_vols)
        mAP = compute_mAP(iou_mat, n_gt, n_pred)
        px  = pixel_metrics(gt, pred)

        r = dict(
            file=file_list[i], n_gt=n_gt, n_pred=n_pred,
            aji=aji,
            obj_f1=obj["f1"], obj_prec=obj["precision"],
            obj_rec=obj["recall"], obj_iou=obj["mean_iou"],
            mAP=mAP,
            pixel_dice=px["pixel_dice"], pixel_iou=px["pixel_iou"],
            pixel_prec=px["pixel_prec"], pixel_rec=px["pixel_rec"],
            time_sec=dt,
        )

        # F1 at all IoU thresholds (for per-threshold analysis)
        for t in TAUS:
            obj_t = obj_metrics_at_thresh(iou_mat, n_gt, n_pred, t)
            r[f"f1_{t:.1f}"] = obj_t["f1"]

        results.append(r)
        print(f"  GT={n_gt}  Pred={n_pred}  "
              f"AJI={100*aji:.2f}%  F1@.5={100*obj['f1']:.2f}%  "
              f"mAP={100*mAP:.2f}%  PxIoU={100*px['pixel_iou']:.2f}%  ({dt:.1f}s)")

        if experiment:
            experiment.log_metrics({
                f"per_stack/{file_list[i]}/aji":        aji,
                f"per_stack/{file_list[i]}/obj_f1":     obj["f1"],
                f"per_stack/{file_list[i]}/obj_prec":   obj["precision"],
                f"per_stack/{file_list[i]}/obj_rec":    obj["recall"],
                f"per_stack/{file_list[i]}/obj_iou":    obj["mean_iou"],
                f"per_stack/{file_list[i]}/mAP":        mAP,
                f"per_stack/{file_list[i]}/pixel_dice": px["pixel_dice"],
                f"per_stack/{file_list[i]}/pixel_iou":  px["pixel_iou"],
                f"per_stack/{file_list[i]}/n_pred":     n_pred,
                f"per_stack/{file_list[i]}/n_gt":       n_gt,
                f"per_stack/{file_list[i]}/time_sec":   dt,
            })

        # Save prediction TIF
        out_path = os.path.join(RESULTS_DIR, file_list[i])
        tifffile.imwrite(out_path, pred.astype(np.uint16), compression="zlib")

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {config.predict_on} ({len(X_raw)} stacks)")
    print(f"{'='*70}")

    keys = [("aji",        "AJI"),
            ("obj_f1",     "Obj F1 @0.5"),
            ("obj_prec",   "Obj Prec @0.5"),
            ("obj_rec",    "Obj Rec @0.5"),
            ("obj_iou",    "Obj IoU @0.5"),
            ("mAP",        "mAP @0.5:0.95"),
            ("pixel_dice", "Pixel Dice"),
            ("pixel_iou",  "Pixel IoU"),
            ("pixel_prec", "Pixel Prec"),
            ("pixel_rec",  "Pixel Rec")]

    summary = {}
    print()
    for k, lbl in keys:
        v = [r[k] for r in results]
        mean_v, std_v = np.mean(v), np.std(v)
        print(f"  {lbl:16s}  {100*mean_v:6.2f}% +/- {100*std_v:.2f}%")
        summary[k] = {"mean": float(mean_v), "std": float(std_v)}

    # Per-stack table
    print(f"\n  {'File':40s} {'GT':>4s} {'Pred':>5s} "
          f"{'AJI%':>6s} {'ObjF1':>6s} {'ObjPr':>6s} {'ObjRc':>6s} "
          f"{'ObjIoU':>7s} {'mAP':>6s} {'Dice':>5s} {'PxIoU':>6s} {'PxPr':>5s} {'PxRc':>5s}")
    for r in results:
        print(f"  {r['file']:40s} {r['n_gt']:4d} {r['n_pred']:5d} "
              f"{100*r['aji']:6.2f} {100*r['obj_f1']:6.2f} "
              f"{100*r['obj_prec']:6.2f} {100*r['obj_rec']:6.2f} "
              f"{100*r['obj_iou']:7.2f} {100*r['mAP']:6.2f} "
              f"{100*r['pixel_dice']:5.2f} {100*r['pixel_iou']:6.2f} "
              f"{100*r['pixel_prec']:5.2f} {100*r['pixel_rec']:5.2f}")

    # ── Log summary to Comet ─────────────────────────────────────────────
    if experiment:
        for k, v in summary.items():
            if isinstance(v, dict):
                experiment.log_metric(f"summary/{k}_mean", v["mean"])
                experiment.log_metric(f"summary/{k}_std",  v["std"])
            else:
                experiment.log_metric(f"summary/{k}", v)

        for t in TAUS:
            vals = [r[f"f1_{t:.1f}"] for r in results]
            experiment.log_metric(f"dataset/F1@{t:.1f}_mean", float(np.mean(vals)))

    # ══════════════════════════════════════════════════════════════════════
    #  Results dir on NAS (predictions + visualizations + error analysis)
    # ══════════════════════════════════════════════════════════════════════
    # ── Save results JSON ────────────────────────────────────────────────
    results_out = {
        "run_name":    config.run_name,
        "model_type":  config.model_type,
        "checkpoint":  str(config.checkpoint_path),
        "segmentation_mode": config.segmentation_mode,
        "dataset_id":  config.dataset_id,
        "predict_on":  config.predict_on,
        "per_stack":   results,
        "summary":     summary,
    }
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    if experiment:
        experiment.log_asset(results_path, file_name="results.json")

    # ══════════════════════════════════════════════════════════════════════
    #  Visualization
    # ══════════════════════════════════════════════════════════════════════
    for idx in range(len(X_raw)):
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
            f"mAP={100*r['mAP']:.2f}%  "
            f"Dice={100*r['pixel_dice']:.2f}%  "
            f"GT={r['n_gt']}  Pred={r['n_pred']}",
            fontsize=12, y=1.0)

        for row, z in enumerate(zs):
            gs, ps = gt[z], pred[z]
            ng  = len(np.unique(gs[gs > 0]))
            np_ = len(np.unique(ps[ps > 0]))

            ax[row, 0].imshow(img[z], cmap='gray')
            ax[row, 0].set_title(f"z={z}/{nz}"); ax[row, 0].axis('off')

            ax[row, 1].imshow(img[z], cmap='gray')
            ax[row, 1].imshow(gs, cmap='viridis', alpha=.7, interpolation='none')
            ax[row, 1].set_title(f"GT (n={ng})"); ax[row, 1].axis('off')

            ax[row, 2].imshow(img[z], cmap='gray')
            ax[row, 2].imshow(ps, cmap='plasma', alpha=.7, interpolation='none')
            ax[row, 2].set_title(f"Pred (n={np_})"); ax[row, 2].axis('off')

            ov = tp_fp_fn_overlay(gs, ps)
            ax[row, 3].imshow(img[z], cmap='gray')
            ax[row, 3].imshow(ov, alpha=.5)
            ax[row, 3].set_title("TP/FP/FN"); ax[row, 3].axis('off')

        plt.tight_layout()

        fig_path = os.path.join(RESULTS_DIR,
                                f"vis_{file_list[idx].replace('.tif', '.png')}")
        fig.savefig(fig_path, dpi=120, bbox_inches='tight')
        print(f"Saved visualization: {fig_path}")

        if experiment:
            experiment.log_image(fig_path, name=file_list[idx])

        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    #  Z-slice error analysis + save predictions
    # ══════════════════════════════════════════════════════════════════════

    def match_instances_3d(gt, pred, thresh=0.5):
        """3D IoU matching → matched/unmatched ID sets."""
        gt_ids   = set(np.unique(gt));   gt_ids.discard(0)
        pred_ids = set(np.unique(pred)); pred_ids.discard(0)

        if not gt_ids or not pred_ids:
            return set(), set(), gt_ids, pred_ids

        gt_list   = sorted(gt_ids)
        pred_list = sorted(pred_ids)
        n_gt, n_pred = len(gt_list), len(pred_list)

        gt_remap   = np.zeros(max(gt_list) + 1, dtype=np.int32)
        pred_remap = np.zeros(max(pred_list) + 1, dtype=np.int32)
        for i, gid in enumerate(gt_list, 1):   gt_remap[gid] = i
        for j, pid in enumerate(pred_list, 1): pred_remap[pid] = j

        n_p1  = np.int64(n_pred + 1)
        codes = gt_remap[gt.ravel()].astype(np.int64) * n_p1 + \
                pred_remap[pred.ravel()].astype(np.int64)
        table = np.bincount(codes, minlength=int((n_gt + 1) * n_p1)) \
                  .reshape(n_gt + 1, int(n_p1))

        overlap   = table[1:, 1:].astype(np.float64)
        gt_vols   = table[1:, :].sum(1).astype(np.float64)
        pred_vols = table[:, 1:].sum(0).astype(np.float64)
        union     = gt_vols[:, None] + pred_vols[None, :] - overlap
        iou       = np.where(union > 0, overlap / union, 0.0)

        matched_gt, matched_pred = set(), set()
        w = iou.copy()
        while True:
            idx = np.unravel_index(np.argmax(w), w.shape)
            if w[idx] < thresh:
                break
            matched_gt.add(gt_list[idx[0]])
            matched_pred.add(pred_list[idx[1]])
            w[idx[0], :] = 0.0
            w[:, idx[1]] = 0.0

        fn_ids = gt_ids - matched_gt
        fp_ids = pred_ids - matched_pred
        return matched_gt, matched_pred, fn_ids, fp_ids

    def per_slice_errors(gt, pred, fn_ids, fp_ids):
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

    # ── Run error analysis ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Z-SLICE ERROR ANALYSIS")
    print(f"{'='*65}")

    all_errors = []

    for idx in range(len(X_raw)):
        fname = file_list[idx]
        pred = preds_all[idx]
        gt   = Y_raw[idx]
        r    = results[idx]

        print(f"\n── {fname} (GT={r['n_gt']}, Pred={r['n_pred']}) ──")

        # 3D matching + per-slice errors
        matched_gt, matched_pred, fn_ids, fp_ids = match_instances_3d(
            gt, pred, thresh=0.5)
        errors = per_slice_errors(gt, pred, fn_ids, fp_ids)
        all_errors.append(errors)

        print(f"  TP={len(matched_gt)}  FP={len(fp_ids)}  FN={len(fn_ids)}  "
              f"nz={len(errors['dice'])}")

    # ── Aggregated Z-error profile plot ───────────────────────────────────
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
    fig.suptitle(f"Z-slice error profile — {config.run_name}\n"
                 f"({len(all_errors)} test stacks, mean per-slice)",
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
    z_profile_path = os.path.join(RESULTS_DIR, "z_error_profile.png")
    fig.savefig(z_profile_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {z_profile_path}")

    if experiment:
        experiment.log_image(z_profile_path, name="z_error_profile")

    plt.close(fig)

    print(f"All results in: {RESULTS_DIR}")

    if experiment:
        experiment.end()
        print("Comet experiment ended.")


if __name__ == "__main__":
    main()