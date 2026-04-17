"""
eval_crossdomain.py — Run nnU-Net inference on cross-domain test sources.

Loads a cross-domain trained model and evaluates on each test_source dataset
using that dataset's original split.json test set and original TIF files.

Usage:
    python eval_crossdomain.py                        # eval current config
    python eval_crossdomain.py --no-tta               # without test-time augmentation
    python eval_crossdomain.py --no-viz               # skip visualizations
    python eval_crossdomain.py --experiment 101       # specific experiment by ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import config

# Reuse everything from eval.py
from eval import (
    load_nnunet_model, zscore_normalize, predict_full_volume,
    binary_to_instances, evaluate_volume, tp_fp_fn_overlay,
    _match_instances_3d, _per_slice_errors,
    TAUS, MAP_THRESHOLDS,
)
import numpy as np
import gc, time, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tifffile import imread, imwrite as tif_write
from pathlib import Path

TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_SLICES = 10


def find_experiment(exp_id: int) -> dict | None:
    for exp in config.cross_domain_experiments:
        if exp["dataset_id"] == exp_id:
            return exp
    return None


def get_checkpoint_path(exp: dict) -> str:
    return os.path.join(
        config.nnunet_results,
        exp["dataset_name"],
        f"nnUNetTrainer_Comet__nnUNetPlans__{config.configuration}",
        f"fold_{config.folds[0]}",
        "checkpoint_best.pth",
    )


def get_source_dataset_dir(dataset_name: str) -> str:
    """Map nnU-Net dataset name back to original TIF dataset directory."""
    raw_dir = os.path.join(config.nnunet_raw, dataset_name)
    ds_json_path = os.path.join(raw_dir, "dataset.json")

    if os.path.exists(ds_json_path):
        with open(ds_json_path) as f:
            ds = json.load(f)
        # Check if there's dataset_id that maps to a known path
        mapping = {
            "Dataset001_HumanNeuronsDodt":     "/NAS/mmaiurov/datasets_benchmark/human_neurons_dodt/",
            "Dataset002_MiceNeuronsDodt":      "/NAS/mmaiurov/datasets_benchmark/mice_neurons_dodt/",
            "Dataset003_RatNeuronsDodt":       "/NAS/mmaiurov/datasets_benchmark/rat_neurons_dodt/",
            "Dataset004_HumanNeuronsOblique":  "/NAS/mmaiurov/datasets_benchmark/human_neurons_oblique/",
            "Dataset005_RatNeuronsOblique":    "/NAS/mmaiurov/datasets_benchmark/rat_neurons_oblique/",
            "Dataset006_MiceNeuronsOblique":   "/NAS/mmaiurov/datasets_benchmark/mice_neurons_oblique/",
        }
        return mapping.get(dataset_name, "")
    return ""


def eval_on_source(model, patch_size, intensity_props, device,
                   source_name: str, xd_name: str,
                   no_tta: bool, no_viz: bool, step_size: float):
    """Evaluate cross-domain model on a single source dataset's test set."""

    dataset_dir = get_source_dataset_dir(source_name)
    if not dataset_dir or not os.path.exists(dataset_dir):
        print(f"  ❌  Dataset dir not found for {source_name}: {dataset_dir}")
        return None

    split_path = os.path.join(dataset_dir, "split.json")
    with open(split_path) as f:
        split = json.load(f)
    file_list = split["test"]

    if not file_list:
        print(f"  ⚠️  No test files for {source_name}")
        return None

    print(f"\n  ── {source_name} ({len(file_list)} test stacks) ──")

    mirror_axes = () if no_tta else (0, 1, 2)
    results, preds_all = [], []
    X_raw, Y_raw = [], []

    for i, fname in enumerate(file_list):
        print(f"    Stack {i+1}/{len(file_list)}: {fname}")
        t0 = time.time()

        img = imread(os.path.join(dataset_dir, "images", fname))
        gt  = imread(os.path.join(dataset_dir, "masks", fname))
        if img.ndim == 4: img = img[..., 0]
        if gt.ndim == 4:  gt  = gt[..., 0]
        X_raw.append(img)
        Y_raw.append(gt)

        img_norm = zscore_normalize(img, intensity_props)
        prob = predict_full_volume(model, img_norm, patch_size, device,
                                   step_size=step_size, mirror_axes=mirror_axes)
        pred = binary_to_instances(prob, config.prob_threshold, config.min_instance_volume)
        preds_all.append(pred)

        r = evaluate_volume(gt, pred)
        r["file"] = fname

        results.append(r)
        dt = time.time() - t0
        print(f"    GT={r['n_gt']}  Pred={r['n_pred']}  "
              f"AJI={100*r['aji']:.2f}%  F1@.5={100*r['obj_f1']:.2f}%  "
              f"mAP={100*r['mAP']:.2f}%  PxIoU={100*r['px_iou']:.2f}%  ({dt:.1f}s)")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary for this source ──────────────────────────────────────────
    keys = [("aji", "AJI"), ("obj_f1", "Obj F1 @0.5"), ("obj_prec", "Obj Prec"),
            ("obj_rec", "Obj Rec"), ("obj_iou", "Obj IoU"), ("mAP", "mAP (0.50:0.95)"),
            ("px_dice", "Pixel Dice"), ("px_iou", "Pixel IoU"),
            ("px_prec", "Pixel Prec"), ("px_rec", "Pixel Rec")]

    print(f"\n    Summary ({source_name}):")
    for k, lbl in keys:
        v = [r[k] for r in results]
        print(f"      {lbl:16s}  {100*np.mean(v):6.2f}% ± {100*np.std(v):.2f}%")

    # ── Save results ─────────────────────────────────────────────────────
    MODEL_NAME = f"nnunet_xd_{xd_name}"
    pred_dir = os.path.join(dataset_dir, "results", MODEL_NAME)
    os.makedirs(pred_dir, exist_ok=True)

    # Save JSON
    summary = {
        "source_dataset": source_name,
        "xd_model": xd_name,
        "subset": "test",
        "tta": not no_tta,
        "per_stack": results,
        "mean": {k: float(np.mean([r[k] for r in results])) for k, _ in keys},
        "std":  {k: float(np.std([r[k] for r in results]))  for k, _ in keys},
    }
    json_path = os.path.join(pred_dir, "eval_test.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    ✅  Results: {json_path}")

    # Save pred TIFFs + Z-error analysis
    all_errors = []
    for idx in range(len(file_list)):
        pred = preds_all[idx]
        gt = Y_raw[idx]
        fname = file_list[idx]

        tif_write(os.path.join(pred_dir, f"pred_{fname}"), pred.astype(np.uint16))
        print(f"    Saved: pred_{fname}")

        matched_gt, matched_pred, fn_ids, fp_ids = _match_instances_3d(gt, pred, thresh=0.5)
        errors = _per_slice_errors(gt, pred, fn_ids, fp_ids)
        all_errors.append(errors)

    # ── Slice visualization (same as eval.py) ────────────────────────────
    if not no_viz:
        viz_dir = os.path.join(pred_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)

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
                f"GT={r['n_gt']}  Pred={r['n_pred']}\n"
                f"Model: {xd_name}",
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
            fig_path = os.path.join(viz_dir, f"{Path(file_list[idx]).stem}.png")
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {fig_path}")

    # ── Z-error plot ─────────────────────────────────────────────────────
    max_nz = max(len(e["dice"]) for e in all_errors)
    agg_keys_z = ["fp_pixels", "fn_pixels", "fp_objects", "fn_objects", "dice"]
    agg = {k: np.zeros(max_nz) for k in agg_keys_z}
    agg_count = np.zeros(max_nz)
    for errors in all_errors:
        nz = len(errors["dice"])
        for k in agg_keys_z:
            agg[k][:nz] += errors[k]
        agg_count[:nz] += 1
    for k in agg_keys_z:
        mask = agg_count > 0
        agg[k][mask] /= agg_count[mask]

    z_ax = np.arange(max_nz)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, max_nz * 0.1)), sharey=True)
    fig.suptitle(f"Z-error profile — {MODEL_NAME} on {source_name}", fontsize=13)

    axes[0].plot(agg["fp_pixels"], z_ax, "b-", lw=1.5, label="FP px")
    axes[0].plot(agg["fn_pixels"], z_ax, "r-", lw=1.5, label="FN px")
    axes[0].set_ylabel("Z slice"); axes[0].set_title("Pixel errors"); axes[0].legend(); axes[0].invert_yaxis()

    axes[1].plot(agg["fp_objects"], z_ax, "b-", lw=1.5, label="FP obj")
    axes[1].plot(agg["fn_objects"], z_ax, "r-", lw=1.5, label="FN obj")
    axes[1].set_title("Object errors"); axes[1].legend(); axes[1].invert_yaxis()

    axes[2].plot(agg["dice"], z_ax, "g-", lw=1.5)
    axes[2].set_title("Dice"); axes[2].set_xlim(0, 1); axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(pred_dir, "z_error_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=None,
                        help="Experiment ID (e.g. 101). Default: use current config dataset_id")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--step-size", type=float, default=0.5)
    args = parser.parse_args()

    # Find experiment
    if args.experiment:
        exp = find_experiment(args.experiment)
    else:
        exp = find_experiment(config.nnunet_dataset_id)

    if exp is None:
        print(f"❌  Experiment not found. Available:")
        for e in config.cross_domain_experiments:
            print(f"    {e['dataset_id']}: {e['dataset_name']}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  Cross-Domain Evaluation")
    print(f"  Model: {exp['dataset_name']}")
    print(f"  Description: {exp['description']}")
    print(f"  Test sources: {[s['name'] for s in exp['test_sources']]}")
    print(f"  TTA: {'OFF' if args.no_tta else 'ON'}")
    print(f"{'='*70}")

    # Load model
    ckpt = get_checkpoint_path(exp)
    if not os.path.exists(ckpt):
        print(f"❌  Checkpoint not found: {ckpt}")
        sys.exit(1)

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    model, patch_size, intensity_props = load_nnunet_model(ckpt, device)

    # Eval on each test source
    all_summaries = {}
    for src in exp["test_sources"]:
        src_name = src["name"]
        summary = eval_on_source(
            model, patch_size, intensity_props, device,
            source_name=src_name,
            xd_name=exp["dataset_name"],
            no_tta=args.no_tta,
            no_viz=args.no_viz,
            step_size=args.step_size,
        )
        if summary:
            all_summaries[src_name] = summary

    # ── Final cross-domain summary ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN SUMMARY: {exp['dataset_name']}")
    print(f"{'='*70}")
    print(f"  {'Target Dataset':45s} {'AJI%':>6s} {'F1@.5':>6s} {'PxIoU':>6s} {'mAP':>6s}")

    for name, s in all_summaries.items():
        print(f"  {name:45s} "
              f"{100*s['mean']['aji']:6.2f} "
              f"{100*s['mean']['obj_f1']:6.2f} "
              f"{100*s['mean']['px_iou']:6.2f} "
              f"{100*s['mean']['mAP']:6.2f}")

    # Save combined summary
    combined_path = os.path.join(
        config.nnunet_results, exp["dataset_name"],
        f"crossdomain_eval_summary.json"
    )
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump({
            "experiment": exp,
            "results": {k: v for k, v in all_summaries.items()},
        }, f, indent=2)
    print(f"\n  ✅  Combined summary: {combined_path}")


if __name__ == "__main__":
    main()