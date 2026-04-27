#!/usr/bin/env python3
"""
aggregate_failure_modes.py — Aggregates failure mode CSVs across all datasets
and produces summary tables for Tables 5 (GT perspective) and 6 (Pred perspective).

Input:  evaluation_aggregated/failure_modes_gt_all.csv
        evaluation_aggregated/failure_modes_pred_all.csv
Output: evaluation_aggregated/table5_gt_failure_modes.csv
        evaluation_aggregated/table6_pred_failure_modes.csv
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = "/NAS/mmaiurov/datasets_benchmark/evaluation_aggregated"

# ══════════════════════════════════════════════════════════════════
#  Model name mapping
# ══════════════════════════════════════════════════════════════════

def short_name(m):
    if "nnunet" in m:                              return "nnU-Net"
    if "stardist" in m and "lightaug" in m:        return "StarDist3D (aug)"
    if "stardist" in m and "noaug" in m:           return "StarDist3D (no aug)"
    if "cellpose" in m:                            return "Cellpose"
    if "finetuned" in m or "fine-tuned" in m:      return "\u03BCSAM (fine-tuned)"
    if "pretrained" in m:                          return "\u03BCSAM (pretrained)"
    if "unet3d" in m and "light_aug" in m:         return "3D U-Net (aug)"
    if "unet3d" in m and "no_aug" in m:            return "3D U-Net (no aug)"
    return m

MODEL_ORDER = [
    "nnU-Net",
    "StarDist3D (aug)",
    "StarDist3D (no aug)",
    "Cellpose",
    "3D U-Net (no aug)",
    "3D U-Net (aug)",
    "\u03BCSAM (fine-tuned)",
    "\u03BCSAM (pretrained)",
]

# ══════════════════════════════════════════════════════════════════
#  TABLE 5: GT-perspective failure modes
# ══════════════════════════════════════════════════════════════════

print("=" * 90)
print("TABLE 5: GT-perspective failure modes (% of GT objects, aggregated)")
print("=" * 90)

gt = pd.read_csv(os.path.join(BASE_DIR, "failure_modes_gt_all.csv"))
gt["short"] = gt["model"].apply(short_name)

gt_agg = gt.groupby("short").agg(
    total_gt=("total_gt", "sum"),
    TP_CLEAN=("TP_CLEAN", "sum"),
    TP_BOUNDARY_ERR=("TP_BOUNDARY_ERR", "sum"),
    UNDERSEG=("UNDERSEG", "sum"),
    MISSED=("MISSED", "sum"),
    MERGED=("MERGED", "sum"),
    SPLIT_TARGET=("SPLIT_TARGET", "sum"),
).reset_index()

# Compute percentages
for col in ["TP_CLEAN", "TP_BOUNDARY_ERR", "UNDERSEG", "MISSED", "MERGED", "SPLIT_TARGET"]:
    gt_agg[col + "_pct"] = (gt_agg[col] / gt_agg["total_gt"] * 100).round(1)

# Order
gt_agg["order"] = gt_agg["short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
gt_agg = gt_agg.sort_values("order").drop(columns=["order"])

# Print
print(f"\n{'Model':<25s} {'n':>5s}  {'Clean':>6s}  {'Bound.':>6s}  {'Under':>6s}  {'Missed':>6s}  {'Merged':>6s}  {'Split':>6s}")
print("-" * 90)
for _, r in gt_agg.iterrows():
    print(f"{r['short']:<25s} {r['total_gt']:5.0f}  "
          f"{r['TP_CLEAN_pct']:5.1f}%  {r['TP_BOUNDARY_ERR_pct']:5.1f}%  "
          f"{r['UNDERSEG_pct']:5.1f}%  {r['MISSED_pct']:5.1f}%  "
          f"{r['MERGED_pct']:5.1f}%  {r['SPLIT_TARGET_pct']:5.1f}%")

# Save
t5 = gt_agg[["short", "total_gt",
             "TP_CLEAN_pct", "TP_BOUNDARY_ERR_pct", "UNDERSEG_pct",
             "MISSED_pct", "MERGED_pct", "SPLIT_TARGET_pct"]].copy()
t5.columns = ["Model", "Total GT", "Clean (%)", "Boundary (%)",
              "Underseg. (%)", "Missed (%)", "Merged (%)", "Split target (%)"]
t5.to_csv(os.path.join(BASE_DIR, "table5_gt_failure_modes.csv"), index=False)
print(f"\nSaved: table5_gt_failure_modes.csv")

# ══════════════════════════════════════════════════════════════════
#  Per-dataset breakdown (for supplementary / verification)
# ══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("TABLE 5 — PER DATASET BREAKDOWN")
print("=" * 90)

for ds in gt["dataset"].unique():
    sub = gt[gt["dataset"] == ds].copy()
    sub["order"] = sub["short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    sub = sub.sort_values("order")
    print(f"\n{ds}:")
    print(f"  {'Model':<25s} {'n':>4s}  {'Clean':>6s}  {'Bound.':>6s}  {'Under':>6s}  {'Miss':>6s}  {'Merge':>6s}  {'Split':>6s}")
    for _, r in sub.iterrows():
        t = r["total_gt"]
        print(f"  {r['short']:<25s} {t:4.0f}  "
              f"{r['TP_CLEAN']/t*100:5.1f}%  {r['TP_BOUNDARY_ERR']/t*100:5.1f}%  "
              f"{r['UNDERSEG']/t*100:5.1f}%  {r['MISSED']/t*100:5.1f}%  "
              f"{r['MERGED']/t*100:5.1f}%  {r['SPLIT_TARGET']/t*100:5.1f}%")

# ══════════════════════════════════════════════════════════════════
#  TABLE 6: Prediction-perspective failure modes
# ══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("TABLE 6: Prediction-perspective failure modes (% of predicted objects, aggregated)")
print("=" * 90)

pred = pd.read_csv(os.path.join(BASE_DIR, "failure_modes_pred_all.csv"))
pred["short"] = pred["model"].apply(short_name)

pred_agg = pred.groupby("short").agg(
    total_pred=("total_pred", "sum"),
    TP_CLEAN=("TP_CLEAN", "sum"),
    OVERSEG=("OVERSEG", "sum"),
    SPLIT=("SPLIT", "sum"),
    SPURIOUS=("SPURIOUS", "sum"),
).reset_index()

for col in ["TP_CLEAN", "OVERSEG", "SPLIT", "SPURIOUS"]:
    pred_agg[col + "_pct"] = (pred_agg[col] / pred_agg["total_pred"] * 100).round(1)

pred_agg["order"] = pred_agg["short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
pred_agg = pred_agg.sort_values("order").drop(columns=["order"])

print(f"\n{'Model':<25s} {'n':>6s}  {'Clean':>6s}  {'Overseg':>7s}  {'Split':>6s}  {'Spurious':>8s}")
print("-" * 70)
for _, r in pred_agg.iterrows():
    print(f"{r['short']:<25s} {r['total_pred']:6.0f}  "
          f"{r['TP_CLEAN_pct']:5.1f}%  {r['OVERSEG_pct']:6.1f}%  "
          f"{r['SPLIT_pct']:5.1f}%  {r['SPURIOUS_pct']:7.1f}%")

# Save
t6 = pred_agg[["short", "total_pred",
               "TP_CLEAN_pct", "OVERSEG_pct", "SPLIT_pct", "SPURIOUS_pct"]].copy()
t6.columns = ["Model", "Total Pred", "Clean (%)", "Overseg. (%)",
              "Split (%)", "Spurious (%)"]
t6.to_csv(os.path.join(BASE_DIR, "table6_pred_failure_modes.csv"), index=False)
print(f"\nSaved: table6_pred_failure_modes.csv")

# ══════════════════════════════════════════════════════════════════
#  Per-dataset breakdown (pred)
# ══════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 90)
print("TABLE 6 — PER DATASET BREAKDOWN")
print("=" * 90)

for ds in pred["dataset"].unique():
    sub = pred[pred["dataset"] == ds].copy()
    sub["order"] = sub["short"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    sub = sub.sort_values("order")
    print(f"\n{ds}:")
    print(f"  {'Model':<25s} {'n':>6s}  {'Clean':>6s}  {'Overseg':>7s}  {'Split':>6s}  {'Spur.':>7s}")
    for _, r in sub.iterrows():
        t = r["total_pred"]
        if t == 0:
            print(f"  {r['short']:<25s} {t:6.0f}  {'N/A':>6s}  {'N/A':>7s}  {'N/A':>6s}  {'N/A':>7s}")
            continue
        print(f"  {r['short']:<25s} {t:6.0f}  "
              f"{r['TP_CLEAN']/t*100:5.1f}%  {r['OVERSEG']/t*100:6.1f}%  "
              f"{r['SPLIT']/t*100:5.1f}%  {r['SPURIOUS']/t*100:6.1f}%")

print("\n\nDone.")