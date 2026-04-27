#!/usr/bin/env python3
"""
aggregate_evaluation.py — Collects evaluation CSVs from all dataset×model
combinations into unified summary tables for the Results section.

Usage:
    python aggregate_evaluation.py

Output (saved to BASE_DIR/evaluation_aggregated/):
    - zone_test_all.csv           — within-object start/middle/end Dice per model×dataset
    - boundary_relaxation_all.csv — strict vs relaxed metrics
    - edge_interior_mw_all.csv    — edge vs interior Mann-Whitney
    - depth_spearman_all.csv      — Spearman ρ depth vs error
    - z_profile_summary.csv       — within-object z-profile aggregated by zone
    - metrics_summary_all.csv     — overall metrics per model×dataset
"""

import os
import glob
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit paths if needed
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = "/NAS/mmaiurov/datasets_benchmark"

DATASETS = {
    "rat_neurons_oblique":   {"species": "Rat",   "modality": "Oblique"},
    "rat_neurons_dodt":      {"species": "Rat",   "modality": "DODT"},
    "mice_neurons_oblique":  {"species": "Mice",  "modality": "Oblique"},
    "mice_neurons_dodt":     {"species": "Mice",  "modality": "DODT"},
    "human_neurons_oblique": {"species": "Human", "modality": "Oblique"},
    "human_neurons_dodt":    {"species": "Human", "modality": "DODT"},
}

# CSV files to aggregate (filename → output name)
TARGET_FILES = {
    "within_object_zone_test.csv":       "zone_test",
    "boundary_relaxation_test.csv":      "boundary_relaxation",
    "edge_vs_interior_mannwhitney.csv":  "edge_interior_mw",
    "depth_correlation_spearman.csv":    "depth_spearman",
    "within_object_z_profile.csv":       "z_profile",
    "metrics_summary.csv":               "metrics_summary",
    "per_volume_metrics.csv":            "per_volume_metrics",
    "failure_modes_gt_summary.csv":      "failure_modes_gt",
    "failure_modes_pred_summary.csv":    "failure_modes_pred",
}

OUT_DIR = os.path.join(BASE_DIR, "evaluation_aggregated")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  COLLECT
# ══════════════════════════════════════════════════════════════════════════════

collected = {name: [] for name in TARGET_FILES.values()}

for ds_folder, meta in DATASETS.items():
    ds_path = os.path.join(BASE_DIR, ds_folder)
    eval_dir = os.path.join(ds_path, "results", "evaluation")
    ds_label = f"{meta['species']} {meta['modality']}"

    if not os.path.isdir(eval_dir):
        print(f"[SKIP] {ds_label}: {eval_dir} not found")
        continue

    print(f"[OK]   {ds_label}: {eval_dir}")

    for csv_file, out_name in TARGET_FILES.items():
        fpath = os.path.join(eval_dir, csv_file)
        if os.path.isfile(fpath):
            try:
                df = pd.read_csv(fpath)
                df.insert(0, "dataset", ds_label)
                df.insert(1, "species", meta["species"])
                df.insert(2, "modality", meta["modality"])
                collected[out_name].append(df)
                print(f"       + {csv_file} ({len(df)} rows)")
            except Exception as e:
                print(f"       ! {csv_file}: {e}")
        else:
            print(f"       - {csv_file}: not found")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE & SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"SAVING to {OUT_DIR}")
print(f"{'='*80}\n")

for out_name, frames in collected.items():
    if not frames:
        print(f"  {out_name}: no data")
        continue
    merged = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(OUT_DIR, f"{out_name}_all.csv")
    merged.to_csv(out_path, index=False)
    print(f"  {out_name}_all.csv: {len(merged)} rows, {len(frames)} datasets")

# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

# 1. Zone test summary: compact table for paper
if collected["zone_test"]:
    df_zt = pd.concat(collected["zone_test"], ignore_index=True)
    print("\n" + "="*80)
    print("WITHIN-OBJECT ZONE TEST SUMMARY (start/middle/end Dice)")
    print("="*80)
    for ds in df_zt["dataset"].unique():
        sub = df_zt[df_zt["dataset"] == ds]
        print(f"\n{ds}:")
        print(f"  {'Model':<25s} {'Start':>8s} {'Middle':>8s} {'End':>8s} {'p':>12s} {'Sig':>5s}")
        print(f"  {'-'*65}")
        for _, r in sub.iterrows():
            p_str = f"{r['p']:.1e}" if r['p'] < 0.001 else f"{r['p']:.4f}"
            sig_val = r.get('sig', '')
            sig_str = str(sig_val) if pd.notna(sig_val) else ''
            print(
                f"  {r['model']:<25s} {r.get('start_dice', 0):.4f}   {r.get('middle_dice', 0):.4f}   {r.get('end_dice', 0):.4f}   {p_str:>12s} {sig_str:>5s}")
# 2. Boundary relaxation summary
if collected["boundary_relaxation"]:
    df_br = pd.concat(collected["boundary_relaxation"], ignore_index=True)
    print("\n" + "="*80)
    print("BOUNDARY RELAXATION SUMMARY (strict vs relaxed)")
    print("="*80)

    # Check what columns are available
    cols = df_br.columns.tolist()
    dice_strict_col = [c for c in cols if 'dice' in c.lower() and 'strict' in c.lower() and 'pixel' in c.lower()]
    dice_relaxed_col = [c for c in cols if 'dice' in c.lower() and 'relaxed' in c.lower() and 'pixel' in c.lower()]

    if dice_strict_col and dice_relaxed_col:
        ds_col = dice_strict_col[0]
        dr_col = dice_relaxed_col[0]
        for ds in df_br["dataset"].unique():
            sub = df_br[df_br["dataset"] == ds]
            print(f"\n{ds}:")
            if "model" in sub.columns:
                for mn in sub["model"].unique():
                    ms = sub[sub["model"] == mn]
                    s_mean = ms[ds_col].mean()
                    r_mean = ms[dr_col].mean()
                    delta = r_mean - s_mean
                    print(f"  {mn:<25s} strict={s_mean:.4f}  relaxed={r_mean:.4f}  Δ={delta:+.4f}")
    else:
        print("  Columns not recognized. Available:", cols)

# 3. Z-profile aggregated by zone (if z_profile data available)
if collected["z_profile"]:
    df_zp = pd.concat(collected["z_profile"], ignore_index=True)
    if "zone" in df_zp.columns and "slice_dice" in df_zp.columns:
        print("\n" + "="*80)
        print("Z-PROFILE: MEAN DICE BY ZONE (aggregated)")
        print("="*80)
        agg = df_zp.groupby(["dataset", "model", "zone"])["slice_dice"].agg(["mean", "std", "count"])
        agg = agg.reset_index()
        pivot = agg.pivot_table(index=["dataset", "model"], columns="zone", values="mean")
        if "start" in pivot.columns and "middle" in pivot.columns and "end" in pivot.columns:
            pivot["drop_start"] = pivot["middle"] - pivot["start"]
            pivot["drop_end"] = pivot["middle"] - pivot["end"]
            pivot_path = os.path.join(OUT_DIR, "z_profile_summary.csv")
            pivot.to_csv(pivot_path)
            print(pivot.to_string())
            print(f"\nSaved to {pivot_path}")

print("\n\nDone. All aggregated files in:", OUT_DIR)