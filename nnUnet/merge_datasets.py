"""
merge_datasets.py — Create cross-domain nnU-Net datasets from existing ones.

Reads cross_domain_experiments from config.py, creates new Dataset folders
by symlinking preprocessed files and building combined splits_final.json.

Usage:
    python merge_datasets.py                # create all experiments from config
    python merge_datasets.py --force        # overwrite existing
    python merge_datasets.py --list         # just show what would be created
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import config


def get_dataset_info(dataset_name: str) -> dict:
    """Load plans, splits, and file_map for an existing dataset."""
    raw_dir    = os.path.join(config.nnunet_raw, dataset_name)
    preproc_dir = os.path.join(config.nnunet_preprocessed, dataset_name)

    # Plans
    plans_path = os.path.join(preproc_dir, "nnUNetPlans.json")
    with open(plans_path) as f:
        plans = json.load(f)

    # Splits
    splits_path = os.path.join(preproc_dir, "splits_final.json")
    with open(splits_path) as f:
        splits = json.load(f)

    # File map (case_id → original filename)
    file_map_path = os.path.join(raw_dir, "file_map.json")
    if os.path.exists(file_map_path):
        with open(file_map_path) as f:
            file_map = json.load(f)
    else:
        file_map = {}

    # dataset.json
    ds_json_path = os.path.join(raw_dir, "dataset.json")

    with open(ds_json_path) as f:
        ds_json = json.load(f)

    # List preprocessed cases
    data_dir = os.path.join(preproc_dir, "nnUNetPlans_3d_fullres")
    cases = set()
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".b2nd") and not f.endswith("_seg.b2nd"):
                cases.add(f.replace(".b2nd", ""))

    return {
        "name": dataset_name,
        "raw_dir": raw_dir,
        "preproc_dir": preproc_dir,
        "data_dir": data_dir,
        "plans": plans,
        "splits": splits,
        "file_map": file_map,
        "ds_json": ds_json,
        "cases": cases,
    }


def create_merged_dataset(experiment: dict, force: bool = False):
    """
    Create a merged dataset for cross-domain training.

    experiment dict keys:
        dataset_id: int           — new dataset ID (e.g., 101)
        dataset_name: str         — e.g., "Dataset101_CrossDomain_RatMice_Dodt"
        train_sources: list[dict] — [{name: "Dataset003_...", use: "train"}]
            use: "train" = use train split, "train+val" = use both, "all" = everything
        test_sources: list[dict]  — [{name: "Dataset001_...", use: "test"}]
            use: "test" = use test split, "val" = val split, "all" = everything
        description: str
    """
    ds_name = experiment["dataset_name"]
    ds_id   = experiment["dataset_id"]

    # Output dirs
    raw_dir     = os.path.join(config.nnunet_raw, ds_name)
    preproc_dir = os.path.join(config.nnunet_preprocessed, ds_name)
    data_dir    = os.path.join(preproc_dir, "nnUNetPlans_3d_fullres")

    if os.path.exists(preproc_dir):
        if force:
            print(f"  ⚠️  Removing existing {preproc_dir}")
            shutil.rmtree(preproc_dir)
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir)
        else:
            print(f"  ❌  {preproc_dir} exists. Use --force to overwrite.")
            return

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ── Collect train cases from sources ─────────────────────────────────
    train_case_ids = []
    val_case_ids   = []
    merged_file_map = {}
    case_counter = 0
    source_plans = None

    for src_spec in experiment["train_sources"]:
        src_info = get_dataset_info(src_spec["name"])
        use = src_spec.get("use", "train")

        if source_plans is None:
            source_plans = src_info["plans"]

        src_splits = src_info["splits"][0]  # fold 0
        src_prefix = src_spec["name"].split("_", 1)[0]  # "Dataset003"

        # Determine which cases to use
        if use == "train":
            cases_for_train = src_splits.get("train", [])
            cases_for_val   = src_splits.get("val", [])
        elif use == "train+val":
            cases_for_train = src_splits.get("train", []) + src_splits.get("val", [])
            cases_for_val   = []
        elif use == "all":
            cases_for_train = list(src_info["cases"])
            cases_for_val   = []
        else:
            raise ValueError(f"Unknown use mode: {use}")

        # Symlink train cases
        for old_id in cases_for_train:
            new_id = f"xd_{case_counter:04d}"
            case_counter += 1

            # Symlink .b2nd and .pkl files
            _symlink_case(src_info["data_dir"], data_dir, old_id, new_id)

            train_case_ids.append(new_id)
            orig_name = src_info["file_map"].get(old_id, old_id)
            merged_file_map[new_id] = f"{src_spec['name']}::{orig_name}"

        # Symlink val cases
        for old_id in cases_for_val:
            new_id = f"xd_{case_counter:04d}"
            case_counter += 1

            _symlink_case(src_info["data_dir"], data_dir, old_id, new_id)

            val_case_ids.append(new_id)
            orig_name = src_info["file_map"].get(old_id, old_id)
            merged_file_map[new_id] = f"{src_spec['name']}::{orig_name}"

    # If no explicit val from sources, take last 15% of train as val
    if not val_case_ids:
        n_val = max(1, int(len(train_case_ids) * 0.15))
        val_case_ids = train_case_ids[-n_val:]
        train_case_ids = train_case_ids[:-n_val]

    # ── Handle test sources (for inference later) ────────────────────────
    test_info = []
    for src_spec in experiment.get("test_sources", []):
        src_info = get_dataset_info(src_spec["name"])
        use = src_spec.get("use", "test")
        src_splits = src_info["splits"][0]

        if use == "test":
            test_cases = src_splits.get("val", []) + src_splits.get("train", [])
            # Actually for inference we test on the test split from split.json
            # We just record which dataset to run inference on
            test_info.append({
                "dataset_name": src_spec["name"],
                "use": use,
                "description": src_spec.get("description", ""),
            })
        else:
            test_info.append({
                "dataset_name": src_spec["name"],
                "use": use,
            })

    # ── Create plans (copy from first source, update dataset name) ───────
    merged_plans = json.loads(json.dumps(source_plans))
    merged_plans["dataset_name"] = ds_name

    with open(os.path.join(preproc_dir, "nnUNetPlans.json"), "w") as f:
        json.dump(merged_plans, f, indent=4)

    # ── Create splits_final.json ─────────────────────────────────────────
    splits = [{"train": train_case_ids, "val": val_case_ids}]
    with open(os.path.join(preproc_dir, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

    # ── Create dataset.json ──────────────────────────────────────────────
    ds_json = {
        "channel_names": {"0": "microscopy"},
        "labels": {"background": 0, "neuron": 1},
        "numTraining": len(train_case_ids) + len(val_case_ids),
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "Tiff3DIO",
        "cross_domain_experiment": True,
        "train_sources": [s["name"] for s in experiment["train_sources"]],
        "test_sources": [s["dataset_name"] if isinstance(s, dict) and "dataset_name" in s
                         else s["name"] for s in experiment.get("test_sources", [])],
        "description": experiment.get("description", ""),
    }
    with open(os.path.join(raw_dir, "dataset.json"), "w") as f:
        json.dump(ds_json, f, indent=4)
    shutil.copy2(
        os.path.join(raw_dir, "dataset.json"),
        os.path.join(preproc_dir, "dataset.json"),
    )
    # ── Save file map + test info ────────────────────────────────────────
    with open(os.path.join(raw_dir, "file_map.json"), "w") as f:
        json.dump(merged_file_map, f, indent=4)

    with open(os.path.join(raw_dir, "test_sources.json"), "w") as f:
        json.dump(test_info, f, indent=4)

    # ── Copy dataset_fingerprint.json from first source ──────────────────
    first_src = get_dataset_info(experiment["train_sources"][0]["name"])
    fp_src = os.path.join(first_src["preproc_dir"], "dataset_fingerprint.json")
    if os.path.exists(fp_src):
        shutil.copy2(fp_src, os.path.join(preproc_dir, "dataset_fingerprint.json"))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  ✅  {ds_name}")
    print(f"  Train: {len(train_case_ids)} cases from "
          f"{[s['name'] for s in experiment['train_sources']]}")
    print(f"  Val:   {len(val_case_ids)} cases")
    print(f"  Test:  {[t['dataset_name'] for t in test_info]}")
    print(f"  Description: {experiment.get('description', '')}")
    print(f"  {'='*60}")


def _symlink_case(src_dir: str, dst_dir: str, old_id: str, new_id: str):
    """Symlink .b2nd, _seg.b2nd, and .pkl files from src to dst with new ID."""
    suffixes = [".b2nd", "_seg.b2nd", ".pkl"]
    for suffix in suffixes:
        src = os.path.join(src_dir, f"{old_id}{suffix}")
        dst = os.path.join(dst_dir, f"{new_id}{suffix}")
        if os.path.exists(src):
            os.symlink(os.path.abspath(src), dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="Just show experiments")
    args = parser.parse_args()

    experiments = config.cross_domain_experiments

    if not experiments:
        print("No cross_domain_experiments defined in config.py")
        return

    for exp in experiments:
        print(f"\n{'━'*60}")
        print(f"  Experiment: {exp['dataset_name']}")
        print(f"  Train: {[s['name'] for s in exp['train_sources']]}")
        print(f"  Test:  {[s.get('name', s.get('dataset_name', '?')) for s in exp.get('test_sources', [])]}")
        print(f"{'━'*60}")

        if args.list:
            continue

        create_merged_dataset(exp, force=args.force)

    if args.list:
        print("\n  (dry run — use without --list to create)")


if __name__ == "__main__":
    main()