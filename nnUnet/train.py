"""
train.py — Run nnU-Net training from config.py (same UX as StarDist/Cellpose).

Usage:
    python train.py              # preprocess + train
    python train.py --skip-prep  # train only (data already preprocessed)

Steps:
  1. Read config.py
  2. Verify dataset is converted (run convert_dataset.py first)
  3. Run nnU-Net preprocessing + planning
  4. Copy splits_final.json to enforce our train/val split
  5. Launch training with Comet logging via custom trainer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import config


def run_cmd(cmd: str, env: dict | None = None):
    """Run a shell command, streaming output."""
    merged_env = {**os.environ, **(env or {})}
    print(f"\n{'─'*60}")
    print(f"  ▶  {cmd}")
    print(f"{'─'*60}\n")
    result = subprocess.run(cmd, shell=True, env=merged_env)
    if result.returncode != 0:
        print(f"\n❌  Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def verify_dataset():
    """Check that convert_dataset.py has been run."""
    ds_dir = os.path.join(config.nnunet_raw, config.nnunet_dataset_name)
    ds_json = os.path.join(ds_dir, "dataset.json")

    if not os.path.exists(ds_json):
        print(f"❌  Dataset not found at {ds_dir}")
        print(f"   Run: python convert_dataset.py")
        sys.exit(1)

    with open(ds_json) as f:
        ds = json.load(f)
    print(f"✅  Dataset: {config.nnunet_dataset_name}")
    print(f"   Cases: {ds['numTraining']}, Labels: {ds['labels']}")
    return ds_dir


def install_custom_trainer():
    """Copy custom trainer to nnU-Net's trainer directory."""
    # Find nnU-Net installation
    import nnunetv2
    nnunet_pkg = Path(nnunetv2.__file__).parent
    trainer_dir = nnunet_pkg / "training" / "nnUNetTrainer"

    src = Path(__file__).parent / "nnUNetTrainer_Comet.py"
    dst = trainer_dir / "nnUNetTrainer_Comet.py"

    if not src.exists():
        print(f"❌  Custom trainer not found: {src}")
        sys.exit(1)

    shutil.copy2(src, dst)
    print(f"✅  Installed custom trainer → {dst}")


def copy_splits_final():
    """Copy our splits_final.json into the preprocessed folder."""
    src_splits = os.path.join(
        config.nnunet_raw, config.nnunet_dataset_name, "splits_final.json"
    )
    dst_dir = os.path.join(config.nnunet_preprocessed, config.nnunet_dataset_name)
    dst_splits = os.path.join(dst_dir, "splits_final.json")

    if not os.path.exists(src_splits):
        print("⚠️  splits_final.json not found in raw dataset, using nnU-Net default splits.")
        return

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src_splits, dst_splits)
    print(f"✅  Copied splits_final.json → {dst_splits}")


def preprocess():
    """Run nnU-Net fingerprint extraction, planning, and preprocessing."""
    ds_id = config.nnunet_dataset_id

    run_cmd(
        f"nnUNetv2_plan_and_preprocess "
        f"-d {ds_id:03d} "
        f"--verify_dataset_integrity "
        f"-np 8"
    )


def train():
    """Run nnU-Net training."""
    ds_id   = config.nnunet_dataset_id
    cfg     = config.configuration
    folds   = config.folds

    # Always use Comet trainer — nnU-Net augmentations are built-in and always active.
    trainer = "nnUNetTrainer_Comet"

    # Pass metadata to trainer via environment variables
    env = {
        "CUDA_VISIBLE_DEVICES":    str(config.gpu_id),
        "NNUNET_DATASET_ID":       config.dataset_id,
        "NNUNET_MODALITY":         config.modality,
        "NNUNET_ANIMAL":           config.animal,
        "NNUNET_VOXEL_SIZE":       str(config.voxel_size),
        "NNUNET_AUG_PRESET":       config.aug_preset,
        "NNUNET_SEG_APPROACH":     config.seg_approach,
        "NNUNET_SEED":             str(config.seed),
    }

    # Comet env vars
    if config.comet_api_key:
        env["COMET_API_KEY"] = config.comet_api_key
    if config.comet_project:
        env["NNUNET_COMET_PROJECT"] = config.comet_project
    if config.comet_workspace:
        env["NNUNET_COMET_WORKSPACE"] = config.comet_workspace

    for fold in folds:
        cmd = (
            f"nnUNetv2_train "
            f"{ds_id:03d} "
            f"{cfg} "
            f"{fold} "
            f"-tr {trainer} "
            f"-p {config.plans}"
        )

        if config.num_epochs != 1000:
            cmd += f" --npz"  # save softmax for ensemble

        run_cmd(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="nnU-Net training from config.py")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip preprocessing (already done)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  nnU-Net Benchmark Training")
    print(f"  Dataset:    {config.dataset_id}")
    print(f"  Modality:   {config.modality}")
    print(f"  Animal:     {config.animal}")
    print(f"  Config:     {config.configuration}")
    print(f"  Aug preset: {config.aug_preset}")
    print(f"  Folds:      {config.folds}")
    print(f"  GPU:        {config.gpu_id}")
    print(f"{'='*60}")

    # Step 1: Verify dataset
    verify_dataset()

    # Step 2: Install custom trainer
    install_custom_trainer()

    # Step 3: Preprocess
    if not args.skip_prep:
        preprocess()

    # Step 4: Copy our split
    copy_splits_final()

    # Step 5: Train
    train()

    print(f"\n{'='*60}")
    print(f"  ✅  Training complete!")
    print(f"  Results: {config.nnunet_results}/{config.nnunet_dataset_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()