"""
train_microsam.py — Fine-tune micro-SAM on 3D neuron stacks.

Two-stage training:
  1. Fine-tune SAM (encoder + mask decoder) via train_sam
  2. Train UNETR instance segmentation decoder via train_instance_segmentation

3D stacks are sliced into 2D images automatically.
Uses split.json for train/val consistency with other models.
All parameters from config_microsam.py; logs to Comet ML.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import subprocess
from glob import glob

import numpy as np
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


# ══════════════════════════════════════════════════════════════════════════════
#  Step 0: Extract 2D slices from 3D TIF stacks
# ══════════════════════════════════════════════════════════════════════════════

def normalize_percentile(img, low=1.0, high=99.8):
    """Percentile normalization to [0, 255] uint8."""
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(img_norm * 255, 0, 255).astype(np.uint8)


def prepare_2d_slices(dataset_dir, split, slices_dir, min_labels=1):
    """Extract 2D slices from 3D TIF stacks for micro-SAM training.

    Creates:
        slices_dir/train/images/*.tif
        slices_dir/train/masks/*.tif
        slices_dir/val/images/*.tif
        slices_dir/val/masks/*.tif

    Returns dict with paths and counts.
    """
    info = {}

    for subset in ["train", "val"]:
        files = split.get(subset, [])
        if not files:
            print(f"  Warning: no files in '{subset}' split")
            continue

        img_dir = os.path.join(slices_dir, subset, "images")
        msk_dir = os.path.join(slices_dir, subset, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        # Check if already prepared
        existing = glob(os.path.join(img_dir, "*.tif"))
        if len(existing) > 0:
            print(f"  {subset}: {len(existing)} slices already exist, skipping extraction")
            info[subset] = {"n_slices": len(existing), "img_dir": img_dir, "msk_dir": msk_dir}
            continue

        n_slices = 0
        for fname in files:
            img_path = os.path.join(dataset_dir, "images", fname)
            msk_path = os.path.join(dataset_dir, "masks", fname)

            img_3d = tifffile.imread(img_path)
            msk_3d = tifffile.imread(msk_path)

            # Handle 4D (Z, Y, X, C)
            if img_3d.ndim == 4:
                img_3d = img_3d[..., 0]
            if msk_3d.ndim == 4:
                msk_3d = msk_3d[..., 0]

            # Normalize whole stack, then slice
            img_norm = normalize_percentile(img_3d)

            stem = fname.replace(".tif", "")
            for z in range(img_3d.shape[0]):
                msk_slice = msk_3d[z]
                n_labels = len(np.unique(msk_slice)) - 1  # exclude bg
                if n_labels < min_labels:
                    continue

                slice_name = f"{stem}_z{z:04d}.tif"
                tifffile.imwrite(os.path.join(img_dir, slice_name),
                                 img_norm[z])
                tifffile.imwrite(os.path.join(msk_dir, slice_name),
                                 msk_slice.astype(np.int32))
                n_slices += 1

        info[subset] = {"n_slices": n_slices, "img_dir": img_dir, "msk_dir": msk_dir}
        print(f"  {subset}: extracted {n_slices} slices from {len(files)} stacks")

    return info


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from micro_sam.training import (
        train_sam,
        train_instance_segmentation,
        default_sam_loader,
    )

    train_unetr = getattr(config, "train_unetr", False)

    print("\n" + "=" * 60)
    print(f"  micro-SAM fine-tuning — {config.model_save_name}")
    print(f"  Base model: {config.model_type}")
    print(f"  Dataset:    {config.dataset_id} ({config.modality})")
    print(f"  Stage 2 (UNETR): {'ON' if train_unetr else 'OFF'}")
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
        experiment.add_tag("finetuning")

        experiment.log_parameters({
            "model_type":           config.model_type,
            "model_save_name":      config.model_save_name,
            "dataset_id":           config.dataset_id,
            "modality":             config.modality,
            "animal":               config.animal,
            "voxel_size":           str(config.voxel_size),
            "sam_n_epochs":         config.sam_n_epochs,
            "sam_lr":               config.sam_lr,
            "sam_early_stopping":   config.sam_early_stopping,
            "sam_batch_size":       config.sam_batch_size,
            "sam_patch_shape":      str(config.sam_patch_shape),
            "sam_freeze":           str(config.sam_freeze),
            "train_unetr":          train_unetr,
            "unetr_n_epochs":       config.unetr_n_epochs if train_unetr else 0,
            "unetr_lr":             config.unetr_lr if train_unetr else 0,
            "min_labels_per_slice": config.min_labels_per_slice,
            "seed":                 config.seed,
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

    print(f"Split: train={len(split['train'])}, val={len(split.get('val', []))}, "
          f"test={len(split['test'])}")

    if experiment:
        experiment.log_asset(split_path, file_name="split.json")
        experiment.log_parameter("n_train_stacks", len(split["train"]))
        experiment.log_parameter("n_val_stacks", len(split.get("val", [])))

    # ── Prepare 2D slices ────────────────────────────────────────────────
    print("\nPreparing 2D slices from 3D stacks...")
    slices_dir = config.slices_dir
    slice_info = prepare_2d_slices(
        dataset_dir=config.in_dataset_dir,
        split=split,
        slices_dir=slices_dir,
        min_labels=config.min_labels_per_slice,
    )

    if experiment:
        for subset, info in slice_info.items():
            experiment.log_parameter(f"n_{subset}_slices", info["n_slices"])

    # ── Directory paths ──────────────────────────────────────────────────
    train_img_dir = slice_info["train"]["img_dir"]
    train_msk_dir = slice_info["train"]["msk_dir"]
    val_img_dir   = slice_info["val"]["img_dir"]
    val_msk_dir   = slice_info["val"]["msk_dir"]

    n_train = len(glob(os.path.join(train_img_dir, "*.tif")))
    n_val   = len(glob(os.path.join(val_img_dir, "*.tif")))
    print(f"\nTrain: {n_train} slices")
    print(f"Val:   {n_val} slices")

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 1: Fine-tune SAM (encoder + mask decoder)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Stage 1: Fine-tune SAM encoder + mask decoder")
    print("=" * 60 + "\n")

    sam_train_loader = default_sam_loader(
        raw_paths=train_img_dir,
        raw_key="*.tif",
        label_paths=train_msk_dir,
        label_key="*.tif",
        patch_shape=config.sam_patch_shape,
        batch_size=config.sam_batch_size,
        with_segmentation_decoder=True,
        is_train=True,
        n_samples=config.n_samples_train,
        shuffle=True,
        num_workers=4,
    )

    sam_val_loader = default_sam_loader(
        raw_paths=val_img_dir,
        raw_key="*.tif",
        label_paths=val_msk_dir,
        label_key="*.tif",
        patch_shape=config.sam_patch_shape,
        batch_size=config.sam_batch_size,
        with_segmentation_decoder=True,
        is_train=False,
        n_samples=config.n_samples_val,
        shuffle=False,
        num_workers=4,
    )

    t0 = time.time()

    train_sam(
        name=config.model_save_name,
        model_type=config.model_type,
        train_loader=sam_train_loader,
        val_loader=sam_val_loader,
        n_epochs=config.sam_n_epochs,
        lr=config.sam_lr,
        early_stopping=config.sam_early_stopping,
        n_objects_per_batch=config.sam_n_objects_per_batch,
        with_segmentation_decoder=True,
        freeze=config.sam_freeze,
        save_root="./checkpoints",
    )

    sam_time = time.time() - t0
    print(f"\nStage 1 completed in {sam_time / 60:.1f} min")

    if experiment:
        experiment.log_metric("sam_train_time_min", sam_time / 60)

    # ── Find SAM checkpoint ──────────────────────────────────────────────
    sam_checkpoint = os.path.join(
        "./checkpoints", "checkpoints", config.model_save_name, "best.pt"
    )
    if not os.path.exists(sam_checkpoint):
        result = subprocess.run(
            ["find", "./checkpoints", "-name", "best.pt", "-path",
             f"*{config.model_save_name}*", "-not", "-path", "*unetr*"],
            capture_output=True, text=True
        )
        candidates = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if candidates:
            sam_checkpoint = candidates[0]
            print(f"  Found checkpoint via search: {sam_checkpoint}")
        else:
            raise FileNotFoundError(
                f"Cannot find best.pt for {config.model_save_name} under ./checkpoints/"
            )
    print(f"SAM checkpoint: {sam_checkpoint}")

    # ══════════════════════════════════════════════════════════════════════
    #  Stage 2 (optional): Train UNETR instance segmentation decoder
    # ══════════════════════════════════════════════════════════════════════
    unetr_time = 0.0
    export_path = sam_checkpoint   # default: use SAM checkpoint directly

    if train_unetr:
        print("\n" + "=" * 60)
        print("  Stage 2: Train UNETR instance segmentation decoder")
        print("=" * 60 + "\n")

        class _StripInstanceChannel:
            """Wraps a dataloader to drop the instance channel from 4ch → 3ch targets."""
            def __init__(self, loader, is_shuffle):
                self.loader = loader
                self.dataset = loader.dataset
                self.shuffle = is_shuffle
            def __iter__(self):
                for x, y in self.loader:
                    yield x, y[:, 1:]
            def __len__(self):
                return len(self.loader)

        _unetr_train = default_sam_loader(
            raw_paths=train_img_dir,
            raw_key="*.tif",
            label_paths=train_msk_dir,
            label_key="*.tif",
            patch_shape=config.unetr_patch_shape,
            batch_size=config.unetr_batch_size,
            with_segmentation_decoder=True,
            is_train=True,
            n_samples=config.n_samples_train,
            shuffle=True,
            num_workers=4,
        )

        _unetr_val = default_sam_loader(
            raw_paths=val_img_dir,
            raw_key="*.tif",
            label_paths=val_msk_dir,
            label_key="*.tif",
            patch_shape=config.unetr_patch_shape,
            batch_size=config.unetr_batch_size,
            with_segmentation_decoder=True,
            is_train=False,
            n_samples=config.n_samples_val,
            shuffle=False,
            num_workers=4,
        )

        unetr_train_loader = _StripInstanceChannel(_unetr_train, is_shuffle=True)
        unetr_val_loader   = _StripInstanceChannel(_unetr_val,   is_shuffle=False)

        t1 = time.time()

        train_instance_segmentation(
            name=config.model_save_name + "_unetr",
            model_type=config.model_type,
            train_loader=unetr_train_loader,
            val_loader=unetr_val_loader,
            n_epochs=config.unetr_n_epochs,
            lr=config.unetr_lr,
            early_stopping=config.unetr_early_stopping,
            checkpoint_path=sam_checkpoint,
            save_root="./checkpoints",
        )

        unetr_time = time.time() - t1
        print(f"\nStage 2 completed in {unetr_time / 60:.1f} min")

        if experiment:
            experiment.log_metric("unetr_train_time_min", unetr_time / 60)

        # Export combined model
        from micro_sam.training import export_instance_segmentation_model

        unetr_checkpoint = os.path.join(
            "./checkpoints", "checkpoints", config.model_save_name + "_unetr", "best.pt"
        )
        if not os.path.exists(unetr_checkpoint):
            result = subprocess.run(
                ["find", "./checkpoints", "-name", "best.pt", "-path", "*_unetr*"],
                capture_output=True, text=True
            )
            candidates = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if candidates:
                unetr_checkpoint = candidates[0]

        export_path = os.path.join("./checkpoints", "exported_model.pt")
        try:
            export_instance_segmentation_model(
                checkpoint_path=unetr_checkpoint,
                export_path=export_path,
                model_type=config.model_type,
            )
            print(f"\nExported model: {export_path}")
        except Exception as e:
            print(f"\nExport failed: {e}")
            export_path = sam_checkpoint
    else:
        print("\nSkipping Stage 2 (train_unetr=False)")
        print(f"SAM checkpoint already has decoder_state for AIS inference.")

    if experiment:
        experiment.log_metric("total_train_time_min", (sam_time + unetr_time) / 60)
        experiment.log_parameter("exported_model_path", export_path)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = sam_time + unetr_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  SAM fine-tune:   {sam_time / 60:.1f} min")
    if train_unetr:
        print(f"  UNETR decoder:   {unetr_time / 60:.1f} min")
    print(f"  Total:           {total_time / 60:.1f} min")
    print(f"")
    print(f"  Model for inference: {export_path}")
    print(f"")
    print(f"  To run inference:")
    print(f"    1. Set checkpoint_path = \"{export_path}\" in config_microsam.py")
    print(f"    2. Set segmentation_mode = \"ais\"")
    print(f"    3. Run: python pred_microsam.py")
    print(f"{'='*60}")

    if experiment:
        experiment.end()
        print("Comet experiment ended.")


if __name__ == "__main__":
    main()