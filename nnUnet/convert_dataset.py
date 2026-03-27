"""
convert_dataset.py — Convert benchmark TIF dataset to nnU-Net v2 format.

Reads split.json from the source dataset, copies train+val images/masks
into nnU-Net's DatasetXXX structure, and creates the required metadata.

Usage:
    python convert_dataset.py          # uses config.py settings
    python convert_dataset.py --force  # overwrite existing dataset

What it does:
  1. Reads split.json (train + val → imagesTr/labelsTr, test → imagesTs)
  2. Converts instance masks → binary (or 3-class) for semantic segmentation
  3. Creates per-file .json with spacing info (required for TIF in nnU-Net)
  4. Creates dataset.json
  5. Creates splits_final.json to match our train/val split exactly
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_split(dataset_dir: str, seed: int = 42,
                        val_ratio: float = 0.15, test_ratio: float = 0.1,
                        skip=None, force_val=None, force_test=None) -> dict:
    """Load existing split.json or create a new one (identical to StarDist pipeline)."""
    split_path = os.path.join(dataset_dir, "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        print(f"✅  Loaded existing split.json: "
              f"{len(split['train'])} train, {len(split['val'])} val, {len(split['test'])} test")
        return split

    # Create new split
    skip = skip or []
    force_val = force_val or []
    force_test = force_test or []

    img_dir = os.path.join(dataset_dir, "images")
    msk_dir = os.path.join(dataset_dir, "masks")
    all_imgs = sorted(f for f in os.listdir(img_dir)
                      if f.endswith((".tif", ".tiff")) and f not in skip)
    all_masks = set(os.listdir(msk_dir))
    files = [f for f in all_imgs if f in all_masks]

    rng = np.random.RandomState(seed)
    remaining = [f for f in files if f not in force_val and f not in force_test]
    rng.shuffle(remaining)

    n_test = max(1, int(len(files) * test_ratio))
    n_val  = max(1, int(len(files) * val_ratio))

    n_test_extra = max(0, n_test - len(force_test))
    n_val_extra  = max(0, n_val  - len(force_val))

    test  = force_test + remaining[:n_test_extra]
    val   = force_val  + remaining[n_test_extra:n_test_extra + n_val_extra]
    train = remaining[n_test_extra + n_val_extra:]

    split = {
        "train": sorted(train), "val": sorted(val), "test": sorted(test),
        "seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio,
        "skip": skip, "force_val": force_val, "force_test": force_test,
    }
    with open(split_path, "w") as f:
        json.dump(split, f, indent=4)
    print(f"✅  Created split.json: {len(train)} train, {len(val)} val, {len(test)} test")
    return split


def instance_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert instance mask (0=bg, 1..N=instances) to binary (0=bg, 1=fg)."""
    return (mask > 0).astype(np.uint8)


def instance_to_threeclass(mask: np.ndarray, border_width: int = 2) -> np.ndarray:
    """Convert instance mask to 3-class: 0=bg, 1=border, 2=core."""
    from scipy.ndimage import binary_erosion, generate_binary_structure

    out = np.zeros_like(mask, dtype=np.uint8)
    struct = generate_binary_structure(mask.ndim, 1)

    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        obj = mask == label_id
        eroded = binary_erosion(obj, structure=struct, iterations=border_width)
        border = obj & ~eroded
        out[eroded] = 2   # core
        out[border] = 1   # border

    return out


def make_spacing_json(path: str, spacing: tuple):
    """Create the .json companion file required by nnU-Net for TIF files."""
    json_path = path.replace(".tif", ".json").replace(".tiff", ".json")
    with open(json_path, "w") as f:
        json.dump({"spacing": list(spacing)}, f)


def make_case_id(filename: str, index: int) -> str:
    """Create a clean case identifier from filename."""
    stem = Path(filename).stem
    # Replace characters that might cause issues
    clean = stem.replace("-", "").replace("_", "").replace(" ", "")
    return f"case_{index:04d}"


# ─────────────────────────────────────────────────────────────────────────────
#  Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(force: bool = False):
    src_dir   = config.in_dataset_dir
    dst_dir   = os.path.join(config.nnunet_raw, config.nnunet_dataset_name)
    img_dir   = os.path.join(src_dir, "images")
    msk_dir   = os.path.join(src_dir, "masks")
    spacing   = config.voxel_size  # (Z, Y, X)
    approach  = config.seg_approach

    if os.path.exists(dst_dir):
        if force:
            print(f"⚠️  Removing existing {dst_dir}")
            shutil.rmtree(dst_dir)
        else:
            print(f"❌  {dst_dir} already exists. Use --force to overwrite.")
            sys.exit(1)

    # Load split
    split = get_or_create_split(
        src_dir, seed=config.seed,
        val_ratio=config.val_ratio, test_ratio=config.test_ratio,
        skip=config.skip, force_val=config.force_val, force_test=config.force_test,
    )

    # Create dirs
    images_tr = os.path.join(dst_dir, "imagesTr")
    labels_tr = os.path.join(dst_dir, "labelsTr")
    images_ts = os.path.join(dst_dir, "imagesTs")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    # Mapping: filename → case_id (for splits_final.json)
    train_case_ids = []
    val_case_ids   = []
    file_map       = {}   # case_id → original filename
    case_idx       = 0

    # ── Process train + val → imagesTr / labelsTr ────────────────────────
    for subset, case_list in [("train", split["train"]), ("val", split["val"])]:
        for fname in case_list:
            case_id = make_case_id(fname, case_idx)
            file_map[case_id] = fname
            case_idx += 1

            if subset == "train":
                train_case_ids.append(case_id)
            else:
                val_case_ids.append(case_id)

            # Image: copy with _0000 suffix (channel 0)
            img_path = os.path.join(img_dir, fname)
            img = imread(img_path)

            # Ensure 3D (Z, Y, X) — drop channel dim if present
            if img.ndim == 4:
                img = img[..., 0]
            assert img.ndim == 3, f"Expected 3D image, got shape {img.shape}"

            dst_img = os.path.join(images_tr, f"{case_id}_0000.tif")
            imwrite(dst_img, img)
            make_spacing_json(dst_img, spacing)

            # Mask: convert instances → semantic
            msk_path = os.path.join(msk_dir, fname)
            msk = imread(msk_path)
            if msk.ndim == 4:
                msk = msk[..., 0]

            if approach == "binary":
                sem = instance_to_binary(msk)
            elif approach == "threeclass":
                sem = instance_to_threeclass(msk, border_width=config.border_width)
            else:
                raise ValueError(f"Unknown seg_approach: {approach}")

            dst_msk = os.path.join(labels_tr, f"{case_id}.tif")
            imwrite(dst_msk, sem)
            make_spacing_json(dst_msk, spacing)

            n_inst = len(np.unique(msk)) - (1 if 0 in msk else 0)
            print(f"  {subset:5s}  {case_id}  ←  {fname}  "
                  f"shape={img.shape}  instances={n_inst}  "
                  f"labels={sorted(np.unique(sem).tolist())}")

    # ── Process test → imagesTs ──────────────────────────────────────────
    for fname in split["test"]:
        case_id = make_case_id(fname, case_idx)
        file_map[case_id] = fname
        case_idx += 1

        img = imread(os.path.join(img_dir, fname))
        if img.ndim == 4:
            img = img[..., 0]

        dst_img = os.path.join(images_ts, f"{case_id}_0000.tif")
        imwrite(dst_img, img)
        make_spacing_json(dst_img, spacing)
        print(f"  test   {case_id}  ←  {fname}  shape={img.shape}")

    # ── dataset.json ─────────────────────────────────────────────────────
    if approach == "binary":
        labels_dict = {"background": 0, "neuron": 1}
    else:
        labels_dict = {"background": 0, "border": 1, "core": 2}

    dataset_json = {
        "channel_names": {"0": config.modality},
        "labels": labels_dict,
        "numTraining": len(train_case_ids) + len(val_case_ids),
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "Tiff3DIO",
        # Extra metadata (not used by nnU-Net, but useful for us)
        "dataset_id": config.dataset_id,
        "imaging_modality": config.modality,  # NOT "modality" — nnU-Net reserves that key
        "animal": config.animal,
        "voxel_size_um": list(config.voxel_size),
        "seg_approach": approach,
    }

    with open(os.path.join(dst_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    # ── file_map.json (our metadata — maps case_id → original filename) ──
    with open(os.path.join(dst_dir, "file_map.json"), "w") as f:
        json.dump(file_map, f, indent=4)

    # ── splits_final.json — force nnU-Net to use our train/val split ─────
    # nnU-Net expects this in the preprocessed folder, but we also save it
    # alongside the raw data. The train.py script will copy it.
    splits = [
        {
            "train": train_case_ids,
            "val":   val_case_ids,
        }
    ]
    # Save alongside raw dataset (train.py will copy to preprocessed)
    with open(os.path.join(dst_dir, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

    print(f"\n{'='*60}")
    print(f"  ✅  Dataset created: {dst_dir}")
    print(f"  Train:  {len(train_case_ids)} cases")
    print(f"  Val:    {len(val_case_ids)} cases")
    print(f"  Test:   {len(split['test'])} cases (imagesTs)")
    print(f"  Labels: {labels_dict}")
    print(f"  Approach: {approach}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing dataset")
    args = parser.parse_args()
    convert(force=args.force)