"""
dataset_tif.py — TIF stack data loader for 3D U-Net benchmark.

Reads directly from raw dataset (images/ + masks/ folders).
Uses split.json for train/val/test partitioning (same file as StarDist/Cellpose).
Applies percentile normalisation and augmentations from augmentations.py on the fly.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tifffile import imread

import config
from augmentations import augmenter


# ══════════════════════════════════════════════════════════════════════════════
#  Split utilities (shared logic with StarDist / Cellpose)
# ══════════════════════════════════════════════════════════════════════════════

def auto_split(dataset_dir: str,
               val_ratio: float = 0.15,
               test_ratio: float = 0.10,
               seed: int = 42,
               skip: list | None = None,
               force_val: list | None = None,
               force_test: list | None = None) -> dict:
    """
    Deterministic train/val/test split by whole stacks.
    Scans dataset_dir/images/*.tif, shuffles with seed, returns split dict.
    """
    skip       = set(skip or [])
    force_val  = set(force_val or [])
    force_test = set(force_test or [])

    dataset_dir = Path(dataset_dir)
    images = sorted(f.name for f in (dataset_dir / "images").glob("*.tif"))
    masks  = sorted(f.name for f in (dataset_dir / "masks").glob("*.tif"))
    assert images == masks, (
        f"Image/mask mismatch.\n  images: {images}\n  masks:  {masks}")

    names = [n for n in images if n not in skip]
    print(f"Dataset: {len(names)} stacks (skipped {len(skip)})")

    pool = [n for n in names
            if n not in force_val and n not in force_test]

    rng = random.Random(seed)
    rng.shuffle(pool)

    n_total = len(pool)
    n_test  = int(round(test_ratio * (n_total + len(force_val) + len(force_test))))
    n_val   = int(round(val_ratio  * (n_total + len(force_val) + len(force_test))))
    n_test  = max(0, n_test - len(force_test))
    n_val   = max(0, n_val  - len(force_val))

    test_names  = sorted(force_test) + sorted(pool[:n_test])
    val_names   = sorted(force_val)  + sorted(pool[n_test:n_test + n_val])
    train_names = sorted(pool[n_test + n_val:])

    assert len(train_names) >= 1, "Need at least 1 training stack!"
    assert len(val_names)   >= 1, "Need at least 1 validation stack!"

    split = {
        "train":      train_names,
        "val":        val_names,
        "test":       test_names,
        "seed":       seed,
        "val_ratio":  val_ratio,
        "test_ratio": test_ratio,
        "skip":       sorted(skip),
        "force_val":  sorted(force_val),
        "force_test": sorted(force_test),
    }

    print(f"  train: {len(train_names)}  val: {len(val_names)}  "
          f"test: {len(test_names)}")
    for subset in ("train", "val", "test"):
        for n in split[subset]:
            print(f"    {subset:6s}  {n}")

    return split


def get_or_create_split(dataset_dir: str, **kwargs) -> dict:
    """Load existing split.json or create a new one."""
    split_path = Path(dataset_dir) / "split.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)
        print(f"Loaded split: {split_path}")
        print(f"  train: {len(split['train'])}  val: {len(split['val'])}  "
              f"test: {len(split.get('test', []))}")
        return split

    split = auto_split(dataset_dir, **kwargs)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Split saved: {split_path}")
    return split


# ══════════════════════════════════════════════════════════════════════════════
#  Normalisation
# ══════════════════════════════════════════════════════════════════════════════

def percentile_norm(vol: np.ndarray,
                    pmin: float = 1.0,
                    pmax: float = 99.8) -> np.ndarray:
    """Percentile-based normalisation to ~[0, 1]."""
    lo, hi = np.percentile(vol, (pmin, pmax))
    if hi - lo < 1e-6:
        return vol.astype(np.float32) * 0.0
    return ((vol.astype(np.float32) - lo) / (hi - lo))


def fill_label_holes(lbl: np.ndarray) -> np.ndarray:
    """Fill holes in each labelled object (binary mask: just fill zeros)."""
    from scipy.ndimage import binary_fill_holes
    if lbl.max() <= 1:
        return binary_fill_holes(lbl > 0).astype(lbl.dtype)
    out = np.zeros_like(lbl)
    for uid in np.unique(lbl):
        if uid == 0:
            continue
        mask = binary_fill_holes(lbl == uid)
        out[mask] = uid
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TIFPatchDataset:
    """
    Reads TIF stacks from images/ and masks/ using split.json,
    applies percentile normalisation + augmentations on-the-fly,
    yields random 3D patches via tf.data.Dataset.

    Compatible interface with the StarDist/Cellpose benchmark pipeline.
    """

    def __init__(self,
                 dataset_dir: str  = config.in_dataset_dir,
                 crop_dim: tuple   = None,
                 batch_size: int   = config.train_batch_size,
                 seed: int         = config.seed,
                 augment_train: bool = True):

        if crop_dim is None:
            crop_dim = config.train_patch_size

        self.Z, self.H, self.W = crop_dim
        self.bs  = batch_size
        self.rng = np.random.default_rng(seed)
        self.augment_train = augment_train

        # Load or create split
        self.split = get_or_create_split(
            dataset_dir,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=seed,
            skip=config.skip,
            force_val=config.force_val,
            force_test=config.force_test,
        )

        self.img_dir  = Path(dataset_dir) / "images"
        self.mask_dir = Path(dataset_dir) / "masks"

        # Pre-load all stacks into memory (most neuron datasets fit)
        self.train_data = self._load_subset("train")
        self.val_data   = self._load_subset("val")
        self.test_data  = self._load_subset("test")

        # Build tf.data pipelines
        self.ds_train = self._make_dataset(self.train_data, augment=augment_train,
                                           shuffle=True, repeat=True)

        # Validation: full grid of all stacks, repeating (keras needs fresh data each epoch)
        self.n_val_patches = self.count_val_patches()
        self.ds_val = self._make_val_dataset()
        print(f"  Val: {self.n_val_patches} grid patches "
              f"({len(self.val_data)} stacks, "
              f"val_steps={-(-self.n_val_patches // self.bs)})")

    # ── Loading ──────────────────────────────────────────────────────────
    def _load_one(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load one TIF stack, normalise image, fill label holes."""
        img = imread(str(self.img_dir / name))     # (Z, Y, X) or (Z, Y, X, C)
        msk = imread(str(self.mask_dir / name))

        # Ensure 3D (Z, Y, X)
        if img.ndim == 4:
            img = img[..., 0]
        if msk.ndim == 4:
            msk = msk[..., 0]

        img = percentile_norm(img, config.norm_percentile_low,
                              config.norm_percentile_high)
        msk = fill_label_holes(msk)
        msk = (msk > 0).astype(np.float32)

        return img, msk

    def _load_subset(self, subset: str) -> list:
        """Load all stacks for a given subset."""
        names = self.split.get(subset, [])
        data = []
        for name in names:
            img, msk = self._load_one(name)
            data.append((img, msk, name))
            print(f"  Loaded {subset:6s} {name}  shape={img.shape}")
        return data

    # ── Patch sampling ───────────────────────────────────────────────────
    def _random_crop(self, img: np.ndarray, msk: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random (Z, H, W) patch."""
        dz, dy, dx = img.shape
        z0 = self.rng.integers(0, max(1, dz - self.Z + 1))
        y0 = self.rng.integers(0, max(1, dy - self.H + 1))
        x0 = self.rng.integers(0, max(1, dx - self.W + 1))
        return (img[z0:z0+self.Z, y0:y0+self.H, x0:x0+self.W],
                msk[z0:z0+self.Z, y0:y0+self.H, x0:x0+self.W])

    def _pad_if_needed(self, img: np.ndarray, msk: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Reflect-pad if any stack dimension < patch dimension."""
        pads = []
        for actual, target in zip(img.shape, (self.Z, self.H, self.W)):
            deficit = max(0, target - actual)
            pads.append((0, deficit))
        if any(d > 0 for _, d in pads):
            img = np.pad(img, pads, mode="reflect")
            msk = np.pad(msk, pads, mode="reflect")
        return img, msk

    # ── Generators ───────────────────────────────────────────────────────
    def _gen(self, data_list, augment, repeat):
        while True:
            idx = self.rng.integers(0, len(data_list))
            img, msk, _ = data_list[idx]
            img, msk = self._pad_if_needed(img, msk)
            img_patch, msk_patch = self._random_crop(img, msk)

            if augment:
                img_patch, msk_patch = augmenter(img_patch, msk_patch)

            # Add channel dim: (Z, H, W) → (Z, H, W, 1)
            yield (img_patch[..., None].astype(np.float32),
                   msk_patch[..., None].astype(np.float32))

            if not repeat:
                break

    def _gen_grid_patches(self, data_list):
        """
        Yield ALL non-overlapping patches covering entire val stacks.
        Deterministic, covers 100% of each volume — no randomness.
        Same grid logic as eval inference.
        """
        pz, py, px = self.Z, self.H, self.W
        for img, msk, name in data_list:
            # Pad to multiple of patch size
            pad_z = (pz - img.shape[0] % pz) % pz
            pad_y = (py - img.shape[1] % py) % py
            pad_x = (px - img.shape[2] % px) % px
            img_p = np.pad(img, [(0, pad_z), (0, pad_y), (0, pad_x)], mode="reflect")
            msk_p = np.pad(msk, [(0, pad_z), (0, pad_y), (0, pad_x)], mode="reflect")

            nz, ny, nx = img_p.shape
            for iz in range(nz // pz):
                for iy in range(ny // py):
                    for ix in range(nx // px):
                        img_patch = img_p[iz*pz:(iz+1)*pz,
                                          iy*py:(iy+1)*py,
                                          ix*px:(ix+1)*px]
                        msk_patch = msk_p[iz*pz:(iz+1)*pz,
                                          iy*py:(iy+1)*py,
                                          ix*px:(ix+1)*px]
                        yield (img_patch[..., None].astype(np.float32),
                               msk_patch[..., None].astype(np.float32))

    def count_val_patches(self) -> int:
        """Count total grid patches across all val stacks."""
        pz, py, px = self.Z, self.H, self.W
        total = 0
        for img, msk, name in self.val_data:
            gz = -(-img.shape[0] // pz)  # ceil div
            gy = -(-img.shape[1] // py)
            gx = -(-img.shape[2] // px)
            total += gz * gy * gx
        return total

    def _make_val_dataset(self) -> tf.data.Dataset:
        """Val dataset: deterministic grid patches, repeating for multi-epoch use."""
        if not self.val_data:
            return None

        out_sig = (
            tf.TensorSpec((self.Z, self.H, self.W, 1), tf.float32),
            tf.TensorSpec((self.Z, self.H, self.W, 1), tf.float32),
        )
        ds = tf.data.Dataset.from_generator(
            lambda: self._gen_grid_patches(self.val_data),
            output_signature=out_sig)
        return ds.batch(self.bs).repeat().prefetch(tf.data.AUTOTUNE)

    def _make_dataset(self, data_list, augment=False,
                      shuffle=True, repeat=True) -> tf.data.Dataset:
        if not data_list:
            return None

        out_sig = (
            tf.TensorSpec((self.Z, self.H, self.W, 1), tf.float32),
            tf.TensorSpec((self.Z, self.H, self.W, 1), tf.float32),
        )

        if repeat:
            # Training: random patches, infinite
            ds = tf.data.Dataset.from_generator(
                lambda: self._gen(data_list, augment, repeat=True),
                output_signature=out_sig)
        else:
            # Validation: deterministic grid covering all stacks
            ds = tf.data.Dataset.from_generator(
                lambda: self._gen_grid_patches(data_list),
                output_signature=out_sig)

        if shuffle:
            ds = ds.shuffle(4 * self.bs, seed=config.seed)
        if repeat:
            ds = ds.repeat()
        return ds.batch(self.bs).prefetch(tf.data.AUTOTUNE)