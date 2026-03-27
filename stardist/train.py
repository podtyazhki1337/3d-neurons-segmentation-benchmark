"""
StarDist 3D training pipeline with GPU support and comprehensive Comet ML logging.

Logs everything needed for publication:
  - Dataset / split metadata
  - Model config (architecture, loss, optimizer, augmentations)
  - Training curves (loss, lr, epoch time)
  - Voxel-level metrics (Dice, IoU)
  - Object-level metrics (F1@IoU thresholds, split/merge rates, count error)
  - Per-stack breakdown + summary CSV
  - Best checkpoint as Comet artifact
"""

from __future__ import print_function, unicode_literals, absolute_import, division

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────────────────────────
import sys
import os
import time
import hashlib
import json
import platform
import shutil
import pathlib
import random
from glob import glob
from pathlib import Path

import numpy as np
import skimage
import skimage.measure
import imageio
from tqdm import tqdm
from tifffile import imread
from skimage.transform import rescale

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

from comet_ml import Experiment

import config
import misc


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GPU configuration  (CALL BEFORE any model creation)
# ──────────────────────────────────────────────────────────────────────────────
def setup_gpu(gpu_id: str = "0", memory_growth: bool = True,
              memory_limit_mb: int | None = None):
    """
    Configure TensorFlow to use a specific GPU with optional memory constraints.

    Parameters
    ----------
    gpu_id : str
        Comma-separated GPU IDs visible to TF (e.g. "0" or "0,1").
    memory_growth : bool
        If True, TF allocates GPU memory incrementally instead of grabbing all.
    memory_limit_mb : int or None
        If set, caps the GPU memory TF can use (in MB).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("⚠️  No GPU found — training will run on CPU.", file=sys.stderr)
        return

    for gpu in gpus:
        try:
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb)])
                print(f"GPU {gpu.name}: memory capped at {memory_limit_mb} MB")
            elif memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU {gpu.name}: memory_growth = True")
        except RuntimeError as e:
            print(f"GPU config error: {e}", file=sys.stderr)

    print(f"✅  GPUs available to TF: {[g.name for g in gpus]}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Comet experiment factory
# ──────────────────────────────────────────────────────────────────────────────
def create_experiment(project_name: str = "stardist3d-neurons",
                      workspace: str = "podtyazhki1337",
                      api_key: str = "Yb0Ffy5WaM3qRxJ3qZAjvcboV",
                      tags: list | None = None) -> Experiment:
    exp = Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace,
        auto_output_logging="simple",
    )
    if tags:
        exp.add_tags(tags)
    return exp


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Dataset / split logging helpers
# ──────────────────────────────────────────────────────────────────────────────
def hash_file_list(paths: list[str]) -> str:
    """Deterministic SHA-256 of sorted file names (for reproducibility)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(str(p)).encode())
    return h.hexdigest()[:16]


def log_dataset_info(experiment: Experiment,
                     dataset_dir: str,
                     split: dict,
                     dataset_id: str = "neurons",
                     modality: str = "dodt",
                     voxel_size_um: tuple = (1.0, 1.0, 1.0),
                     pixel_size_um: float | None = None):
    """Log dataset / split metadata to Comet."""
    img_dir = Path(dataset_dir) / "images"

    def _log_subset(name, file_list):
        paths = [str(img_dir / f) for f in file_list]
        depths = []
        for p in paths:
            try:
                depths.append(imread(p).shape[0])
            except Exception:
                depths.append(-1)
        params = {
            f"n_{name}_stacks":    len(file_list),
            f"{name}_files":       json.dumps(file_list),
            f"{name}_files_hash":  hash_file_list(paths),
        }
        if depths:
            params.update({
                f"{name}_depth_min":  int(np.min(depths)),
                f"{name}_depth_max":  int(np.max(depths)),
                f"{name}_depth_mean": float(np.mean(depths)),
            })
        experiment.log_parameters(params)

    experiment.log_parameters({
        "dataset_id":        dataset_id,
        "modality":          modality,
        "split_unit":        "stack",
        "voxel_size_um_zyx": str(voxel_size_um),
        "pixel_size_um":     pixel_size_um,
        "preprocessing":     "fill_label_holes, percentile_norm (on the fly)",
    })

    _log_subset("train", split["train"])
    _log_subset("val",   split["val"])
    if split.get("test"):
        _log_subset("test", split["test"])


def log_hardware_info(experiment: Experiment):
    """Log GPU model, TF version, mixed precision status."""
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    for g in gpus:
        try:
            details = tf.config.experimental.get_device_details(g)
            gpu_names.append(details.get("device_name", g.name))
        except Exception:
            gpu_names.append(g.name)

    experiment.log_parameters({
        "gpu_devices":       json.dumps(gpu_names) if gpu_names else "CPU_only",
        "n_gpus":            len(gpus),
        "tf_version":        tf.__version__,
        "python_version":    platform.python_version(),
        "mixed_precision":   str(tf.keras.mixed_precision.global_policy().name),
        "os":                platform.platform(),
    })


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Keras callbacks for per-epoch Comet logging
# ──────────────────────────────────────────────────────────────────────────────
class CometEpochLogger(tf.keras.callbacks.Callback):
    """Logs loss, lr, and epoch wall-time to Comet after every epoch."""

    def __init__(self, experiment: Experiment):
        super().__init__()
        self.experiment = experiment
        self._epoch_t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self._epoch_t0 if self._epoch_t0 else 0

        # Extract current learning rate
        opt = self.model.optimizer
        if hasattr(opt, "learning_rate"):
            lr = opt.learning_rate
            if callable(lr):
                lr_val = float(lr(opt.iterations))
            else:
                lr_val = float(lr)
        else:
            lr_val = 0.0

        self.experiment.log_metrics({
            "train_loss":       logs.get("loss", 0),
            "val_loss":         logs.get("val_loss", 0),
            "lr":               lr_val,
            "epoch_time_sec":   elapsed,
        }, epoch=epoch + 1)

        # Log individual loss components if present
        for key in ["dist_loss", "prob_loss", "dist_relevant_loss"]:
            if key in logs:
                self.experiment.log_metric(f"train_{key}", logs[key], epoch=epoch + 1)
            vkey = f"val_{key}"
            if vkey in logs:
                self.experiment.log_metric(vkey, logs[vkey], epoch=epoch + 1)


class DiceCheckpoint(tf.keras.callbacks.Callback):
    """
    Periodically predicts on val set, computes voxel Dice,
    saves best checkpoint and logs to Comet.
    """

    def __init__(self, sd_model, val_x, val_y,
                 experiment: Experiment | None = None,
                 out_dir: str = "dice_ckpts", eval_every: int = 5):
        super().__init__()
        self.sd = sd_model
        self.vx, self.vy = val_x, val_y
        self.experiment = experiment
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        print(f"  DiceCheckpoint: saving to {self.out.resolve()}")
        self.every = eval_every
        self.best = -np.inf

    @staticmethod
    def _dice(gt, pred):
        inter = np.logical_and(gt > 0, pred > 0).sum()
        vol   = (gt > 0).sum() + (pred > 0).sum()
        return (2 * inter) / vol if vol else 1.0

    @staticmethod
    def _iou(gt, pred):
        inter = np.logical_and(gt > 0, pred > 0).sum()
        union = np.logical_or(gt > 0, pred > 0).sum()
        return inter / union if union else 1.0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every:
            return

        dices, ious = [], []
        for i in range(len(self.vx)):
            img = self.vx[i]
            gt  = self.vy[i]
            pr, _ = self.sd.predict_instances(
                img, axes="ZYX",
                n_tiles=self.sd._guess_n_tiles(img),
                show_tile_progress=False)
            dices.append(self._dice(gt, pr))
            ious.append(self._iou(gt, pr))

        mean_dice = float(np.mean(dices))
        mean_iou  = float(np.mean(ious))
        print(f"\n🔍  Epoch {epoch+1}: val_dice={mean_dice:.4f}  val_iou={mean_iou:.4f}")

        if self.experiment:
            self.experiment.log_metrics({
                "val_dice":  mean_dice,
                "val_iou":   mean_iou,
            }, epoch=epoch + 1)

        if mean_dice > self.best:
            self.best = mean_dice
            src = self.sd.logdir / "weights_best.h5"
            dst = self.out / f"bestDice_ep{epoch+1:03d}_{mean_dice:.4f}.h5"
            if src.exists():
                shutil.copy(src, dst)
                print(f"   🟢  Dice improved → saved {dst}")
                if self.experiment:
                    self.experiment.log_model(
                        f"best_dice_ep{epoch+1}", str(dst))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Monkey-patch StarDist3D.train to accept custom callbacks
# ──────────────────────────────────────────────────────────────────────────────
_orig_train = StarDist3D.train


def train_with_callbacks(self, *args, callbacks=None, **kw):
    """Wrapper that injects additional Keras callbacks into model.fit()."""
    cb = list(callbacks or [])
    fit_orig = self.keras_model.fit

    def fit_wrapped(*fa, **fk):
        fk["callbacks"] = list(fk.get("callbacks", [])) + cb
        return fit_orig(*fa, **fk)

    self.keras_model.fit = fit_wrapped
    try:
        return _orig_train(self, *args, **kw)
    finally:
        self.keras_model.fit = fit_orig


StarDist3D.train = train_with_callbacks


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Object-level evaluation (split / merge / count error)
# ──────────────────────────────────────────────────────────────────────────────
def compute_object_metrics(gt_labels, pred_labels, iou_thresholds=(0.25, 0.5)):
    """
    Compute object-level metrics for a single volume.

    Returns dict with:
      - obj_f1_<thr>, obj_precision_<thr>, obj_recall_<thr>  for each threshold
      - split_rate, merge_rate
      - count_error (relative)
      - n_gt, n_pred
      - median_volume_gt, median_volume_pred
    """
    results = {}

    # Per-threshold matching
    for thr in iou_thresholds:
        stats = matching(gt_labels, pred_labels, thresh=thr)
        key = f"{thr:.2f}"
        results[f"obj_f1_{key}"]        = stats.f1
        results[f"obj_precision_{key}"] = stats.precision
        results[f"obj_recall_{key}"]    = stats.recall
        results[f"obj_tp_{key}"]        = stats.tp
        results[f"obj_fp_{key}"]        = stats.fp
        results[f"obj_fn_{key}"]        = stats.fn

    # Instance counts
    n_gt   = len(np.unique(gt_labels))   - (1 if 0 in gt_labels else 0)
    n_pred = len(np.unique(pred_labels)) - (1 if 0 in pred_labels else 0)
    results["n_gt"]   = n_gt
    results["n_pred"] = n_pred
    results["count_error"] = abs(n_pred - n_gt) / n_gt if n_gt > 0 else 0.0

    # Split / merge rates (heuristic via IoU=0.5 matching)
    # split: one GT matched by >1 pred  → FP without corresponding FN
    # merge: one pred covers >1 GT      → FN without corresponding FP
    stats_50 = matching(gt_labels, pred_labels, thresh=0.5)
    # Approximate: splits ~ extra preds beyond TP, merges ~ missed GT beyond TP
    results["split_rate"] = stats_50.fp / max(n_gt, 1)
    results["merge_rate"] = stats_50.fn / max(n_gt, 1)

    # Volume statistics
    gt_props   = skimage.measure.regionprops(gt_labels)
    pred_props = skimage.measure.regionprops(pred_labels)
    gt_vols   = [p.area for p in gt_props]   if gt_props   else [0]
    pred_vols = [p.area for p in pred_props] if pred_props else [0]
    results["median_volume_gt"]   = float(np.median(gt_vols))
    results["median_volume_pred"] = float(np.median(pred_vols))

    # Z-extent
    gt_zext   = [p.bbox[3] - p.bbox[0] for p in gt_props]   if gt_props   else [0]
    pred_zext = [p.bbox[3] - p.bbox[0] for p in pred_props] if pred_props else [0]
    results["median_z_extent_gt"]   = float(np.median(gt_zext))
    results["median_z_extent_pred"] = float(np.median(pred_zext))

    return results


def compute_voxel_metrics(gt, pred):
    """Dice + IoU on the foreground mask."""
    gt_fg   = gt > 0
    pred_fg = pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    vol   = gt_fg.sum() + pred_fg.sum()
    return {
        "dice": (2 * inter / vol) if vol else 1.0,
        "iou":  (inter / union) if union else 1.0,
    }


def neurons_per_slice(labels_3d):
    """Mean unique instances per Z-slice (proxy for 'yield')."""
    counts = []
    for z in range(labels_3d.shape[0]):
        n = len(np.unique(labels_3d[z])) - (1 if 0 in labels_3d[z] else 0)
        counts.append(n)
    return float(np.mean(counts))


def full_evaluation(model, val_x, val_y, experiment: Experiment | None = None,
                    iou_thresholds=(0.25, 0.5), prefix="val"):
    """
    Run full evaluation on the validation set.
    Returns per-stack list of dicts + aggregated summary dict.
    Also logs everything to Comet and saves CSV artifact.
    """
    per_stack = []

    for i in tqdm(range(len(val_x)), desc=f"Evaluating ({prefix})"):
        img = val_x[i]
        gt  = val_y[i]
        pred_labels, details = model.predict_instances(
            img, axes="ZYX",
            n_tiles=model._guess_n_tiles(img),
            show_tile_progress=False)

        row = {"stack_idx": i}
        row.update(compute_voxel_metrics(gt, pred_labels))
        row.update(compute_object_metrics(gt, pred_labels, iou_thresholds))
        row["neurons_per_slice_gt"]   = neurons_per_slice(gt)
        row["neurons_per_slice_pred"] = neurons_per_slice(pred_labels)
        per_stack.append(row)

    # Aggregate
    keys = [k for k in per_stack[0] if k != "stack_idx"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in per_stack]
        summary[f"{prefix}_{k}_mean"] = float(np.mean(vals))
        summary[f"{prefix}_{k}_std"]  = float(np.std(vals))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {prefix.upper()} EVALUATION SUMMARY  ({len(val_x)} stacks)")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:40s} = {v:.4f}")

    # Log to Comet
    if experiment:
        experiment.log_metrics(summary)

        # Save CSV artifact
        import csv
        csv_path = f"{prefix}_per_stack_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_stack[0].keys())
            writer.writeheader()
            writer.writerows(per_stack)
        experiment.log_asset(csv_path, file_name=csv_path)
        print(f"  📄  Per-stack CSV logged as Comet asset: {csv_path}")

    return per_stack, summary


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Data loading — on-the-fly crop + rescale to std_cell_size
# ──────────────────────────────────────────────────────────────────────────────

def compute_std_cell_size(dataset_dir: str, file_list: list,
                          train_patch_size: tuple) -> np.ndarray:
    """
    Auto-compute target cell size from patch size and dataset statistics.

    Logic:
      - Target: cells should occupy ~30% of each patch axis,
        so each patch contains ~3 cells → good training signal.
      - std_cell_size = train_patch_size * 0.3
      - rescale_factor per stack = std_cell_size / per_stack_cell_mean

    This matches the original StarDist workflow where std_cell_size=(5,28,28)
    was used with patch_size=(12,96,96):  28/96 ≈ 0.29.
    """
    TARGET_OCCUPANCY = 0.3  # cells occupy ~30% of patch

    mask_dir = Path(dataset_dir) / "masks"
    per_stack = []
    for fname in file_list:
        y = imageio.volread(mask_dir / fname)
        props = skimage.measure.regionprops(y)
        if not props:
            continue
        exts = []
        for p in props:
            bb = p.bbox
            ndim = len(bb) // 2
            exts.append([bb[d + ndim] - bb[d] for d in range(ndim)])
        per_stack.append(np.mean(exts, axis=0))

    raw_median = np.median(per_stack, axis=0)
    patch = np.array(train_patch_size, dtype=float)
    std = np.maximum(np.round(patch * TARGET_OCCUPANCY), 2).astype(float)

    # Derived rescale factor (median stack)
    factor = std / raw_median

    print(f"  median cell extent (raw): {raw_median.round(1)}")
    print(f"  train_patch_size:         {train_patch_size}")
    print(f"  target occupancy:         {TARGET_OCCUPANCY}")
    print(f"  → std_cell_size:          {std.astype(int)}")
    print(f"  → median rescale factor:  {factor.round(3)}")
    return std


def _measure_cell_means(dataset_dir: str, file_list: list) -> dict:
    """Per-stack mean cell size → used for rescale factor computation."""
    mask_dir = Path(dataset_dir) / "masks"
    out = {}
    for fname in file_list:
        y = imageio.volread(mask_dir / fname)
        props = skimage.measure.regionprops(y)
        if not props:
            continue
        exts = []
        for p in props:
            bb = p.bbox
            ndim = len(bb) // 2
            exts.append([bb[d + ndim] - bb[d] for d in range(ndim)])
        out[fname] = np.mean(exts, axis=0)
    return out


def _crop_to_foreground(vol, mask_vol, padsize):
    """Crop 3D volume to foreground bbox + padding."""
    fg = (mask_vol > 0).astype(np.uint8)
    props = skimage.measure.regionprops(fg)
    if not props:
        return vol
    bb = props[0].bbox
    ndim = len(bb) // 2
    crop = tuple(
        slice(max(0, bb[d] - padsize),
              min(vol.shape[d], bb[d + ndim] + padsize))
        for d in range(ndim))
    return vol[crop]


class StarDistSequence(Sequence):
    """Data loader: reads raw TIFFs → crop → rescale → in memory.

    Pipeline per stack (on repool):
      1. Read raw TIFF
      2. Crop to foreground bbox + padding (removes dead margins)
      3. Rescale to std_cell_size (normalises cell dimensions for StarDist)
      4. Masks: fill_label_holes + uint16
      5. Images: stored raw; percentile-normalised in __getitem__

    No files are copied to disk.

    Parameters
    ----------
    dataset_dir : str
        Root of the dataset (images/ and masks/).
    file_list : list[str]
        Filenames belonging to this subset.
    y : bool
        True = masks, False = images.
    std_cell_size : np.ndarray or None
        Target cell size (Z, Y, X) for rescaling. None = no rescale.
    cell_means : dict
        Per-stack mean cell sizes (from _measure_cell_means).
    padsize : int
        Padding around foreground bbox.
    """

    def __init__(self, dataset_dir, file_list, y=False,
                 std_cell_size=None, cell_means=None,
                 padsize=64,
                 pool_size=None, repool_freq=None, seed=42):
        self.y = y
        self.base_seed = seed
        self.repool_counter = 0
        self.dataset_dir = Path(dataset_dir)
        self.file_list = list(file_list)
        self.std_cell_size = std_cell_size
        self.cell_means = cell_means or {}
        self.padsize = padsize

        subdir = "masks" if y else "images"
        self.data_dir = self.dataset_dir / subdir

        for f in self.file_list:
            assert (self.data_dir / f).exists(), f"Missing: {self.data_dir / f}"

        assert pool_size != 0 and repool_freq != 0
        self.pool_size = pool_size if pool_size else len(self.file_list)
        self.repool_freq = (self.pool_size * repool_freq
                            if repool_freq else self.pool_size)

        self._cell_extents = None
        self.data = None
        info = f"  {subdir}: {len(self.file_list)} files"
        if self.std_cell_size is not None:
            info += f", rescale→({self.std_cell_size.round(1)})"
        print(info)
        self.repool()

    def get_repool_frequency(self):
        return self.repool_freq

    def get_cell_extents(self):
        """Median cell extent per stack (post-rescale, lazy cached)."""
        if self._cell_extents is not None:
            return self._cell_extents
        if self.y:
            extents = []
            for mask in self.data:
                props = skimage.measure.regionprops(mask)
                if not props:
                    continue
                es = []
                for p in props:
                    bb = p.bbox
                    ndim = len(bb) // 2
                    es.append([bb[d + ndim] - bb[d] for d in range(ndim)])
                extents.append(np.median(es, axis=0))
            self._cell_extents = (np.array(extents) if extents
                                  else np.zeros((1, 3)))
        return self._cell_extents

    def on_epoch_end(self):
        pass

    def repool(self):
        """Load a pool of stacks: read → crop → rescale."""
        current_seed = self.base_seed + self.repool_counter
        self.repool_counter += 1

        r_ = random.Random(current_seed)
        sample_ids = list(range(len(self.file_list)))
        self.pool_ids = r_.sample(sample_ids, self.pool_size)

        self.data = []
        for idx in self.pool_ids:
            fname = self.file_list[idx]
            img = imageio.volread(self.data_dir / fname)

            # Crop to foreground
            mask_vol = imageio.volread(self.dataset_dir / "masks" / fname)
            img = _crop_to_foreground(img, mask_vol, self.padsize)

            # Rescale to std_cell_size
            if (self.std_cell_size is not None
                    and fname in self.cell_means):
                factor = self.std_cell_size / self.cell_means[fname]
                order = 0 if self.y else 1
                img = rescale(img, factor, order=order,
                              anti_aliasing=False,
                              preserve_range=True).astype(np.uint16)

            if self.y:
                img = fill_label_holes(img.astype(np.uint16))

            self.data.append(img)

        self._cell_extents = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret = self.data[idx].copy()
        if not self.y:
            ret = normalize(ret.astype(np.float32),
                            config.norm_percentile_low,
                            config.norm_percentile_high,
                            axis=(0, 1, 2))
        return ret


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Dataset split (JSON manifest — no file copying)
# ──────────────────────────────────────────────────────────────────────────────

def auto_split(dataset_dir: str,
               val_ratio: float = 0.15,
               test_ratio: float = 0.0,
               seed: int = 42,
               skip: list | None = None,
               force_val: list | None = None,
               force_test: list | None = None) -> dict:
    """
    Deterministic train / val / test split by whole stacks.

    Scans ``dataset_dir/images/*.tif``, verifies matching masks exist,
    shuffles with the given seed, and returns a split dict.

    No files are copied.  The split dict is saved as ``split.json``
    and used by the data loader to select files at runtime.
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
    n_test  = max(0, n_test  - len(force_test))
    n_val   = max(0, n_val   - len(force_val))

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


def get_or_create_split(dataset_dir: str, split_path: str, **kwargs) -> dict:
    """Load existing split.json or create a new one."""
    p = Path(split_path)
    if p.exists():
        split = misc.get_json(str(p))
        print(f"📄  Loaded split: {p}")
        print(f"  train: {len(split['train'])}  val: {len(split['val'])}  "
              f"test: {len(split.get('test', []))}")
        return split
    split = auto_split(dataset_dir, **kwargs)
    misc.put_json(str(p), split)
    print(f"📄  Split saved: {p}")
    return split


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Augmentations — imported from standalone benchmark module
# ──────────────────────────────────────────────────────────────────────────────
from augmentations import (
    augmenter,
    NO_AUG_CONFIG,
    LIGHT_AUG_CONFIG,
    FULL_CONFIG,
    ACTIVE_CONFIG,
    set_config,
    get_config_json,
)
# Default ACTIVE_CONFIG == LIGHT_AUG_CONFIG.
# Switch via AUG_PRESET flag below or: set_config(NO_AUG_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── All parameters come from config.py ───────────────────────────────
    SEED = config.seed
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    # ── GPU setup ────────────────────────────────────────────────────────
    setup_gpu(gpu_id=config.gpu_id, memory_growth=True)

    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("✅  Mixed precision (float16) enabled")

    # ── Augmentation preset ──────────────────────────────────────────────
    _aug_presets = {
        "NO_AUG":    NO_AUG_CONFIG,
        "LIGHT_AUG": LIGHT_AUG_CONFIG,
        "FULL":      FULL_CONFIG,
    }
    assert config.aug_preset in _aug_presets, \
        f"Unknown aug_preset: {config.aug_preset}"
    set_config(_aug_presets[config.aug_preset])
    print(f"✅  Augmentation preset: {config.aug_preset}")

    # ── Split (JSON only, no file copying) ───────────────────────────────
    split_path = str(Path(config.in_dataset_dir) / "split.json")
    split = get_or_create_split(
        config.in_dataset_dir,
        split_path,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=SEED,
        skip=config.skip,
        force_val=config.force_val,
        force_test=config.force_test,
    )

    # ── Create Comet experiment ──────────────────────────────────────────
    experiment = create_experiment(
        project_name=config.comet_project,
        workspace=config.comet_workspace,
        api_key=config.comet_api_key,
        tags=[config.dataset_id, config.modality, config.animal,
              "stardist3d", f"aug_{config.aug_preset}"],
    )

    # ── Compute std_cell_size from training masks ──────────────────────
    print("\nMeasuring cell sizes from training masks...")
    std_cell_size = compute_std_cell_size(config.in_dataset_dir,
                                          split["train"],
                                          config.train_patch_size)

    # Per-stack cell means for all subsets (needed for rescale factors)
    all_files = split["train"] + split["val"] + split.get("test", [])
    cell_means = _measure_cell_means(config.in_dataset_dir, all_files)

    # Save for pred.py — guarantees identical preprocessing at inference
    preproc_meta = {
        "std_cell_size": [float(v) for v in std_cell_size],
        "cell_means": {k: [float(v) for v in vs]
                       for k, vs in cell_means.items()},
        "padsize": config.padsize,
    }
    preproc_path = str(Path(config.in_dataset_dir) / "preproc_meta.json")
    misc.put_json(preproc_path, preproc_meta)
    print(f"📄  Saved preproc_meta.json → {preproc_path}")

    # ── Load data (read → crop → rescale on the fly) ─────────────────
    def make_loaders(file_list, pool_size=None, repool_freq=None, seed=42):
        common = dict(
            std_cell_size=std_cell_size,
            cell_means=cell_means,
            padsize=config.padsize,
        )
        return (StarDistSequence(config.in_dataset_dir, file_list,
                                 y=False, pool_size=pool_size,
                                 repool_freq=repool_freq, seed=seed,
                                 **common),
                StarDistSequence(config.in_dataset_dir, file_list,
                                 y=True, pool_size=pool_size,
                                 repool_freq=repool_freq, seed=seed,
                                 **common))

    print(f"\nLoading train ({len(split['train'])} stacks)...")
    trn_x, trn_y = make_loaders(split["train"],
                                 config.pool_size, config.repool_freq,
                                 seed=SEED)
    print(f"Loading val ({len(split['val'])} stacks)...")
    val_x, val_y = make_loaders(split["val"], seed=SEED)

    has_test = len(split.get("test", [])) > 0
    if has_test:
        print(f"Loading test ({len(split['test'])} stacks)...")
        tst_x, tst_y = make_loaders(split["test"], seed=SEED)

    # Log split to Comet
    experiment.log_asset(split_path, file_name="split.json")

    # ── Anisotropy ────────────────────────────────────────────────────────
    voxel = np.array(config.voxel_size, dtype=float)

    # Median cell extents from training masks
    train_extents = trn_y.get_cell_extents()
    extents = np.median(train_extents, axis=0)
    print(f"Median cell extent (voxels, ZYX): {extents}")

    if np.allclose(voxel, voxel[0]):
        # Isotropic voxel_size (or placeholder) — derive anisotropy from
        # cell extents: if cells appear 10× wider in XY than Z in voxel
        # space, the voxels must be ~10× coarser in Z.
        anisotropy = tuple(np.max(extents) / extents)
        print(f"⚠️  voxel_size is isotropic {config.voxel_size} — "
              f"using cell-extent-derived anisotropy: {anisotropy}")
        print(f"   Set config.voxel_size to real µm values for best results.")
    else:
        anisotropy = tuple(np.max(voxel) / voxel)
        print(f"Voxel size (µm, ZYX):  {config.voxel_size}")
        print(f"Anisotropy:            {anisotropy}")

    n_channel = 1

    # ── Model config ─────────────────────────────────────────────────────
    grid = tuple(1 if a > 1.5 else 4 for a in anisotropy)
    rays = Rays_GoldenSpiral(config.n_rays, anisotropy=anisotropy)

    use_gpu_opencl = gputools_available()

    conf = Config3D(
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu_opencl,
        n_channel_in     = n_channel,
        train_patch_size = config.train_patch_size,
        train_batch_size = config.train_batch_size,
        unet_n_depth     = config.unet_n_depth,
    )
    print(conf)

    # ── Log EVERYTHING to Comet ──────────────────────────────────────────
    log_hardware_info(experiment)

    log_dataset_info(
        experiment,
        dataset_dir=config.in_dataset_dir,
        split=split,
        dataset_id=config.dataset_id,
        modality=config.modality,
        voxel_size_um=config.voxel_size,
        pixel_size_um=config.pixel_size,
    )

    experiment.log_parameters({
        # Model config
        "model_name":        "StarDist3D",
        "model_version":     "stardist_0.9.x",
        "mode":              "3D",
        "n_rays":            config.n_rays,
        "grid":              str(grid),
        "anisotropy":        str(anisotropy),
        "train_patch_size":  str(config.train_patch_size),
        "train_batch_size":  config.train_batch_size,
        "unet_n_depth":      config.unet_n_depth,
        "n_channel_in":      n_channel,
        "use_gpu_opencl":    use_gpu_opencl,

        # Loss
        "loss_name":         "stardist_default (BCE_prob + MAE_dist)",
        "loss_weights":      "default",

        # Optimizer / schedule
        "optimizer":         "Adam",
        "lr_initial":        config.lr_initial,
        "lr_decay_steps":    config.lr_decay_steps,
        "lr_decay_rate":     config.lr_decay_rate,
        "lr_schedule":       "ExponentialDecay_staircase"
                             if config.lr_staircase else "ExponentialDecay",

        # Augmentations
        "aug_preset":        config.aug_preset,
        "aug_config":        get_config_json(),

        # Training
        "seed":              SEED,
        "dataloader_seed":   SEED,
        "epochs":            config.epochs,
        "mixed_precision":   config.mixed_precision,
        "voxel_size_um":     str(config.voxel_size),
        "std_cell_size":     str(tuple(std_cell_size.round(1))),
        "padsize":           config.padsize,

        # Dataset identifiers
        "animal_train":      config.animal,
        "animal_test":       config.animal,
        "modality_train":    config.modality,
        "modality_test":     config.modality,
    })

    # Log config.py as artifact
    experiment.log_asset("config.py", file_name="config.py")

    # ── Build model ──────────────────────────────────────────────────────
    model = StarDist3D(conf, name=config.model_name, basedir=config.model_basedir)

    # Compute FOV / tile overlap.  mixed_precision float16 breaks
    # scipy.ndimage.zoom inside StarDist's _compute_receptive_field,
    # so we compute it manually and inject the cached value.
    try:
        fov = np.array(model._axes_tile_overlap("ZYX"))
    except (RuntimeError, TypeError):
        # Theoretical RF for U-Net: sum over depths of
        #   pool^depth * n_conv_per_depth * (kernel_size - 1)
        # doubled (encoder + decoder) + 1
        d = config.unet_n_depth
        fov_1d = 2 * sum(2**i * 2 * (3 - 1) for i in range(d)) + 1
        fov = np.array([fov_1d] * 3)
        # Inject into StarDist's cache so _guess_n_tiles works
        model._tile_overlap = {"Z": fov_1d, "Y": fov_1d, "X": fov_1d}
        print(f"⚠️  FOV: scipy incompatible with mixed_precision, "
              f"using theoretical RF={fov_1d}")

    print(f"median object size:     {extents}")
    print(f"network field of view:  ~{fov}")
    if any(extents > fov):
        print("⚠️  WARNING: median object size > network FOV!")
    experiment.log_parameters({
        "network_fov_zyx":      str(tuple(fov)),
        "median_obj_size_zyx":  str(tuple(extents)),
    })

    # ── Optimizer ────────────────────────────────────────────────────────
    schedule = ExponentialDecay(
        initial_learning_rate=config.lr_initial,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
        staircase=config.lr_staircase)
    opt = Adam(learning_rate=schedule)
    model.keras_model.optimizer = opt

    # ── Callbacks ────────────────────────────────────────────────────────
    comet_cb = CometEpochLogger(experiment)
    dice_cb  = DiceCheckpoint(
        model, val_x, val_y,
        experiment=experiment,
        out_dir=str(Path(config.model_basedir) / config.model_name / "dice_ckpts"),
        eval_every=config.dice_eval_every)

    # ── Train ────────────────────────────────────────────────────────────
    steps = trn_x.repool_freq
    experiment.log_parameter("steps_per_epoch", steps)

    t_start = time.time()
    model.train(
        trn_x, trn_y,
        validation_data=(val_x, val_y),
        augmenter=augmenter,
        epochs=config.epochs,
        steps_per_epoch=steps,
        callbacks=[comet_cb, dice_cb],
    )
    train_time = time.time() - t_start
    experiment.log_metric("total_train_time_sec", train_time)
    experiment.log_metric("total_train_time_hr", train_time / 3600)
    print(f"\n⏱️  Training completed in {train_time/3600:.2f} h")

    # ── Optimize NMS thresholds ──────────────────────────────────────────
    print("\nOptimizing thresholds...")
    thr = model.optimize_thresholds(val_x, val_y)
    experiment.log_parameters({
        "nms_thresh":           thr.get("nms", None) if isinstance(thr, dict) else str(thr),
        "prob_thresh":          thr.get("prob", None) if isinstance(thr, dict) else str(thr),
        "optimized_thresholds": str(thr),
    })
    print(f"Optimized thresholds: {thr}")

    # ── Full evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FULL VALIDATION EVALUATION")
    print("=" * 60)

    per_stack, summary = full_evaluation(
        model, val_x, val_y,
        experiment=experiment,
        iou_thresholds=(0.1, 0.25, 0.5, 0.7),
        prefix="val",
    )

    # ── Classic matching_dataset at multiple thresholds ───────────────────
    print("\nComputing matching_dataset across IoU thresholds...")
    Y_val_pred = [
        model.predict_instances(
            val_x[i], axes="ZYX",
            n_tiles=model._guess_n_tiles(val_x[i]),
            show_tile_progress=False)[0]
        for i in tqdm(range(len(val_x)))
    ]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats_all = [matching_dataset(val_y, Y_val_pred, thresh=t,
                                  show_progress=False) for t in taus]
    for t, st in zip(taus, stats_all):
        experiment.log_metrics({
            f"matching_f1_tau{t:.1f}":        st.f1,
            f"matching_precision_tau{t:.1f}": st.precision,
            f"matching_recall_tau{t:.1f}":    st.recall,
        })
        print(f"  tau={t:.1f}  F1={st.f1:.4f}  "
              f"Prec={st.precision:.4f}  Rec={st.recall:.4f}")

    # ── Log tiling info ──────────────────────────────────────────────────
    if len(val_x) > 0:
        sample_tiles = model._guess_n_tiles(val_x[0])
        experiment.log_parameter("tiling_n_tiles_example", str(sample_tiles))

    # ── Log best checkpoint as artifact ──────────────────────────────────
    best_weights = model.logdir / "weights_best.h5"
    if best_weights.exists():
        experiment.log_model("best_checkpoint", str(best_weights))

    # ── FOV reminder ─────────────────────────────────────────────────────
    print(f"\nmedian object size:     {extents}")
    print(f"network field of view:  {fov}")

    # ── Test set evaluation (if present) ─────────────────────────────────
    if has_test:
        print("\n" + "=" * 60)
        print("  TEST SET EVALUATION")
        print("=" * 60)
        test_per_stack, test_summary = full_evaluation(
            model, tst_x, tst_y,
            experiment=experiment,
            iou_thresholds=(0.1, 0.25, 0.5, 0.7),
            prefix="test",
        )

    experiment.end()
    print("\n✅  Experiment finished and logged to Comet.")