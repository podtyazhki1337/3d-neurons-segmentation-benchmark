"""
train.py — 3D U-Net training pipeline for neuron segmentation benchmark.

All parameters come from config.py (no argparse).
Logs everything to Comet ML for reproducibility.
Uses the same split.json / augmentations.py as StarDist and Cellpose.
"""

from __future__ import annotations

import os
import sys
import json
import time

# ── GPU setup (must happen before importing keras) ────────────────────────────
import config

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs available: {[g.name for g in gpus]}")
else:
    print("No GPU found — running on CPU")

if config.mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision (float16) enabled")

from tensorflow import keras as K

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = config.seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Imports ──────────────────────────────────────────────────────────────────
from dataset_tif import TIFPatchDataset
from model import unet_3d, dice_loss, dice_coef
from augmentations import (
    augmenter,
    NO_AUG_CONFIG,
    LIGHT_AUG_CONFIG,
    FULL_CONFIG,
    ACTIVE_CONFIG,
    set_config,
    get_config_json,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Augmentation preset
# ══════════════════════════════════════════════════════════════════════════════

_aug_presets = {
    "NO_AUG":    NO_AUG_CONFIG,
    "LIGHT_AUG": LIGHT_AUG_CONFIG,
    "FULL":      FULL_CONFIG,
}
assert config.aug_preset in _aug_presets, \
    f"Unknown aug_preset: {config.aug_preset}"
set_config(_aug_presets[config.aug_preset])
print(f"Augmentation preset: {config.aug_preset}")


# ══════════════════════════════════════════════════════════════════════════════
#  Comet ML
# ══════════════════════════════════════════════════════════════════════════════

try:
    from comet_ml import Experiment
    experiment = Experiment(
        api_key=config.comet_api_key,
        workspace=config.comet_workspace,
        project_name=config.comet_project,
    )
    experiment.add_tag(config.aug_preset)
    experiment.add_tag(config.modality)
    experiment.add_tag(config.animal)
    experiment.add_tag("unet3d")
except Exception as e:
    print(f"Comet ML init failed: {e} — continuing without logging")
    experiment = None


def log_params():
    """Log all config parameters to Comet."""
    if experiment is None:
        return
    params = {
        # Dataset
        "dataset_id":       config.dataset_id,
        "modality":         config.modality,
        "animal":           config.animal,
        "voxel_size":       str(config.voxel_size),
        "in_dataset_dir":   config.in_dataset_dir,
        # Model
        "model_name":       config.model_name,
        "model_type":       "unet3d",
        "filters":          config.filters,
        "unet_depth":       config.unet_depth,
        "dropout_rate":     config.dropout_rate,
        "use_upsampling":   config.use_upsampling,
        "number_output_classes": config.number_output_classes,
        # Training
        "train_patch_size": str(config.train_patch_size),
        "train_batch_size": config.train_batch_size,
        "epochs":           config.epochs,
        "lr_initial":       config.lr_initial,
        "lr_decay_steps":   config.lr_decay_steps,
        "lr_decay_rate":    config.lr_decay_rate,
        "early_stop_patience": config.early_stop_patience,
        # Augmentation
        "aug_preset":       config.aug_preset,
        "aug_config":       get_config_json(),
        # Hardware
        "gpu_id":           config.gpu_id,
        "mixed_precision":  config.mixed_precision,
        # Reproducibility
        "seed":             config.seed,
    }
    experiment.log_parameters(params)

    # Log config.py as artifact
    try:
        experiment.log_code(file_name="config.py")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════════

class CometLogCallback(K.callbacks.Callback):
    """Log train/val metrics + LR to Comet after each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        if experiment is None or logs is None:
            return
        for k, v in logs.items():
            experiment.log_metric(k, v, step=epoch)
        # Log learning rate
        lr = float(K.backend.get_value(self.model.optimizer.learning_rate))
        experiment.log_metric("lr", lr, step=epoch)


class TimingCallback(K.callbacks.Callback):
    """Track epoch wall-clock time."""

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        if experiment:
            experiment.log_metric("epoch_time_sec", dt, step=epoch)
        print(f"  epoch {epoch} took {dt:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print(f"  U-Net 3D — {config.model_name}")
    print(f"  Dataset:  {config.dataset_id} ({config.modality})")
    print(f"  Aug:      {config.aug_preset}")
    print("=" * 60 + "\n")

    # ── Dataset ──────────────────────────────────────────────────────────
    crop_dim = config.train_patch_size
    tif_data = TIFPatchDataset(
        dataset_dir=config.in_dataset_dir,
        crop_dim=crop_dim,
        batch_size=config.train_batch_size,
        seed=SEED,
        augment_train=(config.aug_preset != "NO_AUG"),
    )

    # Log split to Comet
    if experiment:
        experiment.log_parameter("n_train", len(tif_data.train_data))
        experiment.log_parameter("n_val",   len(tif_data.val_data))
        experiment.log_parameter("n_test",  len(tif_data.test_data))
        experiment.log_parameter("train_files",
                                 str([n for _, _, n in tif_data.train_data]))
        experiment.log_parameter("val_files",
                                 str([n for _, _, n in tif_data.val_data]))
        # Log split.json
        split_path = str(
            os.path.join(config.in_dataset_dir, "split.json"))
        if os.path.exists(split_path):
            experiment.log_asset(split_path, file_name="split.json")

    log_params()

    # ── Model ────────────────────────────────────────────────────────────
    input_shape = crop_dim + (1,)   # (Z, H, W, 1)
    model = unet_3d(
        input_dim=input_shape,
        filters=config.filters,
        number_output_classes=config.number_output_classes,
        use_upsampling=config.use_upsampling,
        dropout_rate=config.dropout_rate,
        model_name=config.model_name,
    )

    # LR schedule
    lr_schedule = K.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.lr_initial,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
        staircase=config.lr_staircase,
    )

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr_schedule),
        loss=dice_loss,
        metrics=[dice_coef],
    )

    n_params = model.count_params()
    print(f"Model parameters: {n_params:,}")
    if experiment:
        experiment.log_parameter("n_params", n_params)

    # ── Callbacks ────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(config.model_basedir, config.model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt = os.path.join(ckpt_dir, "best.weights.h5")
    last_ckpt = os.path.join(ckpt_dir, "last.weights.h5")

    callbacks = [
        K.callbacks.ModelCheckpoint(
            best_ckpt, monitor="val_dice_coef", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1),
        K.callbacks.ModelCheckpoint(
            last_ckpt, save_weights_only=True,
            save_freq="epoch", verbose=0),
        K.callbacks.EarlyStopping(
            monitor="val_dice_coef", mode="max",
            patience=config.early_stop_patience,
            restore_best_weights=True, verbose=1),
        CometLogCallback(),
        TimingCallback(),
    ]

    # ── Steps ────────────────────────────────────────────────────────────
    n_stacks = len(tif_data.train_data)
    steps_per_epoch = max(50, n_stacks * 20 // config.train_batch_size)

    # Val: full grid of all val stacks, ceil(n_patches / batch_size)
    val_steps = -(-tif_data.n_val_patches // config.train_batch_size)

    epochs = 2 if config.quick_demo else config.epochs

    print(f"\nTraining: {epochs} epochs, "
          f"{steps_per_epoch} steps/epoch, "
          f"val_steps={val_steps}, "
          f"batch_size={config.train_batch_size}")

    # ── Fit ──────────────────────────────────────────────────────────────
    model.fit(
        tif_data.ds_train,
        validation_data=tif_data.ds_val,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ── Evaluate best ────────────────────────────────────────────────────
    # EarlyStopping already restored best weights into `model`.
    print("\n\nEvaluating best model on validation set...")
    print("=" * 50)

    val_loss, val_dice = model.evaluate(tif_data.ds_val, steps=val_steps)
    print(f"Best val Dice: {val_dice:.4f}")

    if experiment:
        experiment.log_metric("best_val_dice", val_dice)
        experiment.log_metric("best_val_loss", val_loss)
        try:
            if os.path.exists(best_ckpt):
                experiment.log_asset(best_ckpt)
        except Exception:
            pass

    # ── Save final model ─────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final.keras")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    if experiment:
        experiment.end()
        print("Comet experiment ended.")


if __name__ == "__main__":
    main()