"""
config.py — Single source of truth for all U-Net 3D experiment parameters.

Edit this file before each run.  It is logged to Comet as an artifact
so every experiment is fully reproducible from config alone.

Sections
────────
  1. Paths           – dataset location, model output
  2. Dataset / split  – split ratios, skip/force lists, metadata
  3. Data loading     – normalisation
  4. Model            – architecture, patch, batch
  5. Training         – optimizer, schedule, epochs, callbacks
  6. Augmentation     – preset selector
  7. Hardware         – GPU, mixed precision
  8. Logging          – Comet ML credentials
  9. Reproducibility  – global seed
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Paths
# ══════════════════════════════════════════════════════════════════════════════

# Dataset root — must contain images/ and masks/ with matching .tif names.
# No preprocessing needed: data is read directly at native resolution.
# split.json is saved here automatically on first run.
in_dataset_dir = "/NAS/mmaiurov/datasets_benchmark/mice_neurons_dodt/"

# Model output directory.
model_basedir = "models"
model_name    = "unet3d_mice_neurons_dodt_light_aug"  # subdir for this run's model and logs


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Dataset / split
# ══════════════════════════════════════════════════════════════════════════════

# Ratios for automatic train/val/test split by whole stacks.
# The split is deterministic (controlled by `seed`) and saved to
# in_dataset_dir/split.json.  Delete that file to re-split.
val_ratio  = 0.15
test_ratio = 0.10

# Files to exclude from the dataset entirely.
skip = []

# Force specific filenames into val or test (overrides ratio for those files).
force_val  = []
force_test = []

# Dataset metadata for Comet logging.
dataset_id = "mice_neurons_dodt_light_aug"
modality   = "dodt"               # "oblique" / "dodt" / "fluorescence"
animal     = "mice"                # "rat" / "mouse" / "human"
voxel_size = (1.0, 1.0, 1.0)     # µm  (Z, Y, X) — SET THIS TO REAL VALUES!
pixel_size = None                 # µm  XY pixel size (if known)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Data loading
# ══════════════════════════════════════════════════════════════════════════════

# Percentile normalisation range applied to images on the fly.
norm_percentile_low  = 1
norm_percentile_high = 99.8


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Model
# ══════════════════════════════════════════════════════════════════════════════

# U-Net architecture
filters          = 16               # base number of conv filters
unet_depth       = 4                # encoder/decoder depth (4 poolings)
dropout_rate     = 0.20             # SpatialDropout3D rate
use_upsampling   = False            # True = UpSampling3D, False = Conv3DTranspose
number_output_classes = 1           # binary segmentation

# Patch sampling
train_patch_size = (32, 256, 256)   # (Z, Y, X) — must be divisible by 2^unet_depth
train_batch_size = 8


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Training
# ══════════════════════════════════════════════════════════════════════════════

epochs            = 300
lr_initial        = 3e-4
lr_decay_steps    = 500
lr_decay_rate     = 0.5
lr_staircase      = True

early_stop_patience = 40            # epochs without val_dice improvement
dice_eval_every     = 5             # evaluate on val every N epochs

# Set True to run a quick 2-epoch smoke test.
quick_demo = False


# ══════════════════════════════════════════════════════════════════════════════
#  6.  Augmentation
# ══════════════════════════════════════════════════════════════════════════════

# One of: "NO_AUG", "LIGHT_AUG", "FULL"
# NO_AUG    — identity transform (strict ablation baseline)
# LIGHT_AUG — flips + rot90 + mild intensity + noise (benchmark default)
# FULL      — all transforms including blur, elastic, zoom, cutout
aug_preset = "LIGHT_AUG"


# ══════════════════════════════════════════════════════════════════════════════
#  7.  Hardware
# ══════════════════════════════════════════════════════════════════════════════

gpu_id          = "2"       # comma-separated GPU IDs, e.g. "0,1"
mixed_precision = False     # fp16 — test on your hardware first


# ══════════════════════════════════════════════════════════════════════════════
#  8.  Logging  (Comet ML)
# ══════════════════════════════════════════════════════════════════════════════

comet_api_key     = "Yb0Ffy5WaM3qRxJ3qZAjvcboV"
comet_workspace   = "podtyazhki1337"
comet_project     = "unet3d-neurons"


# ══════════════════════════════════════════════════════════════════════════════
#  9.  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

seed = 42