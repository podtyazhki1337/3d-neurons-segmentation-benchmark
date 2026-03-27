"""
config.py — Single source of truth for all experiment parameters.

Edit this file before each run.  It is logged to Comet as an artifact
so every experiment is fully reproducible from config alone.

Sections
────────
  1. Paths           – dataset location, model output
  2. Dataset / split  – split ratios, skip/force lists, metadata
  3. Data loading     – normalisation, pooling
  4. Model            – architecture, rays, patch, batch
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

# StarDist model output directory.
model_basedir = "models"
model_name    = "stardist_mice_neurons_dodt_lightaug_final"


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
dataset_id = "mice_neurons_dodt_lightaug_final"
modality   = "dodt"               # "oblique" / "dodt" / "fluorescence"
animal     = "mice"                # "rat" / "mouse" / "human"
voxel_size = (1.0, 1.0, 1.0)     # µm  (Z, Y, X) — SET THIS TO REAL VALUES!
                                  # If left isotropic, anisotropy is derived
                                  # from cell extents automatically.
pixel_size = None                 # µm  XY pixel size (if known)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Data loading
# ══════════════════════════════════════════════════════════════════════════════

# Percentile normalisation range applied to images on the fly.
norm_percentile_low  = 1
norm_percentile_high = 99.8

# Padding (voxels) around foreground bbox when cropping stacks.
padsize = 64

# Dataloader pooling (for large datasets that don't fit in RAM).
# None = load everything into memory.
pool_size   = None
repool_freq = 10


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Model
# ══════════════════════════════════════════════════════════════════════════════

n_rays           = 128
train_patch_size = (12, 96, 96)     # original StarDist config for rescaled data
train_batch_size = 20
unet_n_depth     = 2                # depth=2 FOV ≈ (25,93,93), sufficient after rescale


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Training
# ══════════════════════════════════════════════════════════════════════════════

epochs          = 400
lr_initial      = 3e-4
lr_decay_steps  = 700
lr_decay_rate   = 0.5
lr_staircase    = True

# DiceCheckpoint evaluates on val every N epochs and saves best weights.
dice_eval_every = 5

# Set True to run a quick 2-epoch smoke test.
quick_demo      = False


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

gpu_id          = "0"       # comma-separated GPU IDs, e.g. "0,1"
mixed_precision = False     # StarDist loss functions incompatible with fp16


# ══════════════════════════════════════════════════════════════════════════════
#  8.  Logging  (Comet ML)
# ══════════════════════════════════════════════════════════════════════════════

comet_api_key     = "APIKEY"
comet_workspace   = "WORKSPACE"
comet_project     = "stardist3d-neurons"


# ══════════════════════════════════════════════════════════════════════════════
#  9.  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

seed = 42