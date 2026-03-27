"""
config_microsam.py — Single source of truth for micro-SAM experiment parameters.

Edit this file before each run.  It is logged to Comet as an artifact
so every experiment is fully reproducible from config alone.

Sections
────────
  1. Paths            – dataset location, output
  2. Dataset / split   – metadata
  3. Model             – SAM variant, checkpoint, segmentation mode
  4. Inference (3D)    – tiling, batch size, gap closing
  5. Evaluation        – IoU thresholds for matching
  6. Hardware          – GPU
  7. Logging           – Comet ML credentials
  8. Reproducibility   – global seed
"""

from __future__ import annotations

import os


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Paths
# ══════════════════════════════════════════════════════════════════════════════

in_dataset_dir = "/NAS/mmaiurov/datasets_benchmark/mice_neurons_dodt/"

# Where to save predictions (instance label TIFs).
output_dir = "predictions"
run_name   = "microsam_vit_b_lm_finetuned_mice_dodt"


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Dataset / split
# ══════════════════════════════════════════════════════════════════════════════

dataset_id = "mice_neurons_dodt"
modality   = "dodt"
animal     = "mice"
voxel_size = (1.0, 1.0, 1.0)    # µm (Z, Y, X) — placeholder


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Model
# ══════════════════════════════════════════════════════════════════════════════

# SAM model variant:
#   "vit_b_lm"  — ViT-B finetuned for light microscopy (recommended start)
#   "vit_l_lm"  — ViT-L finetuned for LM (heavier, potentially better)
#   "vit_t_lm"  — ViT-Tiny for LM (fastest, less accurate)
#   "vit_b"     — default SAM ViT-B (no microscopy finetuning)
model_type = "vit_b_lm"

# Custom checkpoint path (None = use pretrained from model_type).
# Set this after finetuning to point to your trained model.
checkpoint_path = None

# Segmentation mode:
#   "amg"  — Automatic Mask Generator (works without decoder, SAM default)
#   "ais"  — Automatic Instance Segmentation (needs decoder from finetuning)
#   "apg"  — Automatic Prompt Generator
# For pretrained LM models that ship with a decoder, use "ais".
# For vanilla SAM without decoder, use "amg".
segmentation_mode = "ais"


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Inference (3D)
# ══════════════════════════════════════════════════════════════════════════════

# micro-sam does 3D by segmenting 2D slices and merging across Z.
# Gap closing: number of slices gap allowed when merging objects across Z.
gap_closing   = 2
min_z_extent  = 2       # minimum Z-extent for an object to be kept

# Tiling for large XY images (None = no tiling).
# If your images are >1024 px, consider e.g. (512, 512) with halo (64, 64).
tile_shape = None        # e.g. (512, 512)
halo       = None        # e.g. (64, 64)

# Batch size for computing embeddings across Z-slices.
embedding_batch_size = 1

# Predict on "test" split or "all" files.
predict_on = "test"


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

# IoU thresholds for instance matching (same as StarDist/Cellpose/UNet).
iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ══════════════════════════════════════════════════════════════════════════════
#  6.  Hardware
# ══════════════════════════════════════════════════════════════════════════════

gpu_id = "1"


# ══════════════════════════════════════════════════════════════════════════════
#  7.  Logging  (Comet ML)
# ══════════════════════════════════════════════════════════════════════════════

comet_api_key   = "Yb0Ffy5WaM3qRxJ3qZAjvcboV"
comet_workspace = "podtyazhki1337"
comet_project   = "microsam-neurons"


# ══════════════════════════════════════════════════════════════════════════════
#  8.  Training (fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════

# --- Step 1: fine-tune SAM encoder + mask decoder ---
sam_n_epochs          = 100
sam_lr                = 1e-5
sam_early_stopping    = 15
sam_n_objects_per_batch = 40
sam_freeze            = None       # e.g. ["image_encoder"] to freeze encoder
sam_batch_size        = 1
sam_patch_shape       = (512, 512) # 2D crop from each slice

# --- Step 2: train UNETR instance segmentation decoder ---
unetr_n_epochs        = 100
unetr_lr              = 1e-4
unetr_early_stopping  = 15
unetr_batch_size      = 1
unetr_patch_shape     = (512, 512)
train_unetr = False
# Common
n_samples_train       = 2000     # None = use all slices
n_samples_val         = 300
min_labels_per_slice  = 1         # skip slices with fewer instances

# Model save name (checkpoints go to ./checkpoints/{model_save_name}/)
model_save_name       = "microsam_vit_b_lm_finetuned_mice_dodt"

# Prepared 2D slices directory (auto-generated from 3D stacks).
# Stored on NAS alongside the dataset, same as cellpose/nnunet.
slices_dir            = os.path.join(in_dataset_dir, "slices_2d_microsam")


# ══════════════════════════════════════════════════════════════════════════════
#  9.  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

seed = 42