"""
config.py — Single source of truth for nnU-Net experiment parameters.

Edit this file before each run.  It is logged to Comet as an artifact
so every experiment is fully reproducible from config alone.

Sections
────────
  1. Paths            – dataset location, nnU-Net dirs
  2. Dataset / split   – split ratios, skip/force lists, metadata
  3. nnU-Net dataset   – dataset ID, naming
  4. Model             – configuration, trainer, plans
  5. Training          – folds, epochs
  6. Augmentation      – preset selector (NO_AUG / DEFAULT)
  7. Post-processing   – instance segmentation strategy
  8. Hardware          – GPU
  9. Logging           – Comet ML credentials
 10. Reproducibility   – global seed
"""

from __future__ import annotations
import os

# ══════════════════════════════════════════════════════════════════════════════
#  1.  Paths
# ══════════════════════════════════════════════════════════════════════════════

# Raw dataset — must contain images/ and masks/ with matching .tif names.
in_dataset_dir = "/NAS/mmaiurov/datasets_benchmark/mice_neurons_oblique/"

# nnU-Net directory roots (from env variables, with fallbacks).
nnunet_raw           = os.environ.get("nnUNet_raw",          "/NAS/mmaiurov/nnUNet_data/nnUNet_raw")
nnunet_preprocessed  = os.environ.get("nnUNet_preprocessed", "/NAS/mmaiurov/nnUNet_data/nnUNet_preprocessed")
nnunet_results       = os.environ.get("nnUNet_results",      "/NAS/mmaiurov/nnUNet_data/nnUNet_results")


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Dataset / split
# ══════════════════════════════════════════════════════════════════════════════

val_ratio  = 0.15
test_ratio = 0.1

skip       = []
force_val  = []
force_test = []

# Metadata for Comet logging.
dataset_id = "mice_neurons_oblique"
modality   = "Oblique"
animal     = "mice"
voxel_size = (1.0, 1.0, 1.0)   # µm (Z, Y, X) — placeholder


# ══════════════════════════════════════════════════════════════════════════════
#  3.  nnU-Net dataset
# ══════════════════════════════════════════════════════════════════════════════

# Dataset ID — nnU-Net uses DatasetXXX_Name format.
nnunet_dataset_id   = 6
nnunet_dataset_name = "Dataset006_MiceNeuronsOblique"

# Segmentation approach: "binary" or "threeclass"
#   binary     → 0=bg, 1=neuron → CC post-processing for instances
#   threeclass → 0=bg, 1=border, 2=core → watershed for instances
seg_approach = "binary"

# Border width in voxels (only used if seg_approach="threeclass").
border_width = 2


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Model
# ══════════════════════════════════════════════════════════════════════════════

# nnU-Net configuration to train.
# Options: "2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"
configuration = "3d_fullres"

# Trainer class name (use custom for Comet logging).
trainer = "nnUNetTrainer_Comet"

# Plans identifier (default nnU-Net plans).
plans = "nnUNetPlans"


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Training
# ══════════════════════════════════════════════════════════════════════════════

# Which folds to train. Use [0,1,2,3,4] for full 5-fold CV,
# or ["all"] to train on all data (no CV).
# For benchmark with our own split: ["all"] — we map train+val to trainTr
# and use val as a held-out set via custom splits_final.json.
folds = [0]

# nnU-Net default is 1000. Can override in custom trainer.
num_epochs = 1000


# ══════════════════════════════════════════════════════════════════════════════
#  6.  Augmentation
# ══════════════════════════════════════════════════════════════════════════════

# nnU-Net always uses its built-in augmentation pipeline
# (rotation ±30°, scaling, elastic deform, gamma, mirror, noise, blur).
# This is the standard described in the Nature Methods paper.
# No NO_AUG option — augmentations are integral to the framework.
aug_preset = "DEFAULT"



# ══════════════════════════════════════════════════════════════════════════════
#  7.  Post-processing (instance segmentation)
# ══════════════════════════════════════════════════════════════════════════════

# Probability threshold for binarising softmax output.
prob_threshold = 0.5

# Minimum instance volume in voxels (filter out dust).
min_instance_volume = 15

# Connectivity for connected components (6, 18, or 26).
cc_connectivity = 26


# ══════════════════════════════════════════════════════════════════════════════
#  8.  Hardware
# ══════════════════════════════════════════════════════════════════════════════

gpu_id = 0


# ══════════════════════════════════════════════════════════════════════════════
#  9.  Logging — Comet ML
# ══════════════════════════════════════════════════════════════════════════════

comet_api_key    = ""           # fill in or use COMET_API_KEY env var
comet_project    = ""
comet_workspace  = ""           # fill in


# ══════════════════════════════════════════════════════════════════════════════
# 10.  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

seed = 42

# ══════════════════════════════════════════════════════════════════════════════
# 11.  Cross-domain experiments
# ══════════════════════════════════════════════════════════════════════════════
# Each experiment creates a merged dataset from multiple sources.
# train_sources[].use: "train" = train split, "train+val" = both, "all" = everything
# test_sources[].name: dataset to run inference on (uses its own split.json test set)
#
# After defining experiments:
#   python merge_datasets.py          # create merged datasets
#   python train.py                   # train (set nnunet_dataset_id/name to merged one)
#   python eval.py --subset test      # eval on each test source

cross_domain_experiments = [

    # ── Exp 1: Train on Rat+Mice DODT → Test on Human DODT ──────────────
    {
        "dataset_id": 101,
        "dataset_name": "Dataset101_XD_RatMice_Dodt",
        "description": "Cross-species: rat+mice DODT → human DODT",
        "train_sources": [
            {"name": "Dataset003_RatNeuronsDodt", "use": "train"},
            {"name": "Dataset002_MiceNeuronsDodt", "use": "train"},
        ],
        "test_sources": [
            {"name": "Dataset001_HumanNeuronsDodt", "use": "test"},
        ],
    },

    # ── Exp 2: Train on Rat+Mice Oblique → Test on Human Oblique ────────
    {
        "dataset_id": 102,
        "dataset_name": "Dataset102_XD_RatMice_Oblique",
        "description": "Cross-species: rat+mice oblique → human oblique",
        "train_sources": [
            {"name": "Dataset005_RatNeuronsOblique", "use": "train"},
            {"name": "Dataset006_MiceNeuronsOblique", "use": "train"},
        ],
        "test_sources": [
            {"name": "Dataset004_HumanNeuronsOblique", "use": "test"},
        ],
    },

    # ── Exp 3: Train on all 3 DODT → Test on own test sets ──────────────
    {
        "dataset_id": 103,
        "dataset_name": "Dataset103_XD_AllSpecies_Dodt",
        "description": "Mixed-species DODT: all 3 species combined",
        "train_sources": [
            {"name": "Dataset001_HumanNeuronsDodt", "use": "train"},
            {"name": "Dataset002_MiceNeuronsDodt", "use": "train"},
            {"name": "Dataset003_RatNeuronsDodt", "use": "train"},
        ],
        "test_sources": [
            {"name": "Dataset001_HumanNeuronsDodt", "use": "test"},
            {"name": "Dataset002_MiceNeuronsDodt", "use": "test"},
            {"name": "Dataset003_RatNeuronsDodt", "use": "test"},
        ],
    },

    # ── Exp 4: Train on all 3 Oblique → Test on own test sets ────────────
    {
        "dataset_id": 104,
        "dataset_name": "Dataset104_XD_AllSpecies_Oblique",
        "description": "Mixed-species oblique: all 3 species combined",
        "train_sources": [
            {"name": "Dataset004_HumanNeuronsOblique", "use": "train"},
            {"name": "Dataset005_RatNeuronsOblique", "use": "train"},
            {"name": "Dataset006_MiceNeuronsOblique", "use": "train"},
        ],
        "test_sources": [
            {"name": "Dataset004_HumanNeuronsOblique", "use": "test"},
            {"name": "Dataset005_RatNeuronsOblique", "use": "test"},
            {"name": "Dataset006_MiceNeuronsOblique", "use": "test"},
        ],
    },

    # ── Exp 5: Mixed-modality DODT+Oblique (all species) ─────────────────
    {
        "dataset_id": 105,
        "dataset_name": "Dataset105_XD_AllSpecies_Mixed",
        "description": "Mixed-modality+species: all 6 datasets combined",
        "train_sources": [
            {"name": "Dataset001_HumanNeuronsDodt", "use": "train"},
            {"name": "Dataset002_MiceNeuronsDodt", "use": "train"},
            {"name": "Dataset003_RatNeuronsDodt", "use": "train"},
            {"name": "Dataset004_HumanNeuronsOblique", "use": "train"},
            {"name": "Dataset005_RatNeuronsOblique", "use": "train"},
            {"name": "Dataset006_MiceNeuronsOblique", "use": "train"},
        ],
        "test_sources": [
            {"name": "Dataset001_HumanNeuronsDodt", "use": "test"},
            {"name": "Dataset002_MiceNeuronsDodt", "use": "test"},
            {"name": "Dataset003_RatNeuronsDodt", "use": "test"},
            {"name": "Dataset004_HumanNeuronsOblique", "use": "test"},
            {"name": "Dataset005_RatNeuronsOblique", "use": "test"},
            {"name": "Dataset006_MiceNeuronsOblique", "use": "test"},
        ],
    },
]