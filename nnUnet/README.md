# nnU-Net — 3D Neuron Instance Segmentation

Binary semantic segmentation via nnU-Net v2 with connected-component post-processing for instance labels.

## Setup

```bash
conda create -n nnunet python=3.11 -y && conda activate nnunet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/MIC-DKFZ/nnUNet.git && cd nnUNet && pip install -e .
pip install comet-ml tifffile connected-components-3d scipy
```

Set environment variables (`~/.bashrc`):
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
export nnUNet_compile=0
```

## Files

| File | Purpose |
|---|---|
| `config.py` | All experiment parameters (paths, dataset, training, post-processing, cross-domain) |
| `convert_dataset.py` | TIF dataset → nnU-Net format (reads `split.json`, creates `dataset.json` + `splits_final.json`) |
| `train.py` | Preprocessing + training wrapper with Comet ML logging |
| `eval.py` | Inference + metrics (AJI, Obj F1/Prec/Rec/IoU, COCO mAP, Pixel Dice/IoU) + visualizations |
| `nnUNetTrainer_Comet.py` | Custom trainer with Comet ML experiment tracking |
| `merge_datasets.py` | Create cross-domain datasets from existing preprocessed sources via symlinks |
| `eval_crossdomain.py` | Cross-domain inference on multiple test sources |

## Usage

### Single-dataset training
```bash
# 1. Edit config.py (paths, dataset_id, dataset_name)
# 2. Convert & train
python convert_dataset.py
python train.py
# 3. Evaluate
python eval.py                  # with TTA
python eval.py --no-tta         # without TTA
python eval.py --subset val     # on validation set
```

### Cross-domain training
```bash
# 1. Define experiments in config.py (section 11)
# 2. Create merged datasets & train
python merge_datasets.py
python train.py --skip-prep     # after setting dataset_id in config
# 3. Evaluate
python eval_crossdomain.py --experiment 101
```

## Instance segmentation pipeline

nnU-Net performs **semantic** segmentation (background vs neuron). Instance labels are obtained via 3D connected component analysis (`cc3d`, connectivity=26) of the thresholded binary output, with small objects filtered by `min_instance_volume` (default: 50 voxels).

## Metrics

All metrics use **Hungarian matching** (`scipy.optimize.linear_sum_assignment`) for optimal GT↔prediction assignment. mAP is COCO-style (AP averaged over IoU thresholds 0.50:0.05:0.95).

## Key notes

- `dataset.json` must not contain a `"modality"` key — nnU-Net reserves it for channel counting. Use `"imaging_modality"` instead.
- nnU-Net augmentations (rotation, scaling, elastic deform, gamma, mirror, noise, blur) are always active — no `NO_AUG` ablation.
- Cross-domain datasets use symlinks to preprocessed files — no data duplication.
