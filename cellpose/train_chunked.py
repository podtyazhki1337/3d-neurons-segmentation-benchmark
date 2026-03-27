"""
train_chunked.py — Cellpose training pipeline: slice → chunk → train.

1. Preprocessing: slices 3D TIF stacks into 2D (skips if already done)
2. Training: feeds slices to CLI in RAM-safe chunks

Just set CONFIG and run:
    CUDA_VISIBLE_DEVICES=2 python train_chunked.py
"""

import os, sys, json, shutil, subprocess, random
import numpy as np
from glob import glob
from pathlib import Path
from math import ceil
from tifffile import imread, imwrite
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_DIR     = "/NAS/mmaiurov/datasets_benchmark/mice_neurons_dodt/"
SLICES_DIR      = os.path.join(DATASET_DIR, "slices_2d")
SPLIT_PATH      = os.path.join(DATASET_DIR, "split.json")

# Model
PRETRAINED      = None  # None = from scratch, or path to checkpoint
MODEL_NAME      = "mice_neurons_dodt_chunked"

# Training
TOTAL_EPOCHS    = 500
CHUNK_SIZE      = 400
EPOCHS_PER_CHUNK = None     # None = auto (TOTAL_EPOCHS / n_chunks)
SAVE_EVERY      = 50
MIN_TRAIN_MASKS = 1
MIN_NEURONS_PER_SLICE = 3   # min labelled instances to keep a Z-slice
LR              = 0.2

# Hardware
CUDA_DEVICE     = "0"
SEED            = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Preprocessing: 3D stacks → 2D slices on disk
# ═══════════════════════════════════════════════════════════════════════════════
def slice_dataset(dataset_dir, split_path, slices_dir, min_neurons=1):
    """Slice 3D TIF stacks into 2D images + masks on disk.

    Creates:
        slices_dir/train/<stem>_z<NNNN>.tif
        slices_dir/train/<stem>_z<NNNN>_masks.tif
        slices_dir/val/...

    Skips entirely if slices_dir/train already has .tif files.
    Processes one stack at a time — minimal RAM usage.
    """
    with open(split_path) as f:
        split = json.load(f)

    for subset in ("train", "val"):
        out_dir = os.path.join(slices_dir, subset)

        # Check if already sliced
        existing = [f for f in glob(os.path.join(out_dir, "*.tif"))
                    if "_masks" not in f and "_flows" not in f]
        if existing:
            print(f"  {subset}: {len(existing)} slices already exist in {out_dir} — skipping")
            continue

        os.makedirs(out_dir, exist_ok=True)
        file_list = split[subset]
        total_kept = 0

        for fname in tqdm(file_list, desc=f"Slicing {subset}"):
            stem = Path(fname).stem
            img = imread(os.path.join(dataset_dir, "images", fname))
            msk = imread(os.path.join(dataset_dir, "masks", fname))

            for z in range(img.shape[0]):
                sl_mask = msk[z]
                n_labels = len(np.unique(sl_mask)) - (1 if 0 in sl_mask else 0)
                if n_labels >= min_neurons:
                    imwrite(os.path.join(out_dir, f"{stem}_z{z:04d}.tif"),
                            img[z])
                    imwrite(os.path.join(out_dir, f"{stem}_z{z:04d}_masks.tif"),
                            sl_mask.astype(np.uint16))
                    total_kept += 1

            del img, msk  # free RAM per stack

        print(f"  {subset}: {total_kept} slices saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Training helpers
# ═══════════════════════════════════════════════════════════════════════════════
def find_images(data_dir):
    all_tifs = sorted(glob(os.path.join(data_dir, "*.tif")))
    return [f for f in all_tifs if "_masks" not in f and "_flows" not in f]


def create_chunk_dir(base_dir, chunk_id, image_files):
    chunk_dir = os.path.join(base_dir, f"_chunk_{chunk_id:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    for f in glob(os.path.join(chunk_dir, "*.tif")):
        os.remove(f)
    for img_path in image_files:
        base = os.path.basename(img_path)
        mask_path = img_path.replace(".tif", "_masks.tif")
        dst = os.path.join(chunk_dir, base)
        if not os.path.exists(dst):
            os.symlink(img_path, dst)
        if os.path.exists(mask_path):
            mask_dst = os.path.join(chunk_dir, os.path.basename(mask_path))
            if not os.path.exists(mask_dst):
                os.symlink(mask_path, mask_dst)
    return chunk_dir


def run_cellpose_train(train_dir, val_dir, pretrained, model_name,
                       n_epochs, save_every, lr, min_masks):
    cmd = [
        sys.executable, "-m", "cellpose", "--train",
        "--dir", train_dir,
        "--mask_filter", "_masks",
        "--test_dir", val_dir,
        "--pretrained_model", pretrained,
        "--model_name", model_name,
        "--n_epochs", str(n_epochs),
        "--save_every", str(save_every),
        "--learning_rate", str(lr),
        "--min_train_masks", str(min_masks),
        "--use_gpu",
        "--verbose",
    ]

    print(f"\n{'─'*60}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}\n")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_DEVICE}
    result = subprocess.run(cmd, env=env)
    return result.returncode


def find_latest_model(train_dir, model_name):
    models_dir = os.path.join(train_dir, "models")
    if not os.path.exists(models_dir):
        return None
    exact = os.path.join(models_dir, model_name)
    if os.path.isfile(exact):
        return exact
    candidates = sorted(
        [f for f in Path(models_dir).rglob("*")
         if f.is_file() and f.stat().st_size > 500_000 and "losses" not in f.name],
        key=lambda f: f.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    random.seed(SEED)

    # ── Step 1: Preprocessing ────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  STEP 1: Slice 3D → 2D")
    print(f"{'='*60}")

    slice_dataset(
        dataset_dir=DATASET_DIR,
        split_path=SPLIT_PATH,
        slices_dir=SLICES_DIR,
        min_neurons=MIN_NEURONS_PER_SLICE,
    )

    train_dir = os.path.join(SLICES_DIR, "train")
    val_dir   = os.path.join(SLICES_DIR, "val")

    # ── Step 2: Chunked training ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 2: Chunked training")
    print(f"{'='*60}")

    all_train_images = find_images(train_dir)
    n_total = len(all_train_images)
    n_chunks = ceil(n_total / CHUNK_SIZE)

    print(f"  Train images: {n_total}")
    print(f"  Val images:   {len(find_images(val_dir))}")
    print(f"  Chunk size:   {CHUNK_SIZE}")
    print(f"  Chunks:       {n_chunks}")
    print(f"  GPU:          CUDA_VISIBLE_DEVICES={CUDA_DEVICE}")

    if EPOCHS_PER_CHUNK is not None:
        epc = EPOCHS_PER_CHUNK
    else:
        epc = max(1, TOTAL_EPOCHS // n_chunks)
    print(f"  Epochs/chunk: {epc}")
    print(f"  Total epochs: {epc * n_chunks}")

    random.shuffle(all_train_images)

    chunks = []
    for i in range(0, n_total, CHUNK_SIZE):
        chunks.append(all_train_images[i:i + CHUNK_SIZE])
    print(f"  Chunk sizes:  {[len(c) for c in chunks]}")

    current_model = PRETRAINED if PRETRAINED else "cyto3"

    for ci, chunk_files in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"  CHUNK {ci+1}/{n_chunks}: {len(chunk_files)} images, {epc} epochs")
        print(f"  Model: {current_model}")
        print(f"{'='*60}")

        chunk_dir = create_chunk_dir(SLICES_DIR, ci, chunk_files)
        chunk_model_name = f"{MODEL_NAME}_chunk{ci:03d}"

        ret = run_cellpose_train(
            train_dir=chunk_dir, val_dir=val_dir,
            pretrained=current_model,
            model_name=chunk_model_name,
            n_epochs=epc,
            save_every=min(SAVE_EVERY, epc),
            lr=LR, min_masks=MIN_TRAIN_MASKS)

        if ret != 0:
            print(f"  ❌  Chunk {ci+1} failed with code {ret}")
            sys.exit(1)

        new_model = find_latest_model(chunk_dir, chunk_model_name)
        if new_model:
            current_model = new_model
            print(f"  ✅  Chunk {ci+1} done → {current_model}")
        else:
            print(f"  ⚠️  No model found after chunk {ci+1}, keeping previous")

    # ── Copy final model ─────────────────────────────────────────────────
    final_dir = os.path.join("models", MODEL_NAME)
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, MODEL_NAME)
    if current_model and os.path.isfile(current_model):
        shutil.copy(current_model, final_path)
        print(f"\n💾  Final model: {final_path}")

    # ── Cleanup chunk dirs ───────────────────────────────────────────────
    for ci in range(n_chunks):
        chunk_dir = os.path.join(SLICES_DIR, f"_chunk_{ci:03d}")
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
    print("🗑️  Cleaned up chunk directories")

    print(f"\n✅  Done. {n_chunks} chunks × {epc} epochs = {n_chunks * epc} total")