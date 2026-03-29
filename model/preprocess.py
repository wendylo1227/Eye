"""
Prepares the MRL dataset for hardware training and deployment.

1. Merges 'test' data into 'train' to maximize training set size.
2. Converts all images to 32x32 Grayscale (required for FPGA synchronization).
"""

import os
import shutil
import cv2
import glob
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path('D:/University/Junior/HLS/VGG_Accelerator/data/MRL')
TRAIN_DIR = BASE_DIR / 'train'
TEST_DIR = BASE_DIR / 'test'
VAL_DIR = BASE_DIR / 'val'
OUTPUT_DIR = BASE_DIR / 'dataset_final_32x32'
TARGET_SIZE = (32, 32)

def merge_test_into_train():
    """
    Moves all images from 'test' to 'train'.
    Handles naming conflicts and removes the 'test' directory upon completion.
    """
    print(f"--- Step 1: Merging {TEST_DIR} into {TRAIN_DIR} ---")
    
    if not TEST_DIR.exists():
        print("Test directory not found or already moved. Skipping.")
        return

    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            src_path = Path(root) / file
            
            # Replicate subfolder structure
            relative_path = src_path.relative_to(TEST_DIR)
            dest_path = TRAIN_DIR / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle duplicate filenames
            if dest_path.exists():
                name = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_path.parent / f"{name}_merged_{counter}{suffix}"
                    counter += 1
            
            shutil.move(str(src_path), str(dest_path))

    shutil.rmtree(TEST_DIR)
    print("Merge complete. 'test' folder removed.")

def preprocess_image(src_path, dest_path):
    """Reads image, converts to single-channel grayscale, and resizes to 32x32."""
    img = cv2.imread(str(src_path))
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE)
    cv2.imwrite(str(dest_path), resized)

def process_dataset():
    """Iterates over Train/Val sets and processes all images."""
    print(f"\n--- Step 2: Preprocessing Images (Resize {TARGET_SIZE} & Grayscale) ---")
    
    sets_to_process = ['train', 'val']

    for subset in sets_to_process:
        src_subset_dir = BASE_DIR / subset
        dest_subset_dir = OUTPUT_DIR / subset

        if not src_subset_dir.exists():
            print(f"Warning: {src_subset_dir} does not exist. Skipping.")
            continue

        images = list(src_subset_dir.glob('**/*.*'))
        print(f"Processing {subset}: {len(images)} images found...")

        for img_path in images:
            relative_path = img_path.relative_to(src_subset_dir)
            dest_path = dest_subset_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            preprocess_image(img_path, dest_path)

    print(f"\nSuccess! Processed data saved to: {OUTPUT_DIR.resolve()}")
    print("You can now use this folder for Phase 2 Training.")

if __name__ == "__main__":
    merge_test_into_train()
    process_dataset()