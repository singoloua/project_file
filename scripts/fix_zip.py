"""
fix_zip.py - Creates a Linux-compatible zip file from the split dataset
Converts Windows backslashes to forward slashes in all file paths.
Student: Singo Loua | 240086608
"""

import zipfile
import os
from pathlib import Path
from tqdm import tqdm

SPLIT_DIR = Path(r"C:\beach\data\split")
OUTPUT_ZIP = Path(r"C:\beach\data\split_dataset_fixed.zip")

print("="*60)
print("Creating Linux-compatible zip file...")
print(f"Source : {SPLIT_DIR}")
print(f"Output : {OUTPUT_ZIP}")
print("="*60)

# Count total files first
total = sum(1 for _ in SPLIT_DIR.rglob("*.jpg"))
print(f"Total frames to zip: {total:,}")
print("This will take a few minutes...\n")

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file_path in tqdm(SPLIT_DIR.rglob("*.jpg"), total=total, unit="files"):
        # Create archive name with forward slashes
        arcname = "split/" + "/".join(file_path.relative_to(SPLIT_DIR).parts)
        zf.write(str(file_path), arcname)

size_mb = OUTPUT_ZIP.stat().st_size / (1024*1024)
print(f"\nDone! Zip created: {OUTPUT_ZIP}")
print(f"File size: {size_mb:.1f} MB")
print("\nNow upload split_dataset_fixed.zip to Kaggle instead.")
