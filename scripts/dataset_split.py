r"""
=============================================================================
  D11 - Dataset Splitting (70/15/15 Train/Val/Test)
  Project: Deep Learning and VLM based Real-Time Fighting Detection
           on Beach Safety Management
  Student: Singo Loua | 240086608 | MSc Applied Data Science
  Supervisor: Dr. Ming Jiang | University of Sunderland
=============================================================================

WHAT THIS SCRIPT DOES:
  Takes the thermal frames produced by D10 and splits them into three sets:
    - Train  : 70%  (model learns from this)
    - Val    : 15%  (monitor progress during training)
    - Test   : 15%  (final evaluation only - never seen during training)

  IMPORTANT - Video-level splitting:
  Frames are grouped by their source video before splitting.
  All frames from the same video go into the same split.
  This prevents data leakage (where the model sees the same video
  in both training and testing, which would give falsely high results).

INPUT:
  C:\beach\data\thermal\
    fight\       <- all fighting class frames (subfolders per video)
    no_fight\    <- all non-fighting class frames (subfolders per video)

OUTPUT:
  C:\beach\data\split\
    train\
      fight\     <- 70% of fight video frames
      no_fight\  <- 70% of no_fight video frames
    val\
      fight\     <- 15% of fight video frames
      no_fight\  <- 15% of no_fight video frames
    test\
      fight\     <- 15% of fight video frames
      no_fight\  <- 15% of no_fight video frames

USAGE:
  cd C:\beach
  env\Scripts\activate
  python scripts\dataset_split.py

=============================================================================
"""

import os
import shutil
import random
import logging
from pathlib import Path
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────────

THERMAL_DIR = Path(r"C:\beach\data\thermal")
SPLIT_DIR   = Path(r"C:\beach\data\split")
LOG_DIR     = Path(r"C:\beach\logs")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42          # Fixed seed so split is reproducible
CLASSES     = ["fight", "no_fight"]

# ─── LOGGING ──────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "dataset_split.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_video_groups(class_dir: Path) -> dict:
    """
    Scan a class folder and group frame files by their source video.

    D10 saved frames like:
      fight/beach_youtube/video_name/frame_000001.jpg
      fight/beach_youtube/video_name/frame_000002.jpg

    This function returns a dict:
      { "beach_youtube/video_name": [path1, path2, ...], ... }

    Each key is one video. All frames from that video are its value.
    This is what allows us to split at video level, not frame level.
    """
    groups = defaultdict(list)

    for frame_path in sorted(class_dir.rglob("*.jpg")):
        # Build a unique key from the path relative to the class folder
        # e.g. beach_youtube/BEACH_BRAWL_3/frame_000001.jpg
        # key = beach_youtube/BEACH_BRAWL_3
        relative = frame_path.relative_to(class_dir)
        parts = relative.parts

        if len(parts) >= 2:
            # Group by dataset_name/video_name
            video_key = str(Path(*parts[:-1]))
        else:
            # Frame is directly in class folder (no subfolder)
            video_key = "root"

        groups[video_key].append(frame_path)

    return dict(groups)


def split_groups(groups: dict, seed: int = RANDOM_SEED) -> tuple:
    """
    Split a dict of video groups into train/val/test at video level.

    Returns three dicts: train_groups, val_groups, test_groups
    Each dict has the same structure as the input.
    """
    random.seed(seed)

    video_keys = sorted(groups.keys())
    random.shuffle(video_keys)

    n = len(video_keys)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # test gets the remainder to ensure all videos are assigned
    n_test  = n - n_train - n_val

    train_keys = video_keys[:n_train]
    val_keys   = video_keys[n_train:n_train + n_val]
    test_keys  = video_keys[n_train + n_val:]

    train_groups = {k: groups[k] for k in train_keys}
    val_groups   = {k: groups[k] for k in val_keys}
    test_groups  = {k: groups[k] for k in test_keys}

    return train_groups, val_groups, test_groups


def copy_frames(groups: dict, dest_dir: Path, split_name: str,
                class_name: str) -> int:
    """
    Copy all frames from a group dict into dest_dir.

    Frames are renamed sequentially (frame_000001.jpg etc.)
    so all videos in the split are in one flat folder per class.
    This is the standard format expected by YOLOv8 training.

    Returns total number of frames copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    counter = 0

    for video_key, frame_paths in sorted(groups.items()):
        for frame_path in sorted(frame_paths):
            dest = dest_dir / f"frame_{counter:07d}.jpg"
            shutil.copy2(str(frame_path), str(dest))
            counter += 1

    log.info(f"    {split_name:6s} / {class_name:10s} : {counter:6,} frames "
             f"from {len(groups):4d} videos")
    return counter


def verify_no_leakage(train_g: dict, val_g: dict, test_g: dict,
                      class_name: str) -> None:
    """
    Confirm that no video appears in more than one split.
    Logs a warning if any overlap is found.
    """
    train_set = set(train_g.keys())
    val_set   = set(val_g.keys())
    test_set  = set(test_g.keys())

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set

    if tv or tt or vt:
        log.warning(f"  [{class_name}] DATA LEAKAGE DETECTED:")
        if tv: log.warning(f"    Train/Val overlap : {tv}")
        if tt: log.warning(f"    Train/Test overlap: {tt}")
        if vt: log.warning(f"    Val/Test overlap  : {vt}")
    else:
        log.info(f"    [{class_name}] No data leakage — all splits are clean.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("  D11 Dataset Splitting — 70 / 15 / 15")
    log.info(f"  Student : Singo Loua | 240086608")
    log.info(f"  Seed    : {RANDOM_SEED} (fixed for reproducibility)")
    log.info(f"  Input   : {THERMAL_DIR}")
    log.info(f"  Output  : {SPLIT_DIR}")
    log.info("=" * 65)

    # Track totals for the final summary
    summary = {}

    for class_name in CLASSES:
        class_dir = THERMAL_DIR / class_name

        if not class_dir.exists():
            log.warning(f"Class folder not found: {class_dir} — skipping.")
            continue

        log.info(f"\nProcessing class: {class_name}")
        log.info(f"  Scanning frames in {class_dir} ...")

        # Group frames by source video
        groups = get_video_groups(class_dir)
        total_videos = len(groups)
        total_frames = sum(len(v) for v in groups.values())

        log.info(f"  Found {total_videos} videos, {total_frames:,} frames total")

        if total_videos < 10:
            log.warning(f"  Only {total_videos} videos — split may be uneven. "
                        f"Consider adding more data.")

        # Split at video level
        train_g, val_g, test_g = split_groups(groups)

        log.info(f"  Videos assigned: "
                 f"train={len(train_g)} | val={len(val_g)} | test={len(test_g)}")

        # Verify no leakage
        verify_no_leakage(train_g, val_g, test_g, class_name)

        # Copy frames to split folders
        log.info(f"  Copying frames...")
        n_train = copy_frames(train_g, SPLIT_DIR/"train"/class_name, "train", class_name)
        n_val   = copy_frames(val_g,   SPLIT_DIR/"val"  /class_name, "val",   class_name)
        n_test  = copy_frames(test_g,  SPLIT_DIR/"test" /class_name, "test",  class_name)

        total = n_train + n_val + n_test
        summary[class_name] = {
            "videos": total_videos,
            "total":  total,
            "train":  n_train,
            "val":    n_val,
            "test":   n_test,
        }

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  D11 COMPLETE — Final Dataset Summary")
    log.info("=" * 65)
    log.info(f"\n  {'Class':<12} {'Videos':>7} {'Total':>8} "
             f"{'Train(70%)':>11} {'Val(15%)':>9} {'Test(15%)':>9}")
    log.info("  " + "-" * 60)

    grand_total = grand_train = grand_val = grand_test = 0

    for cls, s in summary.items():
        log.info(f"  {cls:<12} {s['videos']:>7,} {s['total']:>8,} "
                 f"{s['train']:>11,} {s['val']:>9,} {s['test']:>9,}")
        grand_total += s['total']
        grand_train += s['train']
        grand_val   += s['val']
        grand_test  += s['test']

    log.info("  " + "-" * 60)
    log.info(f"  {'TOTAL':<12} {'':>7} {grand_total:>8,} "
             f"{grand_train:>11,} {grand_val:>9,} {grand_test:>9,}")

    log.info(f"\n  Actual split ratios:")
    log.info(f"    Train : {grand_train/grand_total*100:.1f}%")
    log.info(f"    Val   : {grand_val  /grand_total*100:.1f}%")
    log.info(f"    Test  : {grand_test /grand_total*100:.1f}%")

    log.info(f"\n  Output folder : {SPLIT_DIR}")
    log.info(f"  Log file      : {LOG_DIR / 'dataset_split.log'}")
    log.info(f"\n  NEXT STEP: D12 — Train YOLOv8 baseline model")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
