r"""
D12 - YOLOv8 Baseline Training (CPU-Optimised Subset Version)
Student: Singo Loua | 240086608 | CRISP-DM Phase 4: Modelling
"""

import argparse
import logging
import shutil
import random
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

ROOT        = Path(r"C:\beach")
SPLIT_DIR   = ROOT / "data" / "split"
SUBSET_DIR  = ROOT / "data" / "split_subset"
MODELS_DIR  = ROOT / "models"
LOG_DIR     = ROOT / "logs"
RUN_NAME    = "d12_baseline"

DEFAULT_TRAIN_SIZE = 5000
DEFAULT_VAL_SIZE   = 1500
DEFAULT_EPOCHS     = 20
DEFAULT_BATCH      = 8
DEFAULT_IMG_SIZE   = 224
DEFAULT_PATIENCE   = 10
RANDOM_SEED        = 42
MODEL_NAME         = "yolov8n-cls.pt"
WORKERS            = 0

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train_baseline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def create_subset(train_size, val_size):
    random.seed(RANDOM_SEED)
    classes = ["fight", "no_fight"]

    if SUBSET_DIR.exists():
        log.info("Removing previous subset...")
        shutil.rmtree(str(SUBSET_DIR))

    for split, total in [("train", train_size), ("val", val_size)]:
        available = {}
        for cls in classes:
            imgs = list((SPLIT_DIR / split / cls).glob("*.jpg"))
            available[cls] = imgs

        total_available = sum(len(v) for v in available.values())
        log.info(f"\n  {split}: {total_available:,} available, selecting {total:,}")

        per_class = {}
        for cls in classes:
            ratio = len(available[cls]) / total_available
            per_class[cls] = min(int(total * ratio), len(available[cls]))

        allocated = sum(per_class.values())
        if allocated < total:
            per_class["fight"] = min(
                per_class["fight"] + (total - allocated),
                len(available["fight"])
            )

        for cls in classes:
            selected = random.sample(available[cls], per_class[cls])
            dest = SUBSET_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in selected:
                shutil.copy2(str(img), str(dest / img.name))
            log.info(f"    {split}/{cls}: {per_class[cls]:,} images copied")

    log.info(f"\n  Copying test set (full)...")
    for cls in classes:
        src  = SPLIT_DIR / "test" / cls
        dest = SUBSET_DIR / "test" / cls
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(src), str(dest))
        log.info(f"    test/{cls}: {len(list(dest.glob('*.jpg'))):,} images")

    return SUBSET_DIR


def train(train_size, val_size, epochs, batch, img_size, patience):
    log.info("=" * 65)
    log.info("  D12 YOLOv8 Baseline — CPU Subset Training")
    log.info(f"  Student    : Singo Loua | 240086608")
    log.info(f"  Methodology: CRISP-DM Phase 4 — Modelling")
    log.info(f"  Train size : {train_size:,} images (subset of 42,178)")
    log.info(f"  Val size   : {val_size:,} images")
    log.info(f"  Epochs     : {epochs}")
    log.info(f"  Batch      : {batch}")
    log.info(f"  Img size   : {img_size}px")
    log.info(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    log.info("\nCreating stratified subset...")
    subset_path = create_subset(train_size, val_size)
    log.info(f"\nSubset ready. Starting training...")
    log.info("Each epoch should take 5-15 minutes on CPU.\n")

    model = YOLO(MODEL_NAME)

    model.train(
        data     = str(subset_path),
        epochs   = epochs,
        batch    = batch,
        imgsz    = img_size,
        patience = patience,
        workers  = WORKERS,
        device   = "cpu",
        project  = str(MODELS_DIR),
        name     = RUN_NAME,
        exist_ok = True,
        verbose  = True,
        plots    = True,
        save     = True,
        val      = True,
    )

    log.info("\n" + "=" * 65)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Best model: {MODELS_DIR / RUN_NAME / 'weights' / 'best.pt'}")
    log.info("=" * 65)
    evaluate()


def evaluate():
    best = MODELS_DIR / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        log.error(f"No model found at {best}. Run training first.")
        return

    log.info("\nEvaluating on full test set...")
    model = YOLO(str(best))
    metrics = model.val(
        data     = str(SUBSET_DIR),
        split    = "test",
        device   = "cpu",
        workers  = WORKERS,
        plots    = True,
        project  = str(MODELS_DIR),
        name     = RUN_NAME + "_test_eval",
        exist_ok = True,
    )

    log.info("\n" + "=" * 65)
    log.info("  D12 FINAL TEST RESULTS")
    log.info("=" * 65)
    try:
        log.info(f"  Top-1 Accuracy : {metrics.top1*100:.2f}%")
        log.info(f"  Top-5 Accuracy : {metrics.top5*100:.2f}%")
    except Exception:
        log.info("  See results files in models folder.")
    log.info(f"\n  Results: {MODELS_DIR / RUN_NAME}")
    log.info(f"  Log    : {LOG_DIR / 'train_baseline.log'}")
    log.info("\n  NEXT STEP: D13 — Full training on Google Colab (GPU)")
    log.info("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int,   default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val_size",   type=int,   default=DEFAULT_VAL_SIZE)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",      type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",      type=int,   default=DEFAULT_IMG_SIZE)
    parser.add_argument("--patience",   type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--eval_only",  action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        train(args.train_size, args.val_size, args.epochs,
              args.batch, args.imgsz, args.patience)


if __name__ == "__main__":
    main()
