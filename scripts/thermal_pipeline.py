r"""
=============================================================================
  D10 - Thermal Warm-Silhouette Pipeline
  Project: Deep Learning and VLM based Real-Time Fighting Detection
           on Beach Safety Management
  Student: Singo Loua | 240086608 | MSc Applied Data Science
  Supervisor: Dr. Ming Jiang | University of Sunderland
=============================================================================

WHAT THIS SCRIPT DOES:
  Reads every video from all the datasets, detects people using YOLOv8
  segmentation, and converts them to thermal warm-silhouette frames
  (glowing orange/yellow shapes on a dark background) for GDPR compliance.

  All processed frames are saved to C:\beach\data\thermal\ organised by
  dataset and class label (fight / no_fight), ready for D11 splitting.

MY ACTUAL FOLDER STRUCTURE (auto-detected):
  C:\beach\data\raw\
    beach\fighting\                          → fight
    gemini\                                  → fight
    hockey-fight\                            → fight  (1,000 .xvid)
    movies-fight\fights\                     → fight
    movies-fight\noFights\                   → no_fight
    real-life-violence\...\Violence\         → fight  (1,000 videos)
    real-life-violence\...\NonViolence\      → no_fight (1,000 videos)

OUTPUT:
  C:\beach\data\thermal\
    fight\      ← all fighting class frames
    no_fight\   ← all non-fighting class frames

USAGE:
  First activate my environment:
    cd C:\beach
    env\Scripts\activate

  And run:
    python scripts\thermal_pipeline.py

  Preview one video first (recommended):
    python scripts\thermal_pipeline.py --preview "C:\beach\data\raw\beach\fighting\Surfers fight on the beach #fight #viral #surfer #mma #streetfi....mp4"

  Run one dataset only:
    python scripts\thermal_pipeline.py --dataset hockey-fight

=============================================================================
"""

import cv2
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ─── PATHS ────────────────────────────────────────────────────────────────────

ROOT        = Path("C:/beach")
RAW_DIR     = ROOT / "data" / "raw"
OUTPUT_DIR  = ROOT / "data" / "thermal"
LOG_DIR     = ROOT / "logs"

# Output class folders
FIGHT_OUT    = OUTPUT_DIR / "fight"
NOFIGHT_OUT  = OUTPUT_DIR / "no_fight"

# ─── DATASET MAP ──────────────────────────────────────────────────────────────
# Maps each source folder to its class label.
# Uses rglob so it finds videos inside any subfolders automatically.

DATASETS = [
    # ( folder_path,                                    label,      dataset_name     )
    ( RAW_DIR / "beach" / "fighting",                  "fight",    "beach_youtube"  ),
    ( RAW_DIR / "gemini" / "fight",                    "fight",    "gemini_fight"   ),
    ( RAW_DIR / "gemini" / "no_fight", "no_fight", "gemini_nofight" ),
    ( RAW_DIR / "hockey-fight",                         "fight",    "hockey_fight"   ),
    ( RAW_DIR / "movies-fight" / "fights",              "fight",    "movies_fight"   ),
    ( RAW_DIR / "movies-fight" / "noFights",            "no_fight", "movies_nofight" ),
    # Real Life Violence — uses rglob so it finds Violence/ and NonViolence/ inside subfolders
    ( RAW_DIR / "real-life-violence",                   "AUTO",     "real_life"      ),
]

# For real-life-violence we detect the class from the subfolder name
REAL_LIFE_FIGHT_FOLDER    = "Violence"
REAL_LIFE_NOFIGHT_FOLDER  = "NonViolence"

# ─── SETTINGS ─────────────────────────────────────────────────────────────────

DEFAULT_FPS_SAMPLE  = 5      # Extract N frames per second (5 = good balance of coverage vs storage)
DEFAULT_CONFIDENCE  = 0.40   # Min YOLOv8 confidence to count a person detection
MAX_FRAMES_PER_VID  = 200    # Max frames saved per video (keeps dataset manageable)
IMG_SIZE            = (640, 640)
YOLO_MODEL          = "yolov8n-seg.pt"  # Nano seg model — downloads automatically if not present
JPEG_QUALITY        = 92

# Thermal render settings
BACKGROUND_COLOUR   = (10, 10, 20)   # Very dark blue-black (BGR)
GLOW_BOOST          = 1.3            # Brightness multiplier for the warm body glow
BLUR_KERNEL         = 7              # Edge softening (makes it look like real thermal)

# Supported video formats (includes .xvid used by hockey-fight dataset)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".xvid", ".mpg", ".mpeg"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "thermal_pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── THERMAL RENDERING ────────────────────────────────────────────────────────

def render_thermal(frame: np.ndarray, masks: list) -> np.ndarray:
    """
    Convert a video frame to a thermal warm-silhouette representation.

    Detected people become glowing orange/yellow warm shapes on a dark
    background. No face, skin, hair or clothing information is retained.
    Body shape and pose are preserved for fight detection.

    Args:
        frame : original video frame, BGR, shape (H, W, 3)
        masks : list of binary person masks, each shape (H, W)

    Returns:
        thermal frame, BGR, shape (H, W, 3)
    """
    h, w = frame.shape[:2]
    output = np.full((h, w, 3), BACKGROUND_COLOUR, dtype=np.uint8)

    if not masks:
        return output

    heat = np.zeros((h, w), dtype=np.float32)

    for mask in masks:
        # Resize mask to match frame if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        # Soft glow at edges
        blurred = cv2.GaussianBlur(mask, (BLUR_KERNEL, BLUR_KERNEL), 0)
        heat = np.maximum(heat, blurred)

    heat = np.clip(heat * GLOW_BOOST, 0.0, 1.0)
    heat_u8 = (heat * 255).astype(np.uint8)

    # HOT colourmap: black → red → orange → yellow → white
    # This is exactly what real thermal cameras produce for warm bodies
    coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HOT)

    # Blend onto dark background
    alpha = heat[:, :, np.newaxis]
    output = (coloured * alpha + output * (1 - alpha)).astype(np.uint8)
    return output


def get_person_masks(result, h: int, w: int, min_conf: float) -> list:
    """Extract person segmentation masks from a YOLOv8 result."""
    masks = []
    if result.masks is None:
        return masks

    raw_masks = result.masks.data.cpu().numpy()
    classes   = result.boxes.cls.cpu().numpy().astype(int)
    confs     = result.boxes.conf.cpu().numpy()

    for i, (cls, conf) in enumerate(zip(classes, confs)):
        if cls != 0 or conf < min_conf:
            continue
        mask = cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
        masks.append((mask > 0.5).astype(np.float32))

    return masks

# ─── VIDEO PROCESSING ─────────────────────────────────────────────────────────

def process_video(video_path: Path, out_dir: Path, model: YOLO,
                  fps_sample: int, max_frames: int, min_conf: float) -> dict:
    """Process one video file through the thermal pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open: {video_path.name}")
        return {"saved": 0, "people": 0}

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, int(round(video_fps / fps_sample)))

    saved = 0
    people_total = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        frame = cv2.resize(frame, IMG_SIZE)
        results = model(frame, verbose=False, classes=[0])  # only detect persons
        masks = get_person_masks(results[0], IMG_SIZE[1], IMG_SIZE[0], min_conf)
        people_total += len(masks)

        thermal = render_thermal(frame, masks)

        out_path = out_dir / f"frame_{saved:06d}.jpg"
        cv2.imwrite(str(out_path), thermal, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1

    cap.release()
    return {"saved": saved, "people": people_total}

# ─── DATASET PROCESSING ───────────────────────────────────────────────────────

def process_folder(folder: Path, label: str, dataset_name: str,
                   model: YOLO, fps: int, max_frames: int,
                   min_conf: float, only: str = None) -> None:
    """Process all videos in a folder, saving to fight/ or no_fight/ output."""

    if only and dataset_name != only:
        return

    if not folder.exists():
        log.warning(f"Folder not found, skipping: {folder}")
        return

    videos = [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        log.warning(f"No videos found in {folder}")
        return

    out_base = FIGHT_OUT if label == "fight" else NOFIGHT_OUT

    log.info(f"\n{'='*65}")
    log.info(f"  Dataset  : {dataset_name}")
    log.info(f"  Label    : {label}")
    log.info(f"  Videos   : {len(videos)}")
    log.info(f"  Source   : {folder}")
    log.info(f"  Output   : {out_base / dataset_name}")
    log.info(f"{'='*65}")

    total_saved = 0
    total_people = 0

    for vp in tqdm(videos, desc=f"[{dataset_name}]", unit="video"):
        out_dir = out_base / dataset_name / vp.stem
        stats = process_video(vp, out_dir, model, fps, max_frames, min_conf)
        total_saved  += stats["saved"]
        total_people += stats["people"]

    log.info(f"  [{dataset_name}] Done: {total_saved} frames, {total_people} person detections")


def process_real_life_violence(model: YOLO, fps: int,
                                max_frames: int, min_conf: float,
                                only: str = None) -> None:
    """
    Special handler for Real Life Violence dataset.
    Scans all subfolders and detects label from folder name
    (Violence → fight, NonViolence → no_fight).
    """
    if only and only != "real_life":
        return

    base = RAW_DIR / "real-life-violence"
    if not base.exists():
        log.warning(f"real-life-violence folder not found: {base}")
        return

    # Find all Violence and NonViolence folders anywhere inside
    fight_folders    = list(base.rglob(REAL_LIFE_FIGHT_FOLDER))
    nofight_folders  = list(base.rglob(REAL_LIFE_NOFIGHT_FOLDER))

    for folder in fight_folders:
        if folder.is_dir():
            process_folder(folder, "fight", "real_life_violence",
                           model, fps, max_frames, min_conf)

    for folder in nofight_folders:
        if folder.is_dir():
            process_folder(folder, "no_fight", "real_life_nofight",
                           model, fps, max_frames, min_conf)

# ─── PREVIEW MODE ─────────────────────────────────────────────────────────────

def preview(video_path: str, model: YOLO, min_conf: float) -> None:
    """
    Show original vs thermal side-by-side for one video.
    Press Q to quit. Use this BEFORE running the full pipeline
    to check the thermal output looks correct.
    """
    cap = cv2.VideoCapture(video_path)
    log.info(f"Preview: {video_path} — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            continue

        frame = cv2.resize(frame, IMG_SIZE)
        results = model(frame, verbose=False, classes=[0])
        masks = get_person_masks(results[0], IMG_SIZE[1], IMG_SIZE[0], min_conf)
        thermal = render_thermal(frame, masks)

        # Show side by side: original | thermal
        side_by_side = np.hstack([frame, thermal])
        cv2.putText(side_by_side, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(side_by_side, "THERMAL", (IMG_SIZE[0]+10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        cv2.putText(side_by_side, f"People detected: {len(masks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        cv2.imshow("D10 Thermal Preview  [Q = quit]", side_by_side)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── SUMMARY ──────────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print a count of saved frames after pipeline completes."""
    log.info("\n" + "="*65)
    log.info("  D10 PIPELINE COMPLETE — Output Summary")
    log.info("="*65)

    for label_folder in [FIGHT_OUT, NOFIGHT_OUT]:
        if label_folder.exists():
            frames = list(label_folder.rglob("*.jpg"))
            log.info(f"  {label_folder.name:12s}: {len(frames):6,} frames")

    log.info(f"\n  Full output: {OUTPUT_DIR}")
    log.info(f"  Log file  : {LOG_DIR / 'thermal_pipeline.log'}")
    log.info("\n  NEXT STEP : Run D11 dataset splitting script.")
    log.info("="*65)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="D10 Thermal Pipeline — Singo Loua 240086608"
    )
    parser.add_argument("--dataset", default="all",
        help="Dataset to run: all | beach_youtube | gemini | hockey-fight | "
             "movies_fight | movies_nofight | real_life")
    parser.add_argument("--fps",       type=int,   default=DEFAULT_FPS_SAMPLE)
    parser.add_argument("--max_frames",type=int,   default=MAX_FRAMES_PER_VID)
    parser.add_argument("--min_conf",  type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--preview",   type=str,   default=None,
        help="Path to a video to preview before running full pipeline")
    args = parser.parse_args()

    only = None if args.dataset == "all" else args.dataset

    log.info("="*65)
    log.info("  D10 Thermal Warm-Silhouette Pipeline")
    log.info(f"  Student : Singo Loua | 240086608")
    log.info(f"  FPS     : {args.fps} frames/sec sampled")
    log.info(f"  Max     : {args.max_frames} frames/video")
    log.info(f"  Conf    : {args.min_conf} minimum detection confidence")
    log.info("="*65)

    # Create output folders
    FIGHT_OUT.mkdir(parents=True, exist_ok=True)
    NOFIGHT_OUT.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 segmentation model
    # NOTE: you already have yolov8n.pt — this downloads yolov8n-seg.pt (~6MB)
    log.info("\nLoading YOLOv8 segmentation model (yolov8n-seg.pt)...")
    model = YOLO(YOLO_MODEL)
    log.info("  Model ready.\n")

    # Preview mode
    if args.preview:
        preview(args.preview, model, args.min_conf)
        return

    # Process all standard datasets
    for folder, label, name in DATASETS:
        if label == "AUTO":
            continue  # handled separately below
        process_folder(folder, label, name, model,
                       args.fps, args.max_frames, args.min_conf, only)

    # Process Real Life Violence (has nested Violence/NonViolence subfolders)
    process_real_life_violence(model, args.fps, args.max_frames,
                                args.min_conf, only)

    print_summary()


if __name__ == "__main__":
    main()
