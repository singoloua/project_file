r"""
Quick thermal preview test - saves sample frames as images.
Auto-finds the first video in your beach fighting folder.

Run from C:\beach with env active:
    python scripts\preview_test.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path(r"C:\beach\preview_output")
IMG_SIZE    = (640, 640)
NUM_SAMPLES = 5
CONFIDENCE  = 0.35

# Auto-find first video in beach fighting folder
SEARCH_DIR  = Path(r"C:\beach\data\raw\beach\fighting")
EXTS        = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
# ─────────────────────────────────────────────────────────────────────────────

def render_thermal(frame, masks):
    h, w = frame.shape[:2]
    output = np.full((h, w, 3), (10, 10, 20), dtype=np.uint8)
    if not masks:
        return output
    heat = np.zeros((h, w), dtype=np.float32)
    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.float32)
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)
        heat = np.maximum(heat, blurred)
    heat = np.clip(heat * 1.3, 0.0, 1.0)
    heat_u8 = (heat * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HOT)
    alpha = heat[:, :, np.newaxis]
    output = (coloured * alpha + output * (1 - alpha)).astype(np.uint8)
    return output

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all videos in the folder
    if not SEARCH_DIR.exists():
        print(f"ERROR: Folder not found: {SEARCH_DIR}")
        return

    videos = [p for p in SEARCH_DIR.rglob("*") if p.suffix.lower() in EXTS]
    if not videos:
        print(f"ERROR: No video files found in {SEARCH_DIR}")
        return

    video_path = videos[0]
    print(f"\nFound {len(videos)} video(s). Using: {video_path.name}")

    print(f"\nLoading YOLOv8 segmentation model...")
    model = YOLO("yolov8n-seg.pt")
    print("  Model ready.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\nERROR: Cannot open: {video_path}")
        print("Try running: python scripts\\preview_test.py --video <drag video file here>")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total} frames at {fps:.1f} FPS")

    sample_positions = [int(total * i / NUM_SAMPLES) for i in range(NUM_SAMPLES)]

    for i, pos in enumerate(sample_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            print(f"  Sample {i+1}: could not read frame {pos}, skipping.")
            continue

        frame = cv2.resize(frame, IMG_SIZE)
        results = model(frame, verbose=False, classes=[0])

        masks = []
        if results[0].masks is not None:
            raw_masks = results[0].masks.data.cpu().numpy()
            classes   = results[0].boxes.cls.cpu().numpy().astype(int)
            confs     = results[0].boxes.conf.cpu().numpy()
            for j, (cls, conf) in enumerate(zip(classes, confs)):
                if cls == 0 and conf >= CONFIDENCE:
                    mask = cv2.resize(raw_masks[j], IMG_SIZE,
                                      interpolation=cv2.INTER_LINEAR)
                    masks.append((mask > 0.5).astype(np.float32))

        thermal = render_thermal(frame, masks)
        side = np.hstack([frame, thermal])
        cv2.putText(side, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(side, f"THERMAL ({len(masks)} people)", (IMG_SIZE[0]+10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        out_path = OUTPUT_DIR / f"sample_{i+1:02d}.jpg"
        cv2.imwrite(str(out_path), side)
        print(f"  Saved sample {i+1}/{NUM_SAMPLES}: {out_path.name}  ({len(masks)} person(s) detected)")

    cap.release()
    print(f"\nDone! Open this folder to check results:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nLeft = original | Right = thermal silhouette")
    print(f"Orange/yellow glowing shapes = pipeline working correctly")

if __name__ == "__main__":
    main()
