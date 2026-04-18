r"""
=============================================================================
  D14 - LLaVA:13b Visual Language Model Integration
  Three-Tier VLM Processing Pathways
  Project: Deep Learning and VLM based Real-Time Fighting Detection
           on Beach Safety Management
  Student: Singo Loua | 240086608 | MSc Applied Data Science
  Supervisor: Dr. Ming Jiang | University of Sunderland
  Methodology: CRISP-DM Phase 4: Modelling
=============================================================================

WHAT THIS SCRIPT DOES:
  Implements the three-tier VLM processing framework confirmed by
  Dr. Ming Jiang on 16/03/2026. Each pathway sends a different
  type of input to LLaVA:13b and records its reasoning output.

  PATHWAY 1: Raw video frame -> LLaVA:13b
  PATHWAY 2: Thermal silhouette frame -> LLaVA:13b
  PATHWAY 3: YOLOv8 D13 pre-processed + thermal -> LLaVA:13b

  For each pathway, LLaVA:13b is asked:
  - Is there a fight happening?
  - Confidence level (high/medium/low)
  - Description of what it sees
  - Number of people involved

  Results are saved to C:\beach\results\d14_vlm_results.json
  and a comparison report is generated.

REQUIREMENTS:
  - Ollama running locally (ollama serve)
  - llava:13b model pulled (ollama pull llava:13b)
  - D13 model at C:\beach\models\d13_full\weights\best.pt

USAGE:
  First start Ollama in a separate terminal:
    ollama serve

  Then run:
    cd C:\beach
    env\Scripts\activate
    python scripts\d14_vlm_integration.py

  Test on one video:
    python scripts\d14_vlm_integration.py --video "C:\beach\data\raw\beach\fighting\video.mp4"

  Run evaluation on test set samples:
    python scripts\d14_vlm_integration.py --mode evaluate --samples 20

=============================================================================
"""

import cv2
import numpy as np
import json
import base64
import requests
import argparse
import logging
import time
import random
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ─── PATHS ────────────────────────────────────────────────────────────────────

ROOT        = Path(r"C:\beach")
SPLIT_DIR   = ROOT / "data" / "split"
THERMAL_DIR = ROOT / "data" / "thermal"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOG_DIR     = ROOT / "logs"

D13_MODEL   = MODELS_DIR / "d13_full" / "weights" / "best.pt"

# ─── OLLAMA CONFIG ────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
LLAVA_MODEL  = "llava:13b"
TIMEOUT      = 600   # seconds per LLaVA request

# ─── THERMAL SETTINGS ─────────────────────────────────────────────────────────

IMG_SIZE         = (640, 640)
CONFIDENCE       = 0.40
BACKGROUND_COLOUR= (10, 10, 20)
GLOW_BOOST       = 1.3
BLUR_KERNEL      = 7

# ─── VLT-BASED PROMPT (Visual Language Theory from HCI) ──────────────────────
# Designed based on Visual Language Theory principles:
# - Clear visual context setting
# - Structured response format
# - Domain-specific vocabulary
# - Binary classification with confidence

FIGHT_DETECTION_PROMPT = """You are an AI safety system monitoring beach surveillance footage.
Analyse this image carefully and respond in the following exact format:

FIGHT_DETECTED: [YES or NO]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
PEOPLE_COUNT: [number of people visible]
DESCRIPTION: [one sentence describing what you see]
REASONING: [one sentence explaining your fight/no-fight decision]

Focus on: body postures, physical contact, aggressive gestures, crowd dynamics.
Be concise. Respond only in the format above."""

# ─── LOGGING ──────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "d14_vlm.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── THERMAL RENDERING ────────────────────────────────────────────────────────

def render_thermal(frame: np.ndarray, masks: list) -> np.ndarray:
    """Convert frame to thermal warm-silhouette representation."""
    h, w = frame.shape[:2]
    output = np.full((h, w, 3), BACKGROUND_COLOUR, dtype=np.uint8)
    if not masks:
        return output
    heat = np.zeros((h, w), dtype=np.float32)
    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.float32)
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (BLUR_KERNEL, BLUR_KERNEL), 0)
        heat = np.maximum(heat, blurred)
    heat = np.clip(heat * GLOW_BOOST, 0.0, 1.0)
    heat_u8 = (heat * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HOT)
    alpha = heat[:, :, np.newaxis]
    output = (coloured * alpha + output * (1 - alpha)).astype(np.uint8)
    return output


def get_person_masks(result, h: int, w: int) -> list:
    """Extract person segmentation masks from YOLOv8 segmentation result."""
    masks = []
    if result.masks is None:
        return masks
    raw_masks = result.masks.data.cpu().numpy()
    classes   = result.boxes.cls.cpu().numpy().astype(int)
    confs     = result.boxes.conf.cpu().numpy()
    for i, (cls, conf) in enumerate(zip(classes, confs)):
        if cls != 0 or conf < CONFIDENCE:
            continue
        mask = cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
        masks.append((mask > 0.5).astype(np.float32))
    return masks

# ─── IMAGE ENCODING ───────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray) -> str:
    """Convert a numpy frame to base64 string for LLaVA API."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

# ─── LLAVA QUERY ──────────────────────────────────────────────────────────────

def query_llava(frame: np.ndarray, prompt: str = FIGHT_DETECTION_PROMPT) -> dict:
    """
    Send a frame to LLaVA:13b via Ollama and get structured response.

    Uses the VLT-based prompt designed to elicit structured,
    actionable responses for fight detection.

    Returns dict with: raw_response, fight_detected, confidence,
                       people_count, description, reasoning, response_time
    """
    img_b64 = frame_to_base64(frame)
    start_time = time.time()

    payload = {
        "model": LLAVA_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temperature for consistent structured output
            "num_predict": 200,   # Limit response length
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response_time = time.time() - start_time
        raw_text = response.json().get("response", "")

        # Parse structured response
        result = {
            "raw_response": raw_text,
            "response_time": round(response_time, 2),
            "fight_detected": None,
            "confidence": None,
            "people_count": None,
            "description": "",
            "reasoning": "",
            "parse_error": False
        }

        # Extract fields from structured response
        for line in raw_text.upper().split('\n'):
            if "FIGHT_DETECTED:" in line:
                result["fight_detected"] = "YES" in line
            elif "CONFIDENCE:" in line:
                for level in ["HIGH", "MEDIUM", "LOW"]:
                    if level in line:
                        result["confidence"] = level
                        break

        for line in raw_text.split('\n'):
            if "PEOPLE_COUNT:" in line:
                try:
                    nums = [int(s) for s in line.split() if s.isdigit()]
                    if nums:
                        result["people_count"] = nums[0]
                except:
                    pass
            elif "DESCRIPTION:" in line:
                result["description"] = line.replace("DESCRIPTION:", "").strip()
            elif "REASONING:" in line:
                result["reasoning"] = line.replace("REASONING:", "").strip()

        if result["fight_detected"] is None:
            result["parse_error"] = True

        return result

    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama. Make sure 'ollama serve' is running.")
        return {"error": "Ollama not running", "response_time": 0}
    except Exception as e:
        log.error(f"LLaVA query error: {e}")
        return {"error": str(e), "response_time": 0}

# ─── THREE PATHWAYS ───────────────────────────────────────────────────────────

def pathway_1_raw(frame: np.ndarray) -> dict:
    """
    Pathway 1: Raw video frame -> LLaVA:13b
    No preprocessing. Tests VLM baseline on original footage.
    """
    frame_resized = cv2.resize(frame, IMG_SIZE)
    result = query_llava(frame_resized)
    result["pathway"] = 1
    result["pathway_name"] = "Raw Frame"
    return result


def pathway_2_thermal(frame: np.ndarray, seg_model: YOLO) -> dict:
    """
    Pathway 2: Thermal silhouette frame -> LLaVA:13b
    Applies privacy-preserving thermal rendering before VLM query.
    GDPR compliant - no identifiable information sent to LLaVA.
    """
    frame_resized = cv2.resize(frame, IMG_SIZE)
    results = seg_model(frame_resized, verbose=False, classes=[0])
    masks = get_person_masks(results[0], IMG_SIZE[1], IMG_SIZE[0])
    thermal_frame = render_thermal(frame_resized, masks)
    result = query_llava(thermal_frame)
    result["pathway"] = 2
    result["pathway_name"] = "Thermal Frame"
    result["people_detected_yolo"] = len(masks)
    return result


def pathway_3_dl_thermal(frame: np.ndarray,
                          seg_model: YOLO,
                          cls_model: YOLO) -> dict:
    """
    Pathway 3: YOLOv8 D13 pre-processed + thermal -> LLaVA:13b

    Step 1: Run YOLOv8 segmentation to detect people + get masks
    Step 2: Run D13 classification model to get fight/no_fight prediction
    Step 3: Overlay YOLOv8 detection info (bounding boxes, labels) on frame
    Step 4: Apply thermal rendering
    Step 5: Send enriched thermal frame to LLaVA:13b

    This pathway provides LLaVA with pre-annotated context,
    testing whether DL pre-processing improves VLM reasoning.
    """
    frame_resized = cv2.resize(frame, IMG_SIZE)

    # Step 1 - Segmentation
    seg_results = seg_model(frame_resized, verbose=False, classes=[0])
    masks = get_person_masks(seg_results[0], IMG_SIZE[1], IMG_SIZE[0])

    # Step 2 - Classification with D13 model
    cls_results = cls_model(frame_resized, verbose=False)
    cls_probs = cls_results[0].probs
    fight_conf = float(cls_probs.data[0]) if cls_probs is not None else 0.5
    yolo_prediction = "FIGHT" if fight_conf > 0.5 else "NO FIGHT"

    # Step 3 - Draw bounding boxes and D13 prediction on frame
    annotated = frame_resized.copy()
    if seg_results[0].boxes is not None:
        for box in seg_results[0].boxes:
            if int(box.cls) == 0 and float(box.conf) >= CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add D13 prediction overlay
    label = f"YOLOv8: {yolo_prediction} ({fight_conf:.2f})"
    cv2.putText(annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Step 4 - Apply thermal rendering
    thermal_frame = render_thermal(annotated, masks)

    # Add text overlay on thermal frame
    cv2.putText(thermal_frame, f"DL: {yolo_prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Step 5 - Send to LLaVA
    result = query_llava(thermal_frame)
    result["pathway"] = 3
    result["pathway_name"] = "DL Pre-processed + Thermal"
    result["yolo_prediction"] = yolo_prediction
    result["yolo_fight_confidence"] = round(fight_conf, 3)
    result["people_detected_yolo"] = len(masks)
    return result

# ─── EVALUATION MODE ──────────────────────────────────────────────────────────

def evaluate_pathways(samples: int = 20) -> None:
    """
    Run all three pathways on a sample of test images and compare results.

    Selects 'samples' frames from the test set (equal fight/no_fight split),
    runs all three pathways on each, and computes accuracy per pathway.
    Results saved to C:/beach/results/d14_vlm_results.json
    """
    log.info("=" * 65)
    log.info("  D14 VLM Three-Pathway Evaluation")
    log.info(f"  Student : Singo Loua | 240086608")
    log.info(f"  Model   : {LLAVA_MODEL}")
    log.info(f"  Samples : {samples} frames")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    # Check Ollama is running
    try:
        requests.get("http://localhost:11434", timeout=5)
        log.info("Ollama connection: OK")
    except:
        log.error("Ollama is not running! Start it with: ollama serve")
        return

    # Load models
    log.info("\nLoading models...")
    seg_model = YOLO("yolov8n-seg.pt")
    log.info("  Segmentation model loaded")

    if not D13_MODEL.exists():
        log.error(f"D13 model not found at {D13_MODEL}")
        log.error("Please run: xcopy /E /I C:\\beach\\models\\d12_baseline C:\\beach\\models\\d13_full")
        return

    cls_model = YOLO(str(D13_MODEL))
    log.info(f"  D13 classification model loaded")

    # Sample test frames
    per_class = samples // 2
    test_frames = []

    for cls_name, true_label in [("fight", True), ("no_fight", False)]:
        cls_dir = SPLIT_DIR / "test" / cls_name
        if not cls_dir.exists():
            log.warning(f"Test folder not found: {cls_dir}")
            continue
        frames = list(cls_dir.glob("*.jpg"))
        random.seed(42)
        selected = random.sample(frames, min(per_class, len(frames)))
        for f in selected:
            test_frames.append({"path": f, "true_label": true_label, "class": cls_name})

    random.shuffle(test_frames)
    log.info(f"\nSelected {len(test_frames)} test frames ({per_class} fight, {per_class} no_fight)")

    # Run evaluation
    all_results = []
    pathway_stats = {1: [], 2: [], 3: []}

    for idx, item in enumerate(test_frames):
        log.info(f"\nFrame {idx+1}/{len(test_frames)}: {item['path'].name} (True: {item['class']})")
        frame = cv2.imread(str(item["path"]))

        if frame is None:
            log.warning(f"  Could not read frame: {item['path']}")
            continue

        frame_result = {
            "frame": item["path"].name,
            "true_label": item["class"],
            "true_fight": item["true_label"],
            "pathways": {}
        }

        # Pathway 1
        log.info("  Running Pathway 1 (Raw)...")
        p1 = pathway_1_raw(frame)
        frame_result["pathways"]["1"] = p1
        p1_correct = p1.get("fight_detected") == item["true_label"]
        pathway_stats[1].append(p1_correct)
        log.info(f"    P1: fight={p1.get('fight_detected')} conf={p1.get('confidence')} correct={p1_correct} time={p1.get('response_time')}s")

        # Pathway 2
        log.info("  Running Pathway 2 (Thermal)...")
        p2 = pathway_2_thermal(frame, seg_model)
        frame_result["pathways"]["2"] = p2
        p2_correct = p2.get("fight_detected") == item["true_label"]
        pathway_stats[2].append(p2_correct)
        log.info(f"    P2: fight={p2.get('fight_detected')} conf={p2.get('confidence')} correct={p2_correct} time={p2.get('response_time')}s")

        # Pathway 3
        log.info("  Running Pathway 3 (DL+Thermal)...")
        p3 = pathway_3_dl_thermal(frame, seg_model, cls_model)
        frame_result["pathways"]["3"] = p3
        p3_correct = p3.get("fight_detected") == item["true_label"]
        pathway_stats[3].append(p3_correct)
        log.info(f"    P3: fight={p3.get('fight_detected')} yolo={p3.get('yolo_prediction')} correct={p3_correct} time={p3.get('response_time')}s")

        all_results.append(frame_result)

    # Compute accuracy per pathway
    log.info("\n" + "=" * 65)
    log.info("  D14 VLM EVALUATION RESULTS")
    log.info("=" * 65)

    summary = {}
    for pw in [1, 2, 3]:
        stats = pathway_stats[pw]
        if stats:
            acc = sum(stats) / len(stats) * 100
            summary[str(pw)] = {
                "accuracy": round(acc, 2),
                "correct": sum(stats),
                "total": len(stats)
            }
            pw_names = {1: "Raw Frame", 2: "Thermal Frame", 3: "DL+Thermal"}
            log.info(f"  Pathway {pw} ({pw_names[pw]:20s}): {acc:.2f}% ({sum(stats)}/{len(stats)} correct)")

    log.info("=" * 65)

    # Save results
    output = {
        "metadata": {
            "student": "Singo Loua | 240086608",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": LLAVA_MODEL,
            "samples": len(all_results),
            "d13_model": str(D13_MODEL)
        },
        "summary": summary,
        "results": all_results
    }

    out_path = RESULTS_DIR / "d14_vlm_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"\n  Results saved to: {out_path}")
    log.info(f"  Log file        : {LOG_DIR / 'd14_vlm.log'}")
    log.info("\n  NEXT STEP: D15 — Evaluation report + D19 Gradio prototype")
    log.info("=" * 65)

# ─── SINGLE VIDEO MODE ────────────────────────────────────────────────────────

def process_single_video(video_path: str) -> None:
    """
    Run all three pathways on sample frames from a single video.
    Good for testing and demonstration purposes.
    """
    log.info(f"\nProcessing video: {video_path}")

    seg_model = YOLO("yolov8n-seg.pt")
    cls_model = YOLO(str(D13_MODEL)) if D13_MODEL.exists() else None

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25

    # Sample 3 frames from the video
    positions = [int(total * 0.25), int(total * 0.5), int(total * 0.75)]

    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue

        log.info(f"\n--- Frame {i+1}/3 (position {pos}/{total}) ---")

        log.info("  Pathway 1 (Raw)...")
        p1 = pathway_1_raw(frame)
        log.info(f"    Fight: {p1.get('fight_detected')} | Confidence: {p1.get('confidence')}")
        log.info(f"    {p1.get('description', '')}")

        log.info("  Pathway 2 (Thermal)...")
        p2 = pathway_2_thermal(frame, seg_model)
        log.info(f"    Fight: {p2.get('fight_detected')} | Confidence: {p2.get('confidence')}")
        log.info(f"    {p2.get('description', '')}")

        if cls_model:
            log.info("  Pathway 3 (DL+Thermal)...")
            p3 = pathway_3_dl_thermal(frame, seg_model, cls_model)
            log.info(f"    Fight: {p3.get('fight_detected')} | YOLOv8: {p3.get('yolo_prediction')}")
            log.info(f"    {p3.get('description', '')}")

    cap.release()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="D14 LLaVA:13b VLM Integration — Singo Loua 240086608"
    )
    parser.add_argument("--mode", default="evaluate",
                        choices=["evaluate", "video"],
                        help="evaluate: run on test set | video: run on single video")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (for --mode video)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of test frames to evaluate (default: 20)")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  D14 LLaVA:13b VLM Integration")
    log.info(f"  Student    : Singo Loua | 240086608")
    log.info(f"  Methodology: CRISP-DM Phase 4 — Modelling")
    log.info(f"  VLM Model  : {LLAVA_MODEL}")
    log.info(f"  Mode       : {args.mode}")
    log.info("=" * 65)

    if args.mode == "video":
        if not args.video:
            log.error("Please provide --video path")
            return
        process_single_video(args.video)
    else:
        evaluate_pathways(args.samples)


if __name__ == "__main__":
    main()
