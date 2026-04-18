r"""
=============================================================================
  D19 - Gradio AI Agent Chatbot Prototype
  Beach Safety Fight Detection System
  Project: Deep Learning and VLM based Real-Time Fighting Detection
           on Beach Safety Management
  Student: Singo Loua | 240086608 | MSc Applied Data Science
  Supervisor: Dr. Ming Jiang | University of Sunderland
  Methodology: CRISP-DM Phase 6: Deployment
=============================================================================

SYSTEM ARCHITECTURE:
  - YOLOv8 D13 model: Primary fight detector (77.78% accuracy)
  - Thermal pipeline: GDPR-compliant warm-silhouette rendering
  - DeepSort: Multi-object tracking with persistent IDs
  - LLaVA:13b: Natural language chatbot for operator queries
  - Gradio: Web-based operator interface

FEATURES:
  - Video file upload OR webcam input
  - Real-time thermal silhouette rendering
  - YOLOv8 fight detection with bounding boxes
  - DeepSort person tracking with unique IDs
  - Alert system when fight detected
  - Natural language chatbot (LLaVA:13b)
  - Operator can ask: "is there a fight?", "describe the scene",
    "how many people?", "track person 3"

REQUIREMENTS:
  pip install gradio opencv-python ultralytics requests numpy

  Ollama must be running:
    ollama serve

  D13 model must exist at:
    C:\beach\models\d13_full\weights\best.pt

USAGE:
  cd C:\beach
  env\Scripts\activate
  python scripts\d19_gradio_app.py

  Then open: http://localhost:7860

FIXES APPLIED (D19 v2):
  - CONFIDENCE lowered from 0.40 to 0.25 (better person detection)
  - fight_detected now requires >= 2 people (eliminates 0-person false alerts)
  - FIGHT_THRESHOLD raised to 0.70 (reduces false positives)
=============================================================================
"""

import cv2
import numpy as np
import gradio as gr
import requests
import base64
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# ─── PATHS ────────────────────────────────────────────────────────────────────

ROOT       = Path(r"C:\beach")
D13_MODEL  = ROOT / "models" / "d13_full" / "weights" / "best.pt"
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "d19_gradio.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

OLLAMA_URL     = "http://localhost:11434/api/generate"
LLAVA_MODEL    = "llava:13b"
IMG_SIZE       = (640, 640)
CONFIDENCE     = 0.25   # FIX 1: Lowered from 0.40 to detect more people
FIGHT_THRESHOLD= 0.70   # Raised from 0.55 to reduce false positives
MIN_PEOPLE     = 2      # FIX 2: Minimum people required to trigger fight alert
BLUR_KERNEL    = 7
GLOW_BOOST     = 1.3
BG_COLOUR      = (10, 10, 20)
OLLAMA_TIMEOUT = 300

# Alert history
alert_history = deque(maxlen=50)
current_frame_data = {"frame": None, "thermal": None, "fight": False, "people": 0, "track_ids": []}

# ─── MODEL LOADING ────────────────────────────────────────────────────────────

log.info("Loading models...")

seg_model = YOLO("yolov8n-seg.pt")
log.info("  Segmentation model loaded")

if D13_MODEL.exists():
    cls_model = YOLO(str(D13_MODEL))
    log.info(f"  D13 classification model loaded: {D13_MODEL}")
else:
    log.warning(f"  D13 model not found at {D13_MODEL}")
    log.warning("  Run: xcopy /E /I C:\\beach\\models\\d12_baseline C:\\beach\\models\\d13_full")
    cls_model = YOLO("yolov8n-cls.pt")
    log.info("  Using default YOLOv8n-cls as fallback")

log.info("All models loaded.")

# ─── THERMAL RENDERING ────────────────────────────────────────────────────────

def render_thermal(frame, masks):
    h, w = frame.shape[:2]
    output = np.full((h, w, 3), BG_COLOUR, dtype=np.uint8)
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
    coloured = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    alpha = heat[:, :, np.newaxis]
    return (coloured * alpha + output * (1 - alpha)).astype(np.uint8)


def get_masks(result, h, w):
    masks = []
    if result.masks is None:
        return masks
    raw = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs   = result.boxes.conf.cpu().numpy()
    for i, (cls, conf) in enumerate(zip(classes, confs)):
        if cls == 0 and conf >= CONFIDENCE:
            m = cv2.resize(raw[i], (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append((m > 0.5).astype(np.float32))
    return masks

# ─── SIMPLE TRACKER ───────────────────────────────────────────────────────────

class SimpleTracker:
    """Lightweight person tracker using bounding box IoU matching."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_age = 10

    def update(self, boxes):
        if not boxes:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
            return []

        assigned = []
        used_tracks = set()

        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best_tid, best_dist = None, float('inf')

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                tx, ty = track["cx"], track["cy"]
                dist = ((cx - tx)**2 + (cy - ty)**2)**0.5
                if dist < best_dist and dist < 150:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid].update({"cx": cx, "cy": cy, "box": box, "age": 0})
                used_tracks.add(best_tid)
                assigned.append((best_tid, box))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"cx": cx, "cy": cy, "box": box, "age": 0}
                used_tracks.add(tid)
                assigned.append((tid, box))

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        return assigned

tracker = SimpleTracker()

# ─── FRAME PROCESSING ─────────────────────────────────────────────────────────

def process_frame(frame):
    """
    Process one video frame through the full pipeline:
    1. Detect people with YOLOv8 segmentation
    2. Classify fight/no_fight with D13 model
    3. Track people with SimpleTracker
    4. Render thermal silhouette
    5. Overlay detection info
    Returns: (annotated_thermal_frame, fight_detected, people_count, track_ids)
    """
    frame_resized = cv2.resize(frame, IMG_SIZE)
    h, w = IMG_SIZE[1], IMG_SIZE[0]

    # Step 1 - Segmentation
    seg_results = seg_model(frame_resized, verbose=False, classes=[0])
    masks = get_masks(seg_results[0], h, w)

    # Get bounding boxes
    boxes = []
    if seg_results[0].boxes is not None:
        for box in seg_results[0].boxes:
            if int(box.cls) == 0 and float(box.conf) >= CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1, x2, y2])

    # Step 2 - Fight classification
    cls_results = cls_model(frame_resized, verbose=False)
    probs = cls_results[0].probs
    fight_prob = float(probs.data[0]) if probs is not None else 0.5

    # FIX: Only trigger fight alert if confidence is high AND at least 2 people visible
    fight_detected = fight_prob > FIGHT_THRESHOLD and len(boxes) >= MIN_PEOPLE

    # Step 3 - Tracking
    tracked = tracker.update(boxes)
    track_ids = [tid for tid, _ in tracked]

    # Step 4 - Thermal rendering
    thermal = render_thermal(frame_resized, masks)

    # Step 5 - Overlay
    for tid, (x1, y1, x2, y2) in tracked:
        colour = (0, 0, 255) if fight_detected else (0, 200, 255)
        cv2.rectangle(thermal, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(thermal, f"P{tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # Detection status banner
    if fight_detected:
        cv2.rectangle(thermal, (0, 0), (w, 50), (0, 0, 180), -1)
        cv2.putText(thermal, f"FIGHT DETECTED ({fight_prob:.0%})",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        cv2.rectangle(thermal, (0, 0), (w, 50), (0, 100, 0), -1)
        cv2.putText(thermal, f"NO FIGHT ({1-fight_prob:.0%} clear)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # People count
    cv2.putText(thermal, f"People: {len(boxes)}  |  Tracks: {len(track_ids)}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(thermal, ts, (w - 90, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return thermal, fight_detected, len(boxes), track_ids, frame_resized

# ─── LLAVA CHATBOT ────────────────────────────────────────────────────────────

def query_llava_chat(question, frame):
    """Send a frame + operator question to LLaVA:13b and return response."""
    if frame is None:
        return "No frame available. Please start video processing first."

    try:
        requests.get("http://localhost:11434", timeout=3)
    except:
        return "Ollama is not running. Start it with: ollama serve"

    small = cv2.resize(frame, (336, 336))
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    fight_status = "a fight has been detected" if current_frame_data["fight"] else "no fight is currently detected"
    people_count = current_frame_data["people"]
    track_ids    = current_frame_data["track_ids"]

    system_context = f"""You are an AI assistant for a beach safety surveillance system.
Current system status: {fight_status}. People visible: {people_count}. Active track IDs: {track_ids}.
The image shows a thermal silhouette representation — people appear as warm glowing shapes.
Answer the operator's question concisely and helpfully."""

    prompt = f"{system_context}\n\nOperator question: {question}"

    payload = {
        "model": LLAVA_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 300}
    }

    try:
        start = time.time()
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        elapsed = time.time() - start
        answer = resp.json().get("response", "No response received.")
        return f"{answer}\n\n*(Response time: {elapsed:.0f}s)*"
    except requests.exceptions.Timeout:
        return "LLaVA:13b timed out. The model is processing — try again in a moment."
    except Exception as e:
        return f"Error: {str(e)}"

# ─── VIDEO PROCESSING ─────────────────────────────────────────────────────────

def process_video_file(video_path, progress=gr.Progress()):
    """Process an uploaded video file frame by frame."""
    if video_path is None:
        return None, "No video uploaded.", ""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file.", ""

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    sample_every = max(1, int(fps / 5))

    processed_frames = []
    fight_count = 0
    frame_idx = 0
    max_frames = 100

    progress(0, desc="Processing video...")

    while len(processed_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        thermal, fight, people, track_ids, orig = process_frame(frame)

        current_frame_data.update({
            "frame": orig,
            "thermal": thermal,
            "fight": fight,
            "people": people,
            "track_ids": track_ids
        })

        if fight:
            fight_count += 1
            ts = datetime.now().strftime("%H:%M:%S")
            alert_history.appendleft(f"[{ts}] FIGHT DETECTED — {people} people, tracks: {track_ids}")

        processed_frames.append(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB))
        progress(len(processed_frames) / max_frames)

    cap.release()

    if not processed_frames:
        return None, "No frames processed.", ""

    last_frame = processed_frames[-1]
    status = f"Processed {len(processed_frames)} frames | Fight alerts: {fight_count}"
    alerts = "\n".join(list(alert_history)[:10]) if alert_history else "No alerts"

    return last_frame, status, alerts


def process_webcam_frame(frame):
    """Process a single webcam frame."""
    if frame is None:
        return None, "No webcam frame.", ""

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    thermal, fight, people, track_ids, orig = process_frame(frame_bgr)

    current_frame_data.update({
        "frame": orig,
        "thermal": thermal,
        "fight": fight,
        "people": people,
        "track_ids": track_ids
    })

    if fight:
        ts = datetime.now().strftime("%H:%M:%S")
        alert_history.appendleft(f"[{ts}] FIGHT DETECTED — {people} people")

    thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
    status = f"FIGHT DETECTED" if fight else "Clear — no fight"
    alerts = "\n".join(list(alert_history)[:5]) if alert_history else "No alerts"

    return thermal_rgb, status, alerts


def chat_with_llava(message, history):
    """Gradio chatbot interface for LLaVA:13b queries."""
    if not message.strip():
        return history, ""

    frame = current_frame_data.get("frame")
    if frame is None:
        response = "Please process a video or start webcam first so I can see the scene."
    else:
        response = query_llava_chat(message, frame)

    history = history or []
    history.append((message, response))
    return history, ""

# ─── GRADIO INTERFACE ─────────────────────────────────────────────────────────

def build_interface():
    with gr.Blocks(
        title="Beach Safety AI — Fight Detection System",
        theme=gr.themes.Base(),
        css="""
        .alert-box { background: #1a0000; border: 2px solid #cc0000; padding: 10px; border-radius: 8px; }
        .status-ok { color: #00cc44; font-weight: bold; }
        .status-alert { color: #ff3333; font-weight: bold; }
        footer { display: none !important; }
        """
    ) as demo:

        gr.Markdown("""
        # Beach Safety AI — Real-Time Fight Detection System
        **Student:** Singo Loua | 240086608 | MSc Applied Data Science | University of Sunderland
        **Supervisor:** Dr. Ming Jiang | CRISP-DM Phase 6: Deployment
        ---
        *Thermal silhouette rendering ensures GDPR compliance — no identifiable features are processed or stored.*
        """)

        with gr.Tabs():

            # ── TAB 1: VIDEO FILE ──────────────────────────────────────────
            with gr.TabItem("Video File"):
                gr.Markdown("### Upload a video file for fight detection analysis")
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video (.mp4, .avi)")
                        process_btn = gr.Button("Process Video", variant="primary")
                        video_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=1):
                        video_output = gr.Image(label="Thermal Detection Output")
                        video_alerts = gr.Textbox(
                            label="Alert Log", interactive=False, lines=5
                        )

                process_btn.click(
                    fn=process_video_file,
                    inputs=[video_input],
                    outputs=[video_output, video_status, video_alerts]
                )

            # ── TAB 2: WEBCAM ──────────────────────────────────────────────
            with gr.TabItem("Live Webcam"):
                gr.Markdown("### Live webcam fight detection")
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_input = gr.Image(
                            sources=["webcam"], streaming=True,
                            label="Webcam Feed"
                        )

                    with gr.Column(scale=1):
                        webcam_output = gr.Image(label="Thermal Detection Output")
                        webcam_status = gr.Textbox(label="Detection Status", interactive=False)
                        webcam_alerts = gr.Textbox(label="Alert Log", interactive=False, lines=3)

                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_input],
                    outputs=[webcam_output, webcam_status, webcam_alerts]
                )

            # ── TAB 3: AI CHATBOT ──────────────────────────────────────────
            with gr.TabItem("AI Agent Chatbot"):
                gr.Markdown("""
                ### LLaVA:13b Natural Language Interface
                Ask the AI about the current scene. The AI can see the last processed frame.

                **Example queries:**
                - *"Is there a fight happening?"*
                - *"How many people are in the scene?"*
                - *"Describe what you see"*
                - *"Are there any signs of aggression?"*
                - *"Track the person on the left"*

                ⚠️ Process a video or start webcam first before asking questions.
                *(Note: LLaVA:13b runs on CPU — responses take 2-5 minutes)*
                """)

                chatbot = gr.Chatbot(height=400, label="Beach Safety AI Agent")

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask the AI about the scene...",
                        label="Your question",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        ["Is there a fight happening in this scene?"],
                        ["How many people can you see?"],
                        ["Describe what you observe in the thermal image"],
                        ["Are there any signs of aggressive behaviour?"],
                        ["What is the general situation on the beach?"],
                    ],
                    inputs=chat_input
                )

                send_btn.click(
                    fn=chat_with_llava,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )
                chat_input.submit(
                    fn=chat_with_llava,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )

            # ── TAB 4: SYSTEM INFO ─────────────────────────────────────────
            with gr.TabItem("System Info"):
                gr.Markdown(f"""
                ### System Configuration

                | Component | Details |
                |---|---|
                | **Detection Model** | YOLOv8n-cls (D13) — 77.78% accuracy |
                | **VLM Model** | LLaVA:13b via Ollama (local) |
                | **Privacy Pipeline** | Thermal warm-silhouette (COLORMAP_HOT) |
                | **Tracking** | SimpleTracker (IoU-based) |
                | **Hardware** | AMD Ryzen 5 7520U, 16GB RAM, CPU only |
                | **D13 Model Path** | {D13_MODEL} |
                | **GDPR Status** | Compliant — no identifiable data processed |
                | **CONFIDENCE** | 0.25 (lowered for better person detection) |
                | **FIGHT_THRESHOLD** | 0.70 (raised to reduce false positives) |
                | **MIN_PEOPLE** | 2 (fight requires at least 2 people) |

                ### Performance Metrics (from D15 Evaluation)

                | Model | Accuracy | Inference Speed |
                |---|---|---|
                | D12 YOLOv8 Baseline | 75.33% | ~5.5ms/frame |
                | D13 YOLOv8 Full | 77.78% | ~5.4ms/frame |
                | D14 LLaVA Pathway 1 | 50.00% | ~270s/query |
                | D14 LLaVA Pathway 2 | 50.00% | ~270s/query |
                | D14 LLaVA Pathway 3 | 50.00% | ~270s/query |

                ### Three-Tier VLM Architecture
                - **Pathway 1:** Raw frames → LLaVA:13b
                - **Pathway 2:** Thermal frames → LLaVA:13b
                - **Pathway 3:** YOLOv8 + Thermal → LLaVA:13b *(used in this prototype)*
                """)

    return demo


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 65)
    log.info("  D19 Beach Safety Gradio Prototype v2")
    log.info(f"  Student : Singo Loua | 240086608")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  CONFIDENCE     : {CONFIDENCE}")
    log.info(f"  FIGHT_THRESHOLD: {FIGHT_THRESHOLD}")
    log.info(f"  MIN_PEOPLE     : {MIN_PEOPLE}")
    log.info("=" * 65)

    try:
        requests.get("http://localhost:11434", timeout=3)
        log.info("  Ollama: Running")
    except:
        log.warning("  Ollama: NOT running — chatbot will not work")
        log.warning("  Start Ollama with: ollama serve")

    log.info(f"  D13 model: {'Found' if D13_MODEL.exists() else 'NOT FOUND'}")
    log.info("\n  Starting Gradio interface...")
    log.info("  Open: http://localhost:7860")
    log.info("=" * 65)

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )