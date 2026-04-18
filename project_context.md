# PROJECT CONTEXT — Singo Loua | 240086608
# MSc Applied Data Science | University of Sunderland
# Last Updated: 10 April 2026

---

## 1. STUDENT DETAILS

| Field | Value |
|---|---|
| Name | Singo Loua |
| Student ID | 240086608 |
| Email | bi76tf@student.sunderland.ac.uk |
| Programme | MSc Applied Data Science (Full Time) |
| Year | Year 2 — Final Year Dissertation |
| Supervisor | Dr. Ming Jiang |
| Module | PROM02 — Computing Master's Project |
| Module Leader | Dr. Thomas Thu Yein |
| Institution | University of Sunderland |
| Academic Year | 2025/26 |
| GitHub | https://github.com/singoloua/project_file.git |

---

## 2. PROJECT TITLE

**"Deep Learning and Visual Language Model (VLM) based Real-Time Fighting Detection on Beach Safety Management"**

Original title (before supervisor meeting 20/02/2026):
"Real-time snatching and fighting detection in crowd surveillance systems"

---

## 3. RESEARCH QUESTION

Can a VLM AI system detect fighting in beach surveillance videos accurately in real-time while maintaining individual privacy through thermal silhouette rendering, allowing users to query using natural language, and achieving quick inference speed with minimal false positives?

---

## 4. SUPERVISOR MEETINGS

### Meeting 1 — 20/02/2026
- Changed project title to beach fighting detection
- New requirements added:
  - AI agent inside the project
  - Focus on beach environment
  - Thermal warm-shape pipeline for GDPR/privacy (people may be undressed)
  - Gradio chatbot for operator to track people during live recording
  - VLM integration (LLaVA)

### Meeting 2 — 04/03/2026
- Showed Dr. Ming Jiang the collected datasets
- He provided ethics review instructions via email:
  - Online Ethics Review System on University of Sunderland website
  - Applicant Guide PDF: https://www.london.sunderland.ac.uk/images/external-websites/www/research/helpsheets/Applicant-Guide-Online-Ethics-Review.pdf

### Meeting 3 — 16/03/2026
- Three-tier VLM processing architecture introduced (Pathways 1, 2, 3)
- VLT (Visual Language Theory from HCI) research directive added
- Dr. Ming Jiang sent LLaVA resource: https://zilliz.com/blog/llava-visual-instruction-training

---

## 5. SYSTEM ARCHITECTURE

### Three-Tier VLM Processing Pathways
| Pathway | Input | Description |
|---|---|---|
| 1 | Raw video frames | Frames passed directly to VLM — establishes baseline |
| 2 | Thermal-rendered frames | Privacy-compliant warm shapes passed to VLM |
| 3 | DL pre-processed + thermal | YOLOv8/DeepSort annotated frames, then thermally rendered, then VLM |

### Core Components
| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | DeepSort |
| Visual Language Model | LLaVA:13b via Ollama (fully local — GDPR compliant) |
| Privacy Pipeline | OpenCV thermal warm-silhouette rendering |
| AI Agent Interface | Gradio chatbot |
| Development | Python, PyTorch, OpenCV, Jupyter Notebook |
| Hardware | Lenovo laptop, AMD Ryzen 5 7520U, Radeon Graphics, 16GB RAM, NO NVIDIA GPU |
| Project Location | C:\beach (Windows 11 Home) |
| Virtual Environment | C:\beach\env\ |
| IDE | VS Code |

### Why LLaVA:13b was chosen
- Fully local via Ollama — no data leaves machine (GDPR compliant)
- Academically citable
- Open-weight and free
- Runs on CPU (no GPU needed)

---

## 6. DATASETS

All raw data at: C:\beach\data\raw\

| Folder | Label | Videos | Notes |
|---|---|---|---|
| beach\fighting\ | fight | 5 | YouTube beach fight footage via yt-dlp |
| gemini\fight\ | fight | 12 | Synthetic Gemini Veo 2 fight videos |
| gemini\no_fight\ | no_fight | 9 | Synthetic Gemini Veo 2 no_fight videos |
| hockey-fight\ | fight | 1,000 | .xvid format |
| movies-fight\fights\ | fight | 100 | |
| movies-fight\noFights\ | no_fight | 101 | |
| real-life-violence\...\Violence\ | fight | 1,000 | Nested subfolder |
| real-life-violence\...\NonViolence\ | no_fight | 1,000 | Nested subfolder |

### Gemini correction (10/04/2026)
Originally all gemini videos were in one folder labelled as fight.
Singo manually split into gemini\fight\ (12) and gemini\no_fight\ (9).
thermal_pipeline.py line 77 updated to:
  ( RAW_DIR / "gemini" / "fight",    "fight",    "gemini_fight"   ),
  ( RAW_DIR / "gemini" / "no_fight", "no_fight", "gemini_nofight" ),

---

## 7. DELIVERABLES PROGRESS

### Completed
| Deliverable | Description | Date |
|---|---|---|
| D1-D6 | Planning phase | Jan-Feb 2026 |
| D6 | PROM02 Planning Review submitted | 23/02/2026 |
| D7 | Literature Review Chapter 2 drafted | Feb-Mar 2026 |
| D9 | All datasets collected | Mar 2026 |
| D10 | Thermal pipeline built and run | 09/04/2026 |
| D11 | Dataset split 70/15/15 (corrected) | 10/04/2026 |
| Ethics | Ethics approval submitted and RECEIVED | Apr 2026 |

### Milestone 3 — ACHIEVED 10 April 2026

### D10 Final Numbers
- fight: 34,779 frames from 2,121 videos
- no_fight: 25,659 frames from 1,110 videos
- Total: ~60,438 thermal frames
- Script: C:\beach\scripts\thermal_pipeline.py
- Output: C:\beach\data\thermal\

### D11 Final Numbers (corrected)
- Total: 60,438 frames from 3,231 videos
- Train: 42,178 (69.8%)
- Val: 9,262 (15.3%)
- Test: 8,998 (14.9%)
- Zero data leakage confirmed
- Seed: 42
- Script: C:\beach\scripts\dataset_split.py
- Output: C:\beach\data\split\

### Outstanding
- D8: Architecture specification
- D12: Train YOLOv8 baseline model on C:\beach\data\split\
- D13: Temporal extension model
- D14: Hyperparameter tuning + Gradio chatbot
- D15: Evaluation report
- D16: Ethics impact summary
- D17: Comparative analysis
- D18: Real-time inference pipeline
- D19: Working Gradio prototype
- D20-D23: Dissertation writing and submission (deadline 8 May 2026, Viva 11 May 2026)

---

## 8. FOLDER STRUCTURE

```
C:\beach\
  env\                          <- Python virtual environment
  data\
    raw\                        <- Original videos
      beach\fighting\           <- 5 videos
      gemini\fight\             <- 12 videos
      gemini\no_fight\          <- 9 videos
      hockey-fight\             <- 1,000 .xvid videos
      movies-fight\fights\      <- 100 videos
      movies-fight\noFights\    <- 101 videos
      real-life-violence\...\Violence\     <- 1,000 videos
      real-life-violence\...\NonViolence\  <- 1,000 videos
    thermal\                    <- D10 output
      fight\                    <- 34,779 frames
      no_fight\                 <- 25,659 frames
    split\                      <- D11 output
      train\fight\              <- 24,217 frames
      train\no_fight\           <- 17,961 frames
      val\fight\                <- 5,495 frames
      val\no_fight\             <- 3,767 frames
      test\fight\               <- 5,067 frames
      test\no_fight\            <- 3,931 frames
  scripts\
    thermal_pipeline.py         <- D10 (updated line 77 for gemini)
    dataset_split.py            <- D11
    preview_test.py             <- Visual testing tool
  logs\
    thermal_pipeline.log
    dataset_split.log
  preview_output\               <- Sample thermal frames
  yolov8n.pt                    <- Downloaded
  yolov8n-seg.pt                <- Downloaded by D10
  project_context.md            <- This file
```

---

## 9. SCRIPTS SUMMARY

### thermal_pipeline.py (D10)
- Detects people with YOLOv8n-seg, renders thermal silhouettes
- COLORMAP_HOT on dark background
- 5 FPS, max 200 frames/video, 0.4 confidence
- Run all: python scripts\thermal_pipeline.py
- Run one: python scripts\thermal_pipeline.py --dataset gemini_nofight

### dataset_split.py (D11)
- Video-level splitting to prevent data leakage
- 70/15/15 train/val/test, seed 42
- Run: python scripts\dataset_split.py

### preview_test.py
- Saves 5 side-by-side images to C:\beach\preview_output\
- Run: python scripts\preview_test.py

---

## 10. KEY DECISIONS

| Decision | Reason |
|---|---|
| LLaVA:13b | Local/GDPR/citable/free |
| YOLOv8n-seg | Fastest seg model for CPU |
| 5 FPS sampling | Balance coverage vs storage |
| Max 200 frames/video | 16GB RAM constraint |
| Unbalanced classes 57/43 | Not severe; F1/mAP handle it |
| Video-level split | No data leakage |
| Seed 42 | Reproducibility |
| COLORMAP_HOT | Matches real thermal cameras |
| Gradio | Supervisor requirement |

---

## 11. ERRORS FIXED

| Error | Fix |
|---|---|
| SyntaxError unicodeescape \N | Changed to r""" docstring |
| Cannot open video file | Auto-find video instead of hardcoded name |
| cv2.imshow not appearing | Save images to preview_output instead |
| gemini all labelled fight | Split into fight/no_fight subfolders, updated line 77 |

---

## 12. LITERATURE REVIEW — COMPLETED

Chapter 2 covers 5 themes, 13 papers:
1. Violence/fight detection (Qi et al., Verma, Ojha et al., Evany Anne & Sivakumaran)
2. YOLO architectures (Solak et al., Fatima & Ahmed, Redmon & Farhadi, Jocher et al.)
3. VLMs — CLIP and LLaVA
4. Thermal imaging for privacy
5. GDPR and anonymisation

---

## 13. ETHICS — COMPLETE
- Submitted and received April 2026
- Via Online Ethics Review System, University of Sunderland
- Thermal pipeline = primary GDPR control (Article 5(1)(c) data minimisation)

---

## HOW TO USE THIS FILE

Start of every Claude session:
1. Upload this file
2. Say: "Here is my project context"
3. End of session: say "Update my project context"
4. Download, replace old file, push to GitHub

Save to: C:\beach\project_context.md and GitHub
