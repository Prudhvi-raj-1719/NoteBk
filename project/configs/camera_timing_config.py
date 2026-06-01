"""CCTV camera timing anchors and project paths (single source of truth).

Manually verified from CCTV overlay timestamps. Do not use OCR for start times.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_REPORTS_DIR = OUTPUTS_DIR / "reports"
OUTPUTS_CHARTS_DIR = OUTPUTS_DIR / "charts"

MODEL_PATH = PROJECT_ROOT / "yolo11m.pt"
CCTV_FOOTAGE_DIR = DATA_DIR / "CCTV Footage"
POS_CSV_PATH = DATA_DIR / "Brigade_Bangalore_10_April_26.csv"

# Real-world datetime when each recording starts (first frame).
CAMERA_START_TIMES: dict[str, str] = {
    "CAM1": "2026-04-10 20:10:27",
    "CAM2": "2026-04-10 20:10:02",
    "CAM5": "2026-04-10 20:09:48",
}

CAMERA_VIDEO_FILES: dict[str, Path] = {
    "CAM1": CCTV_FOOTAGE_DIR / "CAM 1.mp4",
    "CAM2": CCTV_FOOTAGE_DIR / "CAM 2.mp4",
    "CAM3": CCTV_FOOTAGE_DIR / "CAM 3.mp4",
    "CAM5": CCTV_FOOTAGE_DIR / "CAM 5.mp4",
}

# POS sale day for cross-checking normalized event times.
POS_SALE_DATE = "2026-04-10"
