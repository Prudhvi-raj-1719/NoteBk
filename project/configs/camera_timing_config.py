"""CCTV camera timing anchors and project paths (single source of truth).

Manually verified from CCTV overlay timestamps. Do not use OCR for start times.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

ENTRY_STABILITY_FRAMES = 3
EXIT_STABILITY_FRAMES = 3

# CAM3 production cadence (recovery: denser sampling than CAM1/CAM5).
CAM3_PROCESS_EVERY_N_FRAMES = 5

# CAM3-only crossing recovery (injected into CAM3_ENTRY_CONFIG layouts).
CAM3_ENTRY_STABILITY_FRAMES = 2
CAM3_EXIT_STABILITY_FRAMES = 2
CAM3_NEAR_THRESHOLD_Y_TOLERANCE = 0.10
CAM3_THRESHOLD_OBSERVATION_FRAMES = 3
CAM3_THRESHOLD_MOTION_EPS = 0.015
CAM3_ENABLE_LATE_ENTRY_RECOVERY = True
CAM3_LATE_ENTRY_Y_BAND_BELOW = 0.22
CAM3_FOOTAGE2_PROCESS_EVERY_N_FRAMES = 2
CAM3_FOOTAGE2_CONFIDENCE_THRESHOLD = 0.12
CAM3_FOOTAGE2_YOLO_MAX_DET = 100

CAM3_ENABLE_TRACK_LOSS_FLUSH = True
CAM3_TRACK_LOSS_GRACE_FRAMES = 35
CAM3_TRACK_LOSS_MOTION_EPS = 0.5
CAM3_TRACK_LOSS_FLUSH_REQUIRE_STABLE_PENDING = True
CAM3_LATE_ENTRY_REQUIRE_NEAR_THRESHOLD = False

_CAM3_RECOVERY_DEFAULTS: dict[str, Any] = {
    "ENTRY_STABILITY_FRAMES": CAM3_ENTRY_STABILITY_FRAMES,
    "EXIT_STABILITY_FRAMES": CAM3_EXIT_STABILITY_FRAMES,
    "ENABLE_THRESHOLD_RECOVERY": True,
    "NEAR_THRESHOLD_Y_TOLERANCE": CAM3_NEAR_THRESHOLD_Y_TOLERANCE,
    "THRESHOLD_OBSERVATION_FRAMES": CAM3_THRESHOLD_OBSERVATION_FRAMES,
    "THRESHOLD_MOTION_EPS": CAM3_THRESHOLD_MOTION_EPS,
    "ENABLE_TRACK_LOSS_FLUSH": CAM3_ENABLE_TRACK_LOSS_FLUSH,
    "TRACK_LOSS_GRACE_FRAMES": CAM3_TRACK_LOSS_GRACE_FRAMES,
    "TRACK_LOSS_MOTION_EPS": CAM3_TRACK_LOSS_MOTION_EPS,
    "TRACK_LOSS_FLUSH_REQUIRE_STABLE_PENDING": (
        CAM3_TRACK_LOSS_FLUSH_REQUIRE_STABLE_PENDING
    ),
}


def _cam3_layout(base: dict) -> dict:
    merged = dict(_CAM3_RECOVERY_DEFAULTS)
    merged.update(base)
    return merged


CAM3_ENTRY_CONFIG: dict[str, dict] = {
    "BRIGADE": _cam3_layout({
        "ENTRY_LINE_POLYGON": [
            (0.5950, 0.4444),
            (0.4498, 0.7510),
            (0.4714, 0.7759),
            (0.6121, 0.4566),
        ],
        "STORE_REF": (0.55, 0.48),
    }),
    "FOOTAGE2": _cam3_layout({
        "ENTRY_LINE_POLYGON": [
            (0.3139, 0.6090),
            (0.6815, 0.5701),
            (0.6847, 0.5796),
            (0.3145, 0.6201),
        ],
        "STORE_REF": (0.50, 0.35),
        # Thin horizontal vline: separate by image-y, not left/right quad diagonal.
        "ENTRY_LINE_STYLE": "horizontal_y",
        "ENTRY_PLANE_Y_NORM": 0.54,
        # Footage2: top of threshold => OUTSIDE, bottom => INSIDE (retail only; geometry unchanged).
        "INVERT_RETAIL_SEMANTICS": True,
        # Doorway crowd: denser frames + lower conf so lead walker gets a box at crossing.
        "PROCESS_EVERY_N_FRAMES": CAM3_FOOTAGE2_PROCESS_EVERY_N_FRAMES,
        "CONFIDENCE_THRESHOLD": CAM3_FOOTAGE2_CONFIDENCE_THRESHOLD,
        "YOLO_MAX_DET": CAM3_FOOTAGE2_YOLO_MAX_DET,
        "IOU_THRESHOLD": 0.50,
        "NEAR_THRESHOLD_Y_TOLERANCE": 0.18,
        "THRESHOLD_OBSERVATION_FRAMES": 2,
        "THRESHOLD_MOTION_EPS": 0.01,
        "ENABLE_LATE_ENTRY_RECOVERY": CAM3_ENABLE_LATE_ENTRY_RECOVERY,
        "LATE_ENTRY_Y_BAND_BELOW": 0.14,
        "LATE_ENTRY_REQUIRE_NEAR_THRESHOLD": False,
        "TRACK_LOSS_FLUSH_REQUIRE_STABLE_PENDING": True,
    }),
}
