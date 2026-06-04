"""OSNet Re-ID settings (CAM3-only prototype)."""

from __future__ import annotations

import os

# Enabled by default for CAM3 event generation.
CAM3_REID_ENABLED = os.getenv("CAM3_REID_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Time/location REENTRY fallback when OSNet does not match.
REID_HEURISTIC_FALLBACK = os.getenv("REID_HEURISTIC_FALLBACK", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

REID_DEBUG = os.getenv("REID_DEBUG", "0").strip().lower() in ("1", "true", "yes")

REID_MODEL_NAME = os.getenv("REID_MODEL_NAME", "osnet_x1_0")
REID_COSINE_THRESHOLD = float(os.getenv("REID_COSINE_THRESHOLD", "0.80"))
REID_EXIT_CACHE_SECONDS = float(os.getenv("REID_EXIT_CACHE_SECONDS", "180"))
REID_MIN_DWELL_AFTER_EXIT_SECONDS = float(
    os.getenv("REID_MIN_DWELL_AFTER_EXIT_SECONDS", "2.0")
)
REID_MIN_CROP_AREA = int(os.getenv("REID_MIN_CROP_AREA", "2000"))
REID_TRACK_EMBED_HISTORY = int(os.getenv("REID_TRACK_EMBED_HISTORY", "10"))

# CAM3 session-aware exit matching (ENTRY → EXIT across long ByteTrack gaps).
REID_SESSION_MATCH_ENABLED = os.getenv("REID_SESSION_MATCH_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
REID_SESSION_MATCH_THRESHOLD = float(
    os.getenv("REID_SESSION_MATCH_THRESHOLD", "0.85")
)
REID_SESSION_MARGIN_THRESHOLD = float(os.getenv("REID_SESSION_MARGIN_THRESHOLD", "0.10"))
REID_AREA_RATIO_MIN = float(os.getenv("REID_AREA_RATIO_MIN", "0.5"))
REID_AREA_RATIO_MAX = float(os.getenv("REID_AREA_RATIO_MAX", "2.0"))
REID_SESSION_MAX_AGE_SECONDS = float(os.getenv("REID_SESSION_MAX_AGE_SECONDS", "180"))
REID_NEAR_DOORWAY_Y_TOLERANCE = float(os.getenv("REID_NEAR_DOORWAY_Y_TOLERANCE", "0.10"))
# ByteTrack gap re-attach: open INSIDE sessions + lower bar when only one eligible visitor.
REID_FRAGMENT_MATCH_THRESHOLD = float(
    os.getenv("REID_FRAGMENT_MATCH_THRESHOLD", "0.76")
)
REID_FRAGMENT_MARGIN_THRESHOLD = float(
    os.getenv("REID_FRAGMENT_MARGIN_THRESHOLD", "0.05")
)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


REID_DEVICE = os.getenv("REID_DEVICE", "cuda" if _cuda_available() else "cpu")

# Market-1501 OSNet x1.0 weights (torchreid model zoo).
OSNET_X1_0_MARKET1501_URL = (
    "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"
)
