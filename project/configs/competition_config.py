"""Competition schema constants for NOTEBK event emission."""

from __future__ import annotations

import os

# Challenge / POS store identifier (override via STORE_ID env).
STORE_ID = os.getenv("STORE_ID", "STORE_BLR_002")

DEFAULT_EVENT_CONFIDENCE = 0.85

# NOTEBK camera key -> competition camera_id
CAMERA_COMPETITION_IDS: dict[str, str] = {
    "CAM1": "CAM_SHELF_01",
    "CAM2": "CAM_SHELF_02",
    "CAM3": "CAM_ENTRY_01",
    "CAM5": "CAM_BILLING_01",
}

# CAM3 wall-clock anchor (Brigade); Footage2 clips use clip anchors below.
CAM3_CAMERA_START_TIME = "2026-04-10 20:10:15"

# Footage2 entry clips: synthetic anchors on POS sale day for UTC timestamps.
FOOTAGE2_CAM3_CLIP_STARTS: dict[str, str] = {
    "entry1": "2026-04-10 14:00:00",
    "entry2": "2026-04-10 15:00:00",
}

# REENTRY (CAM3): time / location gates (no appearance embeddings).
REENTRY_TIME_WINDOW_SECONDS = float(
    os.getenv("REENTRY_TIME_WINDOW_SECONDS", "180")
)
REENTRY_LOCATION_MAX_NORM_DIST = float(
    os.getenv("REENTRY_LOCATION_MAX_NORM_DIST", "0.12")
)
REENTRY_ENTRY_Y_TOLERANCE = float(
    os.getenv("REENTRY_ENTRY_Y_TOLERANCE", "0.10")
)

# CAM1 brand polygon name -> competition zone_id
CAM1_ZONE_ID_MAP: dict[str, str] = {
    "Minimalist_top": "MINIMALIST_TOP",
    "FarmStay": "FARMSTAY",
    "TheFaceShop": "THEFACESHOP",
    "GoodVibes": "GOODVIBES",
    "DermaCo": "DERMACO",
    "Minimalist": "MINIMALIST",
    "Aquologica": "AQUOLOGICA",
    "Pilgrim": "PILGRIM",
    "D&K": "D_AND_K",
    "Neutrogena": "NEUTROGENA",
    "Mamaearth": "MAMAEARTH",
}

CAM5_ZONE_ID_MAP: dict[str, str] = {
    "PaymentArea": "BILLING",
    "BillingQueue": "billQ",
}
