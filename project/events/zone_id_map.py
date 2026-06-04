"""Map internal zone / brand names to competition zone_id."""

from __future__ import annotations

from typing import Optional

from configs.competition_config import CAM1_ZONE_ID_MAP, CAM5_ZONE_ID_MAP


def resolve_zone_id(
    notbk_camera_id: str,
    internal_event_type: str,
    zone_name: Optional[str],
) -> Optional[str]:
    if internal_event_type in ("ENTRY", "EXIT", "REENTRY"):
        return None
    if not zone_name:
        if internal_event_type in ("QUEUE_ENTER", "QUEUE_EXIT"):
            return "BILLING"
        return None

    if notbk_camera_id == "CAM5":
        return CAM5_ZONE_ID_MAP.get(zone_name, "BILLING")

    if notbk_camera_id in ("CAM1", "CAM2"):
        mapped = CAM1_ZONE_ID_MAP.get(zone_name)
        if mapped:
            return mapped
        return zone_name

    return zone_name
