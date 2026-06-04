"""Video offset and ISO UTC timestamps for competition events."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from configs.camera_timing_config import CAMERA_START_TIMES, POS_SALE_DATE
from configs.competition_config import (
    CAM3_CAMERA_START_TIME,
    FOOTAGE2_CAM3_CLIP_STARTS,
)

VIDEO_OFFSET_RE = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$"
)


def parse_offset_seconds(offset_timestamp: str) -> float:
    match = VIDEO_OFFSET_RE.match(offset_timestamp.strip())
    if not match:
        raise ValueError(f"Invalid video offset: {offset_timestamp}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = match.group(4) or "0"
    millis = int(fraction.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def is_video_offset(value: str) -> bool:
    return bool(VIDEO_OFFSET_RE.match(value.strip()))


def _parse_anchor(anchor: str) -> datetime:
    return datetime.strptime(anchor, "%Y-%m-%d %H:%M:%S")


def resolve_recording_start(
    notbk_camera_id: str,
    *,
    clip_id: Optional[str] = None,
) -> datetime:
    if clip_id and clip_id in FOOTAGE2_CAM3_CLIP_STARTS:
        return _parse_anchor(FOOTAGE2_CAM3_CLIP_STARTS[clip_id])
    if notbk_camera_id == "CAM3" and notbk_camera_id not in CAMERA_START_TIMES:
        return _parse_anchor(CAM3_CAMERA_START_TIME)
    if notbk_camera_id in CAMERA_START_TIMES:
        return _parse_anchor(CAMERA_START_TIMES[notbk_camera_id])
    return datetime.strptime(f"{POS_SALE_DATE} 00:00:00", "%Y-%m-%d %H:%M:%S")


def video_offset_to_utc_iso(
    notbk_camera_id: str,
    video_offset: str,
    *,
    clip_id: Optional[str] = None,
) -> str:
    """Convert HH:MM:SS.mmm offset to ISO8601 UTC (naive UTC, suffixed with Z)."""
    start = resolve_recording_start(notbk_camera_id, clip_id=clip_id)
    delta = timedelta(seconds=parse_offset_seconds(video_offset))
    dt = start + delta
    millis = int(dt.microsecond // 1000)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{millis:03d}Z"


def utc_iso_to_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_datetime_for_matching(event: dict) -> Optional[str]:
    """Prefer legacy event_datetime; fall back to competition timestamp."""
    if event.get("event_datetime"):
        return str(event["event_datetime"])
    ts = event.get("timestamp")
    if ts is None:
        return None
    text = str(ts).strip()
    if text.endswith("Z") or text.startswith("20"):
        return text.replace("Z", "") if text.endswith("Z") else text
    return None
