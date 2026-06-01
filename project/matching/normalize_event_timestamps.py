"""Normalize CCTV event video offsets to real-world datetimes using manual anchors."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from configs.camera_timing_config import (
    CAMERA_START_TIMES,
    CAMERA_VIDEO_FILES,
    POS_SALE_DATE,
    OUTPUTS_DIR,
    OUTPUTS_REPORTS_DIR,
)

VIDEO_OFFSET_RE = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$"
)
ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
)

REPORT_PATH = OUTPUTS_REPORTS_DIR / "event_time_normalization_report.txt"

# CAM2 output already normalized; only validate, do not rewrite.
SKIP_NORMALIZE_CAMERAS = frozenset({"CAM2"})

CAMERA_IO = {
    "CAM1": {
        "input": OUTPUTS_DIR / "cam1_events.jsonl",
        "output": OUTPUTS_DIR / "cam1_events_normalized.jsonl",
    },
    "CAM2": {
        "input": OUTPUTS_DIR / "cam2_events.jsonl",
        "output": OUTPUTS_DIR / "cam2_events_normalized.jsonl",
    },
    "CAM5": {
        "input": OUTPUTS_DIR / "cam5_events.jsonl",
        "output": OUTPUTS_DIR / "cam5_events_normalized.jsonl",
    },
}


@dataclass
class CameraNormalizationResult:
    camera_id: str
    input_path: Path
    output_path: Path
    skipped: bool = False
    skip_reason: str = ""
    events_processed: int = 0
    events_written: int = 0
    offset_converted: int = 0
    iso_copied: int = 0
    already_had_event_datetime: int = 0
    missing_timestamp: int = 0
    invalid_timestamps: int = 0
    invalid_timestamp_samples: List[str] = field(default_factory=list)
    skipped_records: int = 0
    parse_errors: int = 0
    wrong_sale_date: int = 0
    wrong_sale_date_samples: List[str] = field(default_factory=list)
    validation_failures: int = 0
    first_event_datetime: Optional[str] = None
    last_event_datetime: Optional[str] = None
    recording_start: Optional[datetime] = None
    recording_end: Optional[datetime] = None
    validation_notes: List[str] = field(default_factory=list)

    def note_invalid_timestamp(self, value: str) -> None:
        self.invalid_timestamps += 1
        if len(self.invalid_timestamp_samples) < 20:
            self.invalid_timestamp_samples.append(value)

    def note_wrong_sale_date(self, event_dt: str) -> None:
        self.wrong_sale_date += 1
        if len(self.wrong_sale_date_samples) < 10:
            self.wrong_sale_date_samples.append(event_dt)


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


def parse_start_datetime(camera_id: str) -> datetime:
    if camera_id not in CAMERA_START_TIMES:
        raise KeyError(f"No CAMERA_START_TIMES entry for {camera_id}")
    return datetime.strptime(CAMERA_START_TIMES[camera_id], "%Y-%m-%d %H:%M:%S")


def format_event_datetime(dt: datetime) -> str:
    millis = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{millis:03d}"


def offset_to_datetime(camera_id: str, offset_timestamp: str) -> datetime:
    """Video offset (HH:MM:SS.mmm) + CAMERA_START_TIMES -> real-world datetime."""
    start = parse_start_datetime(camera_id)
    delta = timedelta(seconds=parse_offset_seconds(offset_timestamp))
    return start + delta


def parse_iso_timestamp(value: str) -> datetime:
    text = value.strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")


def is_video_offset(value: str) -> bool:
    return bool(VIDEO_OFFSET_RE.match(value.strip()))


def is_iso_datetime(value: str) -> bool:
    return bool(ISO_DATETIME_RE.match(value.strip()))


def is_on_sale_date(event_dt_str: str) -> bool:
    return event_dt_str.strip()[:10] == POS_SALE_DATE


def get_video_recording_window(camera_id: str) -> Tuple[datetime, datetime]:
    video_path = CAMERA_VIDEO_FILES[camera_id]
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for {camera_id}: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    start = parse_start_datetime(camera_id)
    end = start + timedelta(seconds=duration_seconds)
    return start, end


def resolve_event_datetime(
    camera_id: str,
    timestamp: str,
    result: CameraNormalizationResult,
) -> Optional[str]:
    if is_video_offset(timestamp):
        dt = offset_to_datetime(camera_id, timestamp)
        result.offset_converted += 1
        return format_event_datetime(dt)

    if is_iso_datetime(timestamp):
        dt = parse_iso_timestamp(timestamp)
        result.iso_copied += 1
        return format_event_datetime(dt)

    result.note_invalid_timestamp(timestamp)
    return None


def load_events(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    events: List[Dict[str, Any]] = []
    errors = 0
    if not path.exists():
        return events, errors

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                errors += 1
    return events, errors


def update_datetime_bounds(result: CameraNormalizationResult, event_dt: str) -> None:
    if result.first_event_datetime is None or event_dt < result.first_event_datetime:
        result.first_event_datetime = event_dt
    if result.last_event_datetime is None or event_dt > result.last_event_datetime:
        result.last_event_datetime = event_dt


def validate_event_datetime(
    event_dt_str: str,
    recording_start: datetime,
    recording_end: datetime,
    result: CameraNormalizationResult,
) -> bool:
    if not is_on_sale_date(event_dt_str):
        result.note_wrong_sale_date(event_dt_str)
        return False

    event_dt = parse_iso_timestamp(event_dt_str)
    if recording_start <= event_dt <= recording_end + timedelta(milliseconds=1):
        return True

    result.validation_failures += 1
    if len(result.validation_notes) < 10:
        result.validation_notes.append(
            f"Out of recording window: {event_dt_str} "
            f"(allowed {format_event_datetime(recording_start)} "
            f"to {format_event_datetime(recording_end)})"
        )
    return False


def audit_normalized_file(result: CameraNormalizationResult) -> None:
    """Fill stats from an existing normalized JSONL (used when rewrite is skipped)."""
    events, result.parse_errors = load_events(result.output_path)
    if not events:
        result.skipped = True
        result.skip_reason = "normalized output missing or empty"
        return

    try:
        result.recording_start, result.recording_end = get_video_recording_window(
            result.camera_id
        )
    except (FileNotFoundError, RuntimeError) as exc:
        result.skip_reason = f"{result.skip_reason}; video check failed: {exc}"
        result.recording_start = parse_start_datetime(result.camera_id)

    for event in events:
        result.events_processed += 1
        event_dt = event.get("event_datetime")
        if not event_dt:
            result.missing_timestamp += 1
            result.skipped_records += 1
            continue

        event_dt_str = str(event_dt)
        result.events_written += 1
        update_datetime_bounds(result, event_dt_str)

        if not is_on_sale_date(event_dt_str):
            result.note_wrong_sale_date(event_dt_str)

        if result.recording_start and result.recording_end:
            validate_event_datetime(
                event_dt_str,
                result.recording_start,
                result.recording_end,
                result,
            )


def normalize_camera_events(camera_id: str) -> CameraNormalizationResult:
    io = CAMERA_IO[camera_id]
    result = CameraNormalizationResult(
        camera_id=camera_id,
        input_path=io["input"],
        output_path=io["output"],
    )

    if camera_id in SKIP_NORMALIZE_CAMERAS:
        result.skipped = True
        result.skip_reason = "already normalized (rewrite skipped)"
        if result.output_path.exists():
            audit_normalized_file(result)
        else:
            result.skip_reason += "; output file not found"
        return result

    if not result.input_path.exists():
        result.skipped = True
        result.skip_reason = "input file not found"
        result.skipped_records = 1
        return result

    events, result.parse_errors = load_events(result.input_path)
    if not events:
        result.skipped = True
        result.skip_reason = "no events in file"
        return result

    try:
        result.recording_start, result.recording_end = get_video_recording_window(
            camera_id
        )
    except (FileNotFoundError, RuntimeError) as exc:
        result.skipped = True
        result.skip_reason = str(exc)
        return result

    normalized_events: List[Dict[str, Any]] = []

    for event in events:
        result.events_processed += 1
        output_event = dict(event)

        timestamp = output_event.get("timestamp")
        if timestamp is None or str(timestamp).strip() == "":
            result.missing_timestamp += 1
            result.skipped_records += 1
            normalized_events.append(output_event)
            continue

        timestamp_str = str(timestamp).strip()
        event_dt = resolve_event_datetime(camera_id, timestamp_str, result)
        if event_dt is None:
            result.skipped_records += 1
            normalized_events.append(output_event)
            continue

        output_event["event_datetime"] = event_dt
        validate_event_datetime(
            event_dt,
            result.recording_start,
            result.recording_end,
            result,
        )
        update_datetime_bounds(result, event_dt)
        normalized_events.append(output_event)
        result.events_written += 1

    with result.output_path.open("w", encoding="utf-8") as handle:
        for event in normalized_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    return result


def print_camera_summary(result: CameraNormalizationResult) -> None:
    print(f"\n{result.camera_id}:")
    if result.skipped:
        print(f"  normalize: skipped ({result.skip_reason})")
    else:
        print(f"  normalize: completed")

    print(f"  events processed = {result.events_processed}")
    print(f"  events written (with event_datetime) = {result.events_written}")
    print(f"  offset -> event_datetime = {result.offset_converted}")
    if result.iso_copied:
        print(f"  iso timestamps copied = {result.iso_copied}")
    print(f"  first event_datetime = {result.first_event_datetime}")
    print(f"  last event_datetime = {result.last_event_datetime}")
    print(f"  invalid timestamps = {result.invalid_timestamps}")
    print(f"  skipped records = {result.skipped_records}")
    if result.parse_errors:
        print(f"  json parse errors = {result.parse_errors}")
    if result.wrong_sale_date:
        print(
            f"  event_datetime not on {POS_SALE_DATE} = {result.wrong_sale_date}"
        )
    if result.validation_failures:
        print(f"  recording window failures = {result.validation_failures}")


def append_report(lines: List[str], result: CameraNormalizationResult) -> None:
    lines.append("=" * 70)
    lines.append(result.camera_id)
    lines.append("=" * 70)
    lines.append(f"Input: {result.input_path}")
    lines.append(f"Output: {result.output_path}")
    lines.append(f"Anchor: {CAMERA_START_TIMES.get(result.camera_id, 'N/A')}")

    if result.skipped:
        lines.append(f"Normalize: SKIPPED ({result.skip_reason})")
    else:
        lines.append("Normalize: completed")

    if result.recording_start and result.recording_end:
        lines.append(
            f"Recording window: {format_event_datetime(result.recording_start)} "
            f"to {format_event_datetime(result.recording_end)}"
        )

    lines.append(f"Events processed: {result.events_processed}")
    lines.append(f"Events written: {result.events_written}")
    lines.append(f"Offset converted: {result.offset_converted}")
    lines.append(f"First event_datetime: {result.first_event_datetime}")
    lines.append(f"Last event_datetime: {result.last_event_datetime}")
    lines.append(f"Invalid timestamps: {result.invalid_timestamps}")
    if result.invalid_timestamp_samples:
        lines.append("Invalid timestamp samples:")
        for sample in result.invalid_timestamp_samples:
            lines.append(f"  - {sample}")
    lines.append(f"Skipped records: {result.skipped_records}")
    lines.append(f"JSON parse errors (input lines): {result.parse_errors}")
    lines.append(
        f"event_datetime not on {POS_SALE_DATE}: {result.wrong_sale_date}"
    )
    if result.wrong_sale_date_samples:
        lines.append("Wrong-date samples:")
        for sample in result.wrong_sale_date_samples:
            lines.append(f"  - {sample}")
    lines.append(f"Recording window validation failures: {result.validation_failures}")
    if result.validation_notes:
        lines.append("Validation notes:")
        for note in result.validation_notes:
            lines.append(f"  - {note}")
    lines.append("")


def main() -> None:
    print("CCTV event timestamp normalization")
    print("Source of truth: camera_timing_config.CAMERA_START_TIMES")
    print(f"Required event_datetime date: {POS_SALE_DATE}")
    print("\nAnchors:")
    for camera_id in CAMERA_IO:
        print(f"  {camera_id}: {CAMERA_START_TIMES[camera_id]}")

    results = [normalize_camera_events(camera_id) for camera_id in CAMERA_IO]

    all_on_sale_date = all(
        r.wrong_sale_date == 0
        for r in results
        if r.events_written > 0 or r.camera_id in SKIP_NORMALIZE_CAMERAS
    )

    print("\n" + "=" * 70)
    print("NORMALIZATION SUMMARY")
    print("=" * 70)
    for result in results:
        print_camera_summary(result)

    print("\n" + "=" * 70)
    if all_on_sale_date:
        print(f"DATE CHECK: All event_datetime values are on {POS_SALE_DATE}")
    else:
        print(f"DATE CHECK: FAILED — some events are not on {POS_SALE_DATE}")
    print("=" * 70)

    report_lines = [
        "EVENT TIME NORMALIZATION REPORT",
        "",
        "Method: video offset (HH:MM:SS.mmm) + CAMERA_START_TIMES",
        "Config: camera_timing_config.py (single source of truth)",
        "OCR: not used",
        "",
        "CAMERA_START_TIMES:",
    ]
    for camera_id in CAMERA_IO:
        report_lines.append(f"  {camera_id}: {CAMERA_START_TIMES[camera_id]}")
    report_lines.append("")
    report_lines.append(f"Required event_datetime date: {POS_SALE_DATE}")
    report_lines.append(f"Skipped rewrite: {', '.join(sorted(SKIP_NORMALIZE_CAMERAS))}")
    report_lines.append("")

    for result in results:
        append_report(report_lines, result)

    report_lines.append(
        f"Global date check ({POS_SALE_DATE}): "
        + ("PASS" if all_on_sale_date else "FAIL")
    )
    report_lines.append("")
    report_lines.append("CAM3: not in pipeline (no cam3_events.jsonl)")
    report_lines.append("Purchase matching: not built")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
