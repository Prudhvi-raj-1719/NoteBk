"""Inspect CCTV event JSONL timestamps for alignment with POS data."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from configs.camera_timing_config import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR
from typing import Any, Dict, List, Optional, Tuple

REPORT_PATH = OUTPUTS_REPORTS_DIR / "event_timestamp_report.txt"

EVENT_FILES = [
    OUTPUTS_DIR / "cam1_events.jsonl",
    OUTPUTS_DIR / "cam2_events.jsonl",
    OUTPUTS_DIR / "cam3_events.jsonl",
    OUTPUTS_DIR / "cam5_events.jsonl",
]

ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
)
VIDEO_TIMESTAMP_RE = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$"
)
DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class FileInspection:
    path: Path
    exists: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)
    parse_errors: int = 0
    missing_timestamp: int = 0
    timestamp_samples: List[str] = field(default_factory=list)
    classifications: Dict[str, int] = field(default_factory=dict)
    axis_label: str = "unknown"
    format_description: str = "N/A"
    earliest_raw: Optional[str] = None
    latest_raw: Optional[str] = None
    earliest_sort_key: Optional[float] = None
    latest_sort_key: Optional[float] = None
    pos_alignment_note: str = ""


def load_events(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    events: List[Dict[str, Any]] = []
    errors = 0
    if not path.exists():
        return events, errors

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError:
                errors += 1
    return events, errors


def classify_timestamp(value: str) -> str:
    text = value.strip()
    if ISO_DATETIME_RE.match(text):
        return "absolute_iso_datetime"
    if DATE_ONLY_RE.match(text):
        return "absolute_date_only"
    if VIDEO_TIMESTAMP_RE.match(text):
        return "relative_video_offset"
    if text.replace(".", "", 1).isdigit():
        return "numeric_epoch_or_seconds"
    return "unrecognized"


def video_timestamp_to_seconds(value: str) -> float:
    match = VIDEO_TIMESTAMP_RE.match(value.strip())
    if not match:
        raise ValueError(f"Not a video timestamp: {value}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = match.group(4) or "0"
    millis = int(fraction.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def absolute_timestamp_to_sort_key(value: str) -> float:
    text = value.strip().replace(" ", "T")
    if DATE_ONLY_RE.match(text):
        return datetime.fromisoformat(text).timestamp()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").timestamp()


def timestamp_sort_key(value: str, classification: str) -> Optional[float]:
    try:
        if classification == "relative_video_offset":
            return video_timestamp_to_seconds(value)
        if classification.startswith("absolute"):
            return absolute_timestamp_to_sort_key(value)
        if classification == "numeric_epoch_or_seconds":
            return float(value)
    except (ValueError, TypeError):
        return None
    return None


def determine_axis(classifications: Dict[str, int], samples: List[str]) -> Tuple[str, str, str]:
    if not samples:
        return "empty", "No timestamps", "No events to compare with POS."

    relative_count = classifications.get("relative_video_offset", 0)
    absolute_count = classifications.get("absolute_iso_datetime", 0) + classifications.get(
        "absolute_date_only", 0
    )

    if relative_count > 0 and absolute_count > 0:
        return (
            "mixed_relative_and_absolute",
            f"Mixed: {relative_count} video offsets (HH:MM:SS.mmm) + "
            f"{absolute_count} ISO datetimes",
            "NOT on a single time axis. Regenerate events with one format and anchor to POS.",
        )

    if relative_count > 0:
        return (
            "relative_video_timestamps",
            "HH:MM:SS.mmm (elapsed time from start of video)",
            "NOT aligned with POS absolute datetimes without a video start anchor.",
        )
    if absolute_count > 0:
        return (
            "absolute_datetime_timestamps",
            "ISO-style calendar datetime",
            "May align with POS if the anchor date/time matches the recording day.",
        )
    if classifications.get("numeric_epoch_or_seconds", 0) > 0:
        return (
            "numeric_timestamps",
            "Numeric epoch or seconds value",
            "Requires known epoch anchor to compare with POS datetimes.",
        )
    return (
        "unrecognized",
        "Unrecognized timestamp formats",
        "Normalize timestamps before purchase matching.",
    )


def inspect_file(path: Path) -> FileInspection:
    result = FileInspection(path=path, exists=path.exists())
    if not result.exists:
        result.axis_label = "file_missing"
        result.format_description = "File not found"
        result.pos_alignment_note = "Run the corresponding events script to generate this file."
        return result

    result.events, result.parse_errors = load_events(path)
    if not result.events:
        result.axis_label = "empty"
        result.format_description = "File exists but contains no events"
        result.pos_alignment_note = "No CCTV events to match against POS."
        return result

    sortable: List[Tuple[float, str, str]] = []

    for event in result.events:
        ts = event.get("timestamp")
        if ts is None or str(ts).strip() == "":
            result.missing_timestamp += 1
            continue
        ts_str = str(ts)
        if len(result.timestamp_samples) < 5:
            result.timestamp_samples.append(ts_str)

        kind = classify_timestamp(ts_str)
        result.classifications[kind] = result.classifications.get(kind, 0) + 1

        key = timestamp_sort_key(ts_str, kind)
        if key is not None:
            sortable.append((key, ts_str, kind))

    if sortable:
        sortable.sort(key=lambda item: item[0])
        result.earliest_sort_key = sortable[0][0]
        result.latest_sort_key = sortable[-1][0]
        result.earliest_raw = sortable[0][1]
        result.latest_raw = sortable[-1][1]

    unique_kinds = {item[2] for item in sortable}
    if "relative_video_offset" in unique_kinds and (
        "absolute_iso_datetime" in unique_kinds or "absolute_date_only" in unique_kinds
    ):
        result.classifications.pop("mixed_formats", None)

    (
        result.axis_label,
        result.format_description,
        result.pos_alignment_note,
    ) = determine_axis(result.classifications, result.timestamp_samples)

    return result


def format_event(event: Dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)


def print_file_console(result: FileInspection) -> None:
    print("=" * 70)
    print(result.path.name)
    print("=" * 70)

    if not result.exists:
        print("Status: FILE NOT FOUND")
        print(result.pos_alignment_note)
        return

    print(f"Path: {result.path}")
    print(f"Event count: {len(result.events)}")
    print(f"Parse errors: {result.parse_errors}")
    print(f"Missing timestamp field: {result.missing_timestamp}")

    if not result.events:
        print("Status: EMPTY (no events)")
        return

    print(f"Timestamp format: {result.format_description}")
    print(f"Time axis: {result.axis_label}")
    print(f"Earliest timestamp: {result.earliest_raw}")
    print(f"Latest timestamp: {result.latest_raw}")
    print(f"POS alignment: {result.pos_alignment_note}")

    if result.classifications:
        print("Classification counts:")
        for kind, count in sorted(result.classifications.items()):
            print(f"  {kind}: {count}")

    print("\nFirst 20 events:")
    for event in result.events[:20]:
        print(f"  {format_event(event)}")

    print("\nLast 20 events:")
    for event in result.events[-20:]:
        print(f"  {format_event(event)}")


def append_file_report(lines: List[str], result: FileInspection) -> None:
    lines.append("=" * 70)
    lines.append(result.path.name)
    lines.append("=" * 70)
    lines.append(f"Exists: {result.exists}")
    lines.append(f"Event count: {len(result.events)}")
    lines.append(f"Parse errors: {result.parse_errors}")
    lines.append(f"Missing timestamps: {result.missing_timestamp}")

    if not result.exists:
        lines.append(f"Note: {result.pos_alignment_note}")
        lines.append("")
        return

    if not result.events:
        lines.append(f"Note: {result.pos_alignment_note}")
        lines.append("")
        return

    lines.append(f"Timestamp format: {result.format_description}")
    lines.append(f"Time axis classification: {result.axis_label}")
    lines.append(f"Earliest timestamp: {result.earliest_raw}")
    lines.append(f"Latest timestamp: {result.latest_raw}")
    lines.append(f"POS alignment note: {result.pos_alignment_note}")
    lines.append("Timestamp kind counts:")
    for kind, count in sorted(result.classifications.items()):
        lines.append(f"  {kind}: {count}")
    if result.timestamp_samples:
        lines.append("Sample timestamps:")
        for sample in result.timestamp_samples:
            lines.append(f"  - {sample}")

    lines.append("")
    lines.append("First 20 events:")
    for event in result.events[:20]:
        lines.append(f"  {format_event(event)}")

    lines.append("")
    lines.append("Last 20 events:")
    for event in result.events[-20:]:
        lines.append(f"  {format_event(event)}")
    lines.append("")


def build_summary(lines: List[str], results: List[FileInspection]) -> None:
    lines.append("=" * 70)
    lines.append("CROSS-FILE SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("POS transactions use absolute datetimes, e.g.:")
    lines.append("  2026-04-10 16:55:36 (from aggregated_transactions.json)")
    lines.append("")
    lines.append("CCTV event files:")

    for result in results:
        status = (
            "missing"
            if not result.exists
            else "empty"
            if not result.events
            else result.axis_label
        )
        lines.append(
            f"  {result.path.name}: {len(result.events)} events | axis={status}"
        )

    lines.append("")
    lines.append("Recommendation before purchase matching:")
    lines.append(
        "  1. Regenerate cam1_events.jsonl if it still uses placeholder ISO datetimes "
        "instead of video offsets (HH:MM:SS.mmm)."
    )
    lines.append(
        "  2. Use one shared anchor to map video offsets to store local time "
        "(recording start datetime per camera)."
    )
    lines.append(
        "  3. cam3_events.jsonl and cam5_events.jsonl must be populated by running "
        "their event scripts on the same footage day as the POS export."
    )
    lines.append(
        "  4. Only match POS rows when CCTV timestamps are converted to the same "
        "absolute datetime axis as POS transaction_datetime."
    )


def print_summary(results: List[FileInspection]) -> None:
    print("\n" + "=" * 70)
    print("CROSS-FILE SUMMARY")
    print("=" * 70)
    print("\nPOS transactions use absolute datetimes, e.g.:")
    print("  2026-04-10 16:55:36 (from aggregated_transactions.json)")
    print("\nCCTV event files:")
    for result in results:
        if not result.exists:
            status = "missing"
        elif not result.events:
            status = "empty"
        else:
            status = result.axis_label
        print(f"  {result.path.name}: {len(result.events)} events | axis={status}")


def main() -> None:
    results = [inspect_file(path) for path in EVENT_FILES]

    report_lines: List[str] = [
        "CCTV EVENT TIMESTAMP INSPECTION REPORT",
        "",
    ]

    for result in results:
        print_file_console(result)
        append_file_report(report_lines, result)

    print_summary(results)
    build_summary(report_lines, results)

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
