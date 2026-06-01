"""Check whether normalized CCTV events overlap POS transaction times."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.camera_timing_config import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR, POS_SALE_DATE

AGGREGATED_TRANSACTIONS_PATH = OUTPUTS_DIR / "aggregated_transactions.json"
NORMALIZED_EVENT_FILES = (
    OUTPUTS_DIR / "cam1_events_normalized.jsonl",
    OUTPUTS_DIR / "cam2_events_normalized.jsonl",
)
REPORT_PATH = OUTPUTS_REPORTS_DIR / "video_pos_overlap_report.txt"
WINDOW_PADDING = timedelta(minutes=5)

DISPLAY_DT_FMT = "%Y-%m-%d %H:%M:%S"


def parse_event_datetime(value: str) -> datetime:
    text = value.strip().replace(" ", "T")
    if "." in text:
        base, frac = text.split(".", 1)
        millis = int(frac.ljust(3, "0")[:3])
        return datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(
            microsecond=millis * 1000
        )
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")


def parse_transaction_datetime(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")


def format_display_dt(dt: datetime) -> str:
    return dt.strftime(DISPLAY_DT_FMT)


def load_normalized_events(paths: Tuple[Path, ...]) -> Tuple[List[Dict[str, Any]], List[str]]:
    events: List[Dict[str, Any]] = []
    notes: List[str] = []

    for path in paths:
        if not path.exists():
            notes.append(f"Missing file (skipped): {path.name}")
            continue

        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    notes.append(f"{path.name} line {line_no}: JSON error ({exc})")
                    continue

                event_dt = event.get("event_datetime")
                if not event_dt:
                    notes.append(f"{path.name} line {line_no}: missing event_datetime")
                    continue

                events.append(event)
                count += 1

        notes.append(f"Loaded {count} events from {path.name}")

    return events, notes


def event_datetime_bounds(
    events: List[Dict[str, Any]],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    earliest: Optional[datetime] = None
    latest: Optional[datetime] = None

    for event in events:
        dt = parse_event_datetime(str(event["event_datetime"]))
        if earliest is None or dt < earliest:
            earliest = dt
        if latest is None or dt > latest:
            latest = dt

    return earliest, latest


def load_transactions(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def find_matching_invoices(
    transactions: List[Dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []

    for txn in transactions:
        raw_dt = txn.get("transaction_datetime")
        if not raw_dt:
            continue

        txn_dt = parse_transaction_datetime(str(raw_dt))
        if window_start <= txn_dt <= window_end:
            matches.append(
                {
                    "invoice_number": txn.get("invoice_number", ""),
                    "transaction_datetime": str(raw_dt),
                    "transaction_dt": txn_dt,
                    "brand_names": txn.get("brand_names") or [],
                    "total_amount": txn.get("total_amount"),
                }
            )

    matches.sort(key=lambda row: row["transaction_dt"])
    return matches


def format_brands(brands: List[str]) -> str:
    if not brands:
        return "(none listed)"
    return ", ".join(brands)


def build_report_lines(
    load_notes: List[str],
    event_count: int,
    video_start: Optional[datetime],
    video_end: Optional[datetime],
    window_start: Optional[datetime],
    window_end: Optional[datetime],
    matches: List[Dict[str, Any]],
    total_invoices: int,
) -> List[str]:
    lines = [
        "VIDEO / POS TIME OVERLAP ANALYSIS",
        "",
        "Purpose: verify CCTV event times overlap POS before purchase_matching.py",
        "",
        "Inputs:",
        f"  - {AGGREGATED_TRANSACTIONS_PATH.name}",
    ]
    for path in NORMALIZED_EVENT_FILES:
        lines.append(f"  - {path.name}")
    lines.append("")
    lines.append("CCTV event sources:")
    for note in load_notes:
        lines.append(f"  {note}")
    lines.append(f"Total events with event_datetime: {event_count}")
    lines.append("")

    if video_start is None or video_end is None:
        lines.append("No event_datetime values found — cannot compute overlap.")
        return lines

    lines.extend(
        [
            "CCTV timeline (all normalized events):",
            f"  video_start = {format_display_dt(video_start)}",
            f"  video_end   = {format_display_dt(video_end)}",
            "",
            "POS search window (video_start - 5 min to video_end + 5 min):",
            f"  from = {format_display_dt(window_start)}",
            f"  to   = {format_display_dt(window_end)}",
            "",
            f"Total invoices in {AGGREGATED_TRANSACTIONS_PATH.name}: {total_invoices}",
            f"Matching invoices in window: {len(matches)}",
            "",
        ]
    )

    if not matches:
        lines.append(
            "No POS transactions fall inside the padded CCTV window. "
            "Purchase matching on this clip set is unlikely to find pairs "
            "until CCTV and POS times are confirmed aligned."
        )
        return lines

    lines.append("Matching invoices:")
    lines.append("")
    for idx, match in enumerate(matches, start=1):
        amount = match["total_amount"]
        amount_str = f"{amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
        lines.extend(
            [
                f"{idx}. invoice_number: {match['invoice_number']}",
                f"   transaction_datetime: {match['transaction_datetime']}",
                f"   brands purchased: {format_brands(match['brand_names'])}",
                f"   total_amount: {amount_str}",
                "",
            ]
        )

    return lines


def print_summary(
    video_start: Optional[datetime],
    video_end: Optional[datetime],
    window_start: Optional[datetime],
    window_end: Optional[datetime],
    matches: List[Dict[str, Any]],
) -> None:
    print("Video / POS overlap analysis")
    print("=" * 50)

    if video_start is None or video_end is None:
        print("No CCTV event_datetime values found.")
        return

    print("\nCCTV event timeline:")
    print(f"  video_start = {format_display_dt(video_start)}")
    print(f"  video_end   = {format_display_dt(video_end)}")

    pad_min = int(WINDOW_PADDING.total_seconds() // 60)
    print(f"\nPOS search window (video_start - {pad_min} min to video_end + {pad_min} min):")
    print(f"  from = {format_display_dt(window_start)}")
    print(f"  to   = {format_display_dt(window_end)}")

    print(f"\nMatching invoices: {len(matches)}")
    if not matches:
        print("  (none)")
        return

    print("\nDetails:")
    for match in matches:
        amount = match["total_amount"]
        amount_str = f"{amount:.2f}" if isinstance(amount, (int, float)) else str(amount)
        print(f"  invoice: {match['invoice_number']}")
        print(f"    time:   {match['transaction_datetime']}")
        print(f"    brands: {format_brands(match['brand_names'])}")
        print(f"    total:  {amount_str}")
        print()


def main() -> None:
    events, load_notes = load_normalized_events(NORMALIZED_EVENT_FILES)
    video_start, video_end = event_datetime_bounds(events)

    transactions = load_transactions(AGGREGATED_TRANSACTIONS_PATH)

    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    matches: List[Dict[str, Any]] = []

    if video_start is not None and video_end is not None:
        window_start = video_start - WINDOW_PADDING
        window_end = video_end + WINDOW_PADDING
        matches = find_matching_invoices(transactions, window_start, window_end)

    report_lines = build_report_lines(
        load_notes=load_notes,
        event_count=len(events),
        video_start=video_start,
        video_end=video_end,
        window_start=window_start,
        window_end=window_end,
        matches=matches,
        total_invoices=len(transactions),
    )

    if video_start and video_start.strftime("%Y-%m-%d") != POS_SALE_DATE:
        report_lines.append(
            f"WARNING: video_start date is not {POS_SALE_DATE}"
        )

    report_lines.extend(
        [
            "",
            "Conclusion:",
        ]
    )
    if matches:
        report_lines.append(
            f"  {len(matches)} invoice(s) overlap the CCTV window — "
            "purchase_matching.py may find candidates."
        )
    else:
        report_lines.append(
            "  No invoice overlap in the padded window — "
            "review camera anchors or POS day before purchase_matching.py."
        )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print_summary(video_start, video_end, window_start, window_end, matches)
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
