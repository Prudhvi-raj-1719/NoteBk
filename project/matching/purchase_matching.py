"""Match POS invoices to normalized CCTV events by transaction time window."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.camera_timing_config import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR
from events.event_time import event_datetime_for_matching, utc_iso_to_datetime

MATCH_WINDOW_MINUTES = 5

AGGREGATED_TRANSACTIONS_PATH = OUTPUTS_DIR / "aggregated_transactions.json"
NORMALIZED_EVENT_FILES: Dict[str, Path] = {
    "CAM1": OUTPUTS_DIR / "cam1_events_normalized.jsonl",
    "CAM2": OUTPUTS_DIR / "cam2_events_normalized.jsonl",
    "CAM5": OUTPUTS_DIR / "cam5_events_normalized.jsonl",
}
OUTPUT_PATH = OUTPUTS_DIR / "purchase_matches.json"
REPORT_PATH = OUTPUTS_REPORTS_DIR / "purchase_matching_report.txt"

QUEUE_EVENT_TYPES = frozenset(
    {"QUEUE_ENTER", "QUEUE_EXIT", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "BILLING_QUEUE_EXIT"}
)
PAYMENT_EVENT_TYPES = frozenset(
    {
        "PAYMENT_ENTER",
        "PAYMENT_EXIT",
        "ZONE_ENTER",
        "ZONE_EXIT",
        "BILLING_START",
        "BILLING_COMPLETE",
    }
)
ZONE_ENTER = "ZONE_ENTER"
ZONE_DWELL = "ZONE_DWELL"
DWELL_COMPLETED = "DWELL_COMPLETED"
QUEUE_ZONE_NAMES = frozenset({"BillingQueue"})
PAYMENT_ZONE_NAMES = frozenset({"PaymentArea"})


@dataclass
class MatchingStats:
    total_invoices: int = 0
    matched_invoices: int = 0
    unmatched_invoices: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    load_notes: List[str] = field(default_factory=list)


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


def format_transaction_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_transactions(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def load_events_by_camera(
    event_files: Dict[str, Path],
    stats: MatchingStats,
) -> Dict[str, List[Dict[str, Any]]]:
    by_camera: Dict[str, List[Dict[str, Any]]] = {
        camera: [] for camera in event_files
    }

    for camera_id, path in event_files.items():
        if not path.exists():
            stats.load_notes.append(f"{path.name}: missing (0 events)")
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
                    stats.load_notes.append(
                        f"{path.name} line {line_no}: JSON error ({exc})"
                    )
                    continue

                event_dt = event_datetime_for_matching(event)
                if not event_dt:
                    stats.load_notes.append(
                        f"{path.name} line {line_no}: missing timestamp/event_datetime"
                    )
                    continue

                ts_text = str(event_dt)
                if ts_text.endswith("Z"):
                    event["_parsed_datetime"] = utc_iso_to_datetime(ts_text).replace(
                        tzinfo=None
                    )
                else:
                    event["_parsed_datetime"] = parse_event_datetime(ts_text)
                event.setdefault("camera", camera_id)
                by_camera[camera_id].append(event)
                count += 1

        stats.load_notes.append(f"{path.name}: {count} events")

    for events in by_camera.values():
        events.sort(key=lambda item: item["_parsed_datetime"])

    return by_camera


def events_in_window(
    events: List[Dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> List[Dict[str, Any]]:
    return [
        event
        for event in events
        if window_start <= event["_parsed_datetime"] <= window_end
    ]


def export_event(event: Dict[str, Any]) -> Dict[str, Any]:
    exported = {key: value for key, value in event.items() if not key.startswith("_")}
    return exported


def has_queue_activity(events: List[Dict[str, Any]]) -> bool:
    for event in events:
        if event.get("event_type") in QUEUE_EVENT_TYPES:
            return True
        if event.get("zone_id") in ("BILLING", "billQ") and event.get("event_type") in (
            "BILLING_QUEUE_JOIN",
            "BILLING_QUEUE_ABANDON",
            "BILLING_QUEUE_EXIT",
        ):
            return True
        if event.get("zone") in QUEUE_ZONE_NAMES:
            return True
    return False


def has_payment_activity(events: List[Dict[str, Any]]) -> bool:
    for event in events:
        meta = event.get("metadata") or {}
        internal = meta.get("internal_event_type", event.get("event_type", ""))
        if internal in ("PAYMENT_ENTER", "PAYMENT_EXIT"):
            return True
        if event.get("zone") in PAYMENT_ZONE_NAMES:
            return True
        if event.get("zone_id") in ("BILLING", "billQ") and (
            internal in ("PAYMENT_ENTER", "PAYMENT_EXIT")
            or event.get("event_type") in ("BILLING_START", "BILLING_COMPLETE")
        ):
            return True
    return False


def zones_visited(events: List[Dict[str, Any]]) -> List[str]:
    zones: List[str] = []
    seen = set()
    for event in events:
        if event.get("event_type") not in (ZONE_ENTER,):
            continue
        zone = event.get("zone") or (event.get("metadata") or {}).get("sku_zone")
        if not zone or zone in seen:
            continue
        seen.add(zone)
        zones.append(str(zone))
    return zones


def count_zone_interactions(events: List[Dict[str, Any]]) -> int:
    count = 0
    for event in events:
        event_type = event.get("event_type", "")
        meta = event.get("metadata") or {}
        internal = meta.get("internal_event_type", event_type)
        if event_type in (ZONE_ENTER, "ZONE_EXIT", ZONE_DWELL, DWELL_COMPLETED) or internal in (
            "ZONE_ENTER",
            "ZONE_EXIT",
            "DWELL_COMPLETED",
        ):
            count += 1
        elif event_type in QUEUE_EVENT_TYPES | PAYMENT_EVENT_TYPES:
            count += 1
    return count


def temporal_proximity_score(
    nearby_events: List[Dict[str, Any]],
    transaction_dt: datetime,
    window_seconds: float,
) -> float:
    if not nearby_events:
        return 0.0

    min_delta = min(
        abs((event["_parsed_datetime"] - transaction_dt).total_seconds())
        for event in nearby_events
    )
    if window_seconds <= 0:
        return 0.0
    closeness = 1.0 - min(min_delta / window_seconds, 1.0)
    return closeness * 0.35


def compute_confidence_score(
    nearby_events: List[Dict[str, Any]],
    transaction_dt: datetime,
    window_minutes: int,
) -> float:
    if not nearby_events:
        return 0.0

    score = 0.15
    window_seconds = window_minutes * 60.0

    if has_payment_activity(nearby_events):
        score += 0.25
    if has_queue_activity(nearby_events):
        score += 0.20

    zone_interactions = count_zone_interactions(nearby_events)
    if zone_interactions >= 3:
        score += 0.20
    elif zone_interactions >= 1:
        score += 0.10

    score += temporal_proximity_score(
        nearby_events, transaction_dt, window_seconds
    )

    return round(min(score, 1.0), 2)


def build_journey_summary(nearby_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "zones_visited": zones_visited(nearby_events),
        "queue_activity": has_queue_activity(nearby_events),
        "payment_activity": has_payment_activity(nearby_events),
    }


def match_transaction(
    transaction: Dict[str, Any],
    events_by_camera: Dict[str, List[Dict[str, Any]]],
    window: timedelta,
) -> Dict[str, Any]:
    raw_dt = transaction.get("transaction_datetime")
    if not raw_dt:
        return {
            "invoice_number": transaction.get("invoice_number", ""),
            "transaction_datetime": None,
            "match_found": False,
            "reason": "Missing transaction_datetime",
        }

    transaction_dt = parse_transaction_datetime(str(raw_dt))
    window_start = transaction_dt - window
    window_end = transaction_dt + window

    cam1_nearby = events_in_window(
        events_by_camera.get("CAM1", []), window_start, window_end
    )
    cam2_nearby = events_in_window(
        events_by_camera.get("CAM2", []), window_start, window_end
    )
    cam5_nearby = events_in_window(
        events_by_camera.get("CAM5", []), window_start, window_end
    )

    candidate_customer_journey = {
        "cam1_events": [export_event(event) for event in cam1_nearby],
        "cam2_events": [export_event(event) for event in cam2_nearby],
        "cam5_events": [export_event(event) for event in cam5_nearby],
    }

    matching_events = (
        candidate_customer_journey["cam1_events"]
        + candidate_customer_journey["cam2_events"]
        + candidate_customer_journey["cam5_events"]
    )
    matching_events.sort(key=lambda item: item.get("event_datetime", ""))

    brands = transaction.get("brand_names") or []
    base_record: Dict[str, Any] = {
        "invoice_number": transaction.get("invoice_number", ""),
        "transaction_datetime": str(raw_dt),
        "brands_purchased": brands,
        "total_amount": transaction.get("total_amount"),
        "match_window_minutes": MATCH_WINDOW_MINUTES,
        "search_window": {
            "from": format_transaction_datetime(window_start),
            "to": format_transaction_datetime(window_end),
        },
        "candidate_customer_journey": candidate_customer_journey,
    }

    if not matching_events:
        return {
            **base_record,
            "match_found": False,
            "reason": "No CCTV events in time window",
            "matching_events": [],
            "confidence_score": 0.0,
            "journey_summary": {
                "zones_visited": [],
                "queue_activity": False,
                "payment_activity": False,
            },
        }

    all_nearby_parsed = cam1_nearby + cam2_nearby + cam5_nearby
    confidence = compute_confidence_score(
        all_nearby_parsed, transaction_dt, MATCH_WINDOW_MINUTES
    )

    return {
        **base_record,
        "match_found": True,
        "matching_events": matching_events,
        "confidence_score": confidence,
        "journey_summary": build_journey_summary(all_nearby_parsed),
    }


def run_matching() -> Tuple[List[Dict[str, Any]], MatchingStats]:
    stats = MatchingStats()
    events_by_camera = load_events_by_camera(NORMALIZED_EVENT_FILES, stats)
    transactions = load_transactions(AGGREGATED_TRANSACTIONS_PATH)
    stats.total_invoices = len(transactions)

    window = timedelta(minutes=MATCH_WINDOW_MINUTES)
    results: List[Dict[str, Any]] = []

    for transaction in transactions:
        record = match_transaction(transaction, events_by_camera, window)
        results.append(record)

        if record.get("match_found"):
            stats.matched_invoices += 1
            stats.confidence_scores.append(float(record.get("confidence_score", 0.0)))
        else:
            stats.unmatched_invoices += 1

    return results, stats


def average_confidence(scores: List[float]) -> Optional[float]:
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def build_report_lines(results: List[Dict[str, Any]], stats: MatchingStats) -> List[str]:
    avg_conf = average_confidence(stats.confidence_scores)
    lines = [
        "PURCHASE MATCHING REPORT",
        "",
        "Engine: purchase_matching.py",
        f"Match window: transaction_datetime +/- {MATCH_WINDOW_MINUTES} minutes",
        "",
        "Inputs:",
        f"  - {AGGREGATED_TRANSACTIONS_PATH.name}",
    ]
    for path in NORMALIZED_EVENT_FILES.values():
        lines.append(f"  - {path.name}")
    lines.append("")
    lines.append("Event file notes:")
    for note in stats.load_notes:
        lines.append(f"  {note}")
    lines.append("")
    lines.append(f"Total invoices: {stats.total_invoices}")
    lines.append(f"Matched invoices: {stats.matched_invoices}")
    lines.append(f"Unmatched invoices: {stats.unmatched_invoices}")
    if avg_conf is not None:
        lines.append(f"Average confidence (matched only): {avg_conf}")
    else:
        lines.append("Average confidence (matched only): N/A (no matches)")
    lines.append("")
    lines.append("Matched invoice numbers:")
    matched_numbers = [
        row["invoice_number"] for row in results if row.get("match_found")
    ]
    if matched_numbers:
        for invoice in matched_numbers:
            lines.append(f"  - {invoice}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Unmatched invoice numbers:")
    unmatched_numbers = [
        row["invoice_number"] for row in results if not row.get("match_found")
    ]
    if unmatched_numbers:
        for invoice in unmatched_numbers:
            reason = next(
                (
                    row.get("reason", "")
                    for row in results
                    if row.get("invoice_number") == invoice
                ),
                "",
            )
            lines.append(f"  - {invoice} ({reason})")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Output JSON: {OUTPUT_PATH.name}")
    return lines


def print_summary(stats: MatchingStats) -> None:
    avg_conf = average_confidence(stats.confidence_scores)
    print("Purchase matching")
    print("=" * 50)
    print(f"Match window: +/- {MATCH_WINDOW_MINUTES} minutes")
    print(f"Total invoices: {stats.total_invoices}")
    print(f"Matched: {stats.matched_invoices}")
    print(f"Unmatched: {stats.unmatched_invoices}")
    if avg_conf is not None:
        print(f"Average confidence (matched): {avg_conf}")
    else:
        print("Average confidence (matched): N/A")
    print(f"\nWrote: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


def main() -> None:
    results, stats = run_matching()

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(build_report_lines(results, stats)) + "\n",
        encoding="utf-8",
    )
    print_summary(stats)


if __name__ == "__main__":
    main()
