"""Post-hoc staff enrichment for competition JSONL (no YOLO / detection changes).

All events are emitted with is_staff=false. Run enrich_events() after generation
to update rows for identified staff sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


EnrichmentFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def classify_staff_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Stub classifier: leaves is_staff false. Replace with uniform/zone heuristics."""
    return event


def enrich_event_row(
    event: Dict[str, Any],
    classifier: EnrichmentFn = classify_staff_event,
) -> Dict[str, Any]:
    """Return a copy of one competition event, optionally flagged as staff."""
    updated = dict(event)
    updated = classifier(updated)
    if updated.get("is_staff"):
        meta = dict(updated.get("metadata") or {})
        meta.setdefault("staff_enriched", True)
        updated["metadata"] = meta
    return updated


def enrich_events(
    input_path: Path,
    output_path: Path,
    *,
    classifier: EnrichmentFn = classify_staff_event,
) -> int:
    """Read competition JSONL, apply staff rules, write output. Returns rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            dst.write(json.dumps(enrich_event_row(event, classifier)) + "\n")
            written += 1
    return written


def mark_visitor_staff(
    events: List[Dict[str, Any]],
    visitor_id: str,
    *,
    reason: str = "manual",
) -> List[Dict[str, Any]]:
    """Set is_staff=true for every event belonging to visitor_id (session hook)."""
    out: List[Dict[str, Any]] = []
    for event in events:
        row = dict(event)
        if row.get("visitor_id") == visitor_id:
            row["is_staff"] = True
            meta = dict(row.get("metadata") or {})
            meta["staff_reason"] = reason
            row["metadata"] = meta
        out.append(row)
    return out
