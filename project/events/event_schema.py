"""Canonical competition event schema for all NOTEBK camera pipelines."""

from __future__ import annotations

import uuid
from typing import Any, Dict, FrozenSet, Mapping, Optional

from configs.competition_config import DEFAULT_EVENT_CONFIDENCE, STORE_ID

CANONICAL_EVENT_FIELDS: tuple[str, ...] = (
    "event_id",
    "store_id",
    "camera_id",
    "visitor_id",
    "event_type",
    "timestamp",
    "zone_id",
    "dwell_ms",
    "is_staff",
    "confidence",
    "metadata",
)

ALLOWED_EVENT_TYPES: FrozenSet[str] = frozenset(
    {
        "ENTRY",
        "EXIT",
        "ZONE_ENTER",
        "ZONE_EXIT",
        "ZONE_DWELL",
        "BILLING_QUEUE_JOIN",
        "BILLING_QUEUE_EXIT",
        "BILLING_START",
        "BILLING_COMPLETE",
    }
)


def validate_canonical_event(event: Mapping[str, Any]) -> None:
    """Ensure every mandatory top-level field is present with allowed event_type."""
    missing = [key for key in CANONICAL_EVENT_FIELDS if key not in event]
    if missing:
        raise ValueError(f"Event missing mandatory fields: {missing}")
    event_type = str(event["event_type"])
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"Unsupported event_type {event_type!r}; "
            f"allowed: {sorted(ALLOWED_EVENT_TYPES)}"
        )
    if event["metadata"] is None:
        raise ValueError("metadata must be a dict, not null")
    if not isinstance(event["metadata"], dict):
        raise ValueError("metadata must be a dict")


def build_event(
    *,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: str,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = DEFAULT_EVENT_CONFIDENCE,
    metadata: Optional[Dict[str, Any]] = None,
    store_id: str = STORE_ID,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a schema-compliant event row (UUID v4, ISO-8601 UTC timestamp string)."""
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"Unsupported event_type {event_type!r}; "
            f"allowed: {sorted(ALLOWED_EVENT_TYPES)}"
        )

    row: Dict[str, Any] = {
        "event_id": event_id or str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": max(0, int(dwell_ms)),
        "is_staff": bool(is_staff),
        "confidence": float(confidence),
        "metadata": dict(metadata) if metadata else {},
    }
    validate_canonical_event(row)
    return row
