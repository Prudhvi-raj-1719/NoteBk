"""Map internal NOTEBK event_type strings to canonical competition event_type."""

from __future__ import annotations

INTERNAL_TO_COMPETITION: dict[str, str] = {
    "ENTRY": "ENTRY",
    "EXIT": "EXIT",
    "REENTRY": "ENTRY",
    "ZONE_ENTER": "ZONE_ENTER",
    "ZONE_EXIT": "ZONE_EXIT",
    "DWELL_COMPLETED": "ZONE_DWELL",
    "QUEUE_ENTER": "BILLING_QUEUE_JOIN",
    "QUEUE_EXIT": "BILLING_QUEUE_EXIT",
    "PAYMENT_ENTER": "BILLING_START",
    "PAYMENT_EXIT": "BILLING_COMPLETE",
}


def to_competition_event_type(internal_type: str) -> str:
    mapped = INTERNAL_TO_COMPETITION.get(internal_type)
    if mapped is None:
        raise ValueError(
            f"Unsupported internal event_type {internal_type!r}; "
            f"cannot map to canonical schema"
        )
    return mapped
