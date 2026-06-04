# NOTEBK Competition Schema Migration Report

**Date:** 2026-05-31  
**Scope:** Phases 1–4 in `E:\NOTEBK\project` only (Purpple_Vision unchanged).

---

## 1. Files changed

### Created

| Path | Purpose |
|------|---------|
| `configs/competition_config.py` | `store_id`, camera map, zone maps, REENTRY tuning |
| `events/event_emitter.py` | Shared `EventEmitter` + competition JSONL |
| `events/event_time.py` | Video offset → ISO UTC (`Z`) |
| `events/event_type_map.py` | Internal → competition `event_type` |
| `events/zone_id_map.py` | Brand/area → `zone_id` |
| `events/visitor_registry.py` | `VIS_00001` allocation |
| `events/reentry_session.py` | EXIT cache + REENTRY matching (CAM3) |
| `staff/staff_detection.py` | Post-hoc `is_staff` enrichment hook |
| `staff/__init__.py` | Package marker |
| `outputs/reports/competition_schema_migration_report.md` | This report |

### Modified

| Path | Change |
|------|--------|
| `events/cam1_events.py` | `EventLogger` → `EventEmitter` |
| `events/cam3_events.py` | `EventLogger` → `EventEmitter` + REENTRY + clip anchors |
| `events/cam5_events.py` | `EventLogger` → `EventEmitter` |
| `matching/purchase_matching.py` | Read `timestamp` / competition types |
| `matching/normalize_event_timestamps.py` | Import shared time helpers |
| `analytics/retail_analytics.py` | `ZONE_DWELL`, `metadata.sku_zone`, UTC timestamps |

### Unchanged (by design)

| Path | Note |
|------|------|
| `entry_retail.py` | Crossing logic; still emits internal `ENTRY`/`EXIT` |
| Purpple_Vision | Not touched |

---

## 2. Output schema

Every emitted JSONL row:

```json
{
  "event_id": "<uuid4>",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_SHELF_01",
  "visitor_id": "VIS_00001",
  "event_type": "ZONE_ENTER",
  "timestamp": "2026-04-10T20:11:27.800Z",
  "zone_id": "FARMSTAY",
  "dwell_ms": 0,
  "is_staff": false,
  "confidence": 0.85,
  "metadata": {
    "byte_track_id": 6,
    "video_offset": "00:01:00.800",
    "internal_event_type": "ZONE_ENTER",
    "sku_zone": "FarmStay"
  }
}
```

`metadata` always includes `byte_track_id` and `video_offset`. CAM3 transitions include `metadata.debug`.

---

## 3. Event type mapping

| Internal (engine) | Competition `event_type` | Typical `zone_id` |
|-------------------|-------------------------|-------------------|
| `ENTRY` | `ENTRY` | `null` |
| `EXIT` | `EXIT` | `null` |
| `REENTRY` | `REENTRY` | `null` |
| `ZONE_ENTER` | `ZONE_ENTER` | brand slug |
| `ZONE_EXIT` | `ZONE_EXIT` | brand slug |
| `DWELL_COMPLETED` | `ZONE_DWELL` | brand slug |
| `QUEUE_ENTER` | `BILLING_QUEUE_JOIN` | `BILLING` |
| `QUEUE_EXIT` | `BILLING_QUEUE_ABANDON` | `BILLING` |
| `PAYMENT_ENTER` | `ZONE_ENTER` | `BILLING` |
| `PAYMENT_EXIT` | `ZONE_EXIT` | `BILLING` |

---

## 4. Visitor IDs (Phase 2)

- ByteTrack `int` → `VIS_00001`, `VIS_00002`, … per video run.
- Original track id: `metadata.byte_track_id`.
- REENTRY: `VisitorRegistry.link_track()` reuses prior `VIS_*`.

---

## 5. REENTRY (Phase 3, CAM3 only)

**Session rules**

1. `ENTRY` / `REENTRY` opens session (`open_sessions`).
2. `EXIT` closes session and appends `recent_exits` cache (norm x/y, UTC time, `vis_id`).

**Match rules** (no embeddings)

- New internal `ENTRY` near entry line (`ENTRY_PLANE_Y_NORM` ± tolerance).
- Prior `EXIT` within `REENTRY_TIME_WINDOW_SECONDS` (default 180s).
- Centroid within `REENTRY_LOCATION_MAX_NORM_DIST` (normalized 0–1 plane).

**On match**

- Emit `event_type: "REENTRY"` instead of `ENTRY`.
- Reuse `visitor_id` from cache; `metadata.reentry_from_vis` set.
- `engine.entry_count` decremented once so overlay count stays consistent.

**Example**

```json
{
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_ENTRY_01",
  "visitor_id": "VIS_00002",
  "event_type": "REENTRY",
  "timestamp": "2026-04-10T20:12:05.200Z",
  "zone_id": null,
  "dwell_ms": 0,
  "is_staff": false,
  "confidence": 0.85,
  "metadata": {
    "byte_track_id": 14,
    "video_offset": "00:01:06.000",
    "internal_event_type": "ENTRY",
    "reentry_from_vis": "VIS_00002",
    "reentry_match": "time_location_cache",
    "debug": { "previous_side": "STORE", "current_side": "MALL" }
  }
}
```

(Config: `REENTRY_*` in `configs/competition_config.py`, overridable via env.)

---

## 6. Staff (Phase 4)

- All events emitted with `"is_staff": false`.
- Post-pass: `staff/staff_detection.py`
  - `enrich_events(input_path, output_path)` — stub copies rows.
  - `mark_visitor_staff(events, visitor_id)` — session-level hook for future rules.
- Staff logic must **not** run inside YOLO/detection loops.

---

## 7. Sample outputs

### CAM1 ZONE_DWELL

```json
{
  "event_id": "...",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_SHELF_01",
  "visitor_id": "VIS_00001",
  "event_type": "ZONE_DWELL",
  "timestamp": "2026-04-10T20:11:36.000Z",
  "zone_id": "NEUTROGENA",
  "dwell_ms": 7600,
  "is_staff": false,
  "confidence": 0.85,
  "metadata": {
    "byte_track_id": 6,
    "video_offset": "00:01:08.400",
    "internal_event_type": "DWELL_COMPLETED",
    "sku_zone": "Neutrogena"
  }
}
```

### CAM5 queue

```json
{
  "event_type": "BILLING_QUEUE_JOIN",
  "camera_id": "CAM_BILLING_01",
  "zone_id": "BILLING",
  "metadata": {
    "byte_track_id": 1,
    "video_offset": "00:00:12.400",
    "internal_event_type": "QUEUE_ENTER"
  }
}
```

---

## 8. Timestamps

- `timestamp` field is **ISO8601 UTC** with `Z` suffix at emit time.
- `metadata.video_offset` keeps legacy `HH:MM:SS.mmm` for debugging.
- `normalize_event_timestamps.py` still adds `event_datetime` (no `Z`) for older matching scripts when run on legacy offset-only files.
- CAM3 Brigade anchor: `CAM3_CAMERA_START_TIME` in `competition_config.py`.
- Footage2 clips: `FOOTAGE2_CAM3_CLIP_STARTS` per `entry1` / `entry2`.

---

## 9. Run commands

```powershell
Set-Location E:\NOTEBK\project
python events\cam1_events.py
python events\cam3_events.py
python events\cam5_events.py

# Optional staff enrich (stub)
python -c "from staff.staff_detection import enrich_events; from pathlib import Path; enrich_events(Path('outputs/cam1_events.jsonl'), Path('outputs/cam1_events_staff.jsonl'))"
```

---

## 10. Risks / follow-ups

| Item | Note |
|------|------|
| Downstream scripts | Prefer `timestamp` + competition types; `event_datetime` optional via normalize |
| CAM2 | Still legacy `EventLogger` until wired to `EventEmitter` |
| REENTRY false positives | Tune `REENTRY_*` env vars on Footage2 entry clips |
| Purpple ingest | Adapter can mirror `events/event_type_map.py` when migrating |
