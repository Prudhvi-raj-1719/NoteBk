# Migration Report — Phase 1 (Pipeline Port)

**Date:** 2026-05-31  
**Scope:** Port NOTEBK CV event logic into `Purpple_Vision/pipeline` (no `app/`, DB, or API route changes).  
**NOTEBK sources preserved:** `E:\NOTEBK\project\` unchanged.

---

## 1. Files modified / created (Purpple_Vision)

| File | Action |
|------|--------|
| `pipeline/config.py` | **Created** — paths, clip anchors, CAM3/CAM5 geometry, store/camera ID maps |
| `pipeline/video_time.py` | **Created** — frame index → `HH:MM:SS.mmm` offset |
| `pipeline/event_adapter.py` | **Created** — NOTEBK dict → `app.models.Event` |
| `pipeline/emit.py` | **Replaced** (was stub) — `PipelineEmitter` writes adapted JSONL + optional `.notbk.jsonl` mirror |
| `pipeline/entry_exit.py` | **Replaced** (was stub) — CAM3 logic from `project/events/cam3_events.py` + ENTRY/EXIT emission |
| `pipeline/queue.py` | **Replaced** (was stub) — CAM5 logic from `project/events/cam5_events.py` |
| `pipeline/detect.py` | Unchanged (stub) |
| `pipeline/tracker.py` | Unchanged (stub) |
| `pipeline/zones.py` | Unchanged (stub) |
| `pipeline/dwell.py` | Unchanged (stub) |
| `pipeline/staff.py` | Unchanged (stub) |
| `pipeline/pos_loader.py` | Unchanged (stub) |

---

## 2. Imports updated

| Module | Imports |
|--------|---------|
| `event_adapter.py` | `app.models` (`Event`, `EventMetadata`, `EventType`), `pipeline.config` |
| `emit.py` | `app.models.Event`, `pipeline.event_adapter` |
| `entry_exit.py` | `pipeline.config`, `pipeline.emit`, `pipeline.video_time`, `cv2`, `numpy`, `supervision`, `ultralytics` |
| `queue.py` | `pipeline.config`, `pipeline.emit`, `pipeline.entry_exit.EventSink`, `pipeline.video_time`, OpenCV stack |

**Import validation (run from `E:\Purpple_Vision`):**

```powershell
.\.venv\Scripts\python.exe -m py_compile pipeline\config.py pipeline\video_time.py pipeline\event_adapter.py pipeline\emit.py pipeline\entry_exit.py pipeline\queue.py
.\.venv\Scripts\python.exe -c "from pipeline import event_adapter, emit, entry_exit, queue"
```

**Note:** `supervision` was required for `entry_exit` / `queue` but is not yet listed in `requirements.txt` (install manually: `pip install supervision`).

---

## 3. Schema mappings (NOTEBK → Purpple `Event`)

### 3.1 Identifiers

| NOTEBK | Purpple field | Rule |
|--------|---------------|------|
| `visitor_id` (int) | `visitor_id` | `VIS_{track_id}` |
| `camera` (`CAM3`, `CAM5`, …) | `camera_id` | See `CAMERA_PURPPLE_IDS` in `config.py` |
| — | `store_id` | `STORE_BLR_002` (env `STORE_ID`) |
| — | `event_id` | New UUID v4 per event |

**Camera map (`config.CAMERA_PURPPLE_IDS`):**

| NOTEBK | Purpple `camera_id` |
|--------|---------------------|
| CAM1 | CAM_SHELF_01 |
| CAM2 | CAM_SHELF_02 |
| CAM3 | CAM_ENTRY_01 |
| CAM5 | CAM_BILLING_01 |

### 3.2 Timestamps

| NOTEBK | Purpple `timestamp` |
|--------|---------------------|
| `timestamp` = `HH:MM:SS.mmm` (video offset) | UTC = `CAMERA_CLIP_START[camera]` + offset |
| `event_datetime` (ISO, if present) | Used directly (normalized to UTC) |

Clip anchors from NOTEBK `camera_timing_config` (+ CAM3 estimate) live in `pipeline/config.py` → `CAMERA_CLIP_START`.

### 3.3 Event types

| NOTEBK `event_type` | Purpple `EventType` | `zone_id` | Notes |
|---------------------|---------------------|-----------|--------|
| `ENTRY` | `ENTRY` | `null` | Added on CAM3 crossing (not in original NOTEBK cam3 counter-only script) |
| `EXIT` | `EXIT` | `null` | Same |
| `QUEUE_ENTER` | `BILLING_QUEUE_JOIN` | `BILLING` | `metadata.queue_depth` = 1 |
| `QUEUE_EXIT` | `BILLING_QUEUE_ABANDON` | `BILLING` | |
| `PAYMENT_ENTER` | `ZONE_ENTER` | `BILLING` | Zone mapped from `PaymentArea` |
| `PAYMENT_EXIT` | `ZONE_EXIT` | `BILLING` | |
| `DWELL_COMPLETED` | `ZONE_DWELL` | mapped zone | `dwell_ms` = max(seconds×1000, 30000) |
| `ZONE_ENTER` / `ZONE_EXIT` | same | mapped zone | Ready for CAM1/CAM2 port (Phase 2) |

### 3.4 Other fields

| Field | Default / rule |
|-------|----------------|
| `confidence` | `0.85` unless set on NOTEBK row |
| `is_staff` | `false` |
| `dwell_ms` | `0` except `ZONE_DWELL` |
| `metadata.sku_zone` | Original NOTEBK `zone` string when present |

---

## 4. Behavioural notes vs NOTEBK originals

| Topic | NOTEBK `cam3_events.py` | Purpple `entry_exit.py` |
|-------|-------------------------|-------------------------|
| Counting | Entry/exit counts only | Same counts **plus** NOTEBK-shaped `ENTRY`/`EXIT` rows via sink |
| GUI | OpenCV window default | `show_window=False` in CLI; optional `show_window=True` |
| Output | None | `PipelineEmitter` → adapted JSONL |

| Topic | NOTEBK `cam5_events.py` | Purpple `queue.py` |
|-------|-------------------------|---------------------|
| Event logger | Append raw JSONL + prints | `PipelineEmitter.emit_notbk` → adapter → Purpple JSONL |
| Frame skip | Every 10 frames | Preserved |
| Zone logic | PaymentArea > BillingQueue | Preserved |

---

## 5. How to run (Purpple_Vision)

```powershell
cd E:\Purpple_Vision

# CAM3 → data/generated/cam3_events.jsonl (+ cam3_events.notbk.jsonl mirror)
.\.venv\Scripts\python.exe -m pipeline.entry_exit

# CAM5 → data/generated/cam5_events.jsonl
.\.venv\Scripts\python.exe -m pipeline.queue
```

**Data paths:** Copy CCTV into `data/cctv/Brigade_Bangalore/` or set `CCTV_FOOTAGE_DIR`. Default fallback in `config.py` expects:

- `CAM 3.mp4`, `CAM 5.mp4` under that directory  
- Model: `models/yolo11m.pt` or `YOLO_MODEL_PATH`

Link NOTEBK assets (example):

```powershell
mkdir E:\Purpple_Vision\data\cctv\Brigade_Bangalore -Force
# copy or junction from E:\NOTEBK\project\data\CCTV Footage
```

---

## 6. Remaining work (Phase 2+)

| Item | Priority |
|------|----------|
| Port `cam1_events.py` / `cam2_events.py` into `pipeline/zones.py` + `pipeline/dwell.py` | P0 |
| Implement `pipeline/detect.py` / `pipeline/tracker.py` shared modules (dedupe from entry_exit/queue) | P1 |
| Add `supervision` to `requirements.txt` | P1 |
| `pipeline/timestamp_normalize.py` as standalone or inside adapter when only offsets exist | P1 |
| `store_layout.json` from NOTEBK polygons | P1 |
| Wire pipeline JSONL → `POST /events/ingest` (script, no API code change) | P1 |
| Align YOLO11m (NOTEBK) vs YOLOv8n (Purpple `CHOICES.md`) | P2 |
| Port `pos_aggregation.py` → `pos_loader.py` | P2 |
| Port offline `purchase_matching.py`, `retail_analytics.py` | P2 |
| Move validation scripts to `scripts/validation/` | P3 |
| Unit tests: `tests/test_event_adapter.py`, `tests/test_emit.py` | P1 |

---

## 7. Explicit non-changes (per instructions)

- No edits under `app/` except **read-only** use of `app.models` from `event_adapter.py` / `emit.py`
- No changes to `app/db.py`, ingestion routers, or API routes
- No deletion or modification of `E:\NOTEBK\project\**`

---

## 8. Static validation log

| Step | Result |
|------|--------|
| `py_compile` on all new pipeline modules | PASS |
| `from pipeline import event_adapter, emit, entry_exit, queue` | PASS (Purpple `.venv` + `supervision` installed) |
| Adapter smoke: `QUEUE_ENTER` → `BILLING_QUEUE_JOIN`, `CAM_BILLING_01` | PASS |

---

*End of Phase 1 report.*
