# Migration Report — Phase 2 (CAM1/CAM2 Zone Engagement)

**Date:** 2026-05-31  
**Scope:** Port NOTEBK CAM1/CAM2 zone-engagement pipeline into `Purpple_Vision/pipeline` using shared detect/tracker/zones/dwell modules and `event_adapter` + `emit` for all output.  
**NOTEBK sources preserved:** `E:\NOTEBK\project\events\cam1_events.py`, `cam2_events.py` unchanged.

---

## 1. Files modified / created (Purpple_Vision)

| File | Action |
|------|--------|
| `pipeline/config.py` | **Updated** — `CAM1_ZONES`, `CAM2_ZONES`, shared tuning (`MIN_OVERLAP_PCT`, stability, dwell, frame skip) |
| `pipeline/detect.py` | **Replaced** (was stub) — YOLO11m person detection (`load_yolo_model`, `detect_persons`) |
| `pipeline/tracker.py` | **Replaced** (was stub) — ByteTrack (`create_byte_tracker`, `update_tracks`) |
| `pipeline/zones.py` | **Replaced** (was stub) — polygon overlap zone assignment (`assign_track_zones`, etc.) |
| `pipeline/dwell.py` | **Replaced** (was stub) — `ZoneEngagementEngine`, `process_zone_engagement_video` |
| `pipeline/sink.py` | **Created** — `EventSink` protocol (shared by dwell, entry_exit, queue) |
| `pipeline/cam1_processor.py` | **Created** — CAM1 CLI + `process_cam1_video` |
| `pipeline/cam2_processor.py` | **Created** — CAM2 CLI + `process_cam2_video` |
| `pipeline/event_adapter.py` | **Updated** — CAM1/CAM2 `ZONE_ID_MAP` entries; `D&K` → `D_AND_K` |
| `pipeline/entry_exit.py` | **Updated** — uses shared `detect`, `tracker`, `zones`, `sink` (CAM3) |
| `pipeline/queue.py` | **Updated** — `EventSink` import from `pipeline.sink` |
| `pipeline/emit.py` | Unchanged (Phase 1) — all CAM1/CAM2 events flow through `emit_notbk` |
| `pipeline/video_time.py` | Unchanged — used by dwell/entry_exit |

**Not modified (per plan):** `app/`, `database/`, `routers/`.

---

## 2. Event types supported (Phase 2)

| NOTEBK `event_type` | Emitted by | Purpple `EventType` | Notes |
|---------------------|------------|---------------------|--------|
| `ZONE_ENTER` | CAM1, CAM2 | `ZONE_ENTER` | After `MIN_ZONE_STABILITY_FRAMES` (15) on new zone |
| `ZONE_EXIT` | CAM1, CAM2 | `ZONE_EXIT` | On leave or track loss |
| `DWELL_COMPLETED` | CAM1, CAM2 | `ZONE_DWELL` | Only if dwell ≥ `MIN_DWELL_SECONDS` (2s); adapter enforces `dwell_ms` ≥ 30000 |

**Phase 1 types still supported:** `ENTRY`, `EXIT`, `QUEUE_ENTER`/`QUEUE_EXIT`, `PAYMENT_ENTER`/`PAYMENT_EXIT` (CAM3/CAM5).

---

## 3. Schema mapping (CAM1/CAM2)

### 3.1 NOTEBK JSONL row (internal / `.notbk.jsonl` mirror)

```json
{
  "visitor_id": 12,
  "camera": "CAM1",
  "event_type": "ZONE_ENTER",
  "zone": "FarmStay",
  "timestamp": "00:05:12.340"
}
```

`DWELL_COMPLETED` adds `"dwell_seconds": 45.2`.

### 3.2 Purpple `Event` (via `event_adapter` + `emit`)

| NOTEBK field | Purpple field | Rule |
|--------------|---------------|------|
| `visitor_id` | `visitor_id` | `VIS_{id}` |
| `camera` | `camera_id` | `CAM1` → `CAM_SHELF_01`, `CAM2` → `CAM_SHELF_02` |
| `event_type` | `event_type` | See table above |
| `timestamp` (offset) | `timestamp` (UTC) | `CAMERA_CLIP_START[camera]` + offset |
| `zone` | `zone_id` | `ZONE_ID_MAP` or normalized uppercase (`D&K` → `D_AND_K`) |
| `zone` | `metadata.sku_zone` | **Original name preserved** (e.g. `FarmStay`, `LAKME`, `swiss Beauty`) |
| `dwell_seconds` | `dwell_ms` | `max(seconds×1000, 30000)` for `ZONE_DWELL` |

### 3.3 Zone names preserved in events

**CAM1** (`metadata.sku_zone` / NOTEBK `zone`):  
`FarmStay`, `TheFaceShop`, `GoodVibes`, `DermaCo`, `Minimalist`, `Aquologica`, `Pilgrim`, `D&K`, plus `Minimalist_top` (geometry overlap with top shelf).

**CAM2:**  
`LAKME`, `MAYBELLINE`, `FACESCANADA`, `swiss Beauty`, `MARS`, `ALPS`, `LOREAL`, `EASTIND`.

Polygons copied verbatim from NOTEBK `cam1_events.py` / `cam2_events.py` into `pipeline/config.py`.

---

## 4. Architecture (Purpple_Vision)

```
cam1_processor.py / cam2_processor.py
        │
        ▼
process_zone_engagement_video()  [dwell.py]
        │
        ├── detect.py      YOLO11m, class 0, conf 0.35
        ├── tracker.py     ByteTrack
        ├── zones.py       polygon overlap ≥ 10%
        └── dwell.py       ZoneEngagementEngine → EventSink
                                    │
                                    ▼
                          emit.PipelineEmitter.emit_notbk()
                                    │
                                    ▼
                          event_adapter.notbk_event_to_event()
                                    │
                                    ▼
                          cam1_events.jsonl / cam2_events.jsonl (Purpple schema)
                          + optional .notbk.jsonl mirror
```

**No raw JSONL writes** in processors — only `PipelineEmitter`.

---

## 5. CV parameters (parity with NOTEBK)

| Parameter | Value |
|-----------|--------|
| Model | YOLO11m (`config.MODEL_PATH`, default `models/yolo11m.pt`) |
| Tracker | ByteTrack |
| Class | Person (0) |
| Confidence / IoU | 0.35 / 0.5 |
| Zone assignment | Best polygon overlap %; min 10% |
| Stability | 15 consecutive processed frames |
| Min dwell (emit) | 2.0 s |
| Lost track | 30 frames |
| Frame stride | Process every 10 frames |

---

## 6. Validation

**py_compile** (from `E:\Purpple_Vision`):

```powershell
.\.venv\Scripts\python.exe -m py_compile pipeline\detect.py pipeline\tracker.py pipeline\zones.py pipeline\dwell.py pipeline\sink.py pipeline\cam1_processor.py pipeline\cam2_processor.py pipeline\event_adapter.py pipeline\emit.py pipeline\entry_exit.py pipeline\queue.py pipeline\config.py
```

**Import check:**

```powershell
.\.venv\Scripts\python.exe -c "from pipeline import detect, tracker, zones, dwell, cam1_processor, cam2_processor; print('OK')"
```

**Run pipelines (requires CCTV under `data/cctv/Brigade_Bangalore/`):**

```powershell
.\.venv\Scripts\python.exe -m pipeline.cam1_processor
.\.venv\Scripts\python.exe -m pipeline.cam2_processor
```

Outputs default to `data/generated/cam1_events.jsonl` and `cam2_events.jsonl`.

---

## 7. Remaining work

| Item | Priority | Notes |
|------|----------|--------|
| Refactor `queue.py` to use shared `detect` / `tracker` / `zones` | Medium | CAM5 still duplicates YOLO/ByteTrack locally |
| `pipeline/staff.py` | Low | Staff classification not in NOTEBK cam1/cam2 |
| `pipeline/pos_loader.py` | Medium | Port `pos_aggregation.py` |
| Add `supervision` to `requirements.txt` | High | Required for tracker/detect |
| End-to-end run on Brigade footage + compare counts vs NOTEBK JSONL | High | QA after data path setup |
| `store_layout.json` externalization | Low | Zones currently in `config.py` |
| Wire `POST /events/ingest` batch upload from pipeline output | Medium | API already exists; needs script |
| CAM1/CAM2 OpenCV overlay GUI | Low | NOTEBK had full annotation; Purpple runs headless by default (`show_window=False`) |
| Normalize NOTEBK `event_datetime` on pipeline output | Optional | Adapter accepts ISO if present |

---

## 8. Comparison to NOTEBK

| Aspect | NOTEBK `cam1_events.py` | Purpple Phase 2 |
|--------|-------------------------|-----------------|
| Event sink | `EventLogger` → raw `cam1_events.jsonl` | `PipelineEmitter` → adapted JSONL |
| Module layout | Monolithic script | `detect` / `tracker` / `zones` / `dwell` + thin processors |
| CAM3 | Separate script | `entry_exit.py` now shares detect/tracker |
| Zone names | In-script `CAM1_BRANDS` | `config.CAM1_ZONES` / `CAM2_ZONES` |

Logic for stability, dwell, overlap, and event sequencing matches NOTEBK `ZoneEventEngine` behavior.
