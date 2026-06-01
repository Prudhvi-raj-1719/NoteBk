# NOTEBK → Purpple_Vision Migration Plan

**Status:** Planning only — no files have been moved or modified.  
**Compared trees:** `E:\NOTEBK\project` and `E:\Purpple_Vision`  
**Date:** 2026-05-31

---

## 1. Executive summary

| Area | NOTEBK `project/` | Purpple_Vision |
|------|-------------------|----------------|
| **CV pipeline** | Fully implemented standalone scripts (YOLO**11m**, ByteTrack, OpenCV GUIs) | `pipeline/` modules exist as **one-line stubs** |
| **Event output** | Simple JSONL (`visitor_id`, `camera`, video `timestamp`, zone names) | Strict Pydantic `Event` (UUID, `STORE_*`, `CAM_*`, `VIS_*`, UTC, `dwell_ms`, metadata) |
| **POS** | Offline CSV explore + invoice aggregation | `POST /pos/ingest` + `PosTransaction` schema; `pos_loader.py` stub |
| **Matching** | Video-offset normalization + invoice↔CCTV window matching | `pos_correlation.py` (session billing vs POS, different rule) |
| **Analytics** | Offline `retail_analytics.py` (JSON + MD + matplotlib) | Live API: `metrics`, `funnel`, `heatmap`, `anomalies`, `health` |
| **Tests / API / Docker** | None | Complete (FastAPI, SQLite, pytest, compose) |

**Migration strategy:** Port NOTEBK **computer-vision and offline batch logic** into Purpple_Vision `pipeline/` (and thin adapters into `pipeline/emit.py`), then add a **schema adapter** so JSONL can be ingested via `POST /events/ingest`. Keep Purpple_Vision **API, sessions, and correlation** as source of truth for challenge scoring. Archive NOTEBK-only dev/QA scripts under `scripts/` or `tools/`.

---

## 2. Purpple_Vision structure (reference)

```
Purpple_Vision/
├── app/                    # ✅ Implemented — FastAPI intelligence layer
│   ├── main.py, models.py, db.py, ingestion.py, pos_ingestion.py
│   ├── sessions.py, pos_correlation.py
│   ├── metrics.py, funnel.py, heatmap.py, anomalies.py, health.py
│   └── logging_config.py
├── pipeline/               # ⚠️ Stubs only (docstrings, no logic)
│   ├── detect.py, tracker.py, zones.py, entry_exit.py
│   ├── dwell.py, queue.py, staff.py, emit.py, pos_loader.py
├── dashboard/              # ⚠️ Stubs (streamlit_app.py, terminal_dashboard.py)
├── scripts/                # seed_from_sample.py, phase0_import_check.py
├── tests/                  # ✅ Broad API/session/metrics coverage
├── examples/               # API response samples
└── data/                   # Intended: sample_events.jsonl, store_layout.json, DB
```

---

## 3. NOTEBK `project/` structure (reference)

```
project/
├── configs/camera_timing_config.py   # Paths, CCTV files, manual start times
├── bootstrap.py                      # sys.path helper (legacy standalone runs)
├── validation/                       # Interactive zone/line validation (OpenCV)
├── events/                           # Production event generators per camera
├── pos/                              # POS CSV explore + aggregation
├── matching/                         # Timestamp normalize, purchase match, QA tools
├── analytics/retail_analytics.py     # Offline business intelligence
├── outputs/                          # JSON, JSONL, reports, charts
└── data/                             # CCTV Footage + Brigade POS CSV
```

---

## 4. File-by-file mapping

Legend for **Action**:

| Action | Meaning |
|--------|---------|
| **Keep standalone** | Retain as runnable script (e.g. under `scripts/`) with minimal wrapper |
| **Merge** | Fold logic into an existing Purpple module; stubs become implementations |
| **Replace** | Purpple stub is the target; NOTEBK code becomes the implementation |
| **Archive** | Move to `archive/notebk/` or delete after port; not part of runtime |

### 4.1 Config & bootstrap

| NOTEBK file | Action | Purpple_Vision destination | Notes |
|-------------|--------|---------------------------|-------|
| `configs/camera_timing_config.py` | **Merge** | `pipeline/config.py` + `data/stores/brigade_bangalore/timing.json` | Split paths (`PROJECT_ROOT`) from store-specific anchors (`CAMERA_START_TIMES`). Purpple has no timing config today. |
| `configs/__init__.py` | **Archive** | — | Replace with normal package imports from repo root. |
| `bootstrap.py` | **Archive** | — | Not needed when running as `python -m pipeline...` from repo root. |

### 4.2 Validation (interactive dev tools)

| NOTEBK file | Action | Purpple_Vision destination | Notes |
|-------------|--------|---------------------------|-------|
| `validation/cam1_validation.py` | **Keep standalone** | `scripts/validation/cam1_validation.py` | No Purpple equivalent. Used to tune CAM1 brand polygons. |
| `validation/cam2_validation.py` | **Keep standalone** | `scripts/validation/cam2_validation.py` | Same. |
| `validation/cam3_validation.py` | **Keep standalone** | `scripts/validation/cam3_validation.py` | Overlaps conceptually with `entry_exit.py` but is GUI-only validation. |
| `validation/cam5_validation.py` | **Keep standalone** | `scripts/validation/cam5_validation.py` | Overlaps `queue.py` zones; keep for geometry QA. |
| `validation/__init__.py` | **Archive** | — | Empty package marker. |

### 4.3 Events (CV pipeline — core port)

| NOTEBK file | Action | Purpple_Vision destination | Notes |
|-------------|--------|---------------------------|-------|
| `events/cam1_events.py` | **Merge** | `pipeline/zones.py` + `pipeline/dwell.py` + `pipeline/detect.py` + `pipeline/tracker.py` + `pipeline/emit.py` | Shelf zones (CAM1 brands). Extract: polygon overlap, zone stability, dwell, YOLO loop. |
| `events/cam2_events.py` | **Merge** | Same as CAM1 | CAM2 brand polygons + same engine. Consider `pipeline/zones.py` parameterized by `store_layout.json`. |
| `events/cam3_events.py` | **Replace** | `pipeline/entry_exit.py` | Line-crossing ENTRY/EXIT counting (from `cam3_entry_exit.py` lineage). |
| `events/cam5_events.py` | **Replace** | `pipeline/queue.py` (+ payment logic) | QUEUE_ENTER/EXIT, PAYMENT_ENTER/EXIT → map to `BILLING_QUEUE_JOIN` / `BILLING_QUEUE_ABANDON` in emitter. |
| `events/__init__.py` | **Archive** | — | |

**Suggested thin CLI wrappers (optional, post-merge):**

| Wrapper | Purpose |
|---------|---------|
| `scripts/run_pipeline_cam1.py` | Calls shared pipeline with CAM1 layout |
| `scripts/run_pipeline_cam2.py` | CAM2 layout |
| `scripts/run_pipeline_cam5.py` | CAM5 queue/payment |

### 4.4 POS

| NOTEBK file | Action | Purpple_Vision destination | Notes |
|-------------|--------|---------------------------|-------|
| `pos/pos_aggregation.py` | **Merge** | `pipeline/pos_loader.py` | Brigade CSV → invoice-level JSON/CSV. Output must map to `PosTransaction` for ingest or seed script. |
| `pos/explore_pos_data.py` | **Keep standalone** | `scripts/explore_pos_data.py` | One-off schema discovery; not in challenge runtime path. |
| `pos/__init__.py` | **Archive** | — | |

### 4.5 Matching & time alignment

| NOTEBK file | Action | Purpple_Vision destination | Notes |
|-------------|--------|---------------------------|-------|
| `matching/normalize_event_timestamps.py` | **Merge** | `pipeline/timestamp_normalize.py` (new) + used by `pipeline/emit.py` | Purpple expects UTC `timestamp` on ingest. NOTEBK uses video offsets + manual anchors. |
| `matching/purchase_matching.py` | **Keep standalone** → later **Merge** | `pipeline/purchase_matching.py` (new) or `scripts/purchase_matching.py` | **Not** the same as `app/pos_correlation.py` (see §5). Keep for offline invoice↔event journeys until API exposes similar insights. |
| `matching/video_pos_overlap_analysis.py` | **Keep standalone** | `scripts/video_pos_overlap_analysis.py` | Pre-migration QA (overlap window check). |
| `matching/inspect_event_timestamps.py` | **Keep standalone** | `scripts/inspect_event_timestamps.py` | QA for timestamp formats before ingest. |
| `matching/__init__.py` | **Archive** | — | |

### 4.6 Analytics

| NOTEBK file | Action | Purpple_Vision destination | Notes |
|-------------|--------|---------------------------|-------|
| `analytics/retail_analytics.py` | **Merge** (split) | `app/metrics.py`, `app/funnel.py`, `app/heatmap.py` + optional `app/services/offline_analytics.py` | NOTEBK: offline JSON/MD/charts from files. Purpple: DB-backed API. Port **metrics concepts** (brand visits, dwell, revenue-by-brand) into API or a batch report generator that calls same session builder. |
| `analytics/__init__.py` | **Archive** | — | |

### 4.7 Data & outputs (not Python — placement only)

| NOTEBK asset | Action | Purpple_Vision destination |
|--------------|--------|---------------------------|
| `data/CCTV Footage/*.mp4` | **Keep** | `data/cctv/Brigade_Bangalore/` (or path in `.env`) |
| `data/Brigade_Bangalore_10_April_26.csv` | **Keep** | `data/pos/brigade_bangalore_10_april_26.csv` |
| `outputs/*.jsonl` | **Merge** | `data/generated/events/` → transform → ingest |
| `outputs/aggregated_transactions.json` | **Merge** | Feed `scripts/seed_from_sample.py` or `POST /pos/ingest` after mapping |
| `outputs/reports/*.txt` | **Archive** | `docs/reports/notebk/` or CI artifacts |
| `outputs/charts/*.png` | **Archive** | Regenerate from API or offline analytics post-migration |
| `yolo11m.pt` | **Keep** | `data/models/yolo11m.pt` (document choice vs Purpple YOLOv8n default) |

### 4.8 Purpple_Vision modules with **no** NOTEBK counterpart (do not replace)

| Purpple_Vision file | Action | Notes |
|---------------------|--------|-------|
| `app/main.py`, `app/ingestion.py`, `app/db.py` | **Keep** | Ingestion path for migrated events |
| `app/sessions.py` | **Keep** | Session model is authoritative for API |
| `app/pos_correlation.py` | **Keep** | Challenge conversion rule (billing window) |
| `app/metrics.py`, `funnel.py`, `heatmap.py`, `anomalies.py`, `health.py` | **Keep** | Extend with Brigade store_id, not replace |
| `pipeline/staff.py` | **Keep stub → implement** | NOTEBK has no staff classifier; CAM4 notebook logic not in `project/` |
| `dashboard/*` | **Keep** | Optional UI; not in NOTEBK |
| `tests/*` | **Keep** | Add tests for new pipeline emit shape |
| `scripts/seed_from_sample.py` | **Keep** | Use after `emit.py` produces compliant JSONL |

---

## 5. Duplicate functionality

| Capability | NOTEBK | Purpple_Vision | Resolution |
|------------|--------|----------------|------------|
| Person detection | `events/cam*_events.py` (YOLO11m) | `pipeline/detect.py` stub (YOLOv8n planned) | **Replace stub** with NOTEBK loop; decide **v8 vs v11** in `CHOICES.md` |
| Multi-object tracking | supervision `ByteTrack` in event scripts | `pipeline/tracker.py` stub | **Merge** into tracker module |
| Zone assignment (polygons) | Per-camera hardcoded `CAM*_BRANDS` / `CAM5_ZONES` | `pipeline/zones.py` + `store_layout.json` | **Merge**; externalize polygons to JSON |
| Dwell events | `DWELL_COMPLETED` + `dwell_seconds` | `ZONE_DWELL` + `dwell_ms` (≥30s) | **Adapter** in `emit.py` (rename + ms + threshold rules) |
| Entry / exit | `events/cam3_events.py` | `pipeline/entry_exit.py` | **Replace stub** with NOTEBK logic |
| Queue / billing | `events/cam5_events.py` (`QUEUE_*`, `PAYMENT_*`) | `pipeline/queue.py` (`BILLING_QUEUE_*`) | **Replace stub** + event-type mapping |
| Event serialization | Direct `json.dumps` to JSONL | `pipeline/emit.py` | **Replace stub**; single writer for schema compliance |
| POS load | `pos_aggregation.py` | `pos_loader.py` stub + `pos_ingestion.py` | **Merge** aggregation into loader; ingest via API |
| POS ↔ CCTV time overlap | `video_pos_overlap_analysis.py` | — | **Keep standalone** script |
| Purchase / conversion matching | `purchase_matching.py` (invoice ±5 min, multi-cam journey) | `pos_correlation.py` (session `reached_billing`, billing_at vs txn) | **Both** — different products; do not delete either |
| Timestamp normalization | `normalize_event_timestamps.py` | — (ingest expects UTC) | **New** `pipeline/timestamp_normalize.py` |
| Store analytics | `retail_analytics.py` (offline) | `metrics` + `funnel` + `heatmap` + `anomalies` (online) | **Merge concepts**; avoid duplicating formulas differently |
| Validation GUIs | `validation/cam*.py` | — | **Keep standalone** under `scripts/validation/` |
| Config / video paths | `camera_timing_config.py` | `.env` + planned `store_layout.json` | **Merge** into store config |

---

## 6. Missing functionality

### 6.1 Missing in Purpple_Vision (NOTEBK has it)

| Item | NOTEBK source | Priority |
|------|---------------|----------|
| Working YOLO + ByteTrack video processing | All `events/*.py` | **P0** |
| Brand/shelf zone polygons (CAM1, CAM2) | `cam1_events.py`, `cam2_events.py`, validation | **P0** |
| CAM3 entry/exit line crossing | `cam3_events.py` | **P0** |
| CAM5 queue + payment zones | `cam5_events.py` | **P0** |
| Manual camera start-time anchors | `camera_timing_config.py` | **P0** for Brigade footage |
| Video offset → UTC normalization | `normalize_event_timestamps.py` | **P0** before ingest |
| Brigade POS CSV → invoices | `pos_aggregation.py` | **P1** |
| JSONL → Pydantic `Event` adapter | New (`pipeline/emit.py`) | **P0** |
| Offline invoice↔CCTV journey matching | `purchase_matching.py` | **P2** (analytics / ops) |
| Offline retail report + charts | `retail_analytics.py` | **P2** |
| POS/CCTV overlap QA | `video_pos_overlap_analysis.py` | **P2** |
| Interactive validation tools | `validation/*.py` | **P3** (dev) |
| POS schema explorer | `explore_pos_data.py` | **P3** |
| YOLO11m weights path | `yolo11m.pt` | **P1** (model choice) |

### 6.2 Missing in NOTEBK (Purpple_Vision has it)

| Item | Purpple_Vision source | Notes |
|------|----------------------|-------|
| FastAPI app + OpenAPI | `app/main.py` | Keep |
| Strict event schema + validation | `app/models.py` | Migrate toward this |
| SQLite persistence | `app/db.py` | Keep |
| Idempotent batch ingest | `app/ingestion.py`, `pos_ingestion.py` | Target for pipeline output |
| Visitor sessions (ENTRY/EXIT/REENTRY) | `app/sessions.py` | Keep |
| Challenge conversion correlation | `app/pos_correlation.py` | Keep; not replaced by `purchase_matching.py` |
| Metrics / funnel / heatmap / anomalies APIs | `app/*.py` | Keep |
| Health + stale feed warnings | `app/health.py` | Keep |
| Test suite (pytest) | `tests/` | Extend for pipeline |
| Docker / compose | `docker-compose.yml` | Keep |
| `store_layout.json` contract | Documented in DESIGN | NOTEBK uses inline Python dicts — must author JSON |
| Staff detection pipeline | `pipeline/staff.py` | Not in NOTEBK `project/` |
| Seed script for demo data | `scripts/seed_from_sample.py` | Keep |
| Streamlit dashboard | `dashboard/streamlit_app.py` | Optional |

---

## 7. Schema & behavior gaps (must address during port)

| Topic | NOTEBK | Purpple_Vision `Event` |
|-------|--------|------------------------|
| Event ID | None | UUID v4 required |
| Store / camera IDs | `"CAM1"` | `STORE_BLR_*`, `CAM_*` patterns |
| Visitor ID | Integer track id | `VIS_*` string pattern |
| Timestamp | `00:01:25.500` video offset → `event_datetime` | UTC ISO on `timestamp` |
| Dwell event | `DWELL_COMPLETED`, `dwell_seconds` | `ZONE_DWELL`, `dwell_ms` ≥ 30000 |
| Queue | `QUEUE_ENTER` / `QUEUE_EXIT` | `BILLING_QUEUE_JOIN` / `BILLING_QUEUE_ABANDON` |
| Payment | `PAYMENT_ENTER` / `PAYMENT_EXIT` | Map to billing zone + session flags (no direct PAYMENT_* type) |
| Confidence | Often omitted | Required field on `Event` |
| Staff flag | Not set | `is_staff` required |

**Deliverable:** `pipeline/emit.py` implements `notbk_row_to_event()` (or similar) and is the only JSONL writer.

---

## 8. Recommended target layout (post-migration)

```
Purpple_Vision/
├── app/                          # unchanged role
├── pipeline/
│   ├── config.py                 # ← camera_timing_config (paths)
│   ├── detect.py                 # ← YOLO from NOTEBK
│   ├── tracker.py                # ← ByteTrack from NOTEBK
│   ├── zones.py                  # ← CAM1/CAM2 zone engine
│   ├── dwell.py                  # ← dwell stability + ZONE_DWELL
│   ├── entry_exit.py             # ← cam3_events.py
│   ├── queue.py                  # ← cam5_events.py (queue/payment)
│   ├── staff.py                  # (still TBD; notebook/CAM4 if needed)
│   ├── timestamp_normalize.py    # ← normalize_event_timestamps.py
│   ├── emit.py                   # NOTEBK JSONL → challenge Event JSONL
│   ├── pos_loader.py             # ← pos_aggregation.py
│   └── purchase_matching.py      # ← optional offline matcher
├── scripts/
│   ├── validation/               # ← NOTEBK validation/*.py
│   ├── explore_pos_data.py
│   ├── video_pos_overlap_analysis.py
│   └── inspect_event_timestamps.py
├── data/
│   ├── cctv/
│   ├── pos/
│   ├── stores/brigade_bangalore/timing.json
│   └── store_layout.json         # migrate CAM1/CAM2/CAM5 polygons
└── tests/
    └── test_pipeline_emit.py     # new
```

---

## 9. Step-by-step migration checklist

### Phase 0 — Preparation (no code moves)

- [ ] **0.1** Create a migration branch in Purpple_Vision (`feature/notebk-pipeline-port`).
- [ ] **0.2** Copy `data/CCTV Footage` and `Brigade_Bangalore_10_April_26.csv` into Purpple `data/` (or document `.env` paths).
- [ ] **0.3** Decide detector: keep NOTEBK **YOLO11m** vs Purpple **YOLOv8n** (`CHOICES.md` update).
- [ ] **0.4** Author `data/store_layout.json` from NOTEBK `CAM1_BRANDS`, `CAM2` zones, `CAM5_ZONES`, CAM3 doorway geometry.
- [ ] **0.5** Author `data/stores/brigade_bangalore/timing.json` from `CAMERA_START_TIMES`.

### Phase 1 — Config & emit contract (P0)

- [ ] **1.1** Add `pipeline/config.py` from `configs/camera_timing_config.py`.
- [ ] **1.2** Implement `pipeline/emit.py`: map NOTEBK dict rows → `app.models.Event`.
- [ ] **1.3** Add `tests/test_pipeline_emit.py` with golden samples from `project/outputs/cam1_events_normalized.jsonl`.
- [ ] **1.4** Implement `pipeline/timestamp_normalize.py` from `matching/normalize_event_timestamps.py`.

### Phase 2 — Core CV port (P0)

- [ ] **2.1** **Replace** `pipeline/detect.py` + `pipeline/tracker.py` using NOTEBK YOLO/ByteTrack loop (shared).
- [ ] **2.2** **Replace** `pipeline/zones.py` + `pipeline/dwell.py` from `cam1_events.py` / `cam2_events.py` (parameterize by camera/layout).
- [ ] **2.3** **Replace** `pipeline/entry_exit.py` from `events/cam3_events.py`.
- [ ] **2.4** **Replace** `pipeline/queue.py` from `events/cam5_events.py` (map event types in emit).
- [ ] **2.5** Run end-to-end on one short clip; write `data/generated/events.jsonl`.
- [ ] **2.6** Ingest via `POST /events/ingest` or extend `scripts/seed_from_sample.py`; fix validation errors.

### Phase 3 — POS integration (P1)

- [ ] **3.1** **Merge** `pos/pos_aggregation.py` into `pipeline/pos_loader.py`.
- [ ] **3.2** Map aggregated invoices → `PosTransaction`; ingest via `POST /pos/ingest`.
- [ ] **3.3** Run `scripts/video_pos_overlap_analysis.py` (ported) to confirm time alignment.
- [ ] **3.4** Verify `app/pos_correlation.py` conversion rate with real sessions (separate from purchase_matching).

### Phase 4 — API analytics alignment (P2)

- [ ] **4.1** Compare NOTEBK `retail_analytics.py` metrics to `app/metrics.py` / `heatmap.py` — align formulas or document differences.
- [ ] **4.2** Optionally add `app/services/offline_analytics.py` for batch reports OR drop offline duplicate in favor of API.
- [ ] **4.3** Port `purchase_matching.py` to `pipeline/purchase_matching.py` if invoice-level journeys remain a requirement.

### Phase 5 — Dev tooling & cleanup (P3)

- [ ] **5.1** Move `validation/*.py` → `scripts/validation/`.
- [ ] **5.2** Move `explore_pos_data.py`, `inspect_event_timestamps.py` → `scripts/`.
- [ ] **5.3** **Archive** NOTEBK `project/bootstrap.py`, empty `__init__.py` packages, and duplicate root copies.
- [ ] **5.4** Update Purpple `README.md` / `PROJECT_STATE.md` (pipeline no longer stub).
- [ ] **5.5** Add CI job: run pipeline on sample clip + pytest.

### Phase 6 — Verification

- [ ] **6.1** `pytest` full suite green.
- [ ] **6.2** Manual: `GET /stores/{id}/metrics`, `/funnel`, `/heatmap` for Brigade date `2026-04-10`.
- [ ] **6.3** Compare NOTEBK offline charts vs API-derived exports (sanity).
- [ ] **6.4** Tag release / handoff doc for Purpple_Vision-only workflow.

---

## 10. Risk register

| Risk | Mitigation |
|------|------------|
| Event schema mismatch breaks ingest | Implement `emit.py` first with unit tests before full video runs |
| YOLO11m vs YOLOv8n behavior drift | Re-run validation scripts; update `CHOICES.md` |
| Time axis misalignment (CCTV vs POS) | Keep `timing.json` + overlap script; gate ingest on overlap report |
| Duplicating analytics logic | Single source: `build_sessions()` + shared helpers; offline script calls same code |
| Two “matching” systems confused | Document: `pos_correlation` = API conversion; `purchase_matching` = offline invoice journeys |
| `store_layout.json` maintenance | Generate from NOTEBK polygons once; version in git |

---

## 11. Quick reference mapping table (all NOTEBK Python files)

| NOTEBK `project/` file | Action | Purpple_Vision destination |
|------------------------|--------|----------------------------|
| `configs/camera_timing_config.py` | Merge | `pipeline/config.py` + `data/stores/.../timing.json` |
| `configs/__init__.py` | Archive | — |
| `bootstrap.py` | Archive | — |
| `validation/cam1_validation.py` | Keep standalone | `scripts/validation/cam1_validation.py` |
| `validation/cam2_validation.py` | Keep standalone | `scripts/validation/cam2_validation.py` |
| `validation/cam3_validation.py` | Keep standalone | `scripts/validation/cam3_validation.py` |
| `validation/cam5_validation.py` | Keep standalone | `scripts/validation/cam5_validation.py` |
| `events/cam1_events.py` | Merge | `pipeline/zones.py`, `dwell.py`, `detect.py`, `tracker.py`, `emit.py` |
| `events/cam2_events.py` | Merge | Same as CAM1 (layout-driven) |
| `events/cam3_events.py` | Replace | `pipeline/entry_exit.py` |
| `events/cam5_events.py` | Replace | `pipeline/queue.py` |
| `pos/pos_aggregation.py` | Merge | `pipeline/pos_loader.py` |
| `pos/explore_pos_data.py` | Keep standalone | `scripts/explore_pos_data.py` |
| `matching/normalize_event_timestamps.py` | Merge | `pipeline/timestamp_normalize.py` |
| `matching/purchase_matching.py` | Keep standalone → Merge | `pipeline/purchase_matching.py` |
| `matching/video_pos_overlap_analysis.py` | Keep standalone | `scripts/video_pos_overlap_analysis.py` |
| `matching/inspect_event_timestamps.py` | Keep standalone | `scripts/inspect_event_timestamps.py` |
| `analytics/retail_analytics.py` | Merge (split) | `app/metrics.py`, `funnel.py`, `heatmap.py`, optional `app/services/offline_analytics.py` |

---

*End of migration plan — planning document only; no repository changes applied.*
