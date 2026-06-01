# Retail CCTV + POS Pipeline

Package layout for migration to Purpple_Vision.

## Structure

```
project/
├── configs/          # camera_timing_config.py (paths + anchors)
├── validation/       # Zone geometry validation (CAM1/2/3/5)
├── events/           # Event generators (YOLO + ByteTrack)
├── pos/              # POS explore + aggregation
├── matching/         # Timestamp normalization + purchase matching
├── analytics/        # retail_analytics.py
├── outputs/          # JSON, JSONL, reports, charts
│   ├── reports/
│   └── charts/
└── data/
    ├── CCTV Footage/
    └── Brigade_Bangalore_10_April_26.csv
```

## Run scripts

From `project/`:

```powershell
cd E:\NOTEBK\project
..\ .venv\Scripts\python.exe events\cam1_events.py
..\ .venv\Scripts\python.exe matching\normalize_event_timestamps.py
..\ .venv\Scripts\python.exe pos\pos_aggregation.py
..\ .venv\Scripts\python.exe matching\purchase_matching.py
..\ .venv\Scripts\python.exe analytics\retail_analytics.py
```

Outputs are written under `project/outputs/`.
