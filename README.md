# NoteBk — Retail CCTV + POS Analytics

NOTEBK project: computer-vision event pipelines (YOLO11m + ByteTrack), POS matching, and retail analytics for Brigade Bangalore store footage.

## Layout

| Path | Description |
|------|-------------|
| `project/` | Main package (events, matching, analytics, configs) |
| `project/events/` | CAM1–CAM5 event generators |
| `project/outputs/` | Generated JSONL, reports, charts |
| `migration_plan.md` | NOTEBK → Purpple_Vision migration plan |
| `migration_report_phase*.md` | Pipeline port reports |

See [project/README.md](project/README.md) for run instructions.

## Setup

```powershell
cd E:\NOTEBK
python -m venv .venv
.\.venv\Scripts\pip install -r project\requirements.txt   # if present
```

Place CCTV clips under `project/data/CCTV Footage/` (not committed — see `.gitignore`).

## Related

Migrated intelligence layer: [Purpple_Vision](https://github.com/) (separate repo).
