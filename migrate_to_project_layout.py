#!/usr/bin/env python3
"""Move NOTEBK assets into project/ and copy Python modules with updated paths."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

NOTEBK = Path(__file__).resolve().parent
PROJECT = NOTEBK / "project"

OUTPUTS = PROJECT / "outputs"
REPORTS = OUTPUTS / "reports"
CHARTS = OUTPUTS / "charts"

DATA = PROJECT / "data"

CONFIG_IMPORT_EVENTS = (
    "from configs.camera_timing_config import (\n"
    "    CAMERA_VIDEO_FILES,\n"
    "    MODEL_PATH,\n"
    "    OUTPUTS_DIR,\n"
    ")\n"
)

CONFIG_IMPORT_VALIDATION = (
    "from configs.camera_timing_config import CAMERA_VIDEO_FILES, MODEL_PATH\n"
)

PYTHON_LAYOUT = {
    "cam1_validation.py": "validation",
    "cam2_validation.py": "validation",
    "cam3_validation.py": "validation",
    "cam5_validation.py": "validation",
    "cam1_events.py": "events",
    "cam2_events.py": "events",
    "cam5_events.py": "events",
    "cam3_entry_exit.py": ("events", "cam3_events.py"),
    "explore_pos_data.py": "pos",
    "pos_aggregation.py": "pos",
    "normalize_event_timestamps.py": "matching",
    "purchase_matching.py": "matching",
    "video_pos_overlap_analysis.py": "matching",
    "inspect_event_timestamps.py": "matching",
    "retail_analytics.py": "analytics",
}


BOOTSTRAP = (
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "_PROJECT_ROOT = Path(__file__).resolve().parents[1]\n"
    "if str(_PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(_PROJECT_ROOT))\n"
    "\n"
)


def add_bootstrap(content: str) -> str:
    content = content.replace(
        "import bootstrap  # noqa: F401 - adds project root to sys.path\n", ""
    )
    if "_PROJECT_ROOT = Path(__file__).resolve().parents[1]" in content:
        return content
    marker = "from __future__ import annotations\n\n"
    if marker in content:
        return content.replace(marker, marker + BOOTSTRAP, 1)
    return BOOTSTRAP + content


def insert_after_yolo_import(content: str, block: str) -> str:
    marker = "from ultralytics import YOLO\n"
    if block.strip() in content:
        return content
    if marker in content:
        return content.replace(marker, marker + "\n" + block, 1)
    return block + content


def patch_validation(content: str, camera: str) -> str:
    content = insert_after_yolo_import(content, CONFIG_IMPORT_VALIDATION)
    content = re.sub(
        r'VIDEO_PATH = Path\(r"[^"]+"\)\n',
        f'VIDEO_PATH = CAMERA_VIDEO_FILES["{camera}"]\n',
        content,
        count=1,
    )
    content = content.replace('MODEL_PATH = "yolo11m.pt"\n', "")
    return content


def patch_events(content: str, camera: str, events_file: str) -> str:
    content = insert_after_yolo_import(content, CONFIG_IMPORT_EVENTS)
    content = re.sub(
        r'VIDEO_PATH = Path\(r"[^"]+"\)\n',
        f'VIDEO_PATH = CAMERA_VIDEO_FILES["{camera}"]\n',
        content,
        count=1,
    )
    content = re.sub(
        r'EVENTS_PATH = Path\("[^"]+"\)\n',
        f'EVENTS_PATH = OUTPUTS_DIR / "{events_file}"\n',
        content,
        count=1,
    )
    content = content.replace('MODEL_PATH = "yolo11m.pt"\n', "")
    return content


def patch_cam3_events(content: str) -> str:
    content = insert_after_yolo_import(
        content,
        'from configs.camera_timing_config import CAMERA_VIDEO_FILES, MODEL_PATH\n',
    )
    content = re.sub(
        r'VIDEO_PATH = r"[^"]+"\n',
        'VIDEO_PATH = str(CAMERA_VIDEO_FILES["CAM3"])\n',
        content,
        count=1,
    )
    content = content.replace('MODEL_PATH = "yolo11m.pt"\n', "")
    return content


def patch_normalize(content: str) -> str:
    content = content.replace(
        "from camera_timing_config import (",
        "from configs.camera_timing_config import (",
    )
    content = content.replace(
        "REPORT_PATH = PROJECT_ROOT / \"event_time_normalization_report.txt\"",
        "REPORT_PATH = OUTPUTS_REPORTS_DIR / \"event_time_normalization_report.txt\"",
    )
    content = content.replace(
        "    PROJECT_ROOT,\n",
        "    OUTPUTS_DIR,\n    OUTPUTS_REPORTS_DIR,\n",
    )
    content = content.replace(
        '        "input": PROJECT_ROOT / "cam1_events.jsonl",\n'
        '        "output": PROJECT_ROOT / "cam1_events_normalized.jsonl",',
        '        "input": OUTPUTS_DIR / "cam1_events.jsonl",\n'
        '        "output": OUTPUTS_DIR / "cam1_events_normalized.jsonl",',
    )
    content = content.replace(
        '        "input": PROJECT_ROOT / "cam2_events.jsonl",\n'
        '        "output": PROJECT_ROOT / "cam2_events_normalized.jsonl",',
        '        "input": OUTPUTS_DIR / "cam2_events.jsonl",\n'
        '        "output": OUTPUTS_DIR / "cam2_events_normalized.jsonl",',
    )
    content = content.replace(
        '        "input": PROJECT_ROOT / "cam5_events.jsonl",\n'
        '        "output": PROJECT_ROOT / "cam5_events_normalized.jsonl",',
        '        "input": OUTPUTS_DIR / "cam5_events.jsonl",\n'
        '        "output": OUTPUTS_DIR / "cam5_events_normalized.jsonl",',
    )
    return content


def patch_purchase_matching(content: str) -> str:
    content = content.replace(
        "from camera_timing_config import PROJECT_ROOT",
        "from configs.camera_timing_config import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR",
    )
    replacements = {
        "AGGREGATED_TRANSACTIONS_PATH = PROJECT_ROOT / \"aggregated_transactions.json\"":
        'AGGREGATED_TRANSACTIONS_PATH = OUTPUTS_DIR / "aggregated_transactions.json"',
        '"CAM1": PROJECT_ROOT / "cam1_events_normalized.jsonl"':
        '"CAM1": OUTPUTS_DIR / "cam1_events_normalized.jsonl"',
        '"CAM2": PROJECT_ROOT / "cam2_events_normalized.jsonl"':
        '"CAM2": OUTPUTS_DIR / "cam2_events_normalized.jsonl"',
        '"CAM5": PROJECT_ROOT / "cam5_events_normalized.jsonl"':
        '"CAM5": OUTPUTS_DIR / "cam5_events_normalized.jsonl"',
        "OUTPUT_PATH = PROJECT_ROOT / \"purchase_matches.json\"":
        'OUTPUT_PATH = OUTPUTS_DIR / "purchase_matches.json"',
        "REPORT_PATH = PROJECT_ROOT / \"purchase_matching_report.txt\"":
        'REPORT_PATH = OUTPUTS_REPORTS_DIR / "purchase_matching_report.txt"',
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content


def patch_video_overlap(content: str) -> str:
    content = content.replace(
        "from camera_timing_config import POS_SALE_DATE, PROJECT_ROOT",
        "from configs.camera_timing_config import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR, POS_SALE_DATE",
    )
    content = content.replace(
        "AGGREGATED_TRANSACTIONS_PATH = PROJECT_ROOT / \"aggregated_transactions.json\"",
        'AGGREGATED_TRANSACTIONS_PATH = OUTPUTS_DIR / "aggregated_transactions.json"',
    )
    content = content.replace(
        "    PROJECT_ROOT / \"cam1_events_normalized.jsonl\",\n"
        "    PROJECT_ROOT / \"cam2_events_normalized.jsonl\",",
        '    OUTPUTS_DIR / "cam1_events_normalized.jsonl",\n'
        '    OUTPUTS_DIR / "cam2_events_normalized.jsonl",',
    )
    content = content.replace(
        "REPORT_PATH = PROJECT_ROOT / \"video_pos_overlap_report.txt\"",
        'REPORT_PATH = OUTPUTS_REPORTS_DIR / "video_pos_overlap_report.txt"',
    )
    return content


def patch_inspect_timestamps(content: str) -> str:
    content = content.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parent\n",
        "",
    )
    content = content.replace(
        "REPORT_PATH = PROJECT_ROOT / \"event_timestamp_report.txt\"\n",
        'REPORT_PATH = OUTPUTS_REPORTS_DIR / "event_timestamp_report.txt"\n',
    )
    if "from configs.camera_timing_config import" not in content:
        content = content.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\n\n"
            "from configs.camera_timing_config import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR\n",
        )
    content = content.replace(
        "    PROJECT_ROOT / \"cam1_events.jsonl\",\n",
        '    OUTPUTS_DIR / "cam1_events.jsonl",\n',
    )
    content = content.replace(
        "    PROJECT_ROOT / \"cam2_events.jsonl\",\n",
        '    OUTPUTS_DIR / "cam2_events.jsonl",\n',
    )
    content = content.replace(
        "    PROJECT_ROOT / \"cam3_events.jsonl\",\n",
        '    OUTPUTS_DIR / "cam3_events.jsonl",\n',
    )
    content = content.replace(
        "    PROJECT_ROOT / \"cam5_events.jsonl\",\n",
        '    OUTPUTS_DIR / "cam5_events.jsonl",\n',
    )
    return content


def patch_retail_analytics(content: str) -> str:
    content = content.replace(
        "from camera_timing_config import PROJECT_ROOT",
        "from configs.camera_timing_config import OUTPUTS_CHARTS_DIR, OUTPUTS_DIR, PROJECT_ROOT",
    )
    replacements = {
        '"CAM1": PROJECT_ROOT / "cam1_events_normalized.jsonl"':
        '"CAM1": OUTPUTS_DIR / "cam1_events_normalized.jsonl"',
        '"CAM2": PROJECT_ROOT / "cam2_events_normalized.jsonl"':
        '"CAM2": OUTPUTS_DIR / "cam2_events_normalized.jsonl"',
        '"CAM5": PROJECT_ROOT / "cam5_events_normalized.jsonl"':
        '"CAM5": OUTPUTS_DIR / "cam5_events_normalized.jsonl"',
        "TRANSACTIONS_PATH = PROJECT_ROOT / \"aggregated_transactions.json\"":
        'TRANSACTIONS_PATH = OUTPUTS_DIR / "aggregated_transactions.json"',
        "PURCHASE_MATCHES_PATH = PROJECT_ROOT / \"purchase_matches.json\"":
        'PURCHASE_MATCHES_PATH = OUTPUTS_DIR / "purchase_matches.json"',
        "SUMMARY_PATH = PROJECT_ROOT / \"analytics_summary.json\"":
        'SUMMARY_PATH = OUTPUTS_DIR / "analytics_summary.json"',
        "REPORT_PATH = PROJECT_ROOT / \"analytics_report.md\"":
        'REPORT_PATH = OUTPUTS_DIR / "analytics_report.md"',
        "CHARTS_DIR = PROJECT_ROOT / \"analytics_output\"":
        "CHARTS_DIR = OUTPUTS_CHARTS_DIR",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    return content


def patch_pos_aggregation(content: str) -> str:
    content = content.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parent\n",
        "",
    )
    content = content.replace(
        'INPUT_PATH = PROJECT_ROOT / "Brigade_Bangalore_10_April_26.csv"\n',
        "from configs.camera_timing_config import OUTPUTS_DIR, POS_CSV_PATH\n\n"
        "INPUT_PATH = POS_CSV_PATH\n",
    )
    content = content.replace(
        'OUTPUT_CSV = PROJECT_ROOT / "aggregated_transactions.csv"\n',
        'OUTPUT_CSV = OUTPUTS_DIR / "aggregated_transactions.csv"\n',
    )
    content = content.replace(
        'OUTPUT_JSON = PROJECT_ROOT / "aggregated_transactions.json"\n',
        'OUTPUT_JSON = OUTPUTS_DIR / "aggregated_transactions.json"\n',
    )
    return content


def patch_explore_pos(content: str) -> str:
    content = content.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parent\n",
        "",
    )
    content = content.replace(
        'SCHEMA_REPORT_PATH = PROJECT_ROOT / "pos_schema_report.txt"\n',
        "from configs.camera_timing_config import DATA_DIR, OUTPUTS_REPORTS_DIR\n\n"
        'SCHEMA_REPORT_PATH = OUTPUTS_REPORTS_DIR / "pos_schema_report.txt"\n',
    )
    content = content.replace(
        'COLUMN_SUMMARY_PATH = PROJECT_ROOT / "pos_column_summary.csv"\n',
        'COLUMN_SUMMARY_PATH = OUTPUTS_REPORTS_DIR / "pos_column_summary.csv"\n',
    )
    content = content.replace(
        "    pos_file = locate_pos_file(PROJECT_ROOT)",
        "    pos_file = locate_pos_file(DATA_DIR)",
    )
    return content


def patch_module(name: str, content: str) -> str:
    content = add_bootstrap(content)
    if name.startswith("cam") and name.endswith("_validation.py"):
        camera = name.split("_")[0].upper()
        return patch_validation(content, camera)
    if name == "cam1_events.py":
        return patch_events(content, "CAM1", "cam1_events.jsonl")
    if name == "cam2_events.py":
        return patch_events(content, "CAM2", "cam2_events.jsonl")
    if name == "cam5_events.py":
        return patch_events(content, "CAM5", "cam5_events.jsonl")
    if name == "cam3_entry_exit.py":
        return patch_cam3_events(content)
    if name == "normalize_event_timestamps.py":
        return patch_normalize(content)
    if name == "purchase_matching.py":
        return patch_purchase_matching(content)
    if name == "video_pos_overlap_analysis.py":
        return patch_video_overlap(content)
    if name == "inspect_event_timestamps.py":
        return patch_inspect_timestamps(content)
    if name == "retail_analytics.py":
        return patch_retail_analytics(content)
    if name == "pos_aggregation.py":
        return patch_pos_aggregation(content)
    if name == "explore_pos_data.py":
        return patch_explore_pos(content)
    return content


def ensure_dirs() -> None:
    for path in (
        PROJECT / "configs",
        PROJECT / "validation",
        PROJECT / "events",
        PROJECT / "pos",
        PROJECT / "matching",
        PROJECT / "analytics",
        OUTPUTS,
        REPORTS,
        CHARTS,
        DATA,
    ):
        path.mkdir(parents=True, exist_ok=True)


def move_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))


def move_assets() -> None:
    move_if_exists(NOTEBK / "CCTV Footage", DATA / "CCTV Footage")
    move_if_exists(NOTEBK / "Brigade_Bangalore_10_April_26.csv", DATA / "Brigade_Bangalore_10_April_26.csv")
    move_if_exists(NOTEBK / "yolo11m.pt", PROJECT / "yolo11m.pt")

    for pattern in ("*.jsonl",):
        for src in NOTEBK.glob(pattern):
            move_if_exists(src, OUTPUTS / src.name)

    for name in (
        "aggregated_transactions.json",
        "aggregated_transactions.csv",
        "purchase_matches.json",
        "analytics_summary.json",
        "analytics_report.md",
    ):
        move_if_exists(NOTEBK / name, OUTPUTS / name)

    for name in (
        "event_time_normalization_report.txt",
        "purchase_matching_report.txt",
        "video_pos_overlap_report.txt",
        "event_timestamp_report.txt",
        "pos_schema_report.txt",
    ):
        move_if_exists(NOTEBK / name, REPORTS / name)

    move_if_exists(NOTEBK / "pos_column_summary.csv", REPORTS / "pos_column_summary.csv")

    charts_src = NOTEBK / "analytics_output"
    if charts_src.is_dir():
        for png in charts_src.glob("*.png"):
            move_if_exists(png, CHARTS / png.name)
        if charts_src.exists() and not any(charts_src.iterdir()):
            charts_src.rmdir()


def patch_existing_project_modules() -> None:
    """Re-apply bootstrap to modules already under project/."""
    for py_file in PROJECT.rglob("*.py"):
        if py_file.name in ("bootstrap.py", "camera_timing_config.py", "__init__.py"):
            continue
        if py_file.parent.name == "configs":
            continue
        content = py_file.read_text(encoding="utf-8")
        updated = add_bootstrap(content)
        if updated != content:
            py_file.write_text(updated, encoding="utf-8")
            print(f"bootstrapped {py_file.relative_to(NOTEBK)}")


def copy_python_modules() -> None:
    event_map = {
        "cam1_events.py": "CAM1",
        "cam2_events.py": "CAM2",
        "cam5_events.py": "CAM5",
    }
    for src_name, dest_info in PYTHON_LAYOUT.items():
        if isinstance(dest_info, tuple):
            dest_folder, dest_name = dest_info
        else:
            dest_folder, dest_name = dest_info, src_name

        src = NOTEBK / src_name
        if not src.exists():
            print(f"skip missing: {src_name}")
            continue

        content = src.read_text(encoding="utf-8")
        content = patch_module(src_name, content)
        dest = PROJECT / dest_folder / dest_name
        dest.write_text(content, encoding="utf-8")
        print(f"wrote {dest.relative_to(NOTEBK)}")


def main() -> None:
    ensure_dirs()
    move_assets()
    copy_python_modules()
    patch_existing_project_modules()
    print("Migration complete. Run scripts from project/ with:")
    print("  cd project && python events/cam1_events.py")


if __name__ == "__main__":
    main()
