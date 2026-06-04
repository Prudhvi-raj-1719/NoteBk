"""Compare bbox-center vs foot-point retail entry engine on Footage2 entry clips."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import supervision as sv
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.camera_timing_config import (
    CAM3_ENTRY_CONFIG,
    CAM3_PROCESS_EVERY_N_FRAMES,
    DATA_DIR,
    MODEL_PATH,
)
from entry_retail import RetailEntryEngine, build_retail_entry_engine

FOOTAGE2_DIR = DATA_DIR / "CCTV Footage_2"
FOOTAGE2_LAYOUT = CAM3_ENTRY_CONFIG["FOOTAGE2"]
VIDEOS = [
    ("entry1", FOOTAGE2_DIR / "entry 1.mp4"),
    ("entry2", FOOTAGE2_DIR / "entry 2.mp4"),
]
CONF, IOU = 0.35, 0.5
PROCESS_EVERY_N_FRAMES = CAM3_PROCESS_EVERY_N_FRAMES


def bbox_center(xyxy) -> Tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def foot_point(xyxy) -> Tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int(y2)


@dataclass
class AnchorStats:
    entry: int = 0
    exit: int = 0
    inits: int = 0
    transitions: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class VideoComparison:
    video: str
    resolution: Tuple[int, int]
    total_frames_read: int = 0
    processed_frames: int = 0
    center: AnchorStats = field(default_factory=AnchorStats)
    foot: AnchorStats = field(default_factory=AnchorStats)
    disagree_detections: int = 0
    total_detections: int = 0
    center_out_foot_in: int = 0
    center_in_foot_out: int = 0


def run_engine_on_detections(
    engine: RetailEntryEngine,
    stats: AnchorStats,
    detections: sv.Detections,
    original_frame: int,
    point_fn,
) -> None:
    if detections.tracker_id is None:
        return
    for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
        tid = int(track_id)
        point = point_fn(xyxy)
        result = engine.update(tid, point, original_frame=original_frame)
        if result.init_debug is not None:
            stats.inits += 1
        if result.transition is not None:
            if result.transition.event_type == "ENTRY":
                stats.entry += 1
            else:
                stats.exit += 1
            stats.transitions.append(
                (original_frame, tid, result.transition.event_type)
            )


def run_video(name: str, path: Path, yolo: YOLO) -> VideoComparison:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_engine = build_retail_entry_engine(FOOTAGE2_LAYOUT, w, h)
    foot_engine = build_retail_entry_engine(FOOTAGE2_LAYOUT, w, h)
    geometry = center_engine.geometry
    tracker = sv.ByteTrack()
    out = VideoComparison(video=name, resolution=(w, h))
    original_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out.total_frames_read += 1

        if original_frame % PROCESS_EVERY_N_FRAMES != 0:
            original_frame += 1
            del frame
            continue

        res = yolo.predict(frame, conf=CONF, iou=IOU, classes=[0], verbose=False)[0]
        dets = tracker.update_with_detections(sv.Detections.from_ultralytics(res))

        if dets.tracker_id is not None:
            for tid, xyxy in zip(dets.tracker_id, dets.xyxy):
                cpt = bbox_center(xyxy)
                fpt = foot_point(xyxy)
                cside = geometry.classify_side(cpt)
                fside = geometry.classify_side(fpt)
                if cside != fside:
                    out.disagree_detections += 1
                    if cside == "MALL" and fside == "STORE":
                        out.center_out_foot_in += 1
                    if cside == "STORE" and fside == "MALL":
                        out.center_in_foot_out += 1
                out.total_detections += 1

        run_engine_on_detections(
            center_engine, out.center, dets, original_frame, bbox_center
        )
        run_engine_on_detections(
            foot_engine, out.foot, dets, original_frame, foot_point
        )

        del frame
        out.processed_frames += 1
        original_frame += 1

    cap.release()
    return out


def print_recommendation(results: List[VideoComparison]) -> None:
    total_disagree = sum(r.disagree_detections for r in results)
    total_dets = sum(r.total_detections for r in results)
    foot_in_center_out = sum(r.center_out_foot_in for r in results)
    foot_only_entry = 0
    center_only_entry = 0
    for r in results:
        only_c = set(r.center.transitions) - set(r.foot.transitions)
        only_f = set(r.foot.transitions) - set(r.center.transitions)
        foot_only_entry += sum(1 for x in only_f if x[2] == "ENTRY")
        center_only_entry += sum(1 for x in only_c if x[2] == "ENTRY")

    print("\n=== RECOMMENDATION ===")
    print(
        f"Frame sampling: every {PROCESS_EVERY_N_FRAMES} frames (CAM3 production cadence). "
        "RetailEntryEngine: half-plane + 3-frame stability + latched INSIDE/OUTSIDE."
    )
    if total_dets == 0:
        print("Insufficient detections to compare anchors.")
        return
    disagree_pct = 100.0 * total_disagree / total_dets
    print(f"Instantaneous side disagreement (center vs foot): {disagree_pct:.1f}%")
    print(f"  center=MALL foot=STORE: {foot_in_center_out}")
    if foot_only_entry > center_only_entry:
        print(
            f"Foot anchor produced more ENTRY transitions ({foot_only_entry} vs "
            f"{center_only_entry} center-only)."
        )
    elif center_only_entry > foot_only_entry:
        print(
            f"Centroid anchor produced more ENTRY-only transitions ({center_only_entry})."
        )
    else:
        print(
            "ENTRY/EXIT counts are similar between anchors under sampled frames. "
            "Centroid remains the production anchor."
        )


def main() -> None:
    print(f"Loading YOLO (process every {PROCESS_EVERY_N_FRAMES} frames)...")
    yolo = YOLO(str(MODEL_PATH))
    results: List[VideoComparison] = []

    for vname, vpath in VIDEOS:
        if not vpath.exists():
            print(f"MISSING: {vpath}")
            continue
        print(f"Processing {vname}...")
        results.append(run_video(vname, vpath, yolo))

    print("\n=== ENTRY/EXIT COUNTS (RetailEntryEngine, Footage2) ===")
    for r in results:
        print(
            f"\n{r.video} ({r.resolution[0]}x{r.resolution[1]}, "
            f"read={r.total_frames_read} processed={r.processed_frames})"
        )
        print(f"  center: entry={r.center.entry} exit={r.center.exit} inits={r.center.inits}")
        print(f"  foot:   entry={r.foot.entry} exit={r.foot.exit} inits={r.foot.inits}")
        print(f"  delta:  entry={r.foot.entry - r.center.entry:+d} exit={r.foot.exit - r.center.exit:+d}")
        pct = 100.0 * r.disagree_detections / max(1, r.total_detections)
        print(f"  instantaneous side disagreement: {r.disagree_detections}/{r.total_detections} ({pct:.1f}%)")
        print(f"    center=MALL foot=STORE: {r.center_out_foot_in}")
        print(f"    center=STORE foot=MALL: {r.center_in_foot_out}")
        c_set = set(r.center.transitions)
        f_set = set(r.foot.transitions)
        print(f"  shared transitions: {len(c_set & f_set)}")
        print(f"  center-only: {len(c_set - f_set)}")
        print(f"  foot-only: {len(f_set - c_set)}")

    print("\n=== TOTAL (both clips) ===")
    print(f"  center: entry={sum(r.center.entry for r in results)} exit={sum(r.center.exit for r in results)}")
    print(f"  foot:   entry={sum(r.foot.entry for r in results)} exit={sum(r.foot.exit for r in results)}")
    print_recommendation(results)


if __name__ == "__main__":
    main()
