"""CAM3 Footage2 crossing audit (one-off diagnostic)."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import supervision as sv
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from configs.camera_timing_config import (
    CAM3_ENTRY_CONFIG,
    CAM3_PROCESS_EVERY_N_FRAMES,
    DATA_DIR,
    MODEL_PATH,
)
from entry_retail import (
    EVENT_ENTRY,
    EVENT_EXIT,
    SIDE_MALL,
    SIDE_STORE,
    STATE_INSIDE,
    STATE_OUTSIDE,
    TrackRetailRecord,
    build_retail_entry_engine,
)

FOOTAGE2_DIR = DATA_DIR / "CCTV Footage_2"
CLIPS = [
    ("entry1", FOOTAGE2_DIR / "entry 1.mp4"),
    ("entry2", FOOTAGE2_DIR / "entry 2.mp4"),
]
CONF, IOU = 0.35, 0.5
N = CAM3_PROCESS_EVERY_N_FRAMES
LAYOUT = CAM3_ENTRY_CONFIG["FOOTAGE2"]


@dataclass
class FrameLog:
    frame: int
    track_id: int
    instant_side: str
    store_state: str
    confirmed_side: str
    pending_side: Optional[str]
    pending_frames: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    cross: float
    num_detections: int


@dataclass
class ClipReport:
    clip: str
    total_frames: int
    processed_frames: int = 0
    max_simultaneous_tracks: int = 0
    entry_events: List[dict] = field(default_factory=list)
    exit_events: List[dict] = field(default_factory=list)
    inits: List[dict] = field(default_factory=list)
    inside_without_entry: List[dict] = field(default_factory=list)
    rejected_transitions: List[dict] = field(default_factory=list)
    crossing_frames: List[FrameLog] = field(default_factory=list)
    multi_track_frames: List[dict] = field(default_factory=list)


def audit_clip(clip_name: str, video_path: Path, yolo: YOLO) -> ClipReport:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    engine = build_retail_entry_engine(LAYOUT, w, h)
    tracker = sv.ByteTrack()
    report = ClipReport(clip=clip_name, total_frames=total)
    prev_store: Dict[int, str] = {}
    original_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if original_frame % N != 0:
            original_frame += 1
            continue

        res = yolo.predict(frame, conf=CONF, iou=IOU, classes=[0], verbose=False)[0]
        dets = tracker.update_with_detections(sv.Detections.from_ultralytics(res))
        n_dets = len(dets) if dets.tracker_id is not None else 0
        report.max_simultaneous_tracks = max(report.max_simultaneous_tracks, n_dets)

        if n_dets >= 2:
            report.multi_track_frames.append(
                {
                    "frame": original_frame,
                    "track_ids": [int(t) for t in dets.tracker_id],
                    "count": n_dets,
                }
            )

        if dets.tracker_id is not None:
            for tid, xyxy in zip(dets.tracker_id, dets.xyxy):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cross = engine.geometry.cross_value(center)
                instant = engine.geometry.classify_side(center)

                prev_state = prev_store.get(tid)
                result = engine.update(tid, center, original_frame=original_frame)
                rec = engine._tracks[tid]

                if result.init_debug:
                    report.inits.append(
                        {
                            "frame": original_frame,
                            "track_id": tid,
                            "initial_side": result.init_debug.initial_side,
                            "initial_store_state": result.init_debug.initial_store_state,
                            "centroid": center,
                            "bbox": (x1, y1, x2, y2),
                        }
                    )
                    if result.init_debug.initial_store_state == STATE_INSIDE:
                        report.inside_without_entry.append(
                            {
                                "frame": original_frame,
                                "track_id": tid,
                                "reason": "init_on_inside_side",
                                "initial_side": result.init_debug.initial_side,
                            }
                        )

                if result.transition:
                    ev = {
                        "frame": original_frame,
                        "track_id": tid,
                        "event_type": result.transition.event_type,
                        **result.transition.debug,
                    }
                    if result.transition.event_type == EVENT_ENTRY:
                        report.entry_events.append(ev)
                    else:
                        report.exit_events.append(ev)

                # store_state flipped without transition event
                if (
                    prev_state == STATE_OUTSIDE
                    and result.store_state == STATE_INSIDE
                    and not result.transition
                ):
                    report.inside_without_entry.append(
                        {
                            "frame": original_frame,
                            "track_id": tid,
                            "reason": "outside_to_inside_no_event",
                            "confirmed_side": rec.confirmed_side,
                            "pending_side": rec.pending_side,
                            "pending_frames": rec.pending_frames,
                            "instant_side": instant,
                        }
                    )

                # pending completed but no event (audit after update - check last commit)
                if rec.pending_frames == 0 and prev_state and instant != rec.confirmed_side:
                    pass

                prev_store[tid] = result.store_state

                near_line = abs(center[1] - LAYOUT["ENTRY_PLANE_Y_NORM"] * h) < 80
                if near_line or rec.pending_side is not None:
                    report.crossing_frames.append(
                        FrameLog(
                            frame=original_frame,
                            track_id=tid,
                            instant_side=instant,
                            store_state=result.store_state,
                            confirmed_side=rec.confirmed_side,
                            pending_side=rec.pending_side,
                            pending_frames=rec.pending_frames,
                            bbox=(x1, y1, x2, y2),
                            centroid=center,
                            cross=cross,
                            num_detections=n_dets,
                        )
                    )

        report.processed_frames += 1
        original_frame += 1

    cap.release()

    # post-pass: find stability failures (simulate pending resets)
    for clip_log in report.crossing_frames:
        pass

    return report


def simulate_rejected(engine_layout, w, h, logs: List[FrameLog]) -> List[dict]:
    """Detect side commits that did not emit ENTRY/EXIT."""
    rejected = []
    by_track: Dict[int, List[FrameLog]] = {}
    for lg in logs:
        by_track.setdefault(lg.track_id, []).append(lg)
    for tid, frames in by_track.items():
        for i in range(1, len(frames)):
            prev, cur = frames[i - 1], frames[i]
            if prev.confirmed_side != cur.confirmed_side:
                # side committed between frames (pending cleared)
                entry_from, entry_to = SIDE_STORE, SIDE_MALL  # footage2 invert
                is_entry_cross = prev.confirmed_side == entry_from and cur.confirmed_side == entry_to
                is_exit_cross = prev.confirmed_side == entry_to and cur.confirmed_side == entry_from
                if is_entry_cross and prev.store_state == STATE_OUTSIDE and cur.store_state == STATE_OUTSIDE:
                    rejected.append(
                        {
                            "track_id": tid,
                            "frame": cur.frame,
                            "reason": "entry_cross_but_still_outside",
                            "previous_side": prev.confirmed_side,
                            "current_side": cur.confirmed_side,
                            "pending_frames": cur.pending_frames,
                        }
                    )
                if is_entry_cross and prev.store_state == STATE_INSIDE:
                    rejected.append(
                        {
                            "track_id": tid,
                            "frame": cur.frame,
                            "reason": "entry_cross_but_already_inside",
                            "previous_side": prev.confirmed_side,
                            "current_side": cur.confirmed_side,
                        }
                    )
    return rejected


def main() -> None:
    print("Loading YOLO...")
    yolo = YOLO(str(MODEL_PATH))
    print(f"Footage2 ENTRY rules (INVERT_RETAIL_SEMANTICS=True):")
    print(f"  ENTRY: {SIDE_STORE} -> {SIDE_MALL} while latched {STATE_OUTSIDE}")
    print(f"  EXIT:  {SIDE_MALL} -> {SIDE_STORE} while latched {STATE_INSIDE}")
    print(f"  Stability: {LAYOUT.get('ENTRY_STABILITY_FRAMES', 3)} processed frames")
    print(f"  Process every {N} frames\n")

    for clip_name, path in CLIPS:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        print("=" * 60)
        print(f"CLIP: {clip_name} ({path.name})")
        r = audit_clip(clip_name, path, yolo)
        print(f"  total_frames={r.total_frames} processed={r.processed_frames}")
        print(f"  max_simultaneous_tracks={r.max_simultaneous_tracks}")
        print(f"  ENTRY events={len(r.entry_events)}")
        print(f"  EXIT events={len(r.exit_events)}")
        print(f"  init rows={len(r.inits)}")
        print(f"  init INSIDE (no ENTRY)={sum(1 for x in r.inits if x['initial_store_state']=='INSIDE')}")
        print(f"  frames with 2+ tracks={len(r.multi_track_frames)}")
        if r.entry_events:
            for e in r.entry_events:
                print(f"    ENTRY frame={e['frame']} tid={e['track_id']} {e.get('previous_side')}->{e.get('current_side')}")
        if r.multi_track_frames[:5]:
            print("  sample multi-track frames:")
            for m in r.multi_track_frames[:8]:
                print(f"    frame={m['frame']} ids={m['track_ids']}")
        if r.inits[:6]:
            print("  sample inits:")
            for ini in r.inits[:8]:
                print(
                    f"    frame={ini['frame']} tid={ini['track_id']} "
                    f"side={ini['initial_side']} state={ini['initial_store_state']}"
                )
        inside_init = [x for x in r.inside_without_entry if x["reason"] == "init_on_inside_side"]
        print(f"  tracks that appear already INSIDE (init): {len(inside_init)}")
        # pending stall: many pending frames but never 3
        pending_stalls = {}
        for lg in r.crossing_frames:
            if lg.pending_side and lg.pending_frames in (1, 2):
                key = (lg.track_id, lg.pending_side)
                pending_stalls[key] = pending_stalls.get(key, 0) + 1
        print(f"  pending 1-2 frame observations (near line): {len(pending_stalls)} track-side keys")


if __name__ == "__main__":
    main()
