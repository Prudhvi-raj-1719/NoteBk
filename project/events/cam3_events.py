"""CAM3 retail entry/exit: YOLO11m + ByteTrack + half-plane RetailEntryEngine."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from configs.camera_timing_config import (
    CAM3_ENTRY_CONFIG,
    CAM3_PROCESS_EVERY_N_FRAMES,
    CAMERA_VIDEO_FILES,
    DATA_DIR,
    MODEL_PATH,
    OUTPUTS_DIR,
)
from entry_retail import (
    STATE_INSIDE,
    STATE_OUTSIDE,
    build_entry_line_geometry,
    build_retail_entry_engine,
    format_video_timestamp,
)
from configs.reid_config import CAM3_REID_ENABLED, REID_HEURISTIC_FALLBACK
from events.event_emitter import EventEmitter
from reid.osnet_reid import OsnetEmbedder
from reid.reid_manager import ReIDManager

_FOOTAGE2 = os.getenv("GENERATE_FOOTAGE2") == "1"
_FOOTAGE2_DIR = DATA_DIR / "CCTV Footage_2"
CAM3_LAYOUT = CAM3_ENTRY_CONFIG["FOOTAGE2"] if _FOOTAGE2 else CAM3_ENTRY_CONFIG["BRIGADE"]
PROCESS_EVERY_N_FRAMES = int(
    CAM3_LAYOUT.get("PROCESS_EVERY_N_FRAMES", CAM3_PROCESS_EVERY_N_FRAMES)
)
CONFIDENCE_THRESHOLD = float(
    CAM3_LAYOUT.get("CONFIDENCE_THRESHOLD", 0.15)
)
YOLO_MAX_DET = int(CAM3_LAYOUT.get("YOLO_MAX_DET", 50))

FOOTAGE2_CLIPS: List[Tuple[str, Path, Path]] = [
    (
        "entry1",
        _FOOTAGE2_DIR / "entry 1.mp4",
        OUTPUTS_DIR / "cam3_footage2_entry1_events.jsonl",
    ),
    (
        "entry2",
        _FOOTAGE2_DIR / "entry 2.mp4",
        OUTPUTS_DIR / "cam3_footage2_entry2_events.jsonl",
    ),
]

CAMERA_ID = "CAM3"
PERSON_CLASS_ID = 0
IOU_THRESHOLD = float(CAM3_LAYOUT.get("IOU_THRESHOLD", 0.55))
YOLO_IMGSZ = 960
BYTE_TRACK_ACTIVATION_THRESHOLD = 0.20
BYTE_TRACK_MIN_CONSECUTIVE_FRAMES = 1
BYTE_TRACK_MAX_TIME_LOST = 90
ENTRY_POLYGON_COLOR: Tuple[int, int, int] = (0, 0, 255)


def denormalize_polygon(
    points: List[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [(int(x * width), int(y * height)) for x, y in points],
        dtype=np.int32,
    )


def create_byte_tracker() -> sv.ByteTrack:
    tracker = sv.ByteTrack()
    if hasattr(tracker, "track_activation_threshold"):
        tracker.track_activation_threshold = BYTE_TRACK_ACTIVATION_THRESHOLD
    if hasattr(tracker, "minimum_consecutive_frames"):
        tracker.minimum_consecutive_frames = BYTE_TRACK_MIN_CONSECUTIVE_FRAMES
    if hasattr(tracker, "max_time_lost"):
        tracker.max_time_lost = BYTE_TRACK_MAX_TIME_LOST
    if hasattr(tracker, "det_thresh"):
        tracker.det_thresh = CONFIDENCE_THRESHOLD
    return tracker


def detect_persons(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    conf: float = CONFIDENCE_THRESHOLD,
    iou: float = IOU_THRESHOLD,
) -> sv.Detections:
    results = yolo_model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAX_DET,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]
    return sv.Detections.from_ultralytics(results)


def draw_entry_polygon(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    label: str = "ENTRY_LINE",
) -> np.ndarray:
    annotated = frame_bgr.copy()
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(
        annotated,
        [pts],
        isClosed=True,
        color=ENTRY_POLYGON_COLOR,
        thickness=2,
    )
    for vertex in polygon:
        cv2.circle(
            annotated,
            (int(vertex[0]), int(vertex[1])),
            5,
            ENTRY_POLYGON_COLOR,
            -1,
            cv2.LINE_AA,
        )
    centroid = polygon.mean(axis=0).astype(int)
    cv2.putText(
        annotated,
        label,
        (int(centroid[0]), int(centroid[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        ENTRY_POLYGON_COLOR,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_stats_overlay(
    frame_bgr: np.ndarray,
    entry_count: int,
    exit_count: int,
    active_tracks: int,
    frame_idx: int,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    lines = [
        f"Entry Count: {entry_count}",
        f"Exit Count: {exit_count}",
        f"Active Tracks: {active_tracks}",
        f"Frame: {frame_idx}",
    ]
    x, y0, line_height = 20, 30, 28
    panel_h = line_height * len(lines) + 16
    cv2.rectangle(annotated, (10, 10), (320, 10 + panel_h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (x, y0 + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def annotate_detections(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    track_states: Dict[int, Tuple[str, Tuple[int, int]]],
) -> np.ndarray:
    annotated = frame_bgr.copy()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels: List[str] = []
    if detections.tracker_id is not None:
        for track_id in detections.tracker_id:
            tid = int(track_id)
            state, _ = track_states.get(tid, (STATE_OUTSIDE, (0, 0)))
            labels.append(f"ID {tid} | {state}")
    else:
        labels = ["ID ?"] * len(detections)

    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    if detections.tracker_id is not None:
        for track_id in detections.tracker_id:
            tid = int(track_id)
            _, center = track_states.get(tid, (STATE_OUTSIDE, (0, 0)))
            cv2.circle(annotated, center, 5, (0, 255, 255), -1, cv2.LINE_AA)

    return annotated


def process_frame_updates(
    engine: Any,
    event_emitter: EventEmitter,
    detections: sv.Detections,
    original_frame: int,
    reid_manager: Any | None = None,
    match_utc: Any | None = None,
) -> Dict[int, Tuple[str, Tuple[int, int]]]:
    track_states: Dict[int, Tuple[str, Tuple[int, int]]] = {}
    active_ids: set[int] = set()

    if detections.tracker_id is not None and len(detections) > 0:
        for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            tid = int(track_id)
            active_ids.add(tid)
            x1, y1, x2, y2 = xyxy
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if reid_manager is not None:
                reid_manager.apply_pending_retail_restore(
                    engine, tid, center, original_frame
                )
            result = engine.update(tid, center, original_frame=original_frame)
            track_states[tid] = (result.store_state, center)
            if result.init_debug is not None:
                event_emitter.log_init_terminal(result.init_debug.to_dict())
            if result.transition is not None:
                event_emitter.emit_transition(
                    result.transition.to_event_row(CAMERA_ID),
                    center=center,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    adjust_engine=engine,
                )
            if reid_manager is not None:
                reid_manager.sync_retail_snapshot_from_engine(
                    engine,
                    tid,
                    match_utc=match_utc,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                )

    if not engine.enable_track_loss_flush:
        return track_states

    for transition in engine.flush_removed_tracks(active_ids, original_frame):
        centroid = transition.debug.get("centroid")
        center = (
            (int(centroid[0]), int(centroid[1]))
            if isinstance(centroid, (list, tuple)) and len(centroid) >= 2
            else (0, 0)
        )
        event_emitter.emit_transition(
            transition.to_event_row(CAMERA_ID),
            center=center,
            bbox=transition.debug.get("bbox"),
            adjust_engine=engine,
        )

    return track_states


def print_converted_coordinates(
    entry_polygon: np.ndarray,
    video_width: int,
    video_height: int,
) -> None:
    coords = [(int(x), int(y)) for x, y in entry_polygon]
    print(f"\nConverted ENTRY_LINE_POLYGON ({video_width} x {video_height}):")
    print(f"  ENTRY_LINE: {coords}")


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _resolve_start_frame(
    *,
    cli_start_frame: int | None = None,
    env_name: str = "CAM3_START_FRAME",
) -> int:
    if cli_start_frame is not None:
        return max(0, cli_start_frame)
    return _env_int(env_name)


def _seek_to_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
    fps: float,
) -> int:
    """Seek to frame_index; return the frame index used for original_frame."""
    if frame_index <= 0:
        return 0

    ms = (frame_index / max(fps, 1e-6)) * 1000.0
    capture.set(cv2.CAP_PROP_POS_MSEC, ms)
    actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if actual < frame_index - 2:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

    if actual >= frame_index - 2:
        return frame_index

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for grabbed in range(frame_index):
        if not capture.grab():
            return grabbed
    return frame_index


def print_startup_config(
    *,
    footage2_mode: bool,
    video_path: Path,
    output_path: Path,
    start_frame: int = 0,
    fps: float = 30.0,
) -> None:
    print(f"Footage2 mode: {footage2_mode}")
    print(f"Video: {video_path.name}")
    print(f"Output: {output_path.name}")
    print(f"Video path: {video_path.resolve()}")
    print(f"Output path: {output_path.resolve()}")
    print(f"Process every N frames: {PROCESS_EVERY_N_FRAMES}")
    if start_frame > 0:
        print(
            f"Start frame: {start_frame} (~{start_frame / max(fps, 1e-6):.1f}s, "
            f"set CAM3_START_FRAME=0 to run from beginning)"
        )
    if os.getenv("CAM3_CROSSING_DEBUG") == "1":
        print("CAM3_CROSSING_DEBUG: enabled")


def process_video(
    video_path: Path,
    events_path: Path,
    yolo_model: YOLO,
    *,
    window_name: str,
    show_window: bool,
    clip_id: str | None = None,
    start_frame: int | None = None,
) -> Tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    requested_start = _resolve_start_frame(cli_start_frame=start_frame)
    start_frame = _seek_to_frame(capture, requested_start, fps)
    if requested_start > 0 and start_frame != requested_start:
        print(
            f"Warning: requested start frame {requested_start}, "
            f"decoder ready at {start_frame}"
        )

    print_startup_config(
        footage2_mode=_FOOTAGE2,
        video_path=video_path,
        output_path=events_path,
        start_frame=start_frame,
        fps=fps,
    )
    print(f"Video resolution: {video_width} x {video_height}")

    entry_polygon = denormalize_polygon(
        CAM3_LAYOUT["ENTRY_LINE_POLYGON"],
        video_width,
        video_height,
    )
    print_converted_coordinates(entry_polygon, video_width, video_height)

    engine = build_retail_entry_engine(
        CAM3_LAYOUT,
        video_width,
        video_height,
        timestamp_fn=lambda frame_idx, f=fps: format_video_timestamp(frame_idx, f),
    )
    entry_y_norm = float(CAM3_LAYOUT.get("ENTRY_PLANE_Y_NORM", 0.54))
    reid_manager: ReIDManager | None = None
    if CAM3_REID_ENABLED:
        entry_geometry = build_entry_line_geometry(
            CAM3_LAYOUT,
            video_width,
            video_height,
        )
        reid_manager = ReIDManager(
            embedder=OsnetEmbedder(),
            entry_geometry=entry_geometry,
        )
        print(
            f"CAM3 OSNet Re-ID: enabled (threshold={reid_manager.cosine_threshold}, "
            f"exit_cache={reid_manager.exit_cache_seconds}s, "
            f"session_match={reid_manager.session_match_enabled}, "
            f"session_threshold={reid_manager.session_match_threshold}, "
            f"heuristic_fallback={REID_HEURISTIC_FALLBACK})"
        )
    else:
        print("CAM3 OSNet Re-ID: disabled (CAM3_REID_ENABLED=0)")

    event_emitter = EventEmitter(
        events_path,
        CAMERA_ID,
        clip_id=clip_id,
        video_width=video_width,
        video_height=video_height,
        enable_reentry=True,
        entry_plane_y_norm=entry_y_norm,
        reid_manager=reid_manager,
        heuristic_reentry_fallback=REID_HEURISTIC_FALLBACK,
    )
    tracker = create_byte_tracker()

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    original_frame = start_frame
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if original_frame % PROCESS_EVERY_N_FRAMES != 0:
                original_frame += 1
                del frame
                continue

            detections = detect_persons(frame, yolo_model)
            detections = tracker.update_with_detections(detections)
            match_utc = None
            if reid_manager is not None:
                match_utc = ReIDManager.video_utc_for_frame(
                    CAMERA_ID,
                    original_frame,
                    fps,
                    clip_id=clip_id,
                )
                reid_manager.update_tracks(frame, detections, match_utc=match_utc)
            track_states = process_frame_updates(
                engine,
                event_emitter,
                detections,
                original_frame,
                reid_manager,
                match_utc=match_utc,
            )
            active_tracks = len(detections) if detections.tracker_id is not None else 0

            if show_window:
                annotated = draw_entry_polygon(frame, entry_polygon)
                annotated = annotate_detections(annotated, detections, track_states)
                annotated = draw_stats_overlay(
                    annotated,
                    engine.entry_count,
                    engine.exit_count,
                    active_tracks,
                    original_frame,
                )
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                del annotated

            del frame, detections, track_states
            original_frame += 1

        if engine.enable_track_loss_flush:
            last_frame = max(0, original_frame - 1)
            for transition in engine.flush_removed_tracks(set(), last_frame):
                centroid = transition.debug.get("centroid")
                center = (
                    (int(centroid[0]), int(centroid[1]))
                    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2
                    else (0, 0)
                )
                event_emitter.emit_transition(
                    transition.to_event_row(CAMERA_ID),
                    center=center,
                    adjust_engine=engine,
                )
    finally:
        capture.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"\nFinal Entry Count: {engine.entry_count}")
    print(f"Final Exit Count: {engine.exit_count}")
    rs = engine.recovery_stats
    print(
        f"Recovery stats: recovered_entries={rs.recovered_entries} "
        f"recovered_exits={rs.recovered_exits} "
        f"born_inside_tracks={rs.born_inside_tracks} "
        f"born_near_threshold_tracks={rs.born_near_threshold_tracks}"
    )
    event_emitter.print_summary_cam3()
    return engine.entry_count, engine.exit_count


def _parse_cli_args() -> Tuple[int | None, str | None]:
    import argparse

    parser = argparse.ArgumentParser(description="CAM3 entry/exit event generator")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First video frame to process (overrides CAM3_START_FRAME env)",
    )
    parser.add_argument(
        "--clip",
        choices=["entry1", "entry2"],
        default=None,
        help="Footage2 clip only (overrides CAM3_FOOTAGE2_CLIP env)",
    )
    args = parser.parse_args()
    return args.start_frame, args.clip


def main() -> None:
    cli_start_frame, cli_clip = _parse_cli_args()
    yolo_model = YOLO(str(MODEL_PATH))
    print("YOLO model loaded successfully")

    if _FOOTAGE2:
        show_window = os.getenv("CAM3_EVENTS_HEADLESS") != "1"
        clip_filter = (cli_clip or os.getenv("CAM3_FOOTAGE2_CLIP", "")).strip().lower()
        clips = list(FOOTAGE2_CLIPS)
        if clip_filter:
            clips = [c for c in FOOTAGE2_CLIPS if c[0].lower() == clip_filter]
            if not clips:
                known = ", ".join(c[0] for c in FOOTAGE2_CLIPS)
                raise ValueError(
                    f"Unknown CAM3_FOOTAGE2_CLIP={clip_filter!r}; choose: {known}"
                )
        results: List[Tuple[str, Path, int, int]] = []
        for clip_id, video_path, events_path in clips:
            if not video_path.exists():
                raise FileNotFoundError(f"Missing Footage2 clip: {video_path}")
            print("\n" + "=" * 60)
            print(f"Processing Footage2 clip: {clip_id}")
            entry_count, exit_count = process_video(
                video_path,
                events_path,
                yolo_model,
                window_name=f"Footage2 CAM3 Events ({clip_id})",
                show_window=show_window,
                clip_id=clip_id,
                start_frame=cli_start_frame,
            )
            results.append((clip_id, events_path, entry_count, exit_count))

        print("\n" + "=" * 60)
        print("Footage2 CAM3 summary (all clips)")
        for clip_id, events_path, entry_count, exit_count in results:
            print(
                f"  {clip_id}: ENTRY={entry_count} EXIT={exit_count} "
                f"-> {events_path.resolve()}"
            )
        return

    video_path = Path(CAMERA_VIDEO_FILES["CAM3"])
    events_path = OUTPUTS_DIR / "cam3_events.jsonl"
    process_video(
        video_path,
        events_path,
        yolo_model,
        window_name="CAM3 Entry Exit",
        show_window=True,
        start_frame=cli_start_frame,
    )


if __name__ == "__main__":
    main()
