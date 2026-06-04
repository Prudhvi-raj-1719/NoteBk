"""Lightweight CAM3 validation: YOLO11m + ByteTrack + RetailEntryEngine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, List, Tuple

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
)
from entry_retail import (
    RetailEntryEngine,
    STATE_INSIDE,
    STATE_OUTSIDE,
    build_retail_entry_engine,
)

_FOOTAGE2 = os.getenv("VALIDATE_FOOTAGE2") == "1"
_FOOTAGE2_DIR = DATA_DIR / "CCTV Footage_2"

if _FOOTAGE2:
    _entry_override = os.getenv("VALIDATION_VIDEO")
    if _entry_override:
        VIDEO_PATH = Path(_entry_override)
    else:
        VIDEO_PATH = _FOOTAGE2_DIR / "entry 1.mp4"
    WINDOW_NAME = "Footage2 Entry Validation"
    CAM3_LAYOUT = CAM3_ENTRY_CONFIG["FOOTAGE2"]
else:
    VIDEO_PATH = CAMERA_VIDEO_FILES["CAM3"]
    WINDOW_NAME = "CAM3 Validation"
    CAM3_LAYOUT = CAM3_ENTRY_CONFIG["BRIGADE"]

PROCESS_EVERY_N_FRAMES = CAM3_PROCESS_EVERY_N_FRAMES

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
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
    return sv.ByteTrack()


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
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]
    return sv.Detections.from_ultralytics(results)


def update_track_states(
    engine: RetailEntryEngine,
    detections: sv.Detections,
    original_frame: int,
) -> Dict[int, Tuple[str, Tuple[int, int]]]:
    track_states: Dict[int, Tuple[str, Tuple[int, int]]] = {}
    if detections.tracker_id is None or len(detections) == 0:
        return track_states

    for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
        tid = int(track_id)
        x1, y1, x2, y2 = xyxy
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        result = engine.update(tid, center, original_frame=original_frame)
        track_states[tid] = (result.store_state, center)
        if result.init_debug is not None:
            d = result.init_debug.to_dict()
            print(f"[INIT] visitor_id={d['visitor_id']} initial_side={d['initial_side']} "
                  f"initial_store_state={d['initial_store_state']}")
        if result.transition is not None:
            row = result.transition.to_event_row("CAM3")
            print(
                f"[{row['event_type']}] visitor_id={row['visitor_id']} "
                f"timestamp={row['timestamp']} debug={row['debug']}"
            )

    return track_states


def draw_entry_polygon(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    label: str = "vline" if _FOOTAGE2 else "ENTRY_LINE",
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


def annotate_detections(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    track_states: Dict[int, Tuple[str, Tuple[int, int]]],
    frame_idx: int,
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

    cv2.putText(
        annotated,
        f"Frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def print_converted_coordinates(
    entry_polygon: np.ndarray,
    video_width: int,
    video_height: int,
) -> None:
    coords = [(int(x), int(y)) for x, y in entry_polygon]
    print(f"\nConverted ENTRY_LINE_POLYGON ({video_width} x {video_height}):")
    print(f"  ENTRY_LINE: {coords}")


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {VIDEO_PATH}")

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {video_width} x {video_height}")
    print(f"Process every N frames: {PROCESS_EVERY_N_FRAMES}")
    print(f"Footage2 validation: {_FOOTAGE2}")

    yolo_model = YOLO(str(MODEL_PATH))
    print("YOLO model loaded successfully")

    entry_polygon = denormalize_polygon(
        CAM3_LAYOUT["ENTRY_LINE_POLYGON"],
        video_width,
        video_height,
    )
    print_converted_coordinates(entry_polygon, video_width, video_height)

    engine = build_retail_entry_engine(CAM3_LAYOUT, video_width, video_height)
    tracker = create_byte_tracker()
    headless = os.getenv("VALIDATION_HEADLESS") == "1"
    save_path = os.getenv("VALIDATION_SAVE_PATH")
    save_frame = int(os.getenv("VALIDATION_SAVE_FRAME", "0"))
    save_only = os.getenv("VALIDATION_SAVE_ONLY") == "1"
    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    original_frame = 0
    processed_frame = 0
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
            track_states = update_track_states(engine, detections, original_frame)

            annotated = draw_entry_polygon(frame, entry_polygon)
            annotated = annotate_detections(
                annotated, detections, track_states, original_frame
            )

            if save_path and original_frame == save_frame:
                out = Path(save_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), annotated)
                print(f"Saved validation preview: {out.resolve()}")
                if save_only:
                    break

            if not headless:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            del frame, annotated, detections, track_states
            processed_frame += 1
            original_frame += 1
    finally:
        capture.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"Processed frames: {processed_frame}")
        print(f"ENTRY count: {engine.entry_count}  EXIT count: {engine.exit_count}")


if __name__ == "__main__":
    main()
