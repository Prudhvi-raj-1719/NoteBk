"""CAM3 entry/exit counter: YOLO11m + ByteTrack doorway transition counting."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from configs.camera_timing_config import CAMERA_VIDEO_FILES, MODEL_PATH

VIDEO_PATH = str(CAMERA_VIDEO_FILES["CAM3"])
WINDOW_NAME = "CAM3 Entry Exit"

PERSON_CLASS_ID = 0
INSIDE_DOORWAY = "INSIDE_DOORWAY"
OUTSIDE_DOORWAY = "OUTSIDE_DOORWAY"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

CAM3_CONFIG = {
    "ENTRY_LINE_POLYGON": [
        (0.5950, 0.4444),
        (0.4498, 0.7510),
        (0.4714, 0.7759),
        (0.6121, 0.4566),
    ],
}

ENTRY_POLYGON_COLOR: Tuple[int, int, int] = (0, 0, 255)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class TrackRecord(TypedDict):
    previous_state: str
    current_state: str
    counted: bool


def denormalize_polygon(
    points: List[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [
            (int(x * width), int(y * height))
            for x, y in points
        ],
        dtype=np.int32,
    )


def create_byte_tracker() -> sv.ByteTrack:
    """Create a ByteTrack multi-object tracker."""
    return sv.ByteTrack()


def detect_persons(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    conf: float = CONFIDENCE_THRESHOLD,
    iou: float = IOU_THRESHOLD,
) -> sv.Detections:
    """Run YOLO and return person-only detections (class 0)."""
    results = yolo_model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]
    return sv.Detections.from_ultralytics(results)


def get_bbox_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
    return (
        (int(x1) + int(x2)) // 2,
        (int(y1) + int(y2)) // 2,
    )


def classify_doorway_state(
    center: Tuple[int, int],
    entry_polygon: np.ndarray,
) -> str:
    inside_polygon = cv2.pointPolygonTest(entry_polygon, center, False)
    if inside_polygon >= 0:
        return INSIDE_DOORWAY
    return OUTSIDE_DOORWAY


class EntryExitCounter:
    """Count doorway entry/exit transitions per track without double counting."""

    def __init__(self) -> None:
        self.track_history: Dict[int, TrackRecord] = {}
        self.entry_count = 0
        self.exit_count = 0

    def update_track(
        self,
        track_id: int,
        center: Tuple[int, int],
        entry_polygon: np.ndarray,
    ) -> str:
        current_state = classify_doorway_state(center, entry_polygon)

        if track_id not in self.track_history:
            self.track_history[track_id] = {
                "previous_state": current_state,
                "current_state": current_state,
                "counted": False,
            }
            return current_state

        record = self.track_history[track_id]
        previous_state = record["current_state"]
        record["previous_state"] = previous_state
        record["current_state"] = current_state

        if previous_state != current_state:
            if previous_state == OUTSIDE_DOORWAY and current_state == INSIDE_DOORWAY:
                self.entry_count += 1
                record["counted"] = True
                logger.info("ENTRY detected -> Track %s", track_id)
            elif previous_state == INSIDE_DOORWAY and current_state == OUTSIDE_DOORWAY:
                self.exit_count += 1
                record["counted"] = True
                logger.info("EXIT detected -> Track %s", track_id)
            else:
                record["counted"] = False
        else:
            record["counted"] = False

        return current_state

    def process_detections(
        self,
        detections: sv.Detections,
        entry_polygon: np.ndarray,
    ) -> Dict[int, Tuple[str, Tuple[int, int]]]:
        track_states: Dict[int, Tuple[str, Tuple[int, int]]] = {}
        if detections.tracker_id is None or len(detections) == 0:
            return track_states

        for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            tid = int(track_id)
            x1, y1, x2, y2 = xyxy
            center = get_bbox_center(x1, y1, x2, y2)
            state = self.update_track(tid, center, entry_polygon)
            track_states[tid] = (state, center)

        return track_states


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
            state, _ = track_states.get(tid, (OUTSIDE_DOORWAY, (0, 0)))
            labels.append(f"ID {tid} | {state}")
    else:
        labels = ["ID ?"] * len(detections)

    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    if detections.tracker_id is not None:
        for track_id in detections.tracker_id:
            tid = int(track_id)
            _, center = track_states.get(tid, (OUTSIDE_DOORWAY, (0, 0)))
            cv2.circle(
                annotated,
                center,
                5,
                (0, 255, 255),
                -1,
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
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {VIDEO_PATH}")

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {video_width} x {video_height}")

    yolo_model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")

    entry_polygon = denormalize_polygon(
        CAM3_CONFIG["ENTRY_LINE_POLYGON"],
        video_width,
        video_height,
    )
    print_converted_coordinates(entry_polygon, video_width, video_height)

    tracker = create_byte_tracker()
    counter = EntryExitCounter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            detections = detect_persons(frame, yolo_model)
            detections = tracker.update_with_detections(detections)
            track_states = counter.process_detections(detections, entry_polygon)
            active_tracks = len(detections) if detections.tracker_id is not None else 0

            annotated = draw_entry_polygon(frame, entry_polygon)
            annotated = annotate_detections(annotated, detections, track_states)
            annotated = draw_stats_overlay(
                annotated,
                counter.entry_count,
                counter.exit_count,
                active_tracks,
                frame_idx,
            )

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_idx += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print(f"\nFinal Entry Count: {counter.entry_count}")
        print(f"Final Exit Count: {counter.exit_count}")


if __name__ == "__main__":
    main()
