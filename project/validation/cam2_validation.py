"""Lightweight CAM2 validation: YOLO11m + ByteTrack zone assignment."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from configs.camera_timing_config import CAMERA_VIDEO_FILES, MODEL_PATH

VIDEO_PATH = CAMERA_VIDEO_FILES["CAM2"]
WINDOW_NAME = "CAM2 Validation"

PERSON_CLASS_ID = 0
OUT_OF_ZONE = "OUT_OF_ZONE"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_OVERLAP_PCT = 5
ZONE_HISTORY_LEN = 15

CAM2_BRANDS: Dict[str, List[Tuple[float, float]]] = {
    "MAYBELLINE": [
        (0.8453, 0.1821),
        (0.7621, 0.8552),
        (0.8973, 0.9500),
        (0.9972, 0.2425),
    ],
    "FACESCANADA": [
        (0.6370, 0.7181),
        (0.6663, 0.1349),
        (0.8210, 0.1721),
        (0.7643, 0.8226),
    ],
    "LAKME": [
        (0.5451, 0.1182),
        (0.6663, 0.1326),
        (0.6349, 0.7195),
        (0.5261, 0.6371),
    ],
    "swiss Beauty": [
        (0.4515, 0.1114),
        (0.4438, 0.5596),
        (0.5215, 0.6198),
        (0.5433, 0.1128),
    ],
    "MARS": [
        (0.3897, 0.1062),
        (0.4536, 0.1134),
        (0.4494, 0.5348),
        (0.3778, 0.4920),
    ],
    "ALPS": [
        (0.3369, 0.1060),
        (0.3887, 0.1051),
        (0.3802, 0.4866),
        (0.3350, 0.4537),
    ],
    "LOREAL": [
        (0.2988, 0.1073),
        (0.3377, 0.1058),
        (0.3346, 0.4502),
        (0.2994, 0.4165),
    ],
    "EASTIND": [
        (0.2769, 0.1088),
        (0.2990, 0.1078),
        (0.2982, 0.4100),
        (0.2768, 0.3905),
    ],
}

ZONE_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 255),
    (255, 128, 0),
    (0, 200, 0),
    (255, 0, 255),
    (0, 140, 255),
    (200, 200, 0),
    (255, 100, 100),
    (100, 255, 100),
    (180, 100, 255),
]


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


def bbox_polygon_overlap_pct(
    xyxy: np.ndarray,
    polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> float:
    x1, y1, x2, y2 = xyxy
    bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    if bbox_area == 0:
        return 0.0

    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(frame_width, int(x2))
    iy2 = min(frame_height, int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    crop_w = ix2 - ix1
    crop_h = iy2 - iy1
    bbox_mask = np.full((crop_h, crop_w), 255, dtype=np.uint8)

    poly_shifted = polygon.copy()
    poly_shifted[:, 0] -= ix1
    poly_shifted[:, 1] -= iy1
    poly_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly_shifted.reshape(-1, 1, 2)], 255)

    overlap_pixels = cv2.countNonZero(cv2.bitwise_and(bbox_mask, poly_mask))
    return (float(overlap_pixels) / bbox_area) * 100.0


def resolve_zone_by_overlap(
    xyxy: np.ndarray,
    zone_polygons: Dict[str, np.ndarray],
    frame_width: int,
    frame_height: int,
) -> Tuple[str, int]:
    best_zone = OUT_OF_ZONE
    best_pct = 0.0
    for zone_name, polygon in zone_polygons.items():
        overlap_pct = bbox_polygon_overlap_pct(xyxy, polygon, frame_width, frame_height)
        if overlap_pct > best_pct:
            best_pct = overlap_pct
            best_zone = zone_name

    rounded_pct = int(round(best_pct))
    if best_pct < MIN_OVERLAP_PCT:
        return OUT_OF_ZONE, rounded_pct
    return best_zone, rounded_pct


def assign_track_zones(
    detections: sv.Detections,
    zone_polygons: Dict[str, np.ndarray],
    frame_width: int,
    frame_height: int,
) -> Dict[int, Tuple[str, int]]:
    if detections.tracker_id is None or len(detections) == 0:
        return {}

    track_zones: Dict[int, Tuple[str, int]] = {}
    for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
        tid = int(track_id)
        track_zones[tid] = resolve_zone_by_overlap(
            xyxy, zone_polygons, frame_width, frame_height
        )
    return track_zones


class ZoneHistory:
    """Keep per-track zone history and return the most frequent recent zone."""

    def __init__(self, maxlen: int = ZONE_HISTORY_LEN) -> None:
        self._histories: Dict[int, Deque[str]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )

    def update(self, track_id: int, zone: str) -> None:
        self._histories[track_id].append(zone)

    def smoothed_zone(self, track_id: int) -> str:
        history = self._histories.get(track_id)
        if not history:
            return OUT_OF_ZONE
        return Counter(history).most_common(1)[0][0]

    def apply_smoothing(
        self,
        frame_zones: Dict[int, Tuple[str, int]],
    ) -> Dict[int, Tuple[str, int]]:
        smoothed: Dict[int, Tuple[str, int]] = {}
        for track_id, (zone, overlap_pct) in frame_zones.items():
            self.update(track_id, zone)
            smoothed[track_id] = (self.smoothed_zone(track_id), overlap_pct)
        return smoothed


def draw_polygon(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
    for vertex in polygon:
        cv2.circle(
            annotated,
            (int(vertex[0]), int(vertex[1])),
            5,
            color,
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
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_zones(frame_bgr: np.ndarray, zone_polygons: Dict[str, np.ndarray]) -> np.ndarray:
    annotated = frame_bgr.copy()
    for i, (zone_name, polygon) in enumerate(zone_polygons.items()):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        annotated = draw_polygon(annotated, polygon, color=color, label=zone_name)
    return annotated


def annotate_detections(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    track_zones: Dict[int, Tuple[str, int]],
    frame_idx: int,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels: List[str] = []
    if detections.tracker_id is not None:
        for track_id in detections.tracker_id:
            tid = int(track_id)
            zone, overlap_pct = track_zones.get(tid, (OUT_OF_ZONE, 0))
            labels.append(f"ID {tid} | {zone} | {overlap_pct}%")
    else:
        labels = ["ID ?"] * len(detections)

    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

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
    zone_polygons: Dict[str, np.ndarray],
    video_width: int,
    video_height: int,
) -> None:
    print(f"\nConverted zone coordinates ({video_width} x {video_height}):")
    for zone_name, polygon in zone_polygons.items():
        coords = [(int(x), int(y)) for x, y in polygon]
        print(f"  {zone_name}: {coords}")


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {VIDEO_PATH}")

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {video_width} x {video_height}")
    print(f"Number of zones: {len(CAM2_BRANDS)}")
    print("Zone assignment mode: POLYGON_OVERLAP")
    print(f"Temporal smoothing window: {ZONE_HISTORY_LEN} frames")

    yolo_model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")

    zone_polygons = {
        name: denormalize_polygon(points, video_width, video_height)
        for name, points in CAM2_BRANDS.items()
    }
    print_converted_coordinates(zone_polygons, video_width, video_height)

    tracker = create_byte_tracker()
    zone_history = ZoneHistory(maxlen=ZONE_HISTORY_LEN)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            detections = detect_persons(frame, yolo_model)
            detections = tracker.update_with_detections(detections)
            frame_zones = assign_track_zones(
                detections, zone_polygons, video_width, video_height
            )
            track_zones = zone_history.apply_smoothing(frame_zones)

            annotated = draw_zones(frame, zone_polygons)
            annotated = annotate_detections(annotated, detections, track_zones, frame_idx)

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_idx += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
