"""Lightweight CAM5 validation: YOLO11m + ByteTrack payment area zone assignment."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from configs.camera_timing_config import CAMERA_VIDEO_FILES, MODEL_PATH

VIDEO_PATH = CAMERA_VIDEO_FILES["CAM5"]
WINDOW_NAME = "CAM5 Validation"

PERSON_CLASS_ID = 0
OUT_OF_ZONE = "OUT_OF_ZONE"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_OVERLAP_PCT = 10

ANNOTATION_WIDTH = 1056
ANNOTATION_HEIGHT = 629

CAM5_ZONES: Dict[str, List[Tuple[float, float]]] = {
    "BillingQueue": [
        (0.0016, 0.1970),
        (0.2625, 0.1934),
        (0.0142, 0.9196),
        (0.0000, 0.7044),
    ],
    "PaymentArea": [
        (0.0763, 0.7326),
        (0.1089, 0.7273),
        (0.2531, 0.2871),
        (0.2267, 0.2871),
        (0.1594, 0.4869),
    ],
}

ZONE_PRIORITY: Tuple[str, ...] = ("PaymentArea", "BillingQueue")

ZONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "PaymentArea": (0, 255, 255),
    "BillingQueue": (255, 0, 0),
}
CENTER_POINT_COLOR: Tuple[int, int, int] = (255, 255, 0)
BOTTOM_CENTER_COLOR: Tuple[int, int, int] = (0, 0, 255)


def denormalize_polygon(
    points: List[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [(int(x * width), int(y * height)) for x, y in points],
        dtype=np.int32,
    )


def build_zone_polygons(video_width: int, video_height: int) -> Dict[str, np.ndarray]:
    return {
        name: denormalize_polygon(points, video_width, video_height)
        for name, points in CAM5_ZONES.items()
    }


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
        int((x1 + x2) / 2),
        int((y1 + y2) / 2),
    )


def get_bottom_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
    return (
        int((x1 + x2) / 2),
        int(y2),
    )


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


def resolve_zone_by_priority(
    xyxy: np.ndarray,
    zone_polygons: Dict[str, np.ndarray],
    frame_width: int,
    frame_height: int,
) -> str:
    inside_payment = (
        bbox_polygon_overlap_pct(
            xyxy, zone_polygons["PaymentArea"], frame_width, frame_height
        )
        >= MIN_OVERLAP_PCT
    )
    inside_queue = (
        bbox_polygon_overlap_pct(
            xyxy, zone_polygons["BillingQueue"], frame_width, frame_height
        )
        >= MIN_OVERLAP_PCT
    )

    if inside_payment:
        return "PaymentArea"
    if inside_queue:
        return "BillingQueue"
    return OUT_OF_ZONE


def assign_track_zones(
    detections: sv.Detections,
    zone_polygons: Dict[str, np.ndarray],
    frame_width: int,
    frame_height: int,
) -> Dict[int, str]:
    if detections.tracker_id is None or len(detections) == 0:
        return {}

    track_zones: Dict[int, str] = {}
    for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
        tid = int(track_id)
        track_zones[tid] = resolve_zone_by_priority(
            xyxy, zone_polygons, frame_width, frame_height
        )
    return track_zones


def get_track_anchor_points(
    detections: sv.Detections,
) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
    anchor_points: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    if detections.tracker_id is None or len(detections) == 0:
        return anchor_points

    for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
        tid = int(track_id)
        x1, y1, x2, y2 = xyxy
        center = get_bbox_center(x1, y1, x2, y2)
        bottom_center = get_bottom_center(x1, y1, x2, y2)
        anchor_points[tid] = (center, bottom_center)
    return anchor_points


def draw_polygon(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=3)
    for index, vertex in enumerate(polygon):
        x, y = int(vertex[0]), int(vertex[1])
        cv2.circle(annotated, (x, y), 8, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"{index}",
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"{index}",
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"({x},{y})",
            (x + 12, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"({x},{y})",
            (x + 12, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    centroid = polygon.mean(axis=0).astype(int)
    cv2.putText(
        annotated,
        label,
        (int(centroid[0]), int(centroid[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (int(centroid[0]), int(centroid[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_zones(frame_bgr: np.ndarray, zone_polygons: Dict[str, np.ndarray]) -> np.ndarray:
    annotated = frame_bgr.copy()
    draw_order = ("BillingQueue", "PaymentArea")
    for zone_name in draw_order:
        polygon = zone_polygons[zone_name]
        color = ZONE_COLORS[zone_name]
        annotated = draw_polygon(annotated, polygon, color=color, label=zone_name)
    return annotated


def annotate_detections(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    track_zones: Dict[int, str],
    anchor_points: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    frame_idx: int,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels: List[str] = []
    if detections.tracker_id is not None:
        for track_id in detections.tracker_id:
            tid = int(track_id)
            zone = track_zones.get(tid, OUT_OF_ZONE)
            labels.append(f"ID {tid} | Zone: {zone}")
    else:
        labels = ["ID ?"] * len(detections)

    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    for tid, (center, bottom_center) in anchor_points.items():
        cv2.circle(annotated, center, 5, CENTER_POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(annotated, bottom_center, 5, BOTTOM_CENTER_COLOR, -1, cv2.LINE_AA)

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
    cv2.putText(
        annotated,
        "Assignment: OVERLAP (PaymentArea priority)",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Yellow = center | Red = bottom-center",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def print_zone_debug(
    zone_polygons: Dict[str, np.ndarray],
    video_width: int,
    video_height: int,
) -> None:
    print(f"\nAnnotation reference: {ANNOTATION_WIDTH} x {ANNOTATION_HEIGHT}")
    print("\nNormalized coordinates:")
    for zone_name in ("BillingQueue", "PaymentArea"):
        print(f"  {zone_name}: {CAM5_ZONES[zone_name]}")
    print("\nDenormalized coordinates:")
    for zone_name in ("BillingQueue", "PaymentArea"):
        coords = [(int(x), int(y)) for x, y in zone_polygons[zone_name]]
        print(f"  {zone_name}: {coords}")
    print("\nZone priority order:")
    for index, zone_name in enumerate(ZONE_PRIORITY, start=1):
        print(f"  {index}. {zone_name}")
    print(f"  {len(ZONE_PRIORITY) + 1}. {OUT_OF_ZONE}")
    print("\nPaymentArea loaded")
    print("BillingQueue loaded")


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {VIDEO_PATH}")

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {video_width} x {video_height}")
    print(f"Number of zones: {len(CAM5_ZONES)}")
    print("Zone assignment mode: OVERLAP_PRIORITY (PaymentArea > BillingQueue)")

    yolo_model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")

    zone_polygons = build_zone_polygons(video_width, video_height)
    print_zone_debug(zone_polygons, video_width, video_height)

    tracker = create_byte_tracker()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            detections = detect_persons(frame, yolo_model)
            detections = tracker.update_with_detections(detections)
            track_zones = assign_track_zones(
                detections, zone_polygons, video_width, video_height
            )
            anchor_points = get_track_anchor_points(detections)

            annotated = draw_zones(frame, zone_polygons)
            annotated = annotate_detections(
                annotated, detections, track_zones, anchor_points, frame_idx
            )

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
