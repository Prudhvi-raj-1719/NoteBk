"""Lightweight CAM1 validation: YOLO11m + ByteTrack zone assignment."""

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

from configs.camera_timing_config import CAMERA_VIDEO_FILES, DATA_DIR, MODEL_PATH

_FOOTAGE2 = os.getenv("VALIDATE_FOOTAGE2") == "1"
_FOOTAGE2_DIR = DATA_DIR / "CCTV Footage_2"

if _FOOTAGE2:
    VIDEO_PATH = _FOOTAGE2_DIR / "zone.mp4"
    WINDOW_NAME = "Footage2 Zone Validation"
else:
    VIDEO_PATH = CAMERA_VIDEO_FILES["CAM1"]
    WINDOW_NAME = "CAM1 Validation"

PERSON_CLASS_ID = 0
OUT_OF_ZONE = "OUT_OF_ZONE"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_OVERLAP_PCT = 10
PROCESS_EVERY_N_FRAMES = 10

_BRIGADE_CAM1_BRANDS: Dict[str, List[Tuple[float, float]]] = {
    "Minimalist_top": [
        (0.7323, 0.0391),
        (0.7298, 0.0651),
        (0.7338, 0.0712),
        (0.7557, 0.0712),
        (0.7821, 0.0625),
        (0.7885, 0.0538),
        (0.7518, 0.0382),
    ],
    "FarmStay": [
        (0.0010, 0.1302),
        (0.0642, 0.7743),
        (0.2168, 0.7077),
        (0.1835, 0.0696),
    ],
    "TheFaceShop": [
        (0.1950, 0.0588),
        (0.2173, 0.6821),
        (0.3727, 0.6024),
        (0.3749, 0.0265),
    ],
    "GoodVibes": [
        (0.3840, 0.0035),
        (0.3786, 0.6108),
        (0.5122, 0.5561),
        (0.5298, 0.0035),
    ],
    "DermaCo": [
        (0.5364, 0.0069),
        (0.5116, 0.5477),
        (0.6191, 0.4993),
        (0.6443, 0.0182),
    ],
    "Minimalist": [
        (0.6488, 0.0208),
        (0.7292, 0.0376),
        (0.6997, 0.4865),
        (0.6212, 0.5208),
    ],
    "Aquologica": [
        (0.7328, 0.0434),
        (0.7872, 0.0562),
        (0.7552, 0.4542),
        (0.7008, 0.4694),
    ],
    "Pilgrim": [
        (0.7945, 0.0557),
        (0.8380, 0.0701),
        (0.8065, 0.4345),
        (0.7589, 0.4470),
    ],
    "D&K": [
        (0.8389, 0.0820),
        (0.8743, 0.0883),
        (0.8446, 0.4173),
        (0.8101, 0.4378),
    ],
}

_FOOTAGE2_ZONE_BRANDS: Dict[str, List[Tuple[float, float]]] = {
    "GoodVibes": [
        (0.7832, 0.0194),
        (0.6825, 0.6882),
        (0.7860, 0.8263),
        (0.9638, 0.0697),
    ],
    "Pilgrim": [
        (0.0069, 0.0564),
        (0.0778, 0.8683),
        (0.2125, 0.7586),
        (0.0910, 0.0078),
    ],
    "Mamaearth": [
        (0.5293, 0.0065),
        (0.4958, 0.4646),
        (0.5759, 0.5661),
        (0.6381, 0.0518),
    ],
    "Cetaphil": [
        (0.9382, 0.2825),
        (0.8004, 0.8488),
        (0.9034, 0.9667),
        (0.9960, 0.6674),
    ],
    "Neutrogena": [
        (0.1209, 0.1564),
        (0.2159, 0.7619),
        (0.3476, 0.6465),
        (0.2798, 0.0307),
    ],
    "DandK": [
        (0.2797, 0.0185),
        (0.3459, 0.6514),
        (0.4241, 0.5763),
        (0.3781, 0.0048),
    ],
    "DermaComp": [
        (0.6407, 0.0219),
        (0.5799, 0.5739),
        (0.6742, 0.6872),
        (0.7780, 0.0403),
    ],
}

CAM1_BRANDS = _FOOTAGE2_ZONE_BRANDS if _FOOTAGE2 else _BRIGADE_CAM1_BRANDS

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
    print(f"Number of zones: {len(CAM1_BRANDS)}")
    print("Zone assignment mode: POLYGON_OVERLAP")
    print(f"Process every N frames: {PROCESS_EVERY_N_FRAMES}")

    yolo_model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")

    zone_polygons = {
        name: denormalize_polygon(points, video_width, video_height)
        for name, points in CAM1_BRANDS.items()
    }
    print_converted_coordinates(zone_polygons, video_width, video_height)

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
            track_zones = assign_track_zones(
                detections, zone_polygons, video_width, video_height
            )

            annotated = draw_zones(frame, zone_polygons)
            annotated = annotate_detections(
                annotated, detections, track_zones, original_frame
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

            del frame, annotated, detections, track_zones
            processed_frame += 1
            original_frame += 1
    finally:
        capture.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
