"""CAM2 visitor event generator: zone transitions from YOLO11m + ByteTrack."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from configs.camera_timing_config import (
    CAMERA_VIDEO_FILES,
    MODEL_PATH,
    OUTPUTS_DIR,
)
from events.event_emitter import EventEmitter

VIDEO_PATH = CAMERA_VIDEO_FILES["CAM2"]
WINDOW_NAME = "CAM2 Events"
EVENTS_PATH = OUTPUTS_DIR / "cam2_events.jsonl"
CAMERA_ID = "CAM2"

PERSON_CLASS_ID = 0
OUT_OF_ZONE = "OUT_OF_ZONE"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
MIN_OVERLAP_PCT = 10
LOST_TRACK_FRAMES = 30
MIN_ZONE_STABILITY_FRAMES = 15
MIN_DWELL_SECONDS = 2.0
PROCESS_EVERY_N_FRAMES = 10

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


def video_seconds(frame_index: int, fps: float) -> float:
    return frame_index / fps


def format_video_timestamp(frame_index: int, fps: float) -> str:
    total_seconds = video_seconds(frame_index, fps)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


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


@dataclass
class TrackState:
    current_zone: str = OUT_OF_ZONE
    zone_enter_time: Optional[float] = None
    last_seen_frame: int = 0
    pending_zone: Optional[str] = None
    pending_zone_start_frame: Optional[int] = None
    pending_consecutive_frames: int = 0


@dataclass
class StabilizationStats:
    ignored_zone_transitions: int = 0
    ignored_short_dwells: int = 0


class ZoneEventEngine:
    def __init__(
        self,
        event_logger: EventEmitter,
        fps: float,
        camera_id: str = CAMERA_ID,
    ) -> None:
        self.event_logger = event_logger
        self.fps = fps if fps > 0 else 30.0
        self.camera_id = camera_id
        self.track_states: Dict[int, TrackState] = {}
        self.stabilization = StabilizationStats()

    def _clear_pending(self, state: TrackState) -> None:
        state.pending_zone = None
        state.pending_zone_start_frame = None
        state.pending_consecutive_frames = 0

    def _reject_pending(
        self,
        track_id: int,
        state: TrackState,
        frame_idx: int,
    ) -> None:
        if state.pending_zone is None:
            return

        print(
            f"[DEBUG] Zone candidate rejected: track={track_id} "
            f"committed={state.current_zone} candidate={state.pending_zone} "
            f"frames={state.pending_consecutive_frames} frame={frame_idx}"
        )
        self.stabilization.ignored_zone_transitions += 1
        self._clear_pending(state)

    def frame_timestamp(self, frame_idx: int) -> str:
        return format_video_timestamp(frame_idx, self.fps)

    def frame_time_seconds(self, frame_idx: int) -> float:
        return video_seconds(frame_idx, self.fps)

    def _emit_zone_exit(self, track_id: int, zone: str, frame_idx: int) -> float:
        exit_time = self.frame_time_seconds(frame_idx)
        self.event_logger.emit_event(
            {
                "visitor_id": track_id,
                "camera": self.camera_id,
                "event_type": "ZONE_EXIT",
                "zone": zone,
                "timestamp": self.frame_timestamp(frame_idx),
            }
        )
        return exit_time

    def _emit_dwell_completed(
        self,
        track_id: int,
        zone: str,
        zone_enter_time: Optional[float],
        exit_time: float,
        frame_idx: int,
    ) -> None:
        dwell_seconds = 0.0
        if zone_enter_time is not None:
            dwell_seconds = max(0.0, exit_time - zone_enter_time)

        if dwell_seconds < MIN_DWELL_SECONDS:
            self.stabilization.ignored_short_dwells += 1
            print(
                f"[DEBUG] Ignored short dwell: track={track_id} zone={zone} "
                f"dwell={dwell_seconds:.1f}s (min={MIN_DWELL_SECONDS}s)"
            )
            return

        self.event_logger.emit_event(
            {
                "visitor_id": track_id,
                "camera": self.camera_id,
                "event_type": "DWELL_COMPLETED",
                "zone": zone,
                "dwell_seconds": round(dwell_seconds, 1),
                "timestamp": self.frame_timestamp(frame_idx),
            }
        )

    def _emit_zone_enter(self, track_id: int, zone: str, frame_idx: int) -> float:
        enter_time = self.frame_time_seconds(frame_idx)
        self.event_logger.emit_event(
            {
                "visitor_id": track_id,
                "camera": self.camera_id,
                "event_type": "ZONE_ENTER",
                "zone": zone,
                "timestamp": self.frame_timestamp(frame_idx),
            }
        )
        return enter_time

    def _leave_zone(
        self,
        track_id: int,
        state: TrackState,
        frame_idx: int,
    ) -> None:
        if state.current_zone == OUT_OF_ZONE:
            return

        exit_time = self._emit_zone_exit(track_id, state.current_zone, frame_idx)
        self._emit_dwell_completed(
            track_id,
            state.current_zone,
            state.zone_enter_time,
            exit_time,
            frame_idx,
        )
        state.current_zone = OUT_OF_ZONE
        state.zone_enter_time = None

    def _handle_zone_change(
        self,
        track_id: int,
        previous_zone: str,
        current_zone: str,
        frame_idx: int,
    ) -> None:
        state = self.track_states[track_id]

        if previous_zone != OUT_OF_ZONE:
            exit_time = self._emit_zone_exit(track_id, previous_zone, frame_idx)
            self._emit_dwell_completed(
                track_id,
                previous_zone,
                state.zone_enter_time,
                exit_time,
                frame_idx,
            )

        if current_zone != OUT_OF_ZONE:
            state.zone_enter_time = self._emit_zone_enter(
                track_id, current_zone, frame_idx
            )
        else:
            state.zone_enter_time = None

        state.current_zone = current_zone
        self._clear_pending(state)

    def _commit_zone_change(
        self,
        track_id: int,
        previous_zone: str,
        new_zone: str,
        frame_idx: int,
    ) -> None:
        print(
            f"[DEBUG] Zone candidate confirmed: track={track_id} "
            f"{previous_zone} -> {new_zone} frame={frame_idx}"
        )
        self._handle_zone_change(track_id, previous_zone, new_zone, frame_idx)

    def _process_detected_zone(
        self,
        track_id: int,
        detected_zone: str,
        frame_idx: int,
    ) -> None:
        state = self.track_states[track_id]
        committed_zone = state.current_zone

        if detected_zone == committed_zone:
            self._reject_pending(track_id, state, frame_idx)
            return

        if state.pending_zone != detected_zone:
            state.pending_zone = detected_zone
            state.pending_zone_start_frame = frame_idx
            state.pending_consecutive_frames = 1
            print(
                f"[DEBUG] Zone candidate detected: track={track_id} "
                f"{committed_zone} -> {detected_zone} frame={frame_idx}"
            )
            return

        state.pending_consecutive_frames += 1
        if state.pending_consecutive_frames >= MIN_ZONE_STABILITY_FRAMES:
            self._commit_zone_change(
                track_id,
                committed_zone,
                detected_zone,
                frame_idx,
            )

    def update_active_tracks(
        self,
        frame_idx: int,
        track_zones: Dict[int, Tuple[str, int]],
    ) -> None:
        active_ids = set(track_zones.keys())

        for track_id, (detected_zone, _) in track_zones.items():
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(last_seen_frame=frame_idx)

            state = self.track_states[track_id]
            self._process_detected_zone(track_id, detected_zone, frame_idx)
            state.last_seen_frame = frame_idx

        lost_track_ids = [
            track_id
            for track_id, state in self.track_states.items()
            if track_id not in active_ids
            and frame_idx - state.last_seen_frame > LOST_TRACK_FRAMES
        ]
        for track_id in lost_track_ids:
            state = self.track_states[track_id]
            self._clear_pending(state)
            self._leave_zone(track_id, state, frame_idx)
            del self.track_states[track_id]

    def finalize(self, frame_idx: int) -> None:
        remaining_ids = list(self.track_states.keys())
        for track_id in remaining_ids:
            state = self.track_states[track_id]
            self._clear_pending(state)
            self._leave_zone(track_id, state, frame_idx)
            del self.track_states[track_id]

    def get_dwell_seconds(self, track_id: int, frame_idx: int) -> float:
        state = self.track_states.get(track_id)
        if state is None or state.current_zone == OUT_OF_ZONE:
            return 0.0
        if state.zone_enter_time is None:
            return 0.0
        return max(0.0, self.frame_time_seconds(frame_idx) - state.zone_enter_time)


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


def draw_track_labels(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    track_id: int,
    zone: str,
    dwell_seconds: float,
) -> None:
    lines = [
        f"ID {track_id}",
        f"Zone: {zone}",
        f"Dwell: {dwell_seconds:.1f}s",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    line_height = 18
    y_offset = y1 - 8 - (len(lines) - 1) * line_height
    if y_offset < 0:
        y_offset = y1 + 20

    for i, line in enumerate(lines):
        y = y_offset + i * line_height
        cv2.putText(
            frame_bgr,
            line,
            (x1, y),
            font,
            scale,
            (255, 255, 255),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (x1, y),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def annotate_detections(
    frame_bgr: np.ndarray,
    detections: sv.Detections,
    event_engine: ZoneEventEngine,
    original_frame: int,
    processed_frame: int,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    box_annotator = sv.BoxAnnotator(thickness=2)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    if detections.tracker_id is not None:
        for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            tid = int(track_id)
            state = event_engine.track_states.get(tid)
            zone = state.current_zone if state else OUT_OF_ZONE
            dwell_seconds = event_engine.get_dwell_seconds(tid, original_frame)
            x1, y1, _, _ = xyxy.astype(int)
            draw_track_labels(annotated, x1, y1, tid, zone, dwell_seconds)

    video_time_seconds = event_engine.frame_time_seconds(original_frame)
    overlay_lines = [
        f"processed_frame: {processed_frame}",
        f"original_frame: {original_frame}",
        f"video_time_seconds: {video_time_seconds:.2f}",
    ]
    for i, text in enumerate(overlay_lines):
        cv2.putText(
            annotated,
            text,
            (20, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    print(f"Video resolution: {video_width} x {video_height}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Number of zones: {len(CAM2_BRANDS)}")
    print("Zone assignment mode: POLYGON_OVERLAP")
    print(f"Zone stability frames: {MIN_ZONE_STABILITY_FRAMES}")
    print(f"Minimum dwell seconds: {MIN_DWELL_SECONDS}")
    print(f"Process every N frames: {PROCESS_EVERY_N_FRAMES}")

    yolo_model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")

    zone_polygons = {
        name: denormalize_polygon(points, video_width, video_height)
        for name, points in CAM2_BRANDS.items()
    }
    print_converted_coordinates(zone_polygons, video_width, video_height)

    event_emitter = EventEmitter(EVENTS_PATH, CAMERA_ID)
    event_engine = ZoneEventEngine(event_emitter, fps=fps)
    tracker = create_byte_tracker()
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
                continue

            detections = detect_persons(frame, yolo_model)
            detections = tracker.update_with_detections(detections)
            track_zones = assign_track_zones(
                detections, zone_polygons, video_width, video_height
            )
            event_engine.update_active_tracks(original_frame, track_zones)

            annotated = draw_zones(frame, zone_polygons)
            annotated = annotate_detections(
                annotated,
                detections,
                event_engine,
                original_frame,
                processed_frame,
            )

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            processed_frame += 1
            original_frame += 1
    finally:
        event_engine.finalize(original_frame)
        capture.release()
        cv2.destroyAllWindows()
        event_emitter.print_summary_cam1(event_engine.stabilization)


if __name__ == "__main__":
    main()
