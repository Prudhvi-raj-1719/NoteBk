"""Retail entry/exit engine: half-plane classification + latched store state."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from configs.camera_timing_config import (
    ENTRY_STABILITY_FRAMES,
    EXIT_STABILITY_FRAMES,
)

SIDE_MALL = "MALL"
SIDE_STORE = "STORE"
STATE_INSIDE = "INSIDE"
STATE_OUTSIDE = "OUTSIDE"
STATE_UNKNOWN = "UNKNOWN"
EVENT_ENTRY = "ENTRY"
EVENT_EXIT = "EXIT"


def format_video_timestamp(frame_index: int, fps: float) -> str:
    """Video offset HH:MM:SS.mmm from frame index and FPS."""
    fps = fps if fps > 0 else 30.0
    total_seconds = frame_index / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _midpoint(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def build_entry_line_geometry(
    layout: Dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> "EntryLineGeometry":
    """Construct geometry from a CAM3_ENTRY_CONFIG layout dict."""
    return EntryLineGeometry(
        layout["ENTRY_LINE_POLYGON"],
        frame_width,
        frame_height,
        layout["STORE_REF"],
        entry_line_style=layout.get("ENTRY_LINE_STYLE", "quad_lr"),
        entry_plane_y_norm=layout.get("ENTRY_PLANE_Y_NORM"),
        invert_store_half_plane=layout.get("INVERT_STORE_HALF_PLANE", False),
    )


def build_retail_entry_engine(
    layout: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    *,
    timestamp_fn: Optional[Callable[[int], str]] = None,
) -> "RetailEntryEngine":
    """Geometry + engine from a CAM3_ENTRY_CONFIG layout dict."""
    geometry = build_entry_line_geometry(layout, frame_width, frame_height)
    return RetailEntryEngine(
        geometry,
        timestamp_fn=timestamp_fn,
        invert_retail_semantics=layout.get("INVERT_RETAIL_SEMANTICS", False),
        entry_stability_frames=layout.get("ENTRY_STABILITY_FRAMES", ENTRY_STABILITY_FRAMES),
        exit_stability_frames=layout.get("EXIT_STABILITY_FRAMES", EXIT_STABILITY_FRAMES),
        enable_threshold_recovery=layout.get("ENABLE_THRESHOLD_RECOVERY", False),
        near_threshold_y_tolerance=layout.get("NEAR_THRESHOLD_Y_TOLERANCE", 0.10),
        threshold_observation_frames=layout.get("THRESHOLD_OBSERVATION_FRAMES", 3),
        threshold_motion_eps=layout.get("THRESHOLD_MOTION_EPS", 0.015),
        enable_track_loss_flush=layout.get("ENABLE_TRACK_LOSS_FLUSH", False),
        track_loss_grace_frames=layout.get("TRACK_LOSS_GRACE_FRAMES", 20),
        track_loss_motion_eps=layout.get("TRACK_LOSS_MOTION_EPS", 0.5),
        track_loss_flush_require_stable_pending=layout.get(
            "TRACK_LOSS_FLUSH_REQUIRE_STABLE_PENDING", True
        ),
        enable_late_entry_recovery=layout.get("ENABLE_LATE_ENTRY_RECOVERY", False),
        late_entry_y_band_below=layout.get("LATE_ENTRY_Y_BAND_BELOW", 0.20),
        late_entry_require_near_threshold=layout.get(
            "LATE_ENTRY_REQUIRE_NEAR_THRESHOLD", False
        ),
    )


class EntryLineGeometry:
    """Entry line A→B; STORE side calibrated with STORE_REF."""

    def __init__(
        self,
        polygon_norm: List[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
        store_ref_norm: Tuple[float, float],
        *,
        entry_line_style: str = "quad_lr",
        entry_plane_y_norm: Optional[float] = None,
        invert_store_half_plane: bool = False,
    ) -> None:
        if len(polygon_norm) != 4:
            raise ValueError("ENTRY_LINE_POLYGON must have exactly 4 vertices")

        self.entry_line_style = entry_line_style
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.store_ref_norm = store_ref_norm
        self.store_ref_px = (
            store_ref_norm[0] * frame_width,
            store_ref_norm[1] * frame_height,
        )

        if entry_line_style == "horizontal_y":
            if entry_plane_y_norm is None:
                ys = [y for _, y in polygon_norm]
                entry_plane_y_norm = (min(ys) + max(ys)) / 2.0
            y_sep = entry_plane_y_norm * frame_height
            self.entry_plane_y_norm = entry_plane_y_norm
            self.A = np.array([0.0, y_sep], dtype=np.float64)
            self.B = np.array([float(frame_width), y_sep], dtype=np.float64)
        elif entry_line_style == "quad_lr":
            verts_px = [
                (x * frame_width, y * frame_height)
                for x, y in polygon_norm
            ]
            left = _midpoint(verts_px[0], verts_px[3])
            right = _midpoint(verts_px[1], verts_px[2])
            self.A = np.array(left, dtype=np.float64)
            self.B = np.array(right, dtype=np.float64)
            self.entry_plane_y_norm = None
        else:
            raise ValueError(f"Unknown entry_line_style: {entry_line_style}")

        store_cross = self.cross_value(self.store_ref_px)
        self.store_ref_cross = store_cross
        self._store_sign = 1.0 if store_cross >= 0 else -1.0
        if store_cross == 0:
            self._store_sign = 1.0
        if invert_store_half_plane:
            self._store_sign *= -1.0

    def cross_value(self, point: Tuple[float, float]) -> float:
        p = np.array(point, dtype=np.float64)
        line = self.B - self.A
        vec = p - self.A
        return float(line[0] * vec[1] - line[1] * vec[0])

    def classify_side(self, point: Tuple[float, float]) -> str:
        cross = self.cross_value(point)
        if cross * self._store_sign >= 0:
            return SIDE_STORE
        return SIDE_MALL

    def debug_snapshot(self, point: Tuple[float, float]) -> Dict[str, Any]:
        cross = self.cross_value(point)
        return {
            "entry_line_style": self.entry_line_style,
            "A": (round(float(self.A[0]), 1), round(float(self.A[1]), 1)),
            "B": (round(float(self.B[0]), 1), round(float(self.B[1]), 1)),
            "STORE_REF_px": (
                round(self.store_ref_px[0], 1),
                round(self.store_ref_px[1], 1),
            ),
            "cross_STORE_REF": round(self.store_ref_cross, 1),
            "store_sign": self._store_sign,
            "cross_point": round(cross, 1),
            "geometric_side": self.classify_side(point),
            "entry_plane_y_norm": self.entry_plane_y_norm,
        }


def _geometry_debug_enabled() -> bool:
    return os.getenv("CAM3_GEOMETRY_DEBUG") == "1"


def _crossing_debug_enabled() -> bool:
    return os.getenv("CAM3_CROSSING_DEBUG") == "1"


def _print_crossing_debug(
    track_id: int,
    point: Tuple[int, int],
    original_frame: int,
    record: "TrackRetailRecord",
) -> None:
    print(
        f"[CROSSING] frame={original_frame} track_id={track_id} "
        f"confirmed_side={record.confirmed_side} pending_side={record.pending_side} "
        f"pending_frames={record.pending_frames} store_state={record.store_state} "
        f"observing={record.observing} centroid={point}"
    )


def _print_geometry_debug(track_id: int, point: Tuple[int, int], geometry: EntryLineGeometry) -> None:
    snap = geometry.debug_snapshot(point)
    print(
        f"[GEOM] visitor_id={track_id} style={snap['entry_line_style']} "
        f"A={snap['A']} B={snap['B']} STORE_REF_px={snap['STORE_REF_px']} "
        f"cross(STORE_REF)={snap['cross_STORE_REF']} store_sign={snap['store_sign']} "
        f"cross(centroid)={snap['cross_point']} side={snap['geometric_side']}"
    )


@dataclass
class RecoveryStats:
    recovered_entries: int = 0
    recovered_exits: int = 0
    born_inside_tracks: int = 0
    born_near_threshold_tracks: int = 0


@dataclass
class RetailInitDebug:
    visitor_id: int
    initial_side: str
    initial_store_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visitor_id": self.visitor_id,
            "initial_side": self.initial_side,
            "initial_store_state": self.initial_store_state,
        }


@dataclass
class RetailTransitionEvent:
    visitor_id: int
    event_type: str
    timestamp: str
    debug: Dict[str, Any]

    def to_event_row(self, camera: str = "CAM3") -> Dict[str, Any]:
        return {
            "visitor_id": self.visitor_id,
            "camera": camera,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "debug": self.debug,
        }


@dataclass
class TrackRetailRecord:
    store_state: str
    confirmed_side: str
    pending_side: Optional[str] = None
    pending_frames: int = 0
    initialized: bool = False
    last_cross_value: float = 0.0
    cross_at_pending_start: float = 0.0
    observing: bool = False
    observation_samples: List[Tuple[float, str]] = field(default_factory=list)
    last_point: Optional[Tuple[int, int]] = None
    prev_point: Optional[Tuple[int, int]] = None
    last_seen_frame: int = 0
    exit_emitted: bool = False


@dataclass
class RetailUpdateResult:
    store_state: str
    point: Tuple[int, int]
    init_debug: Optional[RetailInitDebug] = None
    transition: Optional[RetailTransitionEvent] = None


class RetailEntryEngine:
    """Half-plane retail latch with stability + optional threshold recovery."""

    def __init__(
        self,
        geometry: EntryLineGeometry,
        *,
        entry_stability_frames: int = ENTRY_STABILITY_FRAMES,
        exit_stability_frames: int = EXIT_STABILITY_FRAMES,
        timestamp_fn: Optional[Callable[[int], str]] = None,
        invert_retail_semantics: bool = False,
        enable_threshold_recovery: bool = False,
        near_threshold_y_tolerance: float = 0.10,
        threshold_observation_frames: int = 3,
        threshold_motion_eps: float = 0.015,
        enable_track_loss_flush: bool = False,
        track_loss_grace_frames: int = 20,
        track_loss_motion_eps: float = 0.5,
        track_loss_flush_require_stable_pending: bool = True,
        enable_late_entry_recovery: bool = False,
        late_entry_y_band_below: float = 0.20,
        late_entry_require_near_threshold: bool = False,
    ) -> None:
        self.geometry = geometry
        self.entry_stability_frames = entry_stability_frames
        self.exit_stability_frames = exit_stability_frames
        self._timestamp_fn = timestamp_fn
        self.invert_retail_semantics = invert_retail_semantics
        self.enable_threshold_recovery = enable_threshold_recovery
        self.near_threshold_y_tolerance = near_threshold_y_tolerance
        self.threshold_observation_frames = max(2, threshold_observation_frames)
        self.threshold_motion_eps = threshold_motion_eps
        self.enable_track_loss_flush = enable_track_loss_flush
        self.track_loss_grace_frames = max(1, track_loss_grace_frames)
        self.track_loss_motion_eps = track_loss_motion_eps
        self.track_loss_flush_require_stable_pending = (
            track_loss_flush_require_stable_pending
        )
        self.enable_late_entry_recovery = enable_late_entry_recovery
        self.late_entry_y_band_below = max(0.05, late_entry_y_band_below)
        self.late_entry_require_near_threshold = late_entry_require_near_threshold
        self._tracks: Dict[int, TrackRetailRecord] = {}
        self.entry_count = 0
        self.exit_count = 0
        self.recovery_stats = RecoveryStats()

    def _store_state_for_side(self, side: str) -> str:
        if self.invert_retail_semantics:
            return STATE_OUTSIDE if side == SIDE_STORE else STATE_INSIDE
        return STATE_INSIDE if side == SIDE_STORE else STATE_OUTSIDE

    def _entry_exit_sides(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        entry_from, entry_to = (
            (SIDE_STORE, SIDE_MALL)
            if self.invert_retail_semantics
            else (SIDE_MALL, SIDE_STORE)
        )
        exit_from, exit_to = entry_to, entry_from
        return (entry_from, entry_to), (exit_from, exit_to)

    def _stability_required(self, from_side: str, to_side: str) -> int:
        (entry_from, entry_to), (exit_from, exit_to) = self._entry_exit_sides()
        if from_side == entry_from and to_side == entry_to:
            return self.entry_stability_frames
        if from_side == exit_from and to_side == exit_to:
            return self.exit_stability_frames
        return self.entry_stability_frames

    def _y_norm(self, point: Tuple[int, int]) -> float:
        return point[1] / max(self.geometry.frame_height, 1)

    def _is_near_threshold(self, point: Tuple[int, int]) -> bool:
        if self.geometry.entry_plane_y_norm is not None:
            y_norm = self._y_norm(point)
            return (
                abs(y_norm - self.geometry.entry_plane_y_norm)
                <= self.near_threshold_y_tolerance
            )
        cross = abs(self.geometry.cross_value(point))
        line_len = float(np.linalg.norm(self.geometry.B - self.geometry.A)) or 1.0
        return (cross / line_len) <= self.near_threshold_y_tolerance

    def _is_in_late_entry_doorway_band(self, point: Tuple[int, int]) -> bool:
        """Just inside the horizontal entry plane (YOLO often misses until here)."""
        plane = self.geometry.entry_plane_y_norm
        if plane is None:
            return self._is_near_threshold(point)
        y_norm = self._y_norm(point)
        return plane <= y_norm <= plane + self.late_entry_y_band_below

    def _is_inside_store_floor(self, point: Tuple[int, int]) -> bool:
        """Footage2 horizontal line: inside = at/ below entry plane in image y."""
        plane = self.geometry.entry_plane_y_norm
        if plane is None:
            return self._store_state_for_side(self.geometry.classify_side(point)) == STATE_INSIDE
        y_norm = self._y_norm(point)
        if self.invert_retail_semantics:
            return y_norm >= plane
        return y_norm <= plane

    def _late_entry_recovery_eligible(self, point: Tuple[int, int]) -> bool:
        """Doorway band for recovered ENTRY; optional narrow band at the line only."""
        if self.late_entry_require_near_threshold:
            return self._is_near_threshold(point)
        return self._is_in_late_entry_doorway_band(point)

    def _is_clearly_past_exit_side(self, point: Tuple[int, int], exit_to: str) -> bool:
        """Centroid left the doorway band on the mall/outside side (not line flicker)."""
        if self.geometry.classify_side(point) != exit_to:
            return False
        plane = self.geometry.entry_plane_y_norm
        if plane is None:
            return not self._is_near_threshold(point)
        y_norm = self._y_norm(point)
        margin = self.near_threshold_y_tolerance
        if self.invert_retail_semantics:
            return y_norm < plane - margin
        return y_norm > plane + margin

    def _should_use_threshold_observing(self, point: Tuple[int, int]) -> bool:
        return self._is_near_threshold(point) or self._is_in_late_entry_doorway_band(
            point
        )

    def _timestamp(self, original_frame: int) -> str:
        if self._timestamp_fn:
            return self._timestamp_fn(original_frame)
        return format_video_timestamp(original_frame, 30.0)

    def _touch_track(
        self,
        record: TrackRetailRecord,
        point: Tuple[int, int],
        original_frame: int,
    ) -> None:
        if record.last_point is not None:
            record.prev_point = record.last_point
        record.last_point = point
        record.last_seen_frame = original_frame

    def _is_moving_toward_exit(self, record: TrackRetailRecord) -> bool:
        (_, _), (_, exit_to) = self._entry_exit_sides()
        if record.last_point is None:
            return False
        if self.geometry.classify_side(record.last_point) == exit_to:
            return True
        cross_last = self.geometry.cross_value(record.last_point)
        if record.prev_point is not None:
            cross_prev = self.geometry.cross_value(record.prev_point)
        else:
            cross_prev = record.cross_at_pending_start
        sign = (
            self.geometry._store_sign
            if exit_to == SIDE_STORE
            else -self.geometry._store_sign
        )
        return (cross_last - cross_prev) * sign > self.track_loss_motion_eps

    def _try_track_loss_exit_flush(
        self,
        tid: int,
        current_frame: int,
    ) -> Optional[RetailTransitionEvent]:
        if not self.enable_track_loss_flush:
            return None

        record = self._tracks.get(tid)
        if record is None or record.observing or record.exit_emitted:
            return None

        (_, _), (exit_from, exit_to) = self._entry_exit_sides()
        frame_gap = current_frame - record.last_seen_frame
        if frame_gap <= 0 or frame_gap > self.track_loss_grace_frames:
            return None

        if record.store_state != STATE_INSIDE:
            return None
        if record.confirmed_side != exit_from:
            return None
        if record.last_point is None:
            return None
        last_point = record.last_point
        if self.geometry.classify_side(last_point) != exit_to:
            return None
        # Crowded doorway: do not flush while centroid is still on/near the line.
        if self._is_near_threshold(last_point):
            return None

        min_pending = (
            self.exit_stability_frames
            if self.track_loss_flush_require_stable_pending
            else 1
        )
        stable_pending = (
            record.pending_side == exit_to
            and record.pending_frames >= min_pending
        )
        clearly_outside = self._is_clearly_past_exit_side(last_point, exit_to)
        if not (stable_pending or clearly_outside):
            return None
        if not clearly_outside and not self._is_moving_toward_exit(record):
            return None

        cross_before = record.cross_at_pending_start
        cross_after = (
            self.geometry.cross_value(record.last_point)
            if record.last_point is not None
            else record.last_cross_value
        )
        previous_store_state = record.store_state
        record.confirmed_side = exit_to
        record.store_state = STATE_OUTSIDE
        record.pending_side = None
        record.pending_frames = 0
        record.exit_emitted = True
        self.exit_count += 1
        self.recovery_stats.recovered_exits += 1
        transition = self._make_transition(
            tid,
            EVENT_EXIT,
            record.last_seen_frame,
            previous_side=exit_from,
            current_side=exit_to,
            previous_store_state=previous_store_state,
            current_store_state=record.store_state,
            cross_before=cross_before,
            cross_after=cross_after,
            recovered=True,
            recovery_reason="track_loss_flush",
        )
        if record.last_point is not None:
            transition.debug["centroid"] = list(record.last_point)
        return transition

    def flush_removed_tracks(
        self,
        active_track_ids: set[int],
        current_frame: int,
    ) -> List[RetailTransitionEvent]:
        """Emit recovered EXIT for exit-bound tracks lost before stability."""
        if not self.enable_track_loss_flush:
            return []

        transitions: List[RetailTransitionEvent] = []
        for tid in list(self._tracks.keys()):
            if tid in active_track_ids:
                continue
            transition = self._try_track_loss_exit_flush(tid, current_frame)
            if transition is not None:
                transitions.append(transition)
            del self._tracks[tid]
        return transitions

    def _make_transition(
        self,
        tid: int,
        event_type: str,
        original_frame: int,
        *,
        previous_side: str,
        current_side: str,
        previous_store_state: str,
        current_store_state: str,
        cross_before: float,
        cross_after: float,
        recovered: bool = False,
        recovery_reason: Optional[str] = None,
    ) -> RetailTransitionEvent:
        debug: Dict[str, Any] = {
            "previous_side": previous_side,
            "current_side": current_side,
            "previous_store_state": previous_store_state,
            "current_store_state": current_store_state,
            "cross_value_before": cross_before,
            "cross_value_after": cross_after,
        }
        if recovered:
            debug["recovered"] = True
        if recovery_reason:
            debug["recovery_reason"] = recovery_reason
        return RetailTransitionEvent(
            visitor_id=tid,
            event_type=event_type,
            timestamp=self._timestamp(original_frame),
            debug=debug,
        )

    def _init_late_entry_recovery(
        self,
        tid: int,
        point: Tuple[int, int],
        cross: float,
        instant_side: str,
        original_frame: int,
    ) -> RetailUpdateResult:
        """First sight already inside doorway — recover ENTRY YOLO/ByteTrack missed at line."""
        (entry_from, entry_to), _ = self._entry_exit_sides()
        previous_store_state = STATE_OUTSIDE
        self._tracks[tid] = TrackRetailRecord(
            store_state=STATE_INSIDE,
            confirmed_side=entry_to,
            initialized=True,
            last_cross_value=cross,
            cross_at_pending_start=cross,
        )
        self.entry_count += 1
        self.recovery_stats.recovered_entries += 1
        transition = self._make_transition(
            tid,
            EVENT_ENTRY,
            original_frame,
            previous_side=entry_from,
            current_side=entry_to,
            previous_store_state=previous_store_state,
            current_store_state=STATE_INSIDE,
            cross_before=cross,
            cross_after=cross,
            recovered=True,
            recovery_reason="late_detection_entry",
        )
        result = RetailUpdateResult(
            store_state=STATE_INSIDE,
            point=point,
            transition=transition,
            init_debug=RetailInitDebug(
                visitor_id=tid,
                initial_side=entry_to,
                initial_store_state=STATE_INSIDE,
            ),
        )
        self._touch_track(self._tracks[tid], point, original_frame)
        if _crossing_debug_enabled():
            _print_crossing_debug(tid, point, original_frame, self._tracks[tid])
        return result

    def _init_track_immediate(
        self,
        tid: int,
        point: Tuple[int, int],
        cross: float,
        instant_side: str,
        original_frame: int,
    ) -> RetailUpdateResult:
        store_state = self._store_state_for_side(instant_side)
        (entry_from, entry_to), _ = self._entry_exit_sides()
        if (
            self.enable_late_entry_recovery
            and self._is_inside_store_floor(point)
            and self._late_entry_recovery_eligible(point)
        ):
            return self._init_late_entry_recovery(
                tid, point, cross, entry_to, original_frame
            )
        if store_state == STATE_INSIDE:
            self.recovery_stats.born_inside_tracks += 1
        self._tracks[tid] = TrackRetailRecord(
            store_state=store_state,
            confirmed_side=instant_side,
            initialized=True,
            last_cross_value=cross,
        )
        result = RetailUpdateResult(
            store_state=store_state,
            point=point,
            init_debug=RetailInitDebug(
                visitor_id=tid,
                initial_side=instant_side,
                initial_store_state=store_state,
            ),
        )
        self._touch_track(self._tracks[tid], point, original_frame)
        if _crossing_debug_enabled():
            _print_crossing_debug(tid, point, original_frame, self._tracks[tid])
        return result

    def _init_track_observing(
        self,
        tid: int,
        point: Tuple[int, int],
        cross: float,
        instant_side: str,
        original_frame: int,
    ) -> RetailUpdateResult:
        self.recovery_stats.born_near_threshold_tracks += 1
        y_norm = self._y_norm(point)
        self._tracks[tid] = TrackRetailRecord(
            store_state=STATE_UNKNOWN,
            confirmed_side=instant_side,
            initialized=True,
            observing=True,
            last_cross_value=cross,
            observation_samples=[(y_norm, instant_side)],
        )
        result = RetailUpdateResult(
            store_state=STATE_UNKNOWN,
            point=point,
            init_debug=RetailInitDebug(
                visitor_id=tid,
                initial_side=instant_side,
                initial_store_state=STATE_UNKNOWN,
            ),
        )
        self._touch_track(self._tracks[tid], point, original_frame)
        if _crossing_debug_enabled():
            _print_crossing_debug(tid, point, original_frame, self._tracks[tid])
        return result

    def _resolve_observation(
        self,
        tid: int,
        point: Tuple[int, int],
        cross: float,
        instant_side: str,
        original_frame: int,
    ) -> RetailUpdateResult:
        record = self._tracks[tid]
        y_norm = self._y_norm(point)
        record.observation_samples.append((y_norm, instant_side))
        record.last_cross_value = cross

        if len(record.observation_samples) < self.threshold_observation_frames:
            self._touch_track(record, point, original_frame)
            if _crossing_debug_enabled():
                _print_crossing_debug(tid, point, original_frame, record)
            return RetailUpdateResult(store_state=STATE_UNKNOWN, point=point)

        samples = record.observation_samples
        dy = samples[-1][0] - samples[0][0]
        record.observing = False
        transition: Optional[RetailTransitionEvent] = None

        (entry_from, entry_to), (exit_from, exit_to) = self._entry_exit_sides()

        if dy > self.threshold_motion_eps:
            record.confirmed_side = entry_from
            record.store_state = STATE_OUTSIDE
            if instant_side == entry_to:
                record.confirmed_side = entry_to
                record.store_state = STATE_INSIDE
                self.entry_count += 1
                self.recovery_stats.recovered_entries += 1
                transition = self._make_transition(
                    tid,
                    EVENT_ENTRY,
                    original_frame,
                    previous_side=entry_from,
                    current_side=entry_to,
                    previous_store_state=STATE_OUTSIDE,
                    current_store_state=STATE_INSIDE,
                    cross_before=samples[0][0],
                    cross_after=cross,
                    recovered=True,
                )
        elif dy < -self.threshold_motion_eps:
            record.confirmed_side = exit_from
            record.store_state = STATE_INSIDE
            if instant_side == exit_to:
                record.confirmed_side = exit_to
                record.store_state = STATE_OUTSIDE
                record.exit_emitted = True
                self.exit_count += 1
                self.recovery_stats.recovered_exits += 1
                transition = self._make_transition(
                    tid,
                    EVENT_EXIT,
                    original_frame,
                    previous_side=exit_from,
                    current_side=exit_to,
                    previous_store_state=STATE_INSIDE,
                    current_store_state=STATE_OUTSIDE,
                    cross_before=samples[0][0],
                    cross_after=cross,
                    recovered=True,
                )
        else:
            record.confirmed_side = instant_side
            record.store_state = self._store_state_for_side(instant_side)
            if record.store_state == STATE_INSIDE:
                self.recovery_stats.born_inside_tracks += 1

        self._touch_track(record, point, original_frame)
        result = RetailUpdateResult(
            store_state=record.store_state,
            point=point,
            transition=transition,
        )
        if _crossing_debug_enabled():
            _print_crossing_debug(tid, point, original_frame, record)
        return result

    def _update_committed(
        self,
        tid: int,
        point: Tuple[int, int],
        cross: float,
        instant_side: str,
        original_frame: int,
    ) -> RetailUpdateResult:
        record = self._tracks[tid]
        transition: Optional[RetailTransitionEvent] = None

        if instant_side == record.confirmed_side:
            record.pending_side = None
            record.pending_frames = 0
            record.last_cross_value = cross
            result = RetailUpdateResult(store_state=record.store_state, point=point)
            if _crossing_debug_enabled():
                _print_crossing_debug(tid, point, original_frame, record)
            return result

        if record.pending_side != instant_side:
            record.pending_side = instant_side
            record.pending_frames = 1
            record.cross_at_pending_start = record.last_cross_value
        else:
            record.pending_frames += 1

        record.last_cross_value = cross

        assert record.pending_side is not None
        needed = self._stability_required(record.confirmed_side, record.pending_side)
        if record.pending_frames < needed:
            result = RetailUpdateResult(store_state=record.store_state, point=point)
            if _crossing_debug_enabled():
                _print_crossing_debug(tid, point, original_frame, record)
            return result

        previous_side = record.confirmed_side
        current_side = record.pending_side
        cross_before = record.cross_at_pending_start
        cross_after = cross
        previous_store_state = record.store_state

        record.confirmed_side = current_side
        record.pending_side = None
        record.pending_frames = 0

        (entry_from, entry_to), (exit_from, exit_to) = self._entry_exit_sides()

        if (
            previous_side == entry_from
            and current_side == entry_to
            and record.store_state == STATE_OUTSIDE
        ):
            record.store_state = STATE_INSIDE
            self.entry_count += 1
            transition = self._make_transition(
                tid,
                EVENT_ENTRY,
                original_frame,
                previous_side=previous_side,
                current_side=current_side,
                previous_store_state=previous_store_state,
                current_store_state=record.store_state,
                cross_before=cross_before,
                cross_after=cross_after,
            )
        elif (
            previous_side == exit_from
            and current_side == exit_to
            and record.store_state == STATE_INSIDE
        ):
            record.store_state = STATE_OUTSIDE
            record.exit_emitted = True
            self.exit_count += 1
            transition = self._make_transition(
                tid,
                EVENT_EXIT,
                original_frame,
                previous_side=previous_side,
                current_side=current_side,
                previous_store_state=previous_store_state,
                current_store_state=record.store_state,
                cross_before=cross_before,
                cross_after=cross_after,
            )

        self._touch_track(record, point, original_frame)
        result = RetailUpdateResult(
            store_state=record.store_state,
            point=point,
            transition=transition,
        )
        if _crossing_debug_enabled():
            _print_crossing_debug(tid, point, original_frame, record)
        return result

    def restore_retail_from_session(
        self,
        track_id: int,
        point: Tuple[int, int],
        *,
        original_frame: int = 0,
        store_state: str,
        confirmed_side: str,
        pending_side: Optional[str],
        pending_frames: int,
    ) -> None:
        """Seed crossing state for a new byte track from an open visitor session."""
        tid = int(track_id)
        cross = self.geometry.cross_value(point)
        self._tracks[tid] = TrackRetailRecord(
            store_state=store_state,
            confirmed_side=confirmed_side,
            pending_side=pending_side,
            pending_frames=pending_frames,
            initialized=True,
            last_cross_value=cross,
            cross_at_pending_start=cross,
        )
        self._touch_track(self._tracks[tid], point, original_frame)

    def update(
        self,
        track_id: int,
        point: Tuple[int, int],
        *,
        original_frame: int = 0,
    ) -> RetailUpdateResult:
        cross = self.geometry.cross_value(point)
        instant_side = self.geometry.classify_side(point)
        tid = int(track_id)

        if tid not in self._tracks:
            if _geometry_debug_enabled():
                _print_geometry_debug(tid, point, self.geometry)
            if self.enable_threshold_recovery and self._should_use_threshold_observing(
                point
            ):
                return self._init_track_observing(
                    tid, point, cross, instant_side, original_frame
                )
            return self._init_track_immediate(
                tid, point, cross, instant_side, original_frame
            )

        record = self._tracks[tid]
        if record.observing:
            result = self._resolve_observation(
                tid, point, cross, instant_side, original_frame
            )
            if record.observing:
                return result
            if result.transition is not None:
                return result

        result = self._update_committed(
            tid, point, cross, instant_side, original_frame
        )
        return result

    def process_detections(
        self,
        detections: Any,
        *,
        original_frame: int = 0,
        point_fn: Optional[Callable[[Any], Tuple[int, int]]] = None,
    ) -> Dict[int, Tuple[str, Tuple[int, int]]]:
        """Update all tracks; default anchor is bbox centroid."""
        track_states: Dict[int, Tuple[str, Tuple[int, int]]] = {}
        if detections.tracker_id is None or len(detections) == 0:
            return track_states

        for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            tid = int(track_id)
            if point_fn is not None:
                point = point_fn(xyxy)
            else:
                x1, y1, x2, y2 = xyxy
                point = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                )
            result = self.update(tid, point, original_frame=original_frame)
            track_states[tid] = (result.store_state, point)

        if self.enable_track_loss_flush:
            self.flush_removed_tracks(set(track_states.keys()), original_frame)
        return track_states
