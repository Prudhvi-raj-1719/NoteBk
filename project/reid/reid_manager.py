"""CAM3 track Re-ID: OSNet embeddings, exit cache, ENTRY/REENTRY, session exit matching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import supervision as sv

from configs.reid_config import (
    REID_AREA_RATIO_MAX,
    REID_AREA_RATIO_MIN,
    REID_COSINE_THRESHOLD,
    REID_DEBUG,
    REID_EXIT_CACHE_SECONDS,
    REID_FRAGMENT_MARGIN_THRESHOLD,
    REID_FRAGMENT_MATCH_THRESHOLD,
    REID_MIN_CROP_AREA,
    REID_MIN_DWELL_AFTER_EXIT_SECONDS,
    REID_NEAR_DOORWAY_Y_TOLERANCE,
    REID_SESSION_MARGIN_THRESHOLD,
    REID_SESSION_MATCH_ENABLED,
    REID_SESSION_MATCH_THRESHOLD,
    REID_SESSION_MAX_AGE_SECONDS,
    REID_TRACK_EMBED_HISTORY,
)
from events.event_time import video_offset_to_utc_iso
from events.reentry_session import parse_utc_from_iso
from reid.osnet_reid import OsnetEmbedder
from reid.session_store import RetailStateSnapshot, SessionStore

from entry_retail import STATE_INSIDE, STATE_UNKNOWN, format_video_timestamp

if TYPE_CHECKING:
    from entry_retail import EntryLineGeometry, RetailEntryEngine


@dataclass
class ExitCacheRecord:
    vis_id: str
    byte_track_id: int
    embedding: np.ndarray
    exit_utc: datetime


@dataclass
class ReIDManager:
    embedder: OsnetEmbedder
    cosine_threshold: float = REID_COSINE_THRESHOLD
    exit_cache_seconds: float = REID_EXIT_CACHE_SECONDS
    min_dwell_after_exit_seconds: float = REID_MIN_DWELL_AFTER_EXIT_SECONDS
    min_crop_area: int = REID_MIN_CROP_AREA
    embed_history: int = REID_TRACK_EMBED_HISTORY
    session_match_enabled: bool = REID_SESSION_MATCH_ENABLED
    session_match_threshold: float = REID_SESSION_MATCH_THRESHOLD
    session_margin_threshold: float = REID_SESSION_MARGIN_THRESHOLD
    area_ratio_min: float = REID_AREA_RATIO_MIN
    area_ratio_max: float = REID_AREA_RATIO_MAX
    session_max_age_seconds: float = REID_SESSION_MAX_AGE_SECONDS
    fragment_match_threshold: float = REID_FRAGMENT_MATCH_THRESHOLD
    fragment_margin_threshold: float = REID_FRAGMENT_MARGIN_THRESHOLD
    near_doorway_y_tolerance: float = REID_NEAR_DOORWAY_Y_TOLERANCE
    session_rematch_interval: int = 5
    entry_geometry: Optional["EntryLineGeometry"] = None
    track_to_vis: Dict[int, str] = field(default_factory=dict)
    _track_new_vis_only: Set[int] = field(default_factory=set)
    _session_rematch_tick: Dict[int, int] = field(default_factory=dict)
    track_embeddings: Dict[int, Deque[np.ndarray]] = field(default_factory=dict)
    recent_exits: List[ExitCacheRecord] = field(default_factory=list)
    session_store: SessionStore = field(default_factory=SessionStore)
    _byte_tracks_seen: Set[int] = field(default_factory=set)
    _pending_retail_restore: Dict[int, Tuple[int, str, RetailStateSnapshot]] = field(
        default_factory=dict
    )
    _next_vis_index: int = 1

    @staticmethod
    def video_utc_for_frame(
        camera_id: str,
        frame_index: int,
        fps: float,
        *,
        clip_id: Optional[str] = None,
    ) -> datetime:
        """UTC for a video frame (same anchor + offset as EventEmitter)."""
        offset = format_video_timestamp(frame_index, fps)
        utc_iso = video_offset_to_utc_iso(camera_id, offset, clip_id=clip_id)
        return parse_utc_from_iso(utc_iso)

    def _new_vis_id(self) -> str:
        vis = f"VIS_{self._next_vis_index:05d}"
        self._next_vis_index += 1
        return vis

    def _crop_from_box(self, frame_bgr: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.shape[0] * crop.shape[1] < self.min_crop_area:
            return None
        return crop

    @staticmethod
    def _bbox_tuple(xyxy: np.ndarray) -> Tuple[float, float, float, float]:
        return (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

    @staticmethod
    def _bbox_area(xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        return max((x2 - x1) * (y2 - y1), 1.0)

    def _centroid(self, xyxy: np.ndarray) -> Tuple[int, int]:
        return (
            int((xyxy[0] + xyxy[2]) / 2),
            int((xyxy[1] + xyxy[3]) / 2),
        )

    def _is_near_doorway(self, point: Tuple[int, int]) -> bool:
        if self.entry_geometry is None:
            return False
        geom = self.entry_geometry
        if geom.entry_plane_y_norm is not None:
            y_norm = point[1] / max(geom.frame_height, 1)
            return abs(y_norm - geom.entry_plane_y_norm) <= self.near_doorway_y_tolerance
        cross = abs(geom.cross_value(point))
        line_len = float(np.linalg.norm(geom.B - geom.A)) or 1.0
        return (cross / line_len) <= self.near_doorway_y_tolerance

    def _prototype(self, track_id: int) -> Optional[np.ndarray]:
        history = self.track_embeddings.get(track_id)
        if not history:
            return None
        stack = np.stack(list(history), axis=0)
        proto = stack.mean(axis=0)
        norm = np.linalg.norm(proto)
        if norm < 1e-8:
            return None
        return (proto / norm).astype(np.float32)

    def _log_debug(
        self,
        track_id: int,
        vis_id: str,
        *,
        similarity: Optional[float] = None,
        matched_vis: Optional[str] = None,
        reason: str = "",
    ) -> None:
        if not REID_DEBUG:
            return
        sim_text = "-" if similarity is None else f"{similarity:.3f}"
        match_text = matched_vis if matched_vis else "-"
        print(
            f"[REID_DEBUG] track_id={track_id} vis_id={vis_id} "
            f"similarity={sim_text} matched_vis={match_text} reason={reason}"
        )

    @staticmethod
    def _log_session_match(
        new_track: int,
        matched_vis: str,
        similarity: float,
    ) -> None:
        print(
            f"[SESSION_MATCH]\n"
            f"new_track={new_track}\n"
            f"matched_vis={matched_vis}\n"
            f"similarity={similarity:.2f}"
        )

    @staticmethod
    def _log_session_attempt(
        track_id: int,
        *,
        best_vis: str,
        best_sim: float,
        second_best_sim: float,
        margin: float,
        area_ratio: float,
        session_age: float,
        accepted: bool,
        reason: str,
    ) -> None:
        print(
            f"[SESSION_ATTEMPT]\n"
            f"track={track_id}\n"
            f"best_vis={best_vis}\n"
            f"best_sim={best_sim:.2f}\n"
            f"second_best={second_best_sim:.2f}\n"
            f"margin={margin:.2f}\n"
            f"area_ratio={area_ratio:.2f}\n"
            f"session_age={session_age:.1f}\n"
            f"accepted={accepted}\n"
            f"reason={reason}"
        )

    @staticmethod
    def _log_session_match_accepted(
        track_id: int,
        matched_vis: str,
        best_similarity: float,
        second_best_similarity: float,
        margin: float,
    ) -> None:
        print(
            f"[SESSION_MATCH_ACCEPTED]\n"
            f"track={track_id}\n"
            f"matched_vis={matched_vis}\n"
            f"best_similarity={best_similarity:.2f}\n"
            f"second_best_similarity={second_best_similarity:.2f}\n"
            f"margin={margin:.2f}"
        )

    @staticmethod
    def _log_session_match_rejected(
        track_id: int,
        best_vis: str,
        best_similarity: float,
        second_best_similarity: float,
        margin: float,
        reason: str,
    ) -> None:
        print(
            f"[SESSION_MATCH_REJECTED]\n"
            f"track={track_id}\n"
            f"best_vis={best_vis}\n"
            f"best_similarity={best_similarity:.2f}\n"
            f"second_best_similarity={second_best_similarity:.2f}\n"
            f"margin={margin:.2f}\n"
            f"reason={reason}"
        )

    def _attempt_session_match(
        self,
        tid: int,
        embedding: np.ndarray,
        query_area: float,
        match_utc: datetime,
    ) -> Tuple[Optional[str], float]:
        result = self.session_store.match_embedding(
            embedding,
            query_area=query_area,
            match_utc=match_utc,
            threshold=self.session_match_threshold,
            margin_threshold=self.session_margin_threshold,
            area_ratio_min=self.area_ratio_min,
            area_ratio_max=self.area_ratio_max,
            max_age_seconds=self.session_max_age_seconds,
            cosine_fn=self._cosine,
        )
        self._log_session_attempt(
            tid,
            best_vis=result.best_vis,
            best_sim=result.best_sim,
            second_best_sim=result.second_best_sim,
            margin=result.margin,
            area_ratio=result.area_ratio,
            session_age=result.session_age,
            accepted=result.accepted,
            reason=result.reason,
        )
        if result.accepted and result.matched_vis is not None:
            self._log_session_match_accepted(
                tid,
                result.matched_vis,
                result.best_sim,
                result.second_best_sim,
                result.margin,
            )
            return result.matched_vis, result.best_sim
        self._log_session_match_rejected(
            tid,
            result.best_vis,
            result.best_sim,
            result.second_best_sim,
            result.margin,
            result.reason,
        )
        return None, result.best_sim if result.best_sim >= 0 else -1.0

    def _attempt_fragment_session_match(
        self,
        tid: int,
        embedding: np.ndarray,
        query_area: float,
        match_utc: datetime,
    ) -> Tuple[Optional[str], float]:
        result = self.session_store.match_embedding_fragment(
            embedding,
            query_area=query_area,
            match_utc=match_utc,
            threshold=self.fragment_match_threshold,
            margin_threshold=self.fragment_margin_threshold,
            area_ratio_min=self.area_ratio_min,
            area_ratio_max=self.area_ratio_max,
            max_age_seconds=self.session_max_age_seconds,
            cosine_fn=self._cosine,
            inside_store_state=STATE_INSIDE,
        )
        self._log_session_attempt(
            tid,
            best_vis=result.best_vis,
            best_sim=result.best_sim,
            second_best_sim=result.second_best_sim,
            margin=result.margin,
            area_ratio=result.area_ratio,
            session_age=result.session_age,
            accepted=result.accepted,
            reason=f"fragment_{result.reason}",
        )
        if result.accepted and result.matched_vis is not None:
            self._log_session_match_accepted(
                tid,
                result.matched_vis,
                result.best_sim,
                result.second_best_sim,
                result.margin,
            )
            return result.matched_vis, result.best_sim
        self._log_session_match_rejected(
            tid,
            result.best_vis,
            result.best_sim,
            result.second_best_sim,
            result.margin,
            result.reason,
        )
        return None, result.best_sim if result.best_sim >= 0 else -1.0

    def ensure_inside_visitor_session(
        self,
        byte_track_id: int,
        match_utc: datetime,
        bbox: Optional[List[float]],
    ) -> None:
        """Open a session for INSIDE visitors (no ENTRY) so exit fragments can re-attach."""
        if not self.session_match_enabled:
            return
        tid = int(byte_track_id)
        vis_id = self.track_to_vis.get(tid)
        if not vis_id or vis_id in self.session_store.sessions:
            return
        if self.latest_embedding(tid) is None:
            return
        self.open_active_session(vis_id, tid, match_utc, bbox)
        self._log_debug(tid, vis_id, reason="session_opened_inside")

    @staticmethod
    def _log_session_restore(
        old_track: int,
        new_track: int,
        visitor_id: str,
    ) -> None:
        print(
            f"[SESSION_RESTORE]\n"
            f"old_track={old_track}\n"
            f"new_track={new_track}\n"
            f"visitor_id={visitor_id}"
        )

    def _queue_retail_restore_if_session_matched(
        self,
        new_track: int,
        matched_vis: str,
    ) -> None:
        snapshot = self.session_store.get_retail_snapshot(matched_vis)
        if snapshot is None:
            return
        session = self.session_store.sessions.get(matched_vis)
        if session is None:
            return
        self._pending_retail_restore[new_track] = (
            session.entry_byte_track_id,
            matched_vis,
            snapshot,
        )

    def pop_pending_retail_restore(
        self,
        new_track: int,
    ) -> Optional[Tuple[int, str, RetailStateSnapshot]]:
        return self._pending_retail_restore.pop(int(new_track), None)

    def apply_pending_retail_restore(
        self,
        engine: "RetailEntryEngine",
        new_track: int,
        center: Tuple[int, int],
        original_frame: int,
    ) -> bool:
        pending = self.pop_pending_retail_restore(new_track)
        if pending is None:
            return False
        old_track, visitor_id, snapshot = pending
        engine.restore_retail_from_session(
            new_track,
            center,
            original_frame=original_frame,
            store_state=snapshot.store_state,
            confirmed_side=snapshot.confirmed_side,
            pending_side=snapshot.pending_side,
            pending_frames=snapshot.pending_frames,
        )
        self._log_session_restore(old_track, new_track, visitor_id)
        return True

    def _apply_session_match_to_track(
        self,
        tid: int,
        matched_vis: str,
        sim: float,
    ) -> None:
        self.track_to_vis[tid] = matched_vis
        self._track_new_vis_only.discard(tid)
        self._log_session_match(tid, matched_vis, sim)
        self._queue_retail_restore_if_session_matched(tid, matched_vis)
        self._log_debug(
            tid,
            matched_vis,
            similarity=sim,
            matched_vis=matched_vis,
            reason="session_exit_match",
        )

    def _maybe_rematch_session_for_track(
        self,
        tid: int,
        query_area: float,
        match_utc: datetime,
    ) -> None:
        """Re-attempt session match for provisional track_new_vis IDs."""
        if not self.session_match_enabled:
            return
        if tid not in self._track_new_vis_only:
            return
        vis_id = self.track_to_vis.get(tid)
        if not vis_id or vis_id in self.session_store.sessions:
            return
        if not self.session_store.sessions:
            return
        tick = self._session_rematch_tick.get(tid, 0) + 1
        self._session_rematch_tick[tid] = tick
        if tick % self.session_rematch_interval != 0:
            return
        embedding = self._prototype(tid)
        if embedding is None:
            return
        matched_vis, sim = self._attempt_session_match(
            tid, embedding, query_area, match_utc
        )
        if matched_vis is None or matched_vis == vis_id:
            return
        self._apply_session_match_to_track(tid, matched_vis, sim)

    def sync_retail_snapshot_from_engine(
        self,
        engine: "RetailEntryEngine",
        byte_track_id: int,
        *,
        match_utc: Optional[datetime] = None,
        bbox: Optional[List[float]] = None,
    ) -> None:
        """Mirror latest committed retail state into the open session for this VIS."""
        if not self.session_match_enabled:
            return
        tid = int(byte_track_id)
        vis_id = self.track_to_vis.get(tid)
        if not vis_id:
            return
        record = engine._tracks.get(tid)
        if record is None or record.observing or record.store_state == STATE_UNKNOWN:
            return
        if (
            record.store_state == STATE_INSIDE
            and match_utc is not None
            and vis_id not in self.session_store.sessions
        ):
            self.ensure_inside_visitor_session(tid, match_utc, bbox)
        if vis_id not in self.session_store.sessions:
            return
        self.session_store.update_retail_snapshot(
            vis_id,
            RetailStateSnapshot(
                store_state=record.store_state,
                confirmed_side=record.confirmed_side,
                pending_side=record.pending_side,
                pending_frames=record.pending_frames,
            ),
        )

    def _assign_vis_for_new_track(
        self,
        tid: int,
        point: Tuple[int, int],
        embedding: Optional[np.ndarray],
        query_area: float,
        match_utc: datetime,
    ) -> str:
        is_new_byte_track = tid not in self._byte_tracks_seen
        self._byte_tracks_seen.add(tid)

        if tid in self.track_to_vis:
            return self.track_to_vis[tid]

        if (
            is_new_byte_track
            and self.session_match_enabled
            and embedding is not None
            and self.session_store.sessions
        ):
            matched_vis, sim = self._attempt_session_match(
                tid, embedding, query_area, match_utc
            )
            if matched_vis is not None:
                self._apply_session_match_to_track(tid, matched_vis, sim)
                return matched_vis
            matched_vis, sim = self._attempt_fragment_session_match(
                tid, embedding, query_area, match_utc
            )
            if matched_vis is not None:
                self._apply_session_match_to_track(tid, matched_vis, sim)
                return matched_vis

        vis = self._new_vis_id()
        self.track_to_vis[tid] = vis
        self._track_new_vis_only.add(tid)
        self._log_debug(tid, vis, reason="track_new_vis")
        return vis

    def update_tracks(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
        *,
        match_utc: datetime,
    ) -> None:
        """Crop, embed, assign VIS_*; session-match new tracks against open sessions."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        tid_to_xyxy: Dict[int, np.ndarray] = {}
        tid_to_crop: Dict[int, np.ndarray] = {}
        tid_to_point: Dict[int, Tuple[int, int]] = {}

        for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            tid = int(track_id)
            tid_to_xyxy[tid] = xyxy
            tid_to_point[tid] = self._centroid(xyxy)
            crop = self._crop_from_box(frame_bgr, xyxy)
            if crop is not None:
                tid_to_crop[tid] = crop

        if not tid_to_crop:
            for tid, point in tid_to_point.items():
                if tid not in self.track_to_vis:
                    area = self._bbox_area(tid_to_xyxy[tid])
                    self._assign_vis_for_new_track(tid, point, None, area, match_utc)
            return

        track_ids = list(tid_to_crop.keys())
        embeddings = self.embedder.embed_crops([tid_to_crop[t] for t in track_ids])
        emb_by_tid = {tid: emb.astype(np.float32) for tid, emb in zip(track_ids, embeddings)}

        for tid in tid_to_xyxy:
            point = tid_to_point[tid]
            xyxy = tid_to_xyxy[tid]
            bbox = self._bbox_tuple(xyxy)
            emb = emb_by_tid.get(tid)

            query_area = self._bbox_area(xyxy)

            if tid not in self.track_to_vis:
                self._assign_vis_for_new_track(tid, point, emb, query_area, match_utc)

            if emb is not None:
                history = self.track_embeddings.setdefault(
                    tid, deque(maxlen=self.embed_history)
                )
                history.append(emb)
                self._log_debug(
                    tid,
                    self.track_to_vis[tid],
                    reason="track_embedding_updated",
                )

            if tid in self.track_to_vis:
                self._maybe_rematch_session_for_track(tid, query_area, match_utc)

            vis_id = self.track_to_vis.get(tid)
            if vis_id and vis_id in self.session_store.sessions:
                self.session_store.update_bbox(vis_id, bbox)
                proto = self._prototype(tid)
                if proto is not None:
                    self.session_store.sessions[vis_id].embedding = proto.copy()

    def vis_for_track(self, track_id: int) -> str:
        tid = int(track_id)
        if tid not in self.track_to_vis:
            self.track_to_vis[tid] = self._new_vis_id()
            self._byte_tracks_seen.add(tid)
            self._log_debug(tid, self.track_to_vis[tid], reason="vis_lazy_assign")
        return self.track_to_vis[tid]

    def link_track(self, byte_track_id: int, vis_id: str) -> str:
        tid = int(byte_track_id)
        self.track_to_vis[tid] = vis_id
        self._byte_tracks_seen.add(tid)
        self._track_new_vis_only.discard(tid)
        return vis_id

    def latest_embedding(self, track_id: int) -> Optional[np.ndarray]:
        return self._prototype(int(track_id))

    def open_active_session(
        self,
        visitor_id: str,
        byte_track_id: int,
        entry_utc: datetime,
        bbox: Optional[List[float]],
    ) -> None:
        if not self.session_match_enabled:
            return
        embedding = self.latest_embedding(byte_track_id)
        if embedding is None:
            self._log_debug(
                int(byte_track_id),
                visitor_id,
                reason="session_open_no_embedding",
            )
            return
        if bbox and len(bbox) >= 4:
            bbox_t = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        else:
            bbox_t = (0.0, 0.0, 0.0, 0.0)
        self.session_store.open_session(
            visitor_id,
            entry_utc,
            byte_track_id,
            embedding,
            bbox_t,
        )
        self._log_debug(
            int(byte_track_id),
            visitor_id,
            reason="session_opened",
        )

    def close_active_session(self, visitor_id: str) -> None:
        if not self.session_match_enabled:
            return
        self.session_store.close_session(visitor_id)
        self._log_debug(0, visitor_id, reason="session_closed")

    def _prune_exits(self, now: datetime) -> None:
        now_utc = SessionStore.as_utc_aware(now)
        cutoff = now_utc - timedelta(seconds=self.exit_cache_seconds)
        self.recent_exits = [r for r in self.recent_exits if r.exit_utc >= cutoff]

    def should_emit_exit(self, vis_id: str, event_utc: datetime) -> bool:
        """Skip duplicate EXIT when session already closed (fragment / flush)."""
        if vis_id in self.session_store.sessions:
            return True
        event_utc_aware = SessionStore.as_utc_aware(event_utc)
        self._prune_exits(event_utc_aware)
        return not any(r.vis_id == vis_id for r in self.recent_exits)

    def should_emit_entry(self, vis_id: str, internal_emit_type: str) -> bool:
        """Skip duplicate ENTRY; allow first ENTRY after Re-ID-only inside session."""
        if internal_emit_type == "REENTRY":
            return True
        session = self.session_store.sessions.get(vis_id)
        if session is None:
            return True
        return not session.entry_event_emitted

    def mark_entry_event_emitted(self, vis_id: str) -> None:
        session = self.session_store.sessions.get(vis_id)
        if session is not None:
            session.entry_event_emitted = True

    def record_exit(
        self,
        vis_id: str,
        byte_track_id: int,
        exit_utc: datetime,
    ) -> None:
        embedding = self.latest_embedding(byte_track_id)
        if embedding is None:
            self._log_debug(
                int(byte_track_id),
                vis_id,
                reason="exit_cached_no_embedding",
            )
        else:
            exit_utc_aware = SessionStore.as_utc_aware(exit_utc)
            self.recent_exits.append(
                ExitCacheRecord(
                    vis_id=vis_id,
                    byte_track_id=int(byte_track_id),
                    embedding=embedding.copy(),
                    exit_utc=exit_utc_aware,
                )
            )
            self._prune_exits(exit_utc_aware)
            self._log_debug(
                int(byte_track_id),
                vis_id,
                reason="exit_cached",
            )
        self.close_active_session(vis_id)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def match_entry(
        self,
        byte_track_id: int,
        entry_utc: datetime,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Compare entry embedding to recent EXIT prototypes (short-gap REENTRY)."""
        query = self.latest_embedding(byte_track_id)
        if query is None:
            return None, None

        entry_utc_aware = SessionStore.as_utc_aware(entry_utc)
        self._prune_exits(entry_utc_aware)
        best_vis: Optional[str] = None
        best_sim = -1.0

        for record in self.recent_exits:
            age = (entry_utc_aware - record.exit_utc).total_seconds()
            if age < self.min_dwell_after_exit_seconds:
                continue
            if age > self.exit_cache_seconds:
                continue
            sim = self._cosine(query, record.embedding)
            if sim > best_sim:
                best_sim = sim
                best_vis = record.vis_id

        if best_vis is None or best_sim < self.cosine_threshold:
            return None, best_sim if best_sim >= 0 else None

        self._consume_exit(best_vis)
        return best_vis, best_sim

    def _consume_exit(self, vis_id: str) -> None:
        self.recent_exits = [r for r in self.recent_exits if r.vis_id != vis_id]
