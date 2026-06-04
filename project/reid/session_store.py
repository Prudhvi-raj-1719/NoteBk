"""CAM3 active visitor sessions: ENTRY → EXIT (long-gap exit track re-attach)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RetailStateSnapshot:
    store_state: str
    confirmed_side: str
    pending_side: Optional[str]
    pending_frames: int


@dataclass
class ActiveSession:
    visitor_id: str
    entry_utc: datetime
    entry_byte_track_id: int
    embedding: np.ndarray
    entry_bbox_area: float = 1.0
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    retail_snapshot: Optional[RetailStateSnapshot] = None
    entry_event_emitted: bool = False


@dataclass
class SessionMatchResult:
    matched_vis: Optional[str]
    best_vis: str
    best_sim: float
    second_best_sim: float
    margin: float
    area_ratio: float
    session_age: float
    accepted: bool
    reason: str


@dataclass
class SessionStore:
    """Open sessions keyed by visitor_id (VIS_*)."""

    sessions: Dict[str, ActiveSession] = field(default_factory=dict)

    @staticmethod
    def as_utc_aware(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max((x2 - x1) * (y2 - y1), 1.0)

    def open_session(
        self,
        visitor_id: str,
        entry_utc: datetime,
        entry_byte_track_id: int,
        embedding: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        self.sessions[visitor_id] = ActiveSession(
            visitor_id=visitor_id,
            entry_utc=self.as_utc_aware(entry_utc),
            entry_byte_track_id=int(entry_byte_track_id),
            embedding=embedding.astype(np.float32).copy(),
            entry_bbox_area=self.bbox_area(bbox),
            last_bbox=bbox,
        )

    def close_session(self, visitor_id: str) -> None:
        self.sessions.pop(visitor_id, None)

    def update_bbox(self, visitor_id: str, bbox: Tuple[float, float, float, float]) -> None:
        session = self.sessions.get(visitor_id)
        if session is not None:
            session.last_bbox = bbox

    def update_retail_snapshot(
        self,
        visitor_id: str,
        snapshot: RetailStateSnapshot,
    ) -> None:
        session = self.sessions.get(visitor_id)
        if session is not None:
            session.retail_snapshot = snapshot

    def get_retail_snapshot(self, visitor_id: str) -> Optional[RetailStateSnapshot]:
        session = self.sessions.get(visitor_id)
        if session is None:
            return None
        return session.retail_snapshot

    def match_embedding(
        self,
        query: np.ndarray,
        *,
        query_area: float,
        match_utc: datetime,
        threshold: float,
        margin_threshold: float,
        area_ratio_min: float,
        area_ratio_max: float,
        max_age_seconds: float,
        cosine_fn,
    ) -> SessionMatchResult:
        """Score open sessions; accept only if cosine, margin, area, and age pass.

        match_utc and session.entry_utc must both use the video timeline
        (clip anchor + frame offset), not wall-clock time.
        """
        match_utc = self.as_utc_aware(match_utc)
        scored: List[Tuple[str, float, ActiveSession]] = []
        for vis_id, session in self.sessions.items():
            age = (match_utc - session.entry_utc).total_seconds()
            if age > max_age_seconds:
                continue
            sim = cosine_fn(query, session.embedding)
            scored.append((vis_id, sim, session))

        if not scored:
            return SessionMatchResult(
                matched_vis=None,
                best_vis="-",
                best_sim=-1.0,
                second_best_sim=-1.0,
                margin=0.0,
                area_ratio=0.0,
                session_age=0.0,
                accepted=False,
                reason="no_eligible_sessions",
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        best_vis, best_sim, best_session = scored[0]
        session_age = (match_utc - best_session.entry_utc).total_seconds()
        area_ratio = query_area / max(best_session.entry_bbox_area, 1.0)

        if len(scored) < 2:
            return SessionMatchResult(
                matched_vis=None,
                best_vis=best_vis,
                best_sim=best_sim,
                second_best_sim=-1.0,
                margin=0.0,
                area_ratio=area_ratio,
                session_age=session_age,
                accepted=False,
                reason="no_second_best",
            )

        second_best_sim = scored[1][1]
        margin = best_sim - second_best_sim

        if best_sim < threshold:
            return SessionMatchResult(
                matched_vis=None,
                best_vis=best_vis,
                best_sim=best_sim,
                second_best_sim=second_best_sim,
                margin=margin,
                area_ratio=area_ratio,
                session_age=session_age,
                accepted=False,
                reason="below_threshold",
            )

        if margin < margin_threshold:
            return SessionMatchResult(
                matched_vis=None,
                best_vis=best_vis,
                best_sim=best_sim,
                second_best_sim=second_best_sim,
                margin=margin,
                area_ratio=area_ratio,
                session_age=session_age,
                accepted=False,
                reason="margin_too_small",
            )

        if area_ratio < area_ratio_min or area_ratio > area_ratio_max:
            return SessionMatchResult(
                matched_vis=None,
                best_vis=best_vis,
                best_sim=best_sim,
                second_best_sim=second_best_sim,
                margin=margin,
                area_ratio=area_ratio,
                session_age=session_age,
                accepted=False,
                reason="area_ratio_out_of_range",
            )

        return SessionMatchResult(
            matched_vis=best_vis,
            best_vis=best_vis,
            best_sim=best_sim,
            second_best_sim=second_best_sim,
            margin=margin,
            area_ratio=area_ratio,
            session_age=session_age,
            accepted=True,
            reason="accepted",
        )

    def match_embedding_fragment(
        self,
        query: np.ndarray,
        *,
        query_area: float,
        match_utc: datetime,
        threshold: float,
        margin_threshold: float,
        area_ratio_min: float,
        area_ratio_max: float,
        max_age_seconds: float,
        cosine_fn,
        inside_store_state: str,
    ) -> SessionMatchResult:
        """Match new ByteTrack IDs to open INSIDE visitor sessions (fragmentation path)."""
        match_utc = self.as_utc_aware(match_utc)
        scored: List[Tuple[str, float, ActiveSession]] = []
        for vis_id, session in self.sessions.items():
            snap = session.retail_snapshot
            if snap is None or snap.store_state != inside_store_state:
                continue
            age = (match_utc - session.entry_utc).total_seconds()
            if age > max_age_seconds:
                continue
            sim = cosine_fn(query, session.embedding)
            scored.append((vis_id, sim, session))

        if not scored:
            return SessionMatchResult(
                matched_vis=None,
                best_vis="-",
                best_sim=-1.0,
                second_best_sim=-1.0,
                margin=0.0,
                area_ratio=0.0,
                session_age=0.0,
                accepted=False,
                reason="no_inside_sessions",
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        best_vis, best_sim, best_session = scored[0]
        session_age = (match_utc - best_session.entry_utc).total_seconds()
        area_ratio = query_area / max(best_session.entry_bbox_area, 1.0)

        if len(scored) < 2:
            second_best_sim = -1.0
            margin = best_sim
        else:
            second_best_sim = scored[1][1]
            margin = best_sim - second_best_sim

        if best_sim < threshold:
            reason = "below_threshold"
            accepted = False
            matched_vis = None
        elif len(scored) >= 2 and margin < margin_threshold:
            reason = "margin_too_small"
            accepted = False
            matched_vis = None
        elif area_ratio < area_ratio_min or area_ratio > area_ratio_max:
            reason = "area_ratio_out_of_range"
            accepted = False
            matched_vis = None
        else:
            reason = "fragment_accepted"
            accepted = True
            matched_vis = best_vis

        return SessionMatchResult(
            matched_vis=matched_vis,
            best_vis=best_vis,
            best_sim=best_sim,
            second_best_sim=second_best_sim,
            margin=margin,
            area_ratio=area_ratio,
            session_age=session_age,
            accepted=accepted,
            reason=reason,
        )
