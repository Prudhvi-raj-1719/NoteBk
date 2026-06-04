"""Lightweight REENTRY: time + location + session cache (CAM3, no embeddings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from events.event_time import utc_iso_to_datetime


@dataclass
class ExitCacheRecord:
    vis_id: str
    byte_track_id: int
    exit_utc: datetime
    norm_x: float
    norm_y: float


@dataclass
class ReentrySessionManager:
    time_window_seconds: float
    location_max_norm_dist: float
    entry_plane_y_norm: float
    entry_y_tolerance: float
    recent_exits: List[ExitCacheRecord] = field(default_factory=list)
    open_sessions: dict[str, datetime] = field(default_factory=dict)

    def _norm_point(
        self,
        center: Tuple[int, int],
        video_width: int,
        video_height: int,
    ) -> Tuple[float, float]:
        w = max(video_width, 1)
        h = max(video_height, 1)
        return center[0] / w, center[1] / h

    def _near_entry_boundary(self, norm_x: float, norm_y: float) -> bool:
        return abs(norm_y - self.entry_plane_y_norm) <= self.entry_y_tolerance

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _prune_exits(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.time_window_seconds)
        self.recent_exits = [r for r in self.recent_exits if r.exit_utc >= cutoff]

    def on_entry(self, vis_id: str, entry_utc: datetime) -> None:
        self.open_sessions[vis_id] = entry_utc

    def on_exit(
        self,
        vis_id: str,
        byte_track_id: int,
        center: Tuple[int, int],
        exit_utc: datetime,
        video_width: int,
        video_height: int,
    ) -> None:
        norm = self._norm_point(center, video_width, video_height)
        self.recent_exits.append(
            ExitCacheRecord(
                vis_id=vis_id,
                byte_track_id=byte_track_id,
                exit_utc=exit_utc,
                norm_x=norm[0],
                norm_y=norm[1],
            )
        )
        self.open_sessions.pop(vis_id, None)
        self._prune_exits(exit_utc)

    def match_reentry(
        self,
        center: Tuple[int, int],
        event_utc: datetime,
        video_width: int,
        video_height: int,
    ) -> Optional[str]:
        """Return VIS_* to reuse if a recent EXIT matches time + entry proximity."""
        norm = self._norm_point(center, video_width, video_height)
        if not self._near_entry_boundary(norm[0], norm[1]):
            return None

        self._prune_exits(event_utc)
        best_vis: Optional[str] = None
        best_dist = self.location_max_norm_dist + 1.0

        for record in self.recent_exits:
            if record.vis_id in self.open_sessions:
                continue
            age = (event_utc - record.exit_utc).total_seconds()
            if age < 0 or age > self.time_window_seconds:
                continue
            dist = self._distance(norm, (record.norm_x, record.norm_y))
            if dist <= self.location_max_norm_dist and dist < best_dist:
                best_dist = dist
                best_vis = record.vis_id

        return best_vis

    def consume_reentry(self, vis_id: str) -> None:
        """Remove matched EXIT from cache so the same visitor is not double-matched."""
        self.recent_exits = [r for r in self.recent_exits if r.vis_id != vis_id]


def parse_utc_from_iso(timestamp_iso: str) -> datetime:
    return utc_iso_to_datetime(timestamp_iso)
