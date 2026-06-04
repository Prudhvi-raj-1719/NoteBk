"""ByteTrack track_id -> VIS_00001 style visitor IDs (per run scope)."""

from __future__ import annotations


class VisitorRegistry:
    def __init__(self) -> None:
        self._track_to_vis: dict[int, str] = {}
        self._next_index = 1

    def get_vis_id(self, byte_track_id: int) -> str:
        tid = int(byte_track_id)
        if tid not in self._track_to_vis:
            self._track_to_vis[tid] = f"VIS_{self._next_index:05d}"
            self._next_index += 1
        return self._track_to_vis[tid]

    def link_track(self, byte_track_id: int, vis_id: str) -> str:
        """Assign an existing VIS_* id to a new ByteTrack id (REENTRY)."""
        tid = int(byte_track_id)
        self._track_to_vis[tid] = vis_id
        return vis_id

    def vis_for_track(self, byte_track_id: int) -> str | None:
        return self._track_to_vis.get(int(byte_track_id))
