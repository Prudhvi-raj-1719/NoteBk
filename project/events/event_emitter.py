"""Shared competition-schema JSONL writer for CAM1 / CAM3 / CAM5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from configs.competition_config import (
    CAMERA_COMPETITION_IDS,
    DEFAULT_EVENT_CONFIDENCE,
    REENTRY_ENTRY_Y_TOLERANCE,
    REENTRY_LOCATION_MAX_NORM_DIST,
    REENTRY_TIME_WINDOW_SECONDS,
    STORE_ID,
)
from events.event_schema import build_event
from events.event_time import video_offset_to_utc_iso
from events.event_type_map import to_competition_event_type
from events.reentry_session import ReentrySessionManager, parse_utc_from_iso
from events.visitor_registry import VisitorRegistry
from events.zone_id_map import resolve_zone_id

if TYPE_CHECKING:
    from reid.reid_manager import ReIDManager


@dataclass
class EmitterStats:
    total: int = 0
    entry: int = 0
    exit: int = 0
    reentry: int = 0
    zone_enter: int = 0
    zone_exit: int = 0
    zone_dwell: int = 0
    queue_join: int = 0
    queue_abandon: int = 0
    payment_enter: int = 0
    payment_exit: int = 0
    inits: int = 0

    def bump(self, internal_type: str, competition_type: str) -> None:
        self.total += 1
        if internal_type == "ENTRY":
            self.entry += 1
        elif internal_type == "EXIT":
            self.exit += 1
        elif internal_type == "REENTRY":
            self.reentry += 1
        elif internal_type == "ZONE_ENTER" or (
            internal_type == "PAYMENT_ENTER"
        ):
            if internal_type == "PAYMENT_ENTER":
                self.payment_enter += 1
            else:
                self.zone_enter += 1
        elif internal_type == "ZONE_EXIT" or internal_type == "PAYMENT_EXIT":
            if internal_type == "PAYMENT_EXIT":
                self.payment_exit += 1
            else:
                self.zone_exit += 1
        elif internal_type == "DWELL_COMPLETED":
            self.zone_dwell += 1
        elif internal_type == "QUEUE_ENTER":
            self.queue_join += 1
        elif internal_type == "QUEUE_EXIT":
            self.queue_abandon += 1
        _ = competition_type


class EventEmitter:
    """Writes competition JSONL; accepts internal NOTEBK emit payloads."""

    def __init__(
        self,
        output_path: Path,
        notbk_camera_id: str,
        *,
        clip_id: Optional[str] = None,
        video_width: int = 0,
        video_height: int = 0,
        enable_reentry: bool = False,
        entry_plane_y_norm: float = 0.54,
        confidence: float = DEFAULT_EVENT_CONFIDENCE,
        reid_manager: Optional["ReIDManager"] = None,
        heuristic_reentry_fallback: bool = False,
    ) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")
        self.notbk_camera_id = notbk_camera_id
        self.camera_id = CAMERA_COMPETITION_IDS.get(
            notbk_camera_id, notbk_camera_id
        )
        self.clip_id = clip_id
        self.video_width = video_width
        self.video_height = video_height
        self.confidence = confidence
        self.stats = EmitterStats()
        self.registry = VisitorRegistry()
        self.reid_manager = reid_manager
        self.heuristic_reentry_fallback = heuristic_reentry_fallback
        self.reentry: Optional[ReentrySessionManager] = None
        use_heuristic = enable_reentry and (
            reid_manager is None or heuristic_reentry_fallback
        )
        if use_heuristic and video_width > 0 and video_height > 0:
            self.reentry = ReentrySessionManager(
                time_window_seconds=REENTRY_TIME_WINDOW_SECONDS,
                location_max_norm_dist=REENTRY_LOCATION_MAX_NORM_DIST,
                entry_plane_y_norm=entry_plane_y_norm,
                entry_y_tolerance=REENTRY_ENTRY_Y_TOLERANCE,
            )

    def log_init_terminal(self, init_debug: Dict[str, Any]) -> None:
        self.stats.inits += 1
        print(
            f"[INIT] visitor_id={init_debug.get('visitor_id')} "
            f"initial_side={init_debug.get('initial_side')} "
            f"initial_store_state={init_debug.get('initial_store_state')}"
        )

    def emit_internal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        internal_type = str(payload.get("event_type", ""))
        byte_track_id = int(payload["visitor_id"])
        video_offset = str(payload["timestamp"])
        center = payload.get("center")
        zone_name = payload.get("zone")

        utc_iso = video_offset_to_utc_iso(
            self.notbk_camera_id,
            video_offset,
            clip_id=self.clip_id,
        )
        event_utc = parse_utc_from_iso(utc_iso)

        internal_emit_type = internal_type
        reentry_from_vis: Optional[str] = None
        reid_similarity: Optional[float] = None
        reid_matched_vis: Optional[str] = None

        suppress_write = False
        if self.reid_manager is not None:
            vis_id = self.reid_manager.vis_for_track(byte_track_id)

            if internal_type == "ENTRY":
                matched_vis, sim = self.reid_manager.match_entry(
                    byte_track_id, event_utc
                )
                if matched_vis:
                    vis_id = self.reid_manager.link_track(byte_track_id, matched_vis)
                    internal_emit_type = "REENTRY"
                    reentry_from_vis = matched_vis
                    reid_matched_vis = matched_vis
                    reid_similarity = sim
                    if self.reid_manager and hasattr(self.reid_manager, "_log_debug"):
                        self.reid_manager._log_debug(
                            byte_track_id,
                            vis_id,
                            similarity=sim,
                            matched_vis=matched_vis,
                            reason="entry_reentry_osnet",
                        )
                else:
                    if self.reid_manager:
                        self.reid_manager._log_debug(
                            byte_track_id,
                            vis_id,
                            similarity=sim,
                            matched_vis=None,
                            reason="entry_new_vis",
                        )

            elif internal_type == "EXIT":
                if not self.reid_manager.should_emit_exit(vis_id, event_utc):
                    self.reid_manager._log_debug(
                        byte_track_id,
                        vis_id,
                        reason="exit_suppressed_duplicate",
                    )
                    suppress_write = True
                else:
                    self.reid_manager.record_exit(vis_id, byte_track_id, event_utc)

            if (
                not suppress_write
                and internal_emit_type in ("ENTRY", "REENTRY")
                and not self.reid_manager.should_emit_entry(vis_id, internal_emit_type)
            ):
                self.reid_manager._log_debug(
                    byte_track_id,
                    vis_id,
                    reason="entry_suppressed_open_session",
                )
                suppress_write = True

            if internal_emit_type in ("ENTRY", "REENTRY") and not suppress_write:
                bbox = payload.get("bbox")
                self.reid_manager.open_active_session(
                    vis_id,
                    byte_track_id,
                    event_utc,
                    bbox,
                )

        else:
            vis_id = self.registry.get_vis_id(byte_track_id)

        if (
            internal_type == "ENTRY"
            and internal_emit_type == "ENTRY"
            and self.reentry is not None
            and center is not None
        ):
            matched = self.reentry.match_reentry(
                tuple(center),
                event_utc,
                self.video_width,
                self.video_height,
            )
            if matched:
                vis_id = (
                    self.reid_manager.link_track(byte_track_id, matched)
                    if self.reid_manager
                    else self.registry.link_track(byte_track_id, matched)
                )
                internal_emit_type = "REENTRY"
                reentry_from_vis = matched
                self.reentry.consume_reentry(matched)

        if internal_emit_type in ("ENTRY", "REENTRY") and self.reentry is not None:
            self.reentry.on_entry(vis_id, event_utc)
        elif (
            internal_emit_type == "EXIT"
            and self.reentry is not None
            and center is not None
        ):
            self.reentry.on_exit(
                vis_id,
                byte_track_id,
                tuple(center),
                event_utc,
                self.video_width,
                self.video_height,
            )

        competition_type = to_competition_event_type(internal_emit_type)
        dwell_ms = 0
        if internal_type == "DWELL_COMPLETED":
            dwell_seconds = float(payload.get("dwell_seconds", 0))
            dwell_ms = max(0, int(round(dwell_seconds * 1000)))

        metadata: Dict[str, Any] = {
            "byte_track_id": byte_track_id,
            "video_offset": video_offset,
            "internal_event_type": internal_type,
        }
        if zone_name:
            metadata["sku_zone"] = zone_name
        if self.clip_id:
            metadata["clip_id"] = self.clip_id
        if payload.get("debug"):
            metadata["debug"] = payload["debug"]
        if reid_similarity is not None:
            metadata["reid_similarity"] = round(reid_similarity, 4)
        if reid_matched_vis:
            metadata["reid_matched_vis"] = reid_matched_vis
        if reentry_from_vis:
            metadata["reentry_from_vis"] = reentry_from_vis
            metadata["reentry_match"] = (
                "osnet" if self.reid_manager is not None and reid_matched_vis else "time_location_cache"
            )
        if internal_emit_type == "REENTRY":
            metadata["is_reentry"] = True

        row = build_event(
            camera_id=self.camera_id,
            visitor_id=vis_id,
            event_type=competition_type,
            timestamp=utc_iso,
            zone_id=resolve_zone_id(
                self.notbk_camera_id, internal_emit_type, zone_name
            ),
            dwell_ms=dwell_ms,
            confidence=self.confidence,
            metadata=metadata,
            store_id=STORE_ID,
        )

        if suppress_write:
            metadata["suppressed"] = True
        if not suppress_write:
            self._write_row(row, internal_emit_type, competition_type)
            if (
                self.reid_manager is not None
                and internal_emit_type in ("ENTRY", "REENTRY")
            ):
                self.reid_manager.mark_entry_event_emitted(vis_id)
        row["metadata"] = metadata
        return row

    def emit_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """CAM1/CAM5 internal dict (visitor_id, camera, event_type, timestamp, ...)."""
        return self.emit_internal(event)

    def emit_transition(
        self,
        event_row: Dict[str, Any],
        *,
        center: Optional[Tuple[int, int]] = None,
        bbox: Optional[List[float]] = None,
        adjust_engine: Any = None,
    ) -> Dict[str, Any]:
        """CAM3 transition row from entry_retail."""
        payload = dict(event_row)
        if center is not None:
            payload["center"] = center
        if bbox is not None:
            payload["bbox"] = bbox
        row = self.emit_internal(payload)
        meta = row.get("metadata") or {}
        if adjust_engine is not None and meta.get("is_reentry"):
            if hasattr(adjust_engine, "entry_count") and adjust_engine.entry_count > 0:
                adjust_engine.entry_count -= 1
        if adjust_engine is not None and meta.get("suppressed"):
            internal = meta.get("internal_event_type", "")
            if internal == "EXIT" and getattr(adjust_engine, "exit_count", 0) > 0:
                adjust_engine.exit_count -= 1
            elif internal == "ENTRY" and getattr(adjust_engine, "entry_count", 0) > 0:
                adjust_engine.entry_count -= 1
        return row

    def _write_row(
        self,
        row: Dict[str, Any],
        internal_type: str,
        competition_type: str,
    ) -> None:
        self.stats.bump(internal_type, competition_type)
        print("[EVENT]")
        for key, value in row.items():
            print(f"{key}={value}")
        print()
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def print_summary_cam1(self, stabilization: Any = None) -> None:
        print("=" * 40)
        print(f"Total Events: {self.stats.total}")
        print(f"ZONE_ENTER count: {self.stats.zone_enter}")
        print(f"ZONE_EXIT count: {self.stats.zone_exit}")
        print(f"ZONE_DWELL count: {self.stats.zone_dwell}")
        if stabilization is not None:
            print(
                f"Ignored zone transitions: "
                f"{getattr(stabilization, 'ignored_zone_transitions', 0)}"
            )
            print(
                f"Ignored short dwells: "
                f"{getattr(stabilization, 'ignored_short_dwells', 0)}"
            )
        print(f"Events written to: {self.output_path.resolve()}")

    def print_summary_cam5(self, stabilization: Any = None) -> None:
        print("=" * 40)
        print(f"Total Events: {self.stats.total}")
        print(f"BILLING_QUEUE_JOIN count: {self.stats.queue_join}")
        print(f"BILLING_QUEUE_EXIT count: {self.stats.queue_abandon}")
        print(f"PAYMENT ZONE_ENTER count: {self.stats.payment_enter}")
        print(f"PAYMENT ZONE_EXIT count: {self.stats.payment_exit}")
        print(f"ZONE_DWELL count: {self.stats.zone_dwell}")
        if stabilization is not None:
            print(
                f"Ignored zone transitions: "
                f"{getattr(stabilization, 'ignored_zone_transitions', 0)}"
            )
            print(
                f"Ignored short dwells: "
                f"{getattr(stabilization, 'ignored_short_dwells', 0)}"
            )
        print(f"Events written to: {self.output_path.resolve()}")

    def print_summary_cam3(self) -> None:
        print("=" * 40)
        print(f"ENTRY count: {self.stats.entry}")
        print(f"EXIT count: {self.stats.exit}")
        print(f"REENTRY count: {self.stats.reentry}")
        print(f"Track initializations (terminal only): {self.stats.inits}")
        print(f"Events written to: {self.output_path.resolve()}")
