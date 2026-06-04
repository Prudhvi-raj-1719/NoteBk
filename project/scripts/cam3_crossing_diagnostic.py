"""
CAM3 ENTRY/EXIT crossing diagnostic (read-only replay).

Does not change RetailEntryEngine logic. Replays Footage2 entry 1 (default) with
full per-processed-frame logging and writes a markdown report.

Usage:
  set CAM3_DIAG_CLIP=entry1
  set CAM3_REID_ENABLED=0
  python scripts/cam3_crossing_diagnostic.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from configs.camera_timing_config import (
    CAM3_ENTRY_CONFIG,
    CAM3_PROCESS_EVERY_N_FRAMES,
    DATA_DIR,
    MODEL_PATH,
    OUTPUTS_DIR,
)
from entry_retail import (
    EVENT_ENTRY,
    EVENT_EXIT,
    SIDE_MALL,
    SIDE_STORE,
    STATE_INSIDE,
    STATE_OUTSIDE,
    build_retail_entry_engine,
    format_video_timestamp,
)
from events.cam3_events import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    PERSON_CLASS_ID,
    create_byte_tracker,
    detect_persons,
)

FOOTAGE2_DIR = DATA_DIR / "CCTV Footage_2"
CLIPS = {
    "entry1": (
        FOOTAGE2_DIR / "entry 1.mp4",
        OUTPUTS_DIR / "cam3_footage2_entry1_events.jsonl",
    ),
    "entry2": (
        FOOTAGE2_DIR / "entry 2.mp4",
        OUTPUTS_DIR / "cam3_footage2_entry2_events.jsonl",
    ),
}
REPORT_PATH = OUTPUTS_DIR / "reports" / "cam3_entry1_crossing_diagnostic.md"


@dataclass
class FrameLog:
    frame: int
    timestamp: str
    num_detections: int
    track_ids: List[int]
    entries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrackLifecycle:
    track_id: int
    birth_frame: int
    death_frame: Optional[int] = None
    init_side: str = ""
    init_store_state: str = ""
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    id_changes: List[Tuple[int, int, int]] = field(default_factory=list)


class CrossingDiagnostic:
    def __init__(self, layout: Dict[str, Any], fps: float) -> None:
        self.fps = fps
        self.frame_logs: List[FrameLog] = []
        self.track_lifecycles: Dict[int, TrackLifecycle] = {}
        self.prev_active: Set[int] = set()
        self.pending_started: Dict[int, Tuple[int, str, str]] = {}
        self.rejected_commits: List[Dict[str, Any]] = []
        self.skipped_frame_gaps: List[Dict[str, Any]] = []
        self.multi_person_frames: List[Dict[str, Any]] = []
        self.emitted_events: List[Dict[str, Any]] = []
        self.engine: Any = None

    def ts(self, frame: int) -> str:
        return format_video_timestamp(frame, self.fps)

    def log_detection_frame(
        self,
        frame: int,
        detections: sv.Detections,
    ) -> None:
        ids = (
            [int(t) for t in detections.tracker_id]
            if detections.tracker_id is not None
            else []
        )
        active = set(ids)
        born = active - self.prev_active
        died = self.prev_active - active

        fl = FrameLog(
            frame=frame,
            timestamp=self.ts(frame),
            num_detections=len(ids),
            track_ids=ids,
        )

        if len(ids) == 1 and detections.xyxy is not None and len(detections) >= 1:
            x1, y1, x2, y2 = detections.xyxy[0]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0 and (w / h) > 1.35:
                self.multi_person_frames.append(
                    {
                        "frame": frame,
                        "track_id": ids[0],
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "aspect": round(float(w / h), 2),
                        "note": "single_detection_wide_box",
                    }
                )

        if len(ids) >= 2:
            boxes = detections.xyxy
            for i, tid in enumerate(ids):
                for j, tid2 in enumerate(ids):
                    if j <= i:
                        continue
                    b1, b2 = boxes[i], boxes[j]
                    iou = self._box_iou(b1, b2)
                    if iou > 0.25:
                        self.multi_person_frames.append(
                            {
                                "frame": frame,
                                "track_ids": [tid, tid2],
                                "iou": round(iou, 3),
                                "note": "overlapping_tracks_possible_merge",
                            }
                        )

        for tid in born:
            lc = TrackLifecycle(track_id=tid, birth_frame=frame)
            self.track_lifecycles[tid] = lc
            fl.entries.append({"event": "tracker_birth", "track_id": tid})

        for tid in died:
            if tid in self.track_lifecycles:
                self.track_lifecycles[tid].death_frame = frame
            fl.entries.append({"event": "tracker_death", "track_id": tid})
            self.pending_started.pop(tid, None)

        self.prev_active = active
        self.frame_logs.append(fl)

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def analyze_update(
        self,
        frame: int,
        track_id: int,
        xyxy: np.ndarray,
        result: Any,
        engine: Any,
    ) -> None:
        tid = int(track_id)
        x1, y1, x2, y2 = xyxy
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        record = engine._tracks.get(tid)
        if record is None:
            return

        instant_side = engine.geometry.classify_side(center)
        cross = engine.geometry.cross_value(center)

        entry = {
            "track_id": tid,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "centroid": center,
            "instant_side": instant_side,
            "confirmed_side": record.confirmed_side,
            "pending_side": record.pending_side,
            "pending_frames": record.pending_frames,
            "store_state": record.store_state,
            "cross_value": round(float(cross), 1),
        }

        if result.init_debug is not None:
            entry["init"] = result.init_debug.to_dict()
            if tid in self.track_lifecycles:
                self.track_lifecycles[tid].init_side = result.init_debug.initial_side
                self.track_lifecycles[tid].init_store_state = (
                    result.init_debug.initial_store_state
                )
            if result.init_debug.initial_store_state == STATE_INSIDE:
                entry["crossing_decision"] = (
                    "init_inside_no_entry_latch — track born INSIDE without ENTRY"
                )

        if result.transition is not None:
            entry["transition"] = {
                "event_type": result.transition.event_type,
                "timestamp": result.transition.timestamp,
                "debug": result.transition.debug,
            }
            self.emitted_events.append(
                {
                    "frame": frame,
                    "track_id": tid,
                    "event_type": result.transition.event_type,
                    "timestamp": result.transition.timestamp,
                }
            )
            if tid in self.track_lifecycles:
                self.track_lifecycles[tid].transitions.append(entry["transition"])

        prev_pending = self.pending_started.get(tid)
        if record.pending_side and record.pending_frames == 1:
            self.pending_started[tid] = (
                frame,
                record.confirmed_side,
                record.pending_side,
            )
        elif record.pending_side is None and prev_pending:
            start_frame, from_side, to_side = prev_pending
            if (frame - start_frame) < engine.entry_stability_frames:
                self.rejected_commits.append(
                    {
                        "frame": frame,
                        "track_id": tid,
                        "reason": "stability_reset",
                        "detail": (
                            f"pending {from_side}->{to_side} abandoned after "
                            f"{frame - start_frame} processed frames (<3)"
                        ),
                        "store_state": record.store_state,
                    }
                )
            self.pending_started.pop(tid, None)

        if record.pending_frames > 0:
            needed = engine._stability_required(
                record.confirmed_side, record.pending_side or ""
            )
            entry["stability_needed"] = needed
            if (
                record.pending_frames >= needed
                and result.transition is None
            ):
                entry_from, entry_to = (
                    (SIDE_STORE, SIDE_MALL)
                    if engine.invert_retail_semantics
                    else (SIDE_MALL, SIDE_STORE)
                )
                exit_from, exit_to = entry_to, entry_from
                ps, cs = record.confirmed_side, record.pending_side
                blocked = None
                if ps == entry_from and cs == entry_to:
                    if record.store_state != STATE_OUTSIDE:
                        blocked = "ENTRY_blocked_store_state_not_OUTSIDE"
                elif ps == exit_from and cs == exit_to:
                    if record.store_state != STATE_INSIDE:
                        blocked = "EXIT_blocked_store_state_not_INSIDE"
                else:
                    blocked = "side_pair_not_entry_or_exit_transition"
                if blocked:
                    entry["crossing_decision"] = (
                        f"committed_side_change_but_no_event — {blocked}"
                    )
                    self.rejected_commits.append(
                        {
                            "frame": frame,
                            "track_id": tid,
                            "reason": blocked,
                            "pending_frames": record.pending_frames,
                            "store_state": record.store_state,
                            "confirmed_side": record.confirmed_side,
                            "pending_side": record.pending_side,
                        }
                    )

        if self.frame_logs:
            self.frame_logs[-1].entries.append(entry)


def load_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def run_diagnostic(clip_id: str = "entry1") -> Path:
    video_path, jsonl_path = CLIPS[clip_id]
    layout = CAM3_ENTRY_CONFIG["FOOTAGE2"]
    process_n = CAM3_PROCESS_EVERY_N_FRAMES

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    yolo = YOLO(str(MODEL_PATH))
    tracker = create_byte_tracker()
    engine = build_retail_entry_engine(
        layout,
        w,
        h,
        timestamp_fn=lambda f, _fps=fps: format_video_timestamp(f, _fps),
    )
    diag = CrossingDiagnostic(layout, fps)
    diag.engine = engine

    last_processed_frame = -process_n
    original_frame = 0
    processed_frames: List[int] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if original_frame % process_n != 0:
            original_frame += 1
            continue

        processed_frames.append(original_frame)
        if last_processed_frame >= 0:
            gap = original_frame - last_processed_frame
            if gap > process_n:
                diag.skipped_frame_gaps.append(
                    {
                        "from_frame": last_processed_frame,
                        "to_frame": original_frame,
                        "gap_frames": gap,
                        "note": f"PROCESS_EVERY_N_FRAMES={process_n} may skip crossings",
                    }
                )
        last_processed_frame = original_frame

        detections = detect_persons(frame, yolo)
        detections = tracker.update_with_detections(detections)
        diag.log_detection_frame(original_frame, detections)

        if detections.tracker_id is not None:
            for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                tid = int(track_id)
                center = (
                    int((xyxy[0] + xyxy[2]) / 2),
                    int((xyxy[1] + xyxy[3]) / 2),
                )
                result = engine.update(tid, center, original_frame=original_frame)
                diag.analyze_update(original_frame, tid, xyxy, result, engine)

        original_frame += 1

    cap.release()

    actual_jsonl = load_jsonl_events(jsonl_path)
    report = build_report(
        clip_id=clip_id,
        video_path=video_path,
        jsonl_path=jsonl_path,
        diag=diag,
        engine=engine,
        actual_jsonl=actual_jsonl,
        processed_frames=processed_frames,
        process_n=process_n,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = REPORT_PATH if clip_id == "entry1" else REPORT_PATH.with_name(
        f"cam3_{clip_id}_crossing_diagnostic.md"
    )
    out.write_text(report, encoding="utf-8")
    return out


def build_report(
    *,
    clip_id: str,
    video_path: Path,
    jsonl_path: Path,
    diag: CrossingDiagnostic,
    engine: Any,
    actual_jsonl: List[Dict[str, Any]],
    processed_frames: List[int],
    process_n: int,
) -> str:
    entry_emitted = [e for e in diag.emitted_events if e["event_type"] == EVENT_ENTRY]
    exit_emitted = [e for e in diag.emitted_events if e["event_type"] == EVENT_EXIT]

    init_inside = []
    for fl in diag.frame_logs:
        for ent in fl.entries:
            if isinstance(ent, dict) and ent.get("init", {}).get(
                "initial_store_state"
            ) == STATE_INSIDE:
                init_inside.append(
                    {
                        "frame": fl.frame,
                        "track_id": ent["track_id"],
                        "initial_side": ent["init"]["initial_side"],
                    }
                )

    lines: List[str] = []
    lines.append(f"# CAM3 Crossing Diagnostic — Footage2 `{clip_id}`")
    lines.append("")
    lines.append(f"**Video:** `{video_path}`")
    lines.append(f"**JSONL:** `{jsonl_path}`")
    lines.append(f"**PROCESS_EVERY_N_FRAMES:** {process_n}")
    lines.append("")
    lines.append("## Expected vs actual (user ground truth)")
    lines.append("")
    lines.append("| Metric | Expected | Diagnostic replay | JSONL on disk |")
    lines.append("|--------|----------|-------------------|---------------|")
    lines.append("| ENTRY | 3 | {0} | {1} |".format(
        len(entry_emitted),
        sum(1 for e in actual_jsonl if e.get("event_type") in ("ENTRY", "REENTRY")
              and e.get("metadata", {}).get("internal_event_type") == "ENTRY"),
    ))
    lines.append("| EXIT | 4 | {0} | {1} |".format(
        len(exit_emitted),
        sum(1 for e in actual_jsonl if e.get("event_type") == "EXIT"),
    ))
    lines.append("")
    lines.append("## Emitted transitions (replay)")
    lines.append("")
    for ev in diag.emitted_events:
        lines.append(
            f"- frame **{ev['frame']}** (`{ev['timestamp']}`) "
            f"track **{ev['track_id']}** → **{ev['event_type']}**"
        )
    lines.append("")
    lines.append("## JSONL on disk")
    lines.append("")
    for row in actual_jsonl:
        meta = row.get("metadata", {})
        lines.append(
            f"- `{row.get('timestamp')}` track {meta.get('byte_track_id')} "
            f"**{row.get('event_type')}** visitor {row.get('visitor_id')}"
        )
    lines.append("")
    lines.append("## Root-cause checklist (evidence)")
    lines.append("")

    # 1 YOLO merge
    wide = [m for m in diag.multi_person_frames if m.get("note") == "single_detection_wide_box"]
    lines.append("### 1. YOLO merged multiple people into one detection")
    lines.append("")
    if wide:
        lines.append(f"**Possible** — {len(wide)} processed frames with a single wide box (aspect > 1.35):")
        for m in wide[:15]:
            lines.append(f"- frame {m['frame']} track {m['track_id']} aspect={m['aspect']}")
    else:
        lines.append("**No strong evidence** — no single-detection ultra-wide boxes on processed frames.")
    lines.append("")

    # 2 ByteTrack merge
    overlap = [m for m in diag.multi_person_frames if "overlapping" in m.get("note", "")]
    lines.append("### 2. ByteTrack merged two people into one track")
    lines.append("")
    lines.append(
        f"Tracker births: {len([t for t in diag.track_lifecycles.values() if t.birth_frame >= 0])} | "
        f"deaths: {sum(1 for t in diag.track_lifecycles.values() if t.death_frame)}"
    )
    if overlap:
        lines.append(f"**Possible** — {len(overlap)} frames with overlapping track boxes (IoU>0.25):")
        for m in overlap[:10]:
            lines.append(f"- frame {m['frame']} tracks {m.get('track_ids')}")
    lines.append("")

    # 3 Track ID change
    lines.append("### 3. Track ID changed near entry line")
    lines.append("")
    deaths_near_line = [
        (tid, lc.death_frame)
        for tid, lc in diag.track_lifecycles.items()
        if lc.death_frame is not None
    ]
    births = [(tid, lc.birth_frame) for tid, lc in diag.track_lifecycles.items()]
    lines.append("Track lifecycles (birth → death):")
    for tid, lc in sorted(diag.track_lifecycles.items(), key=lambda x: x[1].birth_frame):
        trans = ", ".join(
            f"{t['event_type']}@{t['debug'].get('current_side', '?')}"
            for t in lc.transitions
        ) or "no_transition"
        lines.append(
            f"- track **{tid}**: born frame {lc.birth_frame} "
            f"init {lc.init_store_state}/{lc.init_side} "
            f"death {lc.death_frame or 'active'} — {trans}"
        )
    lines.append("")

    # 4 Stability
    lines.append("### 4. Stability filter rejected a valid crossing")
    lines.append("")
    if diag.rejected_commits:
        for r in diag.rejected_commits[:20]:
            lines.append(
                f"- frame {r['frame']} track {r['track_id']}: **{r['reason']}** — {r.get('detail', r)}"
            )
    else:
        lines.append("No explicit stability-reset rejections logged.")
    lines.append("")

    # 5 Frame skip
    lines.append("### 5. PROCESS_EVERY_N_FRAMES skipped crossing")
    lines.append("")
    lines.append(
        f"Processed **{len(processed_frames)}** frames (every {process_n}th). "
        "Crossings between sampled frames are invisible to the engine."
    )
    lines.append("")

    # 6 Never committed
    lines.append("### 6. Crossing occurred but transition never committed")
    lines.append("")
    blocked = [r for r in diag.rejected_commits if "blocked" in r.get("reason", "")]
    init_in = init_inside
    if init_in:
        lines.append(
            f"**Confirmed — init INSIDE without ENTRY:** {len(init_in)} tracks"
        )
        for row in init_in:
            lines.append(
                f"- frame {row['frame']} track {row['track_id']} "
                f"(initial_side={row['initial_side']}) — no ENTRY possible until OUTSIDE latched"
            )
    if blocked:
        lines.append("**Blocked commits after stability:**")
        for r in blocked:
            lines.append(f"- frame {r['frame']} track {r['track_id']}: {r['reason']}")
    lines.append("")

    lines.append("## Init-without-ENTRY (primary loss mechanism)")
    lines.append("")
    lines.append(
        "Tracks that first appear with centroid already on MALL/INSIDE side never "
        "satisfy `store_state==OUTSIDE` at ENTRY commit → **ENTRY suppressed**."
    )
    lines.append("")

    lines.append("## Per-processed-frame timeline (excerpt)")
    lines.append("")
    lines.append("```")
    for fl in diag.frame_logs:
        if not any(
            e.get("transition") or e.get("init") or e.get("crossing_decision")
            for e in fl.entries
            if isinstance(e, dict)
        ):
            continue
        lines.append(f"--- frame {fl.frame} ({fl.timestamp}) det={fl.num_detections} ids={fl.track_ids} ---")
        for ent in fl.entries:
            if not isinstance(ent, dict) or "track_id" not in ent:
                continue
            parts = [
                f"track={ent['track_id']}",
                f"side={ent.get('confirmed_side')}",
                f"pending={ent.get('pending_side')}({ent.get('pending_frames')})",
                f"store={ent.get('store_state')}",
            ]
            if ent.get("transition"):
                parts.append(f"EMIT={ent['transition']['event_type']}")
            if ent.get("init"):
                parts.append(f"INIT={ent['init']['initial_store_state']}")
            if ent.get("crossing_decision"):
                parts.append(ent["crossing_decision"])
            lines.append("  " + " ".join(parts))
    lines.append("```")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    missing_entry = 3 - len(entry_emitted)
    missing_exit = 4 - len(exit_emitted)
    lines.append(
        f"Replay engine counts: ENTRY={engine.entry_count} EXIT={engine.exit_count}. "
        f"Gap vs expectation: **{missing_entry} ENTRY**, **{missing_exit} EXIT**."
    )
    if init_inside:
        lines.append(
            f"\nMost likely: **#6 + init INSIDE** — at least {len(init_inside)} track(s) "
            "initialized already INSIDE (geometry), so ENTRY never fires; "
            "paired EXIT may be missing if those people were never latched INSIDE via ENTRY "
            "or shared a ByteTrack ID with another visitor."
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    clip = os.getenv("CAM3_DIAG_CLIP", "entry1")
    os.environ["CAM3_REID_ENABLED"] = "0"
    out = run_diagnostic(clip)
    print(f"Report written: {out.resolve()}")


if __name__ == "__main__":
    main()
