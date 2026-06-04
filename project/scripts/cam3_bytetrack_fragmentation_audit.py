"""
CAM3 ByteTrack fragmentation audit (read-only).

Usage:
  python scripts/cam3_bytetrack_fragmentation_audit.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    STATE_INSIDE,
    STATE_UNKNOWN,
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

VIDEO_PATH = DATA_DIR / "CCTV Footage_2" / "entry 1.mp4"
REPORT_PATH = OUTPUTS_DIR / "reports" / "cam3_bytetrack_fragmentation_audit.md"
LAYOUT = CAM3_ENTRY_CONFIG["FOOTAGE2"]
PROCESS_N = CAM3_PROCESS_EVERY_N_FRAMES

# Re-spawn link: birth within this many processed frames after death, centroid proximity.
RESPAWN_MAX_GAP_PROC = 6
RESPAWN_MAX_NORM_DIST = 0.12


@dataclass
class TrackSpan:
    track_id: int
    birth_frame: int
    death_frame: Optional[int] = None
    last_centroid: Tuple[float, float] = (0.0, 0.0)
    centroids: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)
    retail_inits: int = 0
    born_inside: bool = False
    died_pending_exit: bool = False
    died_pending_entry: bool = False
    died_pending_frames: int = 0
    died_store_inside: bool = False
    transitions: List[str] = field(default_factory=list)


def _norm_centroid(cx: float, cy: float, w: int, h: int) -> Tuple[float, float]:
    return cx / max(w, 1), cy / max(h, 1)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _byte_track_params(tracker: sv.ByteTrack) -> Dict[str, Any]:
    keys = [
        "track_activation_threshold",
        "det_thresh",
        "minimum_matching_threshold",
        "max_time_lost",
        "minimum_consecutive_frames",
        "frame_id",
    ]
    return {k: getattr(tracker, k, None) for k in keys if hasattr(tracker, k)}


def run_audit() -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    yolo = YOLO(str(MODEL_PATH))
    tracker = create_byte_tracker()
    bt_params = _byte_track_params(tracker)
    engine = build_retail_entry_engine(LAYOUT, vw, vh)

    spans: Dict[int, TrackSpan] = {}
    prev_active: set[int] = set()
    respawns: List[Dict[str, Any]] = []
    recent_deaths: List[Tuple[int, int, Tuple[float, float], TrackSpan]] = []

    retail_inits = 0
    born_inside = 0
    emitted_entry = 0
    emitted_exit = 0
    missed_exit_stability = 0
    missed_exit_fragment = 0
    missed_entry_inside = 0
    threshold_observing = 0
    frame_idx = 0
    processed = 0

    (_, _), (exit_from, exit_to) = engine._entry_exit_sides()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % PROCESS_N != 0:
            frame_idx += 1
            continue
        processed += 1

        detections = detect_persons(frame, yolo)
        detections = tracker.update_with_detections(detections)
        active: set[int] = set()

        if detections.tracker_id is not None:
            for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                tid = int(track_id)
                active.add(tid)
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                nc = _norm_centroid(cx, cy, vw, vh)

                if tid not in spans:
                    spans[tid] = TrackSpan(track_id=tid, birth_frame=frame_idx)
                spans[tid].last_centroid = nc
                spans[tid].centroids.append((frame_idx, nc))

                center = (int(cx), int(cy))
                result = engine.update(tid, center, original_frame=frame_idx)
                record = engine._tracks.get(tid)

                if result.init_debug is not None:
                    retail_inits += 1
                    spans[tid].retail_inits += 1
                    if result.init_debug.initial_store_state == STATE_INSIDE:
                        spans[tid].born_inside = True
                        born_inside += 1
                    if record and record.observing:
                        threshold_observing += 1

                if result.transition is not None:
                    spans[tid].transitions.append(result.transition.event_type)
                    if result.transition.event_type == EVENT_ENTRY:
                        emitted_entry += 1
                    elif result.transition.event_type == EVENT_EXIT:
                        emitted_exit += 1

        born = active - prev_active
        died = prev_active - active

        for tid in born:
            if tid not in spans:
                spans[tid] = TrackSpan(track_id=tid, birth_frame=frame_idx)
            nc = spans[tid].last_centroid
            for d_frame, d_tid, d_nc, d_span in recent_deaths:
                gap_proc = (frame_idx - d_frame) // PROCESS_N
                if gap_proc > RESPAWN_MAX_GAP_PROC:
                    continue
                if _dist(nc, d_nc) > RESPAWN_MAX_NORM_DIST:
                    continue
                respawns.append(
                    {
                        "old_id": d_tid,
                        "new_id": tid,
                        "death_frame": d_frame,
                        "birth_frame": frame_idx,
                        "gap_processed_frames": gap_proc,
                        "norm_dist": round(_dist(nc, d_nc), 4),
                        "old_lifetime_proc": (
                            (d_span.death_frame or d_frame) - d_span.birth_frame
                        )
                        // PROCESS_N
                        + 1,
                        "old_transitions": list(d_span.transitions),
                        "old_born_inside": d_span.born_inside,
                        "old_died_pending_exit": d_span.died_pending_exit,
                    }
                )

        for tid in died:
            if tid in spans:
                spans[tid].death_frame = frame_idx
                rec = engine._tracks.get(tid)
                if rec:
                    if rec.store_state == STATE_INSIDE and rec.pending_side == exit_to:
                        spans[tid].died_pending_exit = True
                        spans[tid].died_pending_frames = rec.pending_frames
                        if rec.pending_frames >= 1:
                            missed_exit_stability += 1
                        else:
                            missed_exit_fragment += 1
                    if rec.store_state == STATE_INSIDE and rec.pending_side is None:
                        if spans[tid].born_inside and EVENT_ENTRY not in spans[tid].transitions:
                            missed_entry_inside += 1
                recent_deaths.append(
                    (frame_idx, tid, spans[tid].last_centroid, spans[tid])
                )
            recent_deaths = [
                (f, t, c, s)
                for f, t, c, s in recent_deaths
                if (frame_idx - f) // PROCESS_N <= RESPAWN_MAX_GAP_PROC
            ]

        if getattr(engine, "enable_track_loss_flush", False):
            engine.flush_removed_tracks(active, frame_idx)

        prev_active = active
        frame_idx += 1

    cap.release()

    completed = [s for s in spans.values() if s.death_frame is not None]
    lifetimes_proc = [
        ((s.death_frame or s.birth_frame) - s.birth_frame) // PROCESS_N + 1
        for s in completed
    ]
    avg_lifetime = sum(lifetimes_proc) / len(lifetimes_proc) if lifetimes_proc else 0.0

    track_12_46 = [
        r for r in respawns if r["old_id"] == 12 or r["new_id"] == 46 or (12 in (r["old_id"], r["new_id"]) and 46 in (r["old_id"], r["new_id"]))
    ]

    return {
        "video": str(VIDEO_PATH),
        "total_frames": total_frames,
        "processed_frames": processed,
        "fps": fps,
        "process_n": PROCESS_N,
        "yolo_conf": CONFIDENCE_THRESHOLD,
        "yolo_iou": IOU_THRESHOLD,
        "bt_params": bt_params,
        "byte_births": len(spans),
        "byte_deaths": len(completed),
        "retail_inits": retail_inits,
        "born_inside": born_inside,
        "threshold_observing": threshold_observing,
        "emitted_entry": emitted_entry,
        "emitted_exit": emitted_exit,
        "avg_lifetime_proc": avg_lifetime,
        "avg_lifetime_sec": avg_lifetime * PROCESS_N / fps,
        "respawns": respawns,
        "track_12_46": track_12_46,
        "missed_exit_stability": missed_exit_stability,
        "missed_exit_fragment": missed_exit_fragment,
        "missed_entry_inside": missed_entry_inside,
        "died_pending_exit": sum(1 for s in completed if s.died_pending_exit),
        "stability_frames": (
            engine.entry_stability_frames,
            engine.exit_stability_frames,
        ),
        "recovery_enabled": engine.enable_threshold_recovery,
        "flush_enabled": engine.enable_track_loss_flush,
    }


def write_report(data: Dict[str, Any]) -> None:
    respawns = sorted(
        data["respawns"],
        key=lambda r: (r["gap_processed_frames"], r["norm_dist"]),
    )[:20]

    lines = [
        "# CAM3 ByteTrack Fragmentation Audit — Footage2 entry 1",
        "",
        f"**Video:** `{data['video']}`",
        f"**Frames:** {data['total_frames']} total | **{data['processed_frames']}** processed (every **{data['process_n']}**) @ {data['fps']:.1f} fps",
        "",
        "## 1. Current ByteTrack parameters",
        "",
        "CAM3 uses `sv.ByteTrack()` with **no custom arguments** (supervision defaults).",
        "",
        "| Parameter | Value | Notes |",
        "|-----------|-------|-------|",
    ]

    pmap = {
        "track_activation_threshold": "Min detection score to start a track (activation)",
        "det_thresh": "High-confidence detection threshold inside tracker",
        "minimum_matching_threshold": "IoU/cost match threshold (≈ `match_thresh`)",
        "max_time_lost": "Frames to keep lost track before removal (≈ `track_buffer` at update rate)",
        "minimum_consecutive_frames": "Hits before track is confirmed",
        "frame_id": "Internal frame counter",
    }
    for k, note in pmap.items():
        v = data["bt_params"].get(k, "n/a")
        lines.append(f"| `{k}` | {v} | {note} |")

    lines.extend(
        [
            "",
            "### YOLO detection (CAM3)",
            "",
            f"| Setting | Value |",
            f"|---------|-------|",
            f"| `CONFIDENCE_THRESHOLD` | {data['yolo_conf']} |",
            f"| `IOU_THRESHOLD` | {data['yolo_iou']} |",
            f"| `PERSON_CLASS_ID` | 0 |",
            "",
            "**Important:** `max_time_lost` advances once per **processed** frame (every "
            f"{data['process_n']} video frames), so wall-clock retention ≈ "
            f"`max_time_lost` × {data['process_n']} / fps seconds.",
            "",
            "## 2. Track lifecycle statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| ByteTrack unique IDs (births) | {data['byte_births']} |",
            f"| Track deaths (IDs lost) | {data['byte_deaths']} |",
            f"| Retail engine initializations | {data['retail_inits']} |",
            f"| `born_inside_tracks` | {data['born_inside']} |",
            f"| Near-threshold observing inits | {data['threshold_observing']} |",
            f"| Emitted ENTRY | {data['emitted_entry']} |",
            f"| Emitted EXIT | {data['emitted_exit']} |",
            f"| Avg track lifetime (processed frames) | {data['avg_lifetime_proc']:.2f} |",
            f"| Avg track lifetime (seconds, approx) | {data['avg_lifetime_sec']:.2f} |",
            f"| Re-spawn links (death→birth proximity) | {len(data['respawns'])} |",
            f"| Deaths with pending EXIT (inside→store) | {data['died_pending_exit']} |",
            "",
            "## 3. Top 20 visitor trajectories that changed IDs",
            "",
            "Heuristic: same norm-centroid within **0.12** within **6** processed frames after death.",
            "",
        ]
    )

    if respawns:
        lines.append(
            "| rank | old_id | new_id | death_frame | birth_frame | gap(proc) | "
            "dist | old_life(proc) | old_events | pending_exit@death | born_inside |"
        )
        lines.append(
            "|------|--------|--------|-------------|-------------|-----------|"
            "------|--------------|------------|-------------------|-------------|"
        )
        for i, r in enumerate(respawns, 1):
            lines.append(
                f"| {i} | {r['old_id']} | {r['new_id']} | {r['death_frame']} | "
                f"{r['birth_frame']} | {r['gap_processed_frames']} | {r['norm_dist']} | "
                f"{r['old_lifetime_proc']} | {','.join(r['old_transitions']) or '-'} | "
                f"{r['old_died_pending_exit']} | {r['old_born_inside']} |"
            )
    else:
        lines.append("(no re-spawn pairs matched heuristic)")

    lines.extend(["", "## 4. Track 12 → 46 (user-reported)", ""])
    if data["track_12_46"]:
        for r in data["track_12_46"]:
            lines.append(f"- death **{r['old_id']}** @ frame {r['death_frame']} → birth **{r['new_id']}** @ {r['birth_frame']} (dist={r['norm_dist']})")
    else:
        t12 = [r for r in data["respawns"] if r["old_id"] == 12 or r["new_id"] == 12]
        t46 = [r for r in data["respawns"] if r["old_id"] == 46 or r["new_id"] == 46]
        lines.append("- No direct 12↔46 link in top proximity pairs.")
        if t12:
            lines.append(f"- Track **12** links: {t12[:3]}")
        if t46:
            lines.append(f"- Track **46** links: {t46[:3]}")

    stab_e, stab_x = data["stability_frames"]
    lines.extend(
        [
            "",
            "## 5. Missed ENTRY/EXIT — root-cause attribution",
            "",
            "| Cause | Mechanism | Count (this replay) |",
            "|-------|-----------|---------------------|",
            f"| **A) ByteTrack fragmentation** | ID lost/re-spawn; pending crossing incomplete at death | "
            f"re-spawns={len(data['respawns'])}, deaths w/ pending EXIT={data['died_pending_exit']}, "
            f"fragment exits={data['missed_exit_fragment']} |",
            f"| **B) Stability filter** | `pending_frames` ≥1 but < {stab_x} at track death | "
            f"**{data['missed_exit_stability']}** |",
            f"| **C) Frame skipping** | `PROCESS_EVERY_N_FRAMES={data['process_n']}` — crossing between samples | "
            f"qualitative (fast cross < {data['process_n']} frames) |",
            f"| **D) Threshold recovery** | born INSIDE / UNKNOWN observe | born_inside=**{data['born_inside']}**, "
            f"observe inits=**{data['threshold_observing']}** |",
            "",
            "### Primary driver (entry 1)",
            "",
        ]
    )

    primary = []
    if len(data["respawns"]) >= 10:
        primary.append("**A) ByteTrack fragmentation** — many ID re-spawns; short median lifetimes.")
    if data["missed_exit_stability"] >= data["missed_exit_fragment"]:
        primary.append("**B) Stability filter** — exits pending when IDs die before 2-frame commit.")
    if data["born_inside"] >= 5:
        primary.append("**D) Threshold recovery / init INSIDE** — suppresses ENTRY for IDs born past line.")
    primary.append(
        f"**C) Frame skipping** — amplifies A+B because tracker updates only every {data['process_n']} frames."
    )
    lines.extend(primary)

    lines.extend(
        [
            "",
            "## 6. Recommended parameter changes (do not apply yet)",
            "",
            "### ByteTrack (supervision `ByteTrack(...)`)",
            "",
            "```python",
            "sv.ByteTrack(",
            "    track_activation_threshold=0.30,  # was 0.25 — fewer junk tracks",
            "    minimum_matching_threshold=0.75,  # was 0.8 — easier re-associate after occlusion",
            "    max_time_lost=60,                 # was 30 — longer lost-track retention at proc rate",
            "    minimum_consecutive_frames=2,     # was 1 — reduce one-frame ID flicker",
            ")",
            "```",
            "",
            "At processed-frame cadence, `max_time_lost=60` ≈ "
            f"{60 * data['process_n'] / data['fps']:.1f}s wall time vs "
            f"{30 * data['process_n'] / data['fps']:.1f}s today.",
            "",
            "### YOLO",
            "",
            "- Consider `CONFIDENCE_THRESHOLD=0.40–0.45` to reduce noisy boxes that spawn ephemeral IDs.",
            "",
            "### Crossing (separate from ByteTrack)",
            "",
            f"- Stability already **{stab_e}/{stab_x}** processed frames; keep **track-loss flush** enabled for exit-bound deaths.",
            f"- `PROCESS_EVERY_N_FRAMES={data['process_n']}`: lowering to **3** reduces C) but increases compute.",
            "- Threshold recovery helps ENTRY near line but does not fix fragmentation.",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Running ByteTrack fragmentation audit on entry 1...")
    data = run_audit()
    write_report(data)
    print(f"Report: {REPORT_PATH.resolve()}")
    print(
        f"births={data['byte_births']} deaths={data['byte_deaths']} "
        f"respawns={len(data['respawns'])} retail_inits={data['retail_inits']} "
        f"ENTRY={data['emitted_entry']} EXIT={data['emitted_exit']}"
    )


if __name__ == "__main__":
    main()
