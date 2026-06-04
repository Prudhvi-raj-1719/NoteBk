"""
Read-only OSNet Re-ID audit for CAM3 REENTRY (no threshold / logic changes).

Usage:
  python scripts/cam3_reid_audit.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from configs.camera_timing_config import (
    CAM3_ENTRY_CONFIG,
    CAM3_PROCESS_EVERY_N_FRAMES,
    CAMERA_VIDEO_FILES,
    DATA_DIR,
    MODEL_PATH,
    OUTPUTS_DIR,
)
from configs.reid_config import REID_COSINE_THRESHOLD, REID_EXIT_CACHE_SECONDS
from entry_retail import EVENT_ENTRY, EVENT_EXIT, build_retail_entry_engine, format_video_timestamp
from events.cam3_events import create_byte_tracker, detect_persons
from events.event_emitter import parse_utc_from_iso, video_offset_to_utc_iso
from reid.osnet_reid import OsnetEmbedder
from reid.reid_manager import ReIDManager

REPORT_PATH = OUTPUTS_DIR / "reports" / "cam3_reid_audit_report.md"

AUDIT_VIDEOS: List[Tuple[str, Path, Dict[str, Any]]] = [
    ("brigade", CAMERA_VIDEO_FILES["CAM3"], CAM3_ENTRY_CONFIG["BRIGADE"]),
    ("entry1", DATA_DIR / "CCTV Footage_2" / "entry 1.mp4", CAM3_ENTRY_CONFIG["FOOTAGE2"]),
    ("entry2", DATA_DIR / "CCTV Footage_2" / "entry 2.mp4", CAM3_ENTRY_CONFIG["FOOTAGE2"]),
]


@dataclass
class EntryMatchAudit:
    clip: str
    frame: int
    video_offset: str
    byte_track_id: int
    assigned_vis: str
    crop_w: int
    crop_h: int
    crop_area: int
    cache_size: int
    best_vis: Optional[str]
    best_sim: float
    accepted_reentry: bool
    all_cache_sims: List[Tuple[str, float, float]] = field(default_factory=list)


def _load_jsonl_reentries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(OUTPUTS_DIR.glob("cam3*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith('{"camera"'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event_type") == "REENTRY":
                meta = row.get("metadata") or {}
                rows.append(
                    {
                        "file": path.name,
                        "visitor_id": row.get("visitor_id"),
                        "timestamp": row.get("timestamp"),
                        "byte_track_id": meta.get("byte_track_id"),
                        "reid_similarity": meta.get("reid_similarity"),
                        "reid_matched_vis": meta.get("reid_matched_vis"),
                        "reentry_from_vis": meta.get("reentry_from_vis"),
                        "reentry_match": meta.get("reentry_match"),
                        "clip_id": meta.get("clip_id"),
                    }
                )
    return rows


def _score_exit_cache(
    manager: ReIDManager,
    track_id: int,
    entry_utc: datetime,
) -> Tuple[Optional[np.ndarray], List[Tuple[str, float, float]], float, Optional[str]]:
    """Return query, (vis, sim, age_s), best_sim, best_vis."""
    query = manager.latest_embedding(track_id)
    manager._prune_exits(entry_utc)
    scored: List[Tuple[str, float, float]] = []
    best_vis: Optional[str] = None
    best_sim = -1.0
    if query is None:
        return None, scored, best_sim, best_vis

    for record in manager.recent_exits:
        age = (entry_utc - record.exit_utc).total_seconds()
        if age < 0 or age > manager.exit_cache_seconds:
            continue
        sim = manager._cosine(query, record.embedding)
        scored.append((record.vis_id, sim, age))
        if sim > best_sim:
            best_sim = sim
            best_vis = record.vis_id
    scored.sort(key=lambda x: x[1], reverse=True)
    return query, scored, best_sim, best_vis


def _crop_size(frame: np.ndarray, xyxy: np.ndarray) -> Tuple[int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    cw = max(0, x2 - x1)
    ch = max(0, y2 - y1)
    return cw, ch, cw * ch


def replay_clip(
    clip_id: str,
    video_path: Path,
    layout: Dict[str, Any],
    yolo: YOLO,
) -> List[EntryMatchAudit]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    engine = build_retail_entry_engine(
        layout,
        vw,
        vh,
        timestamp_fn=lambda f, _fps=fps: format_video_timestamp(f, _fps),
    )
    manager = ReIDManager(embedder=OsnetEmbedder())
    tracker = create_byte_tracker()
    audits: List[EntryMatchAudit] = []
    frame_idx = 0
    notbk_camera = "CAM3"
    notbk_clip = clip_id if clip_id != "brigade" else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % CAM3_PROCESS_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        detections = detect_persons(frame, yolo)
        detections = tracker.update_with_detections(detections)
        match_utc = ReIDManager.video_utc_for_frame(
            notbk_camera, frame_idx, fps, clip_id=notbk_clip
        )
        manager.update_tracks(frame, detections, match_utc=match_utc)

        active_ids: set[int] = set()
        boxes: Dict[int, np.ndarray] = {}

        if detections.tracker_id is not None:
            for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                tid = int(track_id)
                active_ids.add(tid)
                boxes[tid] = xyxy
                center = (
                    int((xyxy[0] + xyxy[2]) / 2),
                    int((xyxy[1] + xyxy[3]) / 2),
                )
                result = engine.update(tid, center, original_frame=frame_idx)
                if result.transition is None:
                    continue

                tr = result.transition
                video_offset = tr.timestamp
                utc_iso = video_offset_to_utc_iso(
                    notbk_camera,
                    video_offset,
                    clip_id=notbk_clip,
                )
                event_utc = parse_utc_from_iso(utc_iso)

                if tr.event_type == EVENT_EXIT:
                    vis_id = manager.vis_for_track(tid)
                    manager.record_exit(vis_id, tid, event_utc)
                    continue

                if tr.event_type != EVENT_ENTRY:
                    continue

                cw, ch, area = _crop_size(frame, boxes[tid])
                assigned_vis = manager.vis_for_track(tid)
                _, scored, best_sim, best_vis = _score_exit_cache(
                    manager, tid, event_utc
                )
                matched_vis, match_sim = manager.match_entry(tid, event_utc)
                accepted = matched_vis is not None
                sim_report = (
                    float(match_sim)
                    if match_sim is not None
                    else float(best_sim)
                )

                audits.append(
                    EntryMatchAudit(
                        clip=clip_id,
                        frame=frame_idx,
                        video_offset=video_offset,
                        byte_track_id=tid,
                        assigned_vis=assigned_vis,
                        crop_w=cw,
                        crop_h=ch,
                        crop_area=area,
                        cache_size=len(scored),
                        best_vis=best_vis,
                        best_sim=sim_report,
                        accepted_reentry=accepted,
                        all_cache_sims=scored,
                    )
                )
                if accepted and matched_vis:
                    manager.link_track(tid, matched_vis)

        if getattr(engine, "enable_track_loss_flush", False):
            engine.flush_removed_tracks(active_ids, frame_idx)

        frame_idx += 1

    cap.release()
    return audits


def _histogram(values: List[float], bins: List[float]) -> List[str]:
    lines = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = sum(1 for v in values if lo <= v < hi)
        lines.append(f"| [{lo:.2f}, {hi:.2f}) | {count} |")
    return lines


def main() -> None:
    jsonl_reentries = _load_jsonl_reentries()
    yolo = YOLO(str(MODEL_PATH))
    all_audits: List[EntryMatchAudit] = []

    for clip_id, video_path, layout in AUDIT_VIDEOS:
        if not video_path.exists():
            print(f"Skip missing: {video_path}")
            continue
        print(f"Replaying {clip_id}...")
        all_audits.extend(replay_clip(clip_id, video_path, layout, yolo))

    accepted = [a for a in all_audits if a.accepted_reentry]
    rejected = [a for a in all_audits if not a.accepted_reentry and a.best_sim >= 0]
    new_entries = [a for a in all_audits if not a.accepted_reentry]

    # Near-miss false candidates: high best_sim but rejected
    near_miss = [a for a in new_entries if a.best_sim >= REID_COSINE_THRESHOLD - 0.05]

    reentry_sims = [a.best_sim for a in accepted]
    rejected_best = [a.best_sim for a in new_entries if a.cache_size > 0]

    lines = [
        "# CAM3 OSNet Re-ID Audit",
        "",
        f"**Threshold:** {REID_COSINE_THRESHOLD}",
        f"**Exit cache window:** {REID_EXIT_CACHE_SECONDS}s",
        "",
        "## JSONL REENTRY events on disk",
        "",
        f"Count: **{len(jsonl_reentries)}**",
        "",
    ]
    if jsonl_reentries:
        lines.append("| file | visitor_id | matched_vis | similarity | track | clip |")
        lines.append("|------|------------|---------------|------------|-------|------|")
        for r in jsonl_reentries:
            lines.append(
                f"| {r['file']} | {r['visitor_id']} | {r.get('reid_matched_vis')} | "
                f"{r.get('reid_similarity')} | {r.get('byte_track_id')} | {r.get('clip_id')} |"
            )
    else:
        lines.append("(none in competition-schema JSONL)")

    lines.extend(
        [
            "",
            "## Replay summary (all CAM3 clips)",
            "",
            f"- ENTRY crossings audited: **{len(all_audits)}**",
            f"- OSNet REENTRY accepted: **{len(accepted)}**",
            f"- ENTRY (no OSNet match): **{len(new_entries)}**",
            f"- Near-miss (best_sim ≥ {REID_COSINE_THRESHOLD - 0.05:.2f}, rejected): **{len(near_miss)}**",
            "",
            "## REENTRY similarity bands (accepted matches, replay)",
            "",
        ]
    )
    for label, thr in [("< 0.75", 0.75), ("< 0.80", 0.80)]:
        n = sum(1 for s in reentry_sims if s < thr)
        lines.append(f"- similarity {label}: **{n}** / {len(reentry_sims)}")

    lines.extend(
        [
            "",
            "## Similarity distribution",
            "",
            "### Accepted REENTRY (true positive by system)",
            "",
            "| bin | count |",
            "|-----|-------|",
        ]
    )
    if reentry_sims:
        lines.extend(_histogram(reentry_sims, [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]))
    else:
        lines.append("| (no accepted REENTRY in replay) | 0 |")

    lines.extend(
        [
            "",
            "### Rejected ENTRY — best cache similarity (non-match / true negative)",
            "",
            "| bin | count |",
            "|-----|-------|",
        ]
    )
    if rejected_best:
        lines.extend(_histogram(rejected_best, [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]))
    else:
        lines.append("| (no exit cache at ENTRY) | 0 |")

    lines.extend(["", "## Top 20 REENTRY matches (by similarity)", ""])
    top = sorted(accepted, key=lambda a: a.best_sim, reverse=True)[:20]
    if top:
        lines.append(
            "| rank | clip | frame | track | visitor (pre) | matched_vis | "
            "similarity | crop (w×h) | area | cache |"
        )
        lines.append(
            "|------|------|-------|-------|---------------|-------------|"
            "------------|------------|------|-------|"
        )
        for i, a in enumerate(top, 1):
            lines.append(
                f"| {i} | {a.clip} | {a.frame} | {a.byte_track_id} | {a.assigned_vis} | "
                f"{a.best_vis} | {a.best_sim:.4f} | {a.crop_w}×{a.crop_h} | {a.crop_area} | "
                f"{a.cache_size} |"
            )
    else:
        lines.append("(none)")

    lines.extend(["", "## Flagged case: entry1 track 26 → VIS_00021", ""])
    flagged = [a for a in all_audits if a.clip == "entry1" and a.byte_track_id == 26 and a.accepted_reentry]
    if flagged:
        a = flagged[0]
        lines.append(
            f"- frame {a.frame} ({a.video_offset}): similarity **{a.best_sim:.4f}** "
            f"(threshold {REID_COSINE_THRESHOLD}), crop **{a.crop_w}×{a.crop_h}** (area {a.crop_area})"
        )
        lines.append(f"- exit cache candidates at ENTRY:")
        for vis, sim, age in a.all_cache_sims[:5]:
            lines.append(f"  - {vis}: sim={sim:.4f}, age={age:.1f}s")
    else:
        lines.append("(not reproduced in this replay pass)")

    lines.extend(
        [
            "",
            "## Is 0.65 too low?",
            "",
        ]
    )
    if reentry_sims:
        below_75 = sum(1 for s in reentry_sims if s < 0.75)
        below_80 = sum(1 for s in reentry_sims if s < 0.80)
        min_sim = min(reentry_sims)
        lines.append(
            f"- All accepted REENTRY similarities in replay: min={min_sim:.4f}, "
            f"count={len(reentry_sims)}"
        )
        lines.append(
            f"- **{below_75}** accepted match(es) would be blocked at threshold **0.75**; "
            f"**{below_80}** at **0.80**."
        )
    if near_miss:
        lines.append(
            f"- **{len(near_miss)}** ENTRY event(s) had best_sim within 0.05 of threshold "
            f"but were rejected (borderline true negatives)."
        )
    lines.append(
        "- Cosine threshold **0.65** accepts matches in the high-0.60s where Market-1501 OSNet "
        "often confuses different shoppers with similar clothing/lighting."
    )
    if flagged and flagged[0].best_sim < 0.70:
        lines.append(
            f"- The entry1 false REENTRY at **{flagged[0].best_sim:.4f}** is only **{flagged[0].best_sim - REID_COSINE_THRESHOLD:.4f}** "
            "above threshold — raising to **0.72–0.75** would likely reject it while preserving fewer marginal matches."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {REPORT_PATH.resolve()}")
    print(f"REENTRY accepted (replay): {len(accepted)}")
    if reentry_sims:
        print(f"  sims: min={min(reentry_sims):.4f} max={max(reentry_sims):.4f}")
    print(f"JSONL REENTRY on disk: {len(jsonl_reentries)}")


if __name__ == "__main__":
    main()
