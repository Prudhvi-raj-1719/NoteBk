"""

Compare legacy vs recovery vs recovery+flush CAM3 crossing on Footage2 entry1.



Legacy: process_every=10, stability=3, no threshold recovery, no track-loss flush.

Recovery: stability=2, threshold recovery, track-loss flush disabled.

Recovery+Flush: same as recovery with ENABLE_TRACK_LOSS_FLUSH=True.



Usage:

  python scripts/cam3_recovery_comparison.py

"""



from __future__ import annotations



import copy
import gc
import sys

from pathlib import Path

from typing import Any, Dict, List



_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(_PROJECT_ROOT))



import cv2

from ultralytics import YOLO



from configs.camera_timing_config import (

    CAM3_ENTRY_CONFIG,

    DATA_DIR,

    MODEL_PATH,

)

from entry_retail import EVENT_ENTRY, EVENT_EXIT, build_retail_entry_engine, format_video_timestamp

from events.cam3_events import create_byte_tracker, detect_persons



VIDEO_PATH = DATA_DIR / "CCTV Footage_2" / "entry 1.mp4"

REPORT_PATH = (

    _PROJECT_ROOT / "outputs" / "reports" / "cam3_entry1_recovery_comparison.md"

)



LEGACY_LAYOUT = {

    **CAM3_ENTRY_CONFIG["FOOTAGE2"],

    "ENTRY_STABILITY_FRAMES": 3,

    "EXIT_STABILITY_FRAMES": 3,

    "ENABLE_THRESHOLD_RECOVERY": False,

    "ENABLE_TRACK_LOSS_FLUSH": False,

}

LEGACY_PROCESS_N = 10



RECOVERY_LAYOUT = {

    **copy.deepcopy(CAM3_ENTRY_CONFIG["FOOTAGE2"]),

    "ENABLE_TRACK_LOSS_FLUSH": False,

}

RECOVERY_PROCESS_N = 5



FLUSH_LAYOUT = {

    **copy.deepcopy(CAM3_ENTRY_CONFIG["FOOTAGE2"]),

    "ENABLE_TRACK_LOSS_FLUSH": True,

}

FLUSH_PROCESS_N = 5





def _append_transition(

    events: List[Dict[str, Any]],

    frame_idx: int,

    tid: int,

    transition: Any,

) -> None:

    events.append(

        {

            "frame": frame_idx,

            "timestamp": transition.timestamp,

            "track_id": tid,

            "event_type": transition.event_type,

            "recovered": transition.debug.get("recovered", False),

            "recovery_reason": transition.debug.get("recovery_reason"),

        }

    )





def run_pipeline(
    layout: Dict[str, Any],
    process_every_n: int,
    label: str,
    yolo: YOLO,
) -> Dict[str, Any]:

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():

        raise FileNotFoundError(VIDEO_PATH)



    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0



    tracker = create_byte_tracker()

    engine = build_retail_entry_engine(

        layout,

        w,

        h,

        timestamp_fn=lambda f, _fps=fps: format_video_timestamp(f, _fps),

    )



    events: List[Dict[str, Any]] = []

    frame_idx = 0

    last_processed_frame = 0



    while True:

        ok, frame = cap.read()

        if not ok:

            break

        if frame_idx % process_every_n != 0:

            frame_idx += 1

            continue



        last_processed_frame = frame_idx

        detections = detect_persons(frame, yolo)

        detections = tracker.update_with_detections(detections)

        active_ids: set[int] = set()



        if detections.tracker_id is not None:

            for track_id, xyxy in zip(detections.tracker_id, detections.xyxy):

                tid = int(track_id)

                active_ids.add(tid)

                center = (

                    int((xyxy[0] + xyxy[2]) / 2),

                    int((xyxy[1] + xyxy[3]) / 2),

                )

                result = engine.update(tid, center, original_frame=frame_idx)

                if result.transition is not None:

                    _append_transition(events, frame_idx, tid, result.transition)



        if engine.enable_track_loss_flush:
            for transition in engine.flush_removed_tracks(active_ids, frame_idx):
                _append_transition(events, frame_idx, transition.visitor_id, transition)

        frame_idx += 1

    if engine.enable_track_loss_flush:
        for transition in engine.flush_removed_tracks(set(), last_processed_frame):
            _append_transition(
                events, last_processed_frame, transition.visitor_id, transition
            )



    cap.release()



    entries = [e for e in events if e["event_type"] == EVENT_ENTRY]

    exits = [e for e in events if e["event_type"] == EVENT_EXIT]

    rs = engine.recovery_stats



    return {

        "label": label,

        "process_every_n": process_every_n,

        "entry_stability": layout.get("ENTRY_STABILITY_FRAMES"),

        "recovery_enabled": layout.get("ENABLE_THRESHOLD_RECOVERY", False),

        "track_loss_flush": layout.get("ENABLE_TRACK_LOSS_FLUSH", False),

        "entry_count": len(entries),

        "exit_count": len(exits),

        "engine_entry": engine.entry_count,

        "engine_exit": engine.exit_count,

        "events": events,

        "recovery_stats": {

            "recovered_entries": rs.recovered_entries,

            "recovered_exits": rs.recovered_exits,

            "born_inside_tracks": rs.born_inside_tracks,

            "born_near_threshold_tracks": rs.born_near_threshold_tracks,

        },

    }





def format_events(events: List[Dict[str, Any]]) -> str:

    lines = []

    for e in events:

        rec = ""

        if e.get("recovered"):

            reason = e.get("recovery_reason") or "recovered"

            rec = f" recovered ({reason})"

        lines.append(

            f"- frame {e['frame']} ({e['timestamp']}) track {e['track_id']} "

            f"**{e['event_type']}**{rec}"

        )

    return "\n".join(lines) if lines else "(none)"





def main() -> None:
    yolo = YOLO(str(MODEL_PATH))
    legacy = run_pipeline(LEGACY_LAYOUT, LEGACY_PROCESS_N, "legacy", yolo)
    gc.collect()
    recovery = run_pipeline(RECOVERY_LAYOUT, RECOVERY_PROCESS_N, "recovery", yolo)
    gc.collect()
    flush = run_pipeline(FLUSH_LAYOUT, FLUSH_PROCESS_N, "recovery+flush", yolo)
    del yolo
    gc.collect()



    lines = [

        "# CAM3 Recovery Comparison — Footage2 entry 1",

        "",

        f"**Video:** `{VIDEO_PATH}`",

        "",

        "## Configuration",

        "",

        "| Setting | Legacy | Recovery | Recovery + Flush |",

        "|---------|--------|----------|------------------|",

        f"| PROCESS_EVERY_N_FRAMES | {legacy['process_every_n']} | {recovery['process_every_n']} | {flush['process_every_n']} |",

        f"| ENTRY/EXIT stability | {legacy['entry_stability']} | {recovery['entry_stability']} | {flush['entry_stability']} |",

        f"| Threshold recovery | {legacy['recovery_enabled']} | {recovery['recovery_enabled']} | {flush['recovery_enabled']} |",

        f"| Track-loss flush | {legacy['track_loss_flush']} | {recovery['track_loss_flush']} | {flush['track_loss_flush']} |",

        "",

        "## Event counts",

        "",

        "| Metric | Legacy | Recovery | Recovery + Flush | Expected (manual) |",

        "|--------|--------|----------|------------------|-------------------|",

        f"| ENTRY | {legacy['entry_count']} | {recovery['entry_count']} | {flush['entry_count']} | 3 |",

        f"| EXIT | {legacy['exit_count']} | {recovery['exit_count']} | {flush['exit_count']} | 4 |",

        "",

        "## Recovery diagnostics",

        "",

        "| Counter | Recovery | Recovery + Flush |",

        "|---------|----------|------------------|",

        f"| recovered_entries | {recovery['recovery_stats']['recovered_entries']} | {flush['recovery_stats']['recovered_entries']} |",

        f"| recovered_exits | {recovery['recovery_stats']['recovered_exits']} | {flush['recovery_stats']['recovered_exits']} |",

        f"| born_inside_tracks | {recovery['recovery_stats']['born_inside_tracks']} | {flush['recovery_stats']['born_inside_tracks']} |",

        f"| born_near_threshold_tracks | {recovery['recovery_stats']['born_near_threshold_tracks']} | {flush['recovery_stats']['born_near_threshold_tracks']} |",

        "",

        "## Legacy events",

        "",

        format_events(legacy["events"]),

        "",

        "## Recovery events",

        "",

        format_events(recovery["events"]),

        "",

        "## Recovery + Flush events",

        "",

        format_events(flush["events"]),

        "",

        "## Delta vs legacy",

        "",

        f"- Recovery ENTRY: {recovery['entry_count'] - legacy['entry_count']:+d} | EXIT: {recovery['exit_count'] - legacy['exit_count']:+d}",

        f"- Recovery+Flush ENTRY: {flush['entry_count'] - legacy['entry_count']:+d} | EXIT: {flush['exit_count'] - legacy['exit_count']:+d}",

        f"- Flush vs Recovery ENTRY: {flush['entry_count'] - recovery['entry_count']:+d} | EXIT: {flush['exit_count'] - recovery['exit_count']:+d}",

        "",

    ]



    flush_exits = [

        e for e in flush["events"]

        if e["event_type"] == EVENT_EXIT and e.get("recovery_reason") == "track_loss_flush"

    ]

    if flush_exits:

        lines.append("### Track-loss flush EXIT events")

        lines.append("")

        lines.append(format_events(flush_exits))

        lines.append("")



    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report: {REPORT_PATH.resolve()}")

    print(

        f"Legacy ENTRY={legacy['entry_count']} EXIT={legacy['exit_count']} | "

        f"Recovery ENTRY={recovery['entry_count']} EXIT={recovery['exit_count']} | "

        f"Flush ENTRY={flush['entry_count']} EXIT={flush['exit_count']}"

    )





if __name__ == "__main__":

    main()


