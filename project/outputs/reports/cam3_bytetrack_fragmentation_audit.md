# CAM3 ByteTrack Fragmentation Audit — Footage2 entry 1

**Video:** `E:\NOTEBK\project\data\CCTV Footage_2\entry 1.mp4`
**Frames:** 2636 total | **501** processed (every **5**) @ 25.0 fps

## 1. Current ByteTrack parameters

CAM3 uses `sv.ByteTrack()` with **no custom arguments** (supervision defaults).

| Parameter | Value | Notes |
|-----------|-------|-------|
| `track_activation_threshold` | 0.25 | Min detection score to start a track (activation) |
| `det_thresh` | 0.35 | High-confidence detection threshold inside tracker |
| `minimum_matching_threshold` | 0.8 | IoU/cost match threshold (≈ `match_thresh`) |
| `max_time_lost` | 30 | Frames to keep lost track before removal (≈ `track_buffer` at update rate) |
| `minimum_consecutive_frames` | 1 | Hits before track is confirmed |
| `frame_id` | 0 | Internal frame counter |

### YOLO detection (CAM3)

| Setting | Value |
|---------|-------|
| `CONFIDENCE_THRESHOLD` | 0.35 |
| `IOU_THRESHOLD` | 0.5 |
| `PERSON_CLASS_ID` | 0 |

**Important:** `max_time_lost` advances once per **processed** frame (every 5 video frames), so wall-clock retention ≈ `max_time_lost` × 5 / fps seconds.

## 2. Track lifecycle statistics

| Metric | Value |
|--------|-------|
| ByteTrack unique IDs (births) | 51 |
| Track deaths (IDs lost) | 49 |
| Retail engine initializations | 131 |
| `born_inside_tracks` | 19 |
| Near-threshold observing inits | 5 |
| Emitted ENTRY | 3 |
| Emitted EXIT | 3 |
| Avg track lifetime (processed frames) | 19.92 |
| Avg track lifetime (seconds, approx) | 3.98 |
| Re-spawn links (death→birth proximity) | 135 |
| Deaths with pending EXIT (inside→store) | 0 |

## 3. Top 20 visitor trajectories that changed IDs

Heuristic: same norm-centroid within **0.12** within **6** processed frames after death.

| rank | old_id | new_id | death_frame | birth_frame | gap(proc) | dist | old_life(proc) | old_events | pending_exit@death | born_inside |
|------|--------|--------|-------------|-------------|-----------|------|--------------|------------|-------------------|-------------|
| 1 | 14 | 14 | 1005 | 1010 | 1 | 0.0003 | 65 | ENTRY | False | True |
| 2 | 14 | 14 | 930 | 935 | 1 | 0.0007 | 50 | ENTRY | False | True |
| 3 | 14 | 14 | 1115 | 1120 | 1 | 0.001 | 87 | ENTRY | False | True |
| 4 | 14 | 14 | 2080 | 2085 | 1 | 0.0036 | 280 | ENTRY | False | True |
| 5 | 22 | 22 | 1485 | 1490 | 1 | 0.0041 | 23 | - | False | False |
| 6 | 42 | 42 | 2095 | 2100 | 1 | 0.006 | 3 | - | False | False |
| 7 | 14 | 14 | 1260 | 1265 | 1 | 0.0065 | 116 | ENTRY | False | True |
| 8 | 14 | 14 | 1975 | 1980 | 1 | 0.0072 | 259 | ENTRY | False | True |
| 9 | 19 | 19 | 1470 | 1475 | 1 | 0.01 | 50 | EXIT | False | True |
| 10 | 47 | 46 | 2360 | 2365 | 1 | 0.012 | 10 | - | False | False |
| 11 | 49 | 49 | 2450 | 2455 | 1 | 0.0121 | 3 | - | False | False |
| 12 | 14 | 14 | 2255 | 2260 | 1 | 0.0149 | 315 | ENTRY,EXIT | False | True |
| 13 | 25 | 25 | 1575 | 1580 | 1 | 0.0211 | 2 | - | False | False |
| 14 | 35 | 34 | 1750 | 1755 | 1 | 0.0219 | 6 | - | False | False |
| 15 | 25 | 25 | 1620 | 1625 | 1 | 0.0232 | 11 | - | False | False |
| 16 | 19 | 19 | 1335 | 1340 | 1 | 0.0234 | 23 | EXIT | False | True |
| 17 | 34 | 34 | 1750 | 1755 | 1 | 0.0248 | 7 | - | False | False |
| 18 | 19 | 19 | 1350 | 1355 | 1 | 0.0256 | 26 | EXIT | False | True |
| 19 | 5 | 5 | 310 | 315 | 1 | 0.0266 | 35 | - | False | False |
| 20 | 48 | 48 | 2370 | 2375 | 1 | 0.0308 | 8 | - | False | False |

## 4. Track 12 → 46 (user-reported)

- death **45** @ frame 2255 → birth **46** @ 2275 (dist=0.1068)
- death **14** @ frame 2255 → birth **46** @ 2275 (dist=0.0801)
- death **14** @ frame 2265 → birth **46** @ 2275 (dist=0.0856)
- death **47** @ frame 2360 → birth **46** @ 2365 (dist=0.012)
- death **47** @ frame 2360 → birth **46** @ 2380 (dist=0.0177)
- death **46** @ frame 2370 → birth **46** @ 2380 (dist=0.0057)
- death **46** @ frame 2475 → birth **46** @ 2490 (dist=0.0728)

## 5. Missed ENTRY/EXIT — root-cause attribution

| Cause | Mechanism | Count (this replay) |
|-------|-----------|---------------------|
| **A) ByteTrack fragmentation** | ID lost/re-spawn; pending crossing incomplete at death | re-spawns=135, deaths w/ pending EXIT=0, fragment exits=0 |
| **B) Stability filter** | `pending_frames` ≥1 but < 2 at track death | **0** |
| **C) Frame skipping** | `PROCESS_EVERY_N_FRAMES=5` — crossing between samples | qualitative (fast cross < 5 frames) |
| **D) Threshold recovery** | born INSIDE / UNKNOWN observe | born_inside=**19**, observe inits=**5** |

### Primary driver (entry 1)

**A) ByteTrack fragmentation** — many ID re-spawns; short median lifetimes.
**B) Stability filter** — exits pending when IDs die before 2-frame commit.
**D) Threshold recovery / init INSIDE** — suppresses ENTRY for IDs born past line.
**C) Frame skipping** — amplifies A+B because tracker updates only every 5 frames.

## 6. Recommended parameter changes (do not apply yet)

### ByteTrack (supervision `ByteTrack(...)`)

```python
sv.ByteTrack(
    track_activation_threshold=0.30,  # was 0.25 — fewer junk tracks
    minimum_matching_threshold=0.75,  # was 0.8 — easier re-associate after occlusion
    max_time_lost=60,                 # was 30 — longer lost-track retention at proc rate
    minimum_consecutive_frames=2,     # was 1 — reduce one-frame ID flicker
)
```

At processed-frame cadence, `max_time_lost=60` ≈ 12.0s wall time vs 6.0s today.

### YOLO

- Consider `CONFIDENCE_THRESHOLD=0.40–0.45` to reduce noisy boxes that spawn ephemeral IDs.

### Crossing (separate from ByteTrack)

- Stability already **2/2** processed frames; keep **track-loss flush** enabled for exit-bound deaths.
- `PROCESS_EVERY_N_FRAMES=5`: lowering to **3** reduces C) but increases compute.
- Threshold recovery helps ENTRY near line but does not fix fragmentation.
