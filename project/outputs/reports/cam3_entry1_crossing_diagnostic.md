# CAM3 Crossing Diagnostic — Footage2 `entry1`

**Video:** `E:\NOTEBK\project\data\CCTV Footage_2\entry 1.mp4`
**JSONL:** `E:\NOTEBK\project\outputs\cam3_footage2_entry1_events.jsonl`
**PROCESS_EVERY_N_FRAMES:** 10

## Expected vs actual (user ground truth)

| Metric | Expected | Diagnostic replay | JSONL on disk |
|--------|----------|-------------------|---------------|
| ENTRY | 3 | 2 | 2 |
| EXIT | 4 | 3 | 3 |

## Emitted transitions (replay)

- frame **510** (`00:00:20.400`) track **4** → **ENTRY**
- frame **790** (`00:00:31.600`) track **5** → **ENTRY**
- frame **1350** (`00:00:54.000`) track **9** → **EXIT**
- frame **2210** (`00:01:28.400`) track **5** → **EXIT**
- frame **2230** (`00:01:29.200`) track **23** → **EXIT**

## JSONL on disk

- `2026-04-10T14:00:20.400Z` track 4 **ENTRY** visitor VIS_00004
- `2026-04-10T14:00:31.600Z` track 5 **ENTRY** visitor VIS_00005
- `2026-04-10T14:00:54.000Z` track 9 **EXIT** visitor VIS_00009
- `2026-04-10T14:01:28.400Z` track 5 **EXIT** visitor VIS_00005
- `2026-04-10T14:01:29.200Z` track 23 **EXIT** visitor VIS_00023

## Root-cause checklist (evidence)

### 1. YOLO merged multiple people into one detection

**No strong evidence** — no single-detection ultra-wide boxes on processed frames.

### 2. ByteTrack merged two people into one track

Tracker births: 28 | deaths: 27

### 3. Track ID changed near entry line

Track lifecycles (birth → death):
- track **2**: born frame 170 init INSIDE/MALL death 180 — no_transition
- track **1**: born frame 250 init / death 260 — no_transition
- track **3**: born frame 270 init OUTSIDE/STORE death 280 — no_transition
- track **4**: born frame 430 init OUTSIDE/STORE death 520 — ENTRY@MALL
- track **6**: born frame 1250 init / death 1260 — no_transition
- track **7**: born frame 1250 init / death 1260 — no_transition
- track **8**: born frame 1290 init INSIDE/MALL death 1300 — no_transition
- track **10**: born frame 1310 init INSIDE/MALL death 1350 — no_transition
- track **9**: born frame 1370 init / death 1400 — no_transition
- track **11**: born frame 1460 init / death 1470 — no_transition
- track **13**: born frame 1580 init / death 1590 — no_transition
- track **16**: born frame 1640 init INSIDE/MALL death 1660 — no_transition
- track **12**: born frame 1670 init / death 1680 — no_transition
- track **17**: born frame 1710 init OUTSIDE/STORE death 1730 — no_transition
- track **18**: born frame 1740 init OUTSIDE/STORE death 1750 — no_transition
- track **19**: born frame 1800 init OUTSIDE/STORE death 1830 — no_transition
- track **20**: born frame 1800 init OUTSIDE/STORE death 1820 — no_transition
- track **15**: born frame 1820 init / death 1830 — no_transition
- track **14**: born frame 1880 init / death 1890 — no_transition
- track **21**: born frame 1890 init / death 1900 — no_transition
- track **22**: born frame 2090 init OUTSIDE/STORE death 2110 — no_transition
- track **24**: born frame 2240 init INSIDE/MALL death 2250 — no_transition
- track **5**: born frame 2250 init / death 2260 — no_transition
- track **23**: born frame 2270 init / death 2280 — no_transition
- track **25**: born frame 2320 init OUTSIDE/STORE death 2330 — no_transition
- track **26**: born frame 2340 init OUTSIDE/STORE death 2350 — no_transition
- track **27**: born frame 2390 init / death 2480 — no_transition
- track **28**: born frame 2500 init OUTSIDE/STORE death active — no_transition

### 4. Stability filter rejected a valid crossing

No explicit stability-reset rejections logged.

### 5. PROCESS_EVERY_N_FRAMES skipped crossing

Processed **251** frames (every 10th). Crossings between sampled frames are invisible to the engine.

### 6. Crossing occurred but transition never committed

**Confirmed — init INSIDE without ENTRY:** 7 tracks
- frame 170 track 2 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched
- frame 1290 track 8 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched
- frame 1300 track 9 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched
- frame 1310 track 10 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched
- frame 1640 track 16 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched
- frame 2170 track 23 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched
- frame 2240 track 24 (initial_side=MALL) — no ENTRY possible until OUTSIDE latched

## Init-without-ENTRY (primary loss mechanism)

Tracks that first appear with centroid already on MALL/INSIDE side never satisfy `store_state==OUTSIDE` at ENTRY commit → **ENTRY suppressed**.

## Per-processed-frame timeline (excerpt)

```
--- frame 0 (00:00:00.000) det=1 ids=[1] ---
  track=1 side=None pending=None(None) store=None
  track=1 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 170 (00:00:06.800) det=1 ids=[2] ---
  track=2 side=None pending=None(None) store=None
  track=2 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
--- frame 270 (00:00:10.800) det=1 ids=[3] ---
  track=3 side=None pending=None(None) store=None
  track=3 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 430 (00:00:17.200) det=1 ids=[4] ---
  track=4 side=None pending=None(None) store=None
  track=4 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 510 (00:00:20.400) det=1 ids=[4] ---
  track=4 side=MALL pending=None(0) store=INSIDE EMIT=ENTRY
--- frame 730 (00:00:29.200) det=1 ids=[5] ---
  track=5 side=None pending=None(None) store=None
  track=5 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 790 (00:00:31.600) det=1 ids=[5] ---
  track=5 side=MALL pending=None(0) store=INSIDE EMIT=ENTRY
--- frame 1170 (00:00:46.800) det=2 ids=[6, 5] ---
  track=6 side=None pending=None(None) store=None
  track=6 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
  track=5 side=MALL pending=None(0) store=INSIDE
--- frame 1230 (00:00:49.200) det=3 ids=[5, 7, 6] ---
  track=7 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=7 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
  track=6 side=STORE pending=None(0) store=OUTSIDE
--- frame 1290 (00:00:51.600) det=2 ids=[5, 8] ---
  track=8 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=8 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
--- frame 1300 (00:00:52.000) det=2 ids=[5, 9] ---
  track=9 side=None pending=None(None) store=None
  track=8 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=9 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
--- frame 1310 (00:00:52.400) det=3 ids=[5, 9, 10] ---
  track=10 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=9 side=MALL pending=None(0) store=INSIDE
  track=10 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
--- frame 1350 (00:00:54.000) det=2 ids=[5, 9] ---
  track=10 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=9 side=STORE pending=None(0) store=OUTSIDE EMIT=EXIT
--- frame 1420 (00:00:56.800) det=2 ids=[5, 11] ---
  track=11 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=11 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 1440 (00:00:57.600) det=2 ids=[5, 12] ---
  track=12 side=None pending=None(None) store=None
  track=11 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=12 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 1490 (00:00:59.600) det=3 ids=[5, 13, 12] ---
  track=13 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=13 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
  track=12 side=STORE pending=None(0) store=OUTSIDE
--- frame 1570 (00:01:02.800) det=2 ids=[5, 14] ---
  track=14 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=14 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 1600 (00:01:04.000) det=2 ids=[15, 5] ---
  track=15 side=None pending=None(None) store=None
  track=15 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
  track=5 side=MALL pending=None(0) store=INSIDE
--- frame 1640 (00:01:05.600) det=4 ids=[16, 5, 12, 15] ---
  track=16 side=None pending=None(None) store=None
  track=12 side=None pending=None(None) store=None
  track=16 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
  track=5 side=MALL pending=None(0) store=INSIDE
  track=12 side=STORE pending=None(0) store=OUTSIDE
  track=15 side=STORE pending=None(0) store=OUTSIDE
--- frame 1710 (00:01:08.400) det=2 ids=[5, 17] ---
  track=17 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=17 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 1740 (00:01:09.600) det=2 ids=[5, 18] ---
  track=18 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=18 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 1800 (00:01:12.000) det=5 ids=[5, 15, 19, 20, 21] ---
  track=19 side=None pending=None(None) store=None
  track=20 side=None pending=None(None) store=None
  track=21 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=15 side=STORE pending=None(0) store=OUTSIDE
  track=19 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
  track=20 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
  track=21 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 2090 (00:01:23.600) det=2 ids=[5, 22] ---
  track=5 side=None pending=None(None) store=None
  track=22 side=None pending=None(None) store=None
  track=5 side=MALL pending=None(0) store=INSIDE
  track=22 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 2170 (00:01:26.800) det=1 ids=[23] ---
  track=23 side=None pending=None(None) store=None
  track=23 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
--- frame 2210 (00:01:28.400) det=1 ids=[5] ---
  track=23 side=None pending=None(None) store=None
  track=5 side=STORE pending=None(0) store=OUTSIDE EMIT=EXIT
--- frame 2230 (00:01:29.200) det=2 ids=[23, 5] ---
  track=23 side=None pending=None(None) store=None
  track=23 side=STORE pending=None(0) store=OUTSIDE EMIT=EXIT
  track=5 side=STORE pending=None(0) store=OUTSIDE
--- frame 2240 (00:01:29.600) det=1 ids=[24] ---
  track=24 side=None pending=None(None) store=None
  track=5 side=None pending=None(None) store=None
  track=23 side=None pending=None(None) store=None
  track=24 side=MALL pending=None(0) store=INSIDE INIT=INSIDE init_inside_no_entry_latch — track born INSIDE without ENTRY
--- frame 2320 (00:01:32.800) det=1 ids=[25] ---
  track=25 side=None pending=None(None) store=None
  track=25 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 2340 (00:01:33.600) det=1 ids=[26] ---
  track=26 side=None pending=None(None) store=None
  track=26 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 2350 (00:01:34.000) det=1 ids=[27] ---
  track=27 side=None pending=None(None) store=None
  track=26 side=None pending=None(None) store=None
  track=27 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
--- frame 2500 (00:01:40.000) det=1 ids=[28] ---
  track=28 side=None pending=None(None) store=None
  track=28 side=STORE pending=None(0) store=OUTSIDE INIT=OUTSIDE
```

## Expected-scenario mapping (Footage2 entry 1)

| Expected event | Diagnostic match | Outcome |
|----------------|------------------|---------|
| Red shirt woman **ENTRY** | track **4** — init OUTSIDE @ frame 430, **ENTRY** @ frame 510 (00:00:20.4) | **Emitted** |
| White shirt man **ENTRY** | track **5** — init OUTSIDE @ frame 730, **ENTRY** @ frame 790 (00:00:31.6) | **Emitted** |
| Mother + daughter **EXIT** (×2) | track **9** **EXIT** @ frame 1350 only; tracks **8**, **10** init INSIDE @ 1290–1310, **no EXIT** | **1 of 2 EXIT** |
| Black shirt girl **ENTRY** | tracks **8**, **9**, **10** spawn on MALL/INSIDE @ 1290–1310 (`INIT=INSIDE`) | **Not emitted** |
| Red shirt woman **EXIT** | track **5** **EXIT** @ frame 2210 (00:01:28.4) | **Emitted** |
| White shirt man **EXIT** | track **23** **EXIT** @ frame 2230; track 23 **init INSIDE** @ 2170 (no prior ENTRY on id 23) | **Emitted** (id handoff) |

### Missing ENTRY (black shirt girl) — root cause

**#6 — init INSIDE, transition never committed as ENTRY**

At 00:00:51–52, ByteTrack assigns new IDs **8**, **9**, **10** while centroids are already below the horizontal threshold (MALL / `INSIDE` with `INVERT_RETAIL_SEMANTICS`). Engine latches `store_state=INSIDE` on first sight with **no** `STORE→MALL` crossing while `OUTSIDE`.

ENTRY commit requires `previous_store_state == OUTSIDE`. None of these tracks ever satisfy that after init.

**Not primary:** #1 YOLO merge (no ultra-wide single boxes), #4 stability reset (no resets logged), #5 frame skip alone (init would still be INSIDE on first sampled frame after appear).

### Missing EXIT (second of mother + daughter) — root cause

**#3 + #6 — track ID fragmentation and short INSIDE tracks without EXIT commit**

- track **10**: `INIT=INSIDE` @ frame 1310, tracker death @ frame 1350 (~4 processed samples). **No** `MALL→STORE` stability cycle completed → **no EXIT**.
- track **8**: `INIT=INSIDE` @ frame 1290, death @ frame 1300 (~1–2 samples) → **no EXIT**.
- track **9**: only member of the group with a committed **EXIT** @ frame 1350.

**#5** may contribute: EXIT requires 3 consecutive processed frames on STORE side; a crossing between frames 1340–1350 could be missed with `PROCESS_EVERY_N_FRAMES=10`.

**Not primary:** #2 single merged track for the pair (tracks 8/9/10 are distinct when both visible @ frame 1310, `det=3`).

### Persistent track 5 (white shirt / shared INSIDE carrier)

track **5** stays `INSIDE` from its ENTRY (790) through the mother/daughter window and receives the red-shirt **EXIT** @ 2210. A separate short id **23** appears @ 2170 already **INSIDE** and emits the second late **EXIT** @ 2230 — consistent with **#3 track ID change** near the door for the white-shirt exit.

## Conclusion

Replay engine counts: ENTRY=2 EXIT=3. Gap vs expectation: **1 ENTRY**, **1 EXIT**.

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| 1. YOLO merged people | **Unlikely** | No wide single-detection boxes; `det=2`/`det=3` when group visible |
| 2. ByteTrack merged pair | **Partial** | Separate ids 8/9/10, but only one EXIT committed |
| 3. Track ID changed | **Yes (late EXIT)** | Exit on track 23 vs ENTRY on track 5 |
| 4. Stability rejected crossing | **Unlikely** | No stability-reset rows in log |
| 5. Frame skip | **Possible (2nd EXIT)** | 10-frame gap; short-lived track 10 |
| 6. Never committed / init INSIDE | **Yes (missing ENTRY + 1 EXIT)** | 7× `INIT=INSIDE`; tracks 8/10 never EXIT |

**Primary root cause:** Footage2 geometry + latch rules — visitors first detected **already inside** the store half-plane do not get ENTRY; short ByteTrack ids **8/10** never complete EXIT stability before id death. Secondary: **track 5 vs 23** split for the final white-shirt EXIT.
