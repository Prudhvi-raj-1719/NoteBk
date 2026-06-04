# CAM3 Recovery Comparison — Footage2 entry 1

**Video:** `E:\NOTEBK\project\data\CCTV Footage_2\entry 1.mp4`

## Configuration

| Setting | Legacy | Recovery | Recovery + Flush |
|---------|--------|----------|------------------|
| PROCESS_EVERY_N_FRAMES | 10 | 5 | 5 |
| ENTRY/EXIT stability | 3 | 2 | 2 |
| Threshold recovery | False | True | True |
| Track-loss flush | False | False | True |

## Event counts

| Metric | Legacy | Recovery | Recovery + Flush | Expected (manual) |
|--------|--------|----------|------------------|-------------------|
| ENTRY | 2 | 4 | 3 | 3 |
| EXIT | 3 | 3 | 3 | 4 |

## Recovery diagnostics

| Counter | Recovery | Recovery + Flush |
|---------|----------|------------------|
| recovered_entries | 1 | 1 |
| recovered_exits | 0 | 0 |
| born_inside_tracks | 4 | 19 |
| born_near_threshold_tracks | 2 | 5 |

## Legacy events

- frame 510 (00:00:20.400) track 4 **ENTRY**
- frame 790 (00:00:31.600) track 5 **ENTRY**
- frame 1350 (00:00:54.000) track 9 **EXIT**
- frame 2210 (00:01:28.400) track 5 **EXIT**
- frame 2230 (00:01:29.200) track 23 **EXIT**

## Recovery events

- frame 495 (00:00:19.800) track 12 **ENTRY**
- frame 770 (00:00:30.800) track 14 **ENTRY**
- frame 1240 (00:00:49.600) track 19 **ENTRY** recovered (recovered)
- frame 1330 (00:00:53.200) track 19 **EXIT**
- frame 1335 (00:00:53.400) track 21 **EXIT**
- frame 1625 (00:01:05.000) track 26 **ENTRY**
- frame 2180 (00:01:27.200) track 14 **EXIT**

## Recovery + Flush events

- frame 495 (00:00:19.800) track 12 **ENTRY**
- frame 770 (00:00:30.800) track 14 **ENTRY**
- frame 1330 (00:00:53.200) track 19 **EXIT**
- frame 1335 (00:00:53.400) track 21 **EXIT**
- frame 1635 (00:01:05.400) track 26 **ENTRY** recovered (recovered)
- frame 2180 (00:01:27.200) track 14 **EXIT**

## Delta vs legacy

- Recovery ENTRY: +2 | EXIT: +0
- Recovery+Flush ENTRY: +1 | EXIT: +0
- Flush vs Recovery ENTRY: -1 | EXIT: +0
