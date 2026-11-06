# ISL Clip Recorder — research/track_b/recorder

Standalone collector for personal ISL word samples. It is deliberately **not**
integrated into the SignSage app and must never edit frontend/, .env, render.yaml
or SIGNSAGE_MODEL_PROMOTED. All files live under `research/track_b/recorder/`.

## What it is

A single-page recorder (`recorder.html`) that extracts landmarks with the **exact
same extractor as the SignSage frontend** (`frontend/src/features/translate/landmarks.ts`),
copied verbatim (not imported):

- `@mediapipe/tasks-vision` **1.0.1** (vendored copy: `vision_bundle.mjs` + `assets/wasm/`)
- PoseLandmarker + FaceLandmarker + HandLandmarker, **runningMode VIDEO**
- Pose `numPoses=1`, Face `numFaces=1`, Hands `numHands=2`, **delegate GPU**
- No confidence thresholds set (MediaPipe defaults); no delegate fallback
- Model assets vendored locally (`assets/models/*.task`) so nothing loads from the network
- 543 landmarks/frame: POSE[0:33] + FACE[33:501] + LEFT_HAND[501:522] + RIGHT_HAND[522:543],
  each [x,y,z]; missing parts written as [0,0,0]
- Timestamps: `performance.now()`, VIDEO mode strictly-increasing; frames skipped when
  `video.currentTime` unchanged; detection throttled to 80 ms (same as the frontend loop)
- Handedness follows the image (MediaPipe "Left" = appears on the left). Coordinates are
  **not mirrored**; only the preview is CSS-mirrored (`scaleX(-1)`), exactly like the app.
  Handedness can therefore swap under a mirrored selfie preview — an accepted calibration
  limit, consistent across frames.

Serves on **localhost only** (getUserMedia requires a secure context; localhost qualifies).
Saves **landmarks only** — never raw video or images.

## Files

| File | Purpose |
|---|---|
| `recorder.html` | Standalone app: embedded JS + CSS |
| `vision_bundle.mjs` | `@mediapipe/tasks-vision` 1.0.1 bundle (vendored from frontend/node_modules) |
| `assets/wasm/*` | Tasks-vision WASM bundle (vendored) |
| `assets/models/*.task` | Pose / face / hand landmarker models (vendored from `/mnt/Drive-D/track_b_venv/assets`) |
| `words.json` | 29 labels: 14 avatar words from `backend/app/data/sign_vocabulary.json` + extra ISL words + `rest`, `transition` |
| `convert_and_validate.py` | JSON -> .npy + manifest.csv; validation; runs `features.py` |
| `RECORDER_README.md` | this file |

## Vocabulary (words.json)

14 avatar words (lowercased gloss_ids from `backend/app/data/sign_vocabulary.json`):
`hello, thank you, sorry, please, yes, no, help, water, food, name, you, me, where, how`

Plus (deduplicated): `friend, hospital, school, market, doctor, medicine, good, bad,
morning, night, today, tomorrow, mother, father`

Plus activity labels: `rest` (hands down), `transition` (hands moving, no word).

Total: 27 words + 2 activity labels = 29.

## How to run

1. `cd research/track_b/recorder`
2. `python3 -m http.server 8765`  (serves localhost only)
3. Open `http://localhost:8765/recorder.html` in a browser, grant camera permission.
4. Pick session (e.g. `S1`), pick a word, set duration (default 2.5 s), click **Start**,
   then **Record**. A 3-2-1 countdown shows, then recording runs. Do NOT move during
   the countdown.
5. While recording, the live indicator shows left/right hand detected (green dots) and
   the skeleton overlay draws (same `drawOverlay` as the frontend).
6. After the take: the screen shows the fraction of frames with missing left/right hand.
   If either exceeds 40%, a warning offers **Retake** (same word/session/take), **Save
   anyway**, or **Discard**.
7. **Save** downloads `WORD__SESSION__takeNN__TIMESTAMP.json` containing:
   `(T,543,3) landmarks, label, session, take number, timestamp, effective fps,
   hands-missing fractions, extractor name/version/options`. Per-label per-session take
   counts persist on screen (sessionStorage) across reloads.

## Data protocol (recording plan)

- Session 1: **15 takes per word + 30 rest + 30 transition**.
- Session 2 (a different time, slightly different position/lighting): **10 takes per
  word + 15 rest + 15 transition** — Session 2 is used **only for testing**, never
  training.
- Before recording a sign, check it against the INCLUDE reference clips /
  avatar clip (`research/track_b/data/include_clips/*`, `include_landmarks/*`).
- Keep framing consistent with the live SignSage app (user-facing camera, head and
  hands in frame, similar distance).

## Convert + validate

```
python3 convert_and_validate.py <clip.json or dir of clips> [--out <dir>]
```

For every clip: validates shape `(T, 543, 3)`, writes `<stem>.npy` (float32), appends
`manifest.csv`, and reports per-clip hands-missing fractions. Then runs
`research/track_b/features.py` (`extract_features`, layout `signsage`) on one clip and
prints the output shape, expected `(60, 98)`.

## Testing status

- `convert_and_validate.py` — TESTED on a synthetic clip (shape, .npy, manifest,
  missing-fraction recomputation, `features.py` output `(60, 98)`) and on an existing
  INCLUDE clip (`hello__INCLUDE__00570__MVI_0037.npy`, shape (59,543,3) -> (60,98)).
- Asset serving — TESTED: all vendored files return HTTP 200 on localhost.
- `recorder.html` camera / record / save flow — **NOT TESTED**: no camera device and
  no browser automation available in this environment; the page has not been opened in
  a browser. MediaPipe `delegate: "GPU"` initialization has not been exercised here.
- Take counts, countdown, warning/retake, JSON download — **NOT TESTED** (same reason).
