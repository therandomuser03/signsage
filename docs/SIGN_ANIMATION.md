# Sign animation pipeline — landmark → GLB clip

How the Text→Sign page's avatar motion is produced, reproduced, and what it is NOT.

## Honesty rules (non-negotiable)

- No motion has been validated by a fluent ISL signer. The UI keeps saying so.
- Glosses without a shipped clip render a **"No clip yet"** alert and skip the gloss;
  they never fall back to the old generic articulation demo (removed 2026-11-17).
- Authored approximations (PLEASE, FOOD, NAME, HOW) are hand-keyframed
  approximations — NOT recorded data, NOT validated ISL, `needsReview: true`.
- Retargeted clips are mechanical transfers of recorded MediaPipe landmarks.
  Faithfulness to the recording is bounded by the data (2.5D, low fps, gaps);
  machines cannot vouch for the sign's linguistic correctness.

## Data provenance per clip

| Gloss | Source | Samples | Notes |
|---|---|---|---|
| HELLO | ISLRTC dictionary videos, MediaPipe (T,75,3) | 21 | DTW-medoid retarget; both hands 100% present |
| THANK_YOU | ISLRTC dictionary videos | 21 | DTW-medoid retarget; both hands 100% present |
| WHERE | local single-signer recording (`data/videos/where.mp4`) | 1 | single sample, `needsReview: true`, 6.4 s (slow, authentic data) |
| PLEASE | authored approximation | 0 | no recording exists |
| FOOD | authored approximation | 0 | no recording exists |
| NAME | authored approximation | 0 | no recording exists |
| HOW | authored approximation | 0 | no recording exists |
| SORRY, YES, NO, HELP, WATER, ME, YOU | **none ships** | — | poor one-handed captures only; recordings needed |

Sample schema (docs/sign_data_audit.md): `(T,75,3)` float32 — pose 0–33,
left-hand slot 33–54, right-hand slot 54–75; image-normalized x,y plus
relative z (z-damped ×0.3 on retarget). Handedness labels are unusable on
unmirrored video; physical side is assigned by image x vs shoulder midpoint
(image-left = person's right), with nearest-wrist temporal continuity.

## Pipeline (tools/)

1. `extract_sign_landmarks.py` — video → `(T,75,3)` npy (MediaPipe Tasks API).
2. `signlib/landmarks.py` — aspect correction to pixel units → shoulder-center
   normalization scaled by avatar shoulder width (0.229 m) → gap interpolation
   (≤5 frames) → per-landmark temporal median → DTW medoid selection → idle-trim.
3. `signlib/retarget.py` — per frame:
   - arms: two-bone IK (pole = rest elbow offset, out/down/slightly back);
   - wrist: delta-from-bind swing (wrist→middle-MCP) + twist aligning the palm
     plane to the landmark palm plane (branch chosen nearest the carried palm,
     or exact when an authored palm normal is supplied);
   - fingers: per-bone minimal-arc swing onto the landmark chain directions;
     metacarpals damped to 60%; all expressed as local quaternions vs rest;
   - smoothing: per-component hemisphere-fixed moving average (window 5);
     8-frame ease-in/out from rest baked into the clip;
   - output: absolute local quaternions per bone per frame at source fps.
4. `author_sign_clips.py` — synthetic hand-template keyframes → same retargeter
   (anatomy: index/pinky offsets and hyperextensible-joint-free chains); output
   clips marked `authored-approximation`.
5. `build_sign_clips.py` → `frontend/public/models/avatar/clips/<gloss>.json`
   + `docs/verification/retarget_report.json`.
6. `update_manifest.py` → manifest `clip` entries (`null` = "no clip yet").

`signsage-human.glb` (MakeHuman rig, 164 joints) is NEVER swapped; only poses change.

## Runtime (frontend)

`HumanAvatarViewer.tsx` fetches/caches clips from `/models/avatar/clips/*.json`,
preloads them once the GLB is up, then per gloss applies frames via
`HumanAvatarScene.applyClip()` — absolute local quaternions, slerped between
frames. Ease in/out is baked in, so sequences crossfade naturally through rest.
Stop/Replay/Clear behave as before; the progress indicator and
`unavailableGlosses` ("No clip yet") flow through `useSignAnimation` unchanged.
`/dev/verify-clips` (DEV-only) pins any clip+frame for screenshots.

## Verification (Step 3)

Run:
```
python3 tools/build_sign_clips.py        # FK check per sign + rebuild clips
python3 tools/verify_clips.py            # FK error + pairwise DTW + L/R report
python3 tools/render_verification.mjs    # (frontend/) headless renders start/mid/end
python3 tools/render_landmarks.py        # source landmark panels
python3 tools/compose_side_by_side.py    # side-by-side composites
./backend/.venv/bin/python -m pytest tools/signlib/test_retarget.py
```

Current numbers are in `docs/verification/verification_report.json` and
`retarget_report.json`. FK error ≈ mean wrist+fingertip error vs source
landmarks, in shoulder-widths (0.229 m): HELLO 0.55, THANK_YOU 0.48, WHERE 0.25
(single sample), authored ≈ 0.2 by construction (synthetic ground truth).
Pairwise DTW between all 7 clips: minimum 0.084 (FOOD vs NAME — both are
one-handed chin/chest taps; flagged for signer review to differentiate).
Left/right mapping covered by unit tests (image-left ⇒ physical right ⇒ rig .R).
Distinguishability passes; correctness does NOT — signer review still required.
