# Sign-clip pipeline — making Text → Sign actually sign

**Status:** proposal (2026-11-17). Not built. SignSage today plays a generic, clearly-labelled articulation demo for every gloss because there is no ISL motion data anywhere in the repo (`signsage-human.glb` contains 0 animation clips; the 14 `/models/avatar/signs/*.glb` files referenced by the manifest do not exist).

**Goal:** ship real ISL motion for the existing 14-word vocabulary (HELLO, THANK_YOU, SORRY, PLEASE, YES, NO, HELP, WATER, FOOD, NAME, YOU, ME, WHERE, HOW — the manifest order), matching the signer-adapted scope of the word CNN (`word_v3`). Not signer-independent; UI must keep saying so.

**Non-goal:** generative/rig-learned sign synthesis, signer-independent motion, grammar beyond gloss order. Those are separate research tracks and are explicitly out of scope here.

## Why not "just animate it"

The GLB has a complete MakeHuman-style rig (166 nodes, 38 finger bones verified in the export) plus morph targets, and the codebase already owns a correct bone map (`frontend/src/features/avatar-test/rigMap.ts` — finger chains, sanitised names, curl conventions, connectivity inspection). Motion capture from the signer is the cheapest truthful source of sign data: one signer, the same pipeline the app already uses for recognition. Anything else (manual keyframing of 14 signs × ~50 keyframes, or learned synthesis) is more work and less honest.

## Pipeline

```text
webcam → MediaPipe tasks-vision (pose+face+hands, VIDEO mode)
  → artifacts/<GLOSS>/take_<n>.json   (landmark tracks, mm-normalised, fps)
  → clean/trim/resample                (60 fps, drop occluded frames ≤ 2, interpolate short gaps)
  → retarget onto signsage-human.glb   (rigMap.ts bone map; rest-pose calibration per session)
  → clips/<GLOSS>.glb                  (glTF AnimationClip embedded, LINEAR keys, 30–60 fps)
  → manifest update                    (same SignManifest shape; adds "clip": { "url": ..., "authored": true })
  → playback via AnimationMixer         (inside HumanAvatarViewer / future SignClipPlayer)
```

### Stage details

1. **Capture** (`research/sign_clips/` tools, NOT app runtime): a recorder that wraps the same `@mediapipe/tasks-vision` configuration as `frontend/src/features/translate/landmarks.ts` (identical models, GPU delegate, VIDEO mode). 2–3 takes per gloss, 1920×1080, ≥24 fps, full body, plain background. Raw `.mp4` stays untracked/local; the gitignored capture output is `take_<n>.json` (MediaPipe landmark arrays + timestamps).
2. **Clean**: pick one take per gloss (signer self-selects best); resample to 60 fps; drop leading/trailing rest; interpolate hand gaps ≤ 2 frames; discard or re-record takes with >20 % hand-missing frames (typical during fast finger moves — mitigation is takes, not interpolation).
3. **Retarget** (the hard part; 2–4 days):
   - Rest-pose calibration per capture session: one frame of the signer in A-pose; measure shoulder width, arm lengths, hand size; build signer→rig scale.
   - Pose/shoulder/elbow/wrist: match bone segment directions (`bone_dir` per the rigMap convention — bone length axis is local **Y**, verified for every finger/arm bone), solve minimal rotation from rest orientation; clamp to anatomical ranges (elbow ≥ −150° flexion, no shoulder beyond ±120° abduction).
   - Hands (per finger, per side): local flexion from the angle between consecutive bone segments (wrist→mcp→pip→dip); metacarpals from palm arc; thumb from its own 3-bone chain (different hinge axis than fingers 2–5 — thumb is the weak spot of any retarget; give it the most validation).
   - Face: only `eyeBlinkLeft/Right` and `jawOpen` morph targets driven from the GLB's own morph-target list (browser head-roll/blink was already used by the articulation demo; reuse `setMorphTarget`).
   - Output per gloss: one `THREE.AnimationClip` baked to glTF LINEAR tracks on the actual bone node names (sanitised the same way `sanitizeBoneName` does: strip `.:[\]`).
4. **Export**: Python + `numpy` + `pygltflib` (batch, headless) to write `clips/<GLOSS>.glb`. Alternative JS route (three.js `GLTFExporter` with `animations`) is acceptable if Python authoring goes sideways. Keep one code path — do not ship both.
5. **Playback**: extend `HumanAvatarViewer` (or extract `SignClipPlayer` sharing `HumanAvatarScene`) with `THREE.AnimationMixer`: fetch + cache clips by URL, apply tracks to live bones by name, 0.15 s crossfade between signs, ease back to rest at sequence end. Fallback chain unchanged: clip available → play clip; else → current articulation demo (so builds without clips still work). UI copy changes from "articulation demo" to "signing <GLOSS>" only when ≥1 verified clip played.
6. **Validation gate — before any UI copy change:**
   - Automated: no NaN tracks; bone angular velocity within physical bounds; clip duration within ±20 % of manifest `durationMs`; every clip bone name resolves in the GLB (extend `rigMap.test.ts`); morph targets limited to whitelist.
   - Human: signer reviews all 14 clips side-by-side with the reference take; an ISL-fluent reviewer (not required for release-to-self, required before public deploy) signs off vocabulary coverage.
   - Honest framing stays until this gate passes.

## Sequencing

1. Wire ONE gloss end-to-end (recommend `HELLO`): capture → clean → retarget → bake → play. This de-risks the whole pipeline; everything else is repetition.
2. The remaining 13 glosses in one batch session.
3. Validation gate, then UI copy change, then (separately) consider committing clips to git (`frontend/public/models/avatar/signs/`, ~50–200 KB each — committing is fine, raw captures never are).

## Risks and mitigations

- **Signer data privacy**: signed clips are biometric-ish personal data. Git gets only retargeted bone tracks (no RGB, no face geometry); raw video stays local. Signer consent is the owner deleting/shipping these — never commit raw captures.
- **One-signer limitation** carries to motion: fine for the word-CNN-aligned demo; label it the same way (`signer-adapted`) as today.
- **Retarget drift** (e.g., wrist slide): measure final hand-position error against the capture on a holdout take; if > ~10 cm equivalent, iterate on rest-pose calibration before touching anything else.
- **Finger curl direction**: use the conventions fixed in `humanAvatarScene.ts` (CURL_SIGN = +1 — the earlier −1 was curl-backward claws, fixed 2026-11-17); thumb may need per-side sign.
- **Licensing**: MediaPipe models (Apache-2.0) are fine; the sign content is the signer's own.

## What this does NOT touch

- The generic articulation demo remains as the no-clips fallback.
- The VRM/AvatarController path (`AvatarViewer.tsx`) is untouched — if hand-authored VRM + GLB clips ever ship, it reuses `playSequence` as-is.
- Backend, word CNN, iSign work: unchanged.
