# SignSage — Sign Animation (landmark → avatar clip)

How the Text → Sign avatar gets its motion, how to reproduce it, and what it is **not**.

## 1. Honesty rules (non-negotiable)

- **No motion has been validated by a fluent ISL signer.** The UI says so.
- A gloss without a shipped clip shows **"No clip yet"** and is skipped. There is no generic fallback animation.
- Hand-authored approximations (PLEASE, FOOD, NAME, HOW) are keyframed from a template — not recorded, not validated, `needsReview: true`.
- Retargeted clips are mechanical transfers of recorded MediaPipe landmarks. Fidelity is bounded by the data (2.5D, low frame rate, detection gaps); a machine cannot vouch for linguistic correctness.
- Distinguishability passes automated checks; **correctness does not** — signer review is required.

## 2. Data provenance (as of 2026-11-17)

| Gloss | Source | Samples | Notes |
|---|---|---|---|
| HELLO | ISLRTC dictionary videos | 21 | DTW-medoid retarget; both hands detected in every frame |
| THANK_YOU | ISLRTC dictionary videos | 21 | Same |
| WHERE | Local single-signer recording | 1 | Single sample, `needsReview`, slow (6.4 s) |
| PLEASE, FOOD, NAME, HOW | Authored approximation | 0 | No recording exists |
| SORRY, YES, NO, HELP, WATER, ME, YOU | **No clip ships** | 1–2 poor | One-handed or partial detections; recordings needed |

Sample layout: `(T, 75, 3)` float32 — pose 0–32, left-hand slot 33–53, right-hand slot 54–74; image-normalised `x, y` plus relative `z` (damped ×0.3 on retarget). MediaPipe handedness labels are unusable on unmirrored video; the physical side is assigned from image x versus the shoulder midpoint (image-left = the person's right) with nearest-wrist continuity.

## 3. Pipeline

1. **Extract** — video → `(T,75,3)` landmarks (MediaPipe Tasks API).
2. **Clean** — aspect correction, shoulder-centre normalisation scaled to the avatar shoulder width (0.229 m), gap interpolation (≤ 5 frames), temporal median, DTW-medoid selection, idle trim.
3. **Retarget** per frame — arms by two-bone IK; wrist swing and twist matched to the palm plane; fingers by per-bone minimal-arc swing; metacarpals damped; smoothing window 5; 8-frame ease in/out baked in. Output: absolute local quaternions per bone at source fps.
4. **Author** (fallback) — synthetic hand-template keyframes through the same retargeter, marked `authored-approximation`.
5. **Build** — write `frontend/public/models/avatar/clips/<gloss>.json` and a report; update the manifest (`clip: null` means "No clip yet").
6. **Play** — `HumanAvatarViewer` fetches and caches clips, applies frames through `HumanAvatarScene.applyClip()` with slerp between frames; ease in/out lets sequences crossfade through rest.

The GLB (`signsage-human.glb`, MakeHuman rig, 164 joints, 0 animations) is **never swapped**; only poses change.

> **To verify:** the 2026-11-17 workspace cleanup notes that a `tools/` directory was removed, while the scripts below live under `tools/`. Restore them (or note their new location here) before relying on the commands.

```bash
python3 tools/build_sign_clips.py      # forward-kinematics check per sign + rebuild clips
python3 tools/verify_clips.py          # FK error, pairwise DTW, left/right report
python3 tools/render_verification.mjs  # headless renders (run from frontend/)
python3 tools/render_landmarks.py && python3 tools/compose_side_by_side.py
pytest tools/signlib/test_retarget.py
```

Verification numbers (FK error in shoulder-widths): HELLO 0.55, THANK_YOU 0.48, WHERE 0.25 (single sample), authored ≈ 0.2 by construction. Closest pair by DTW: FOOD vs NAME (0.084; both one-handed chin/chest taps) — flagged for signer review. Generated renders are reproducible and should not be committed.

## 4. Rig conventions

- MakeHuman-style rig: `spine01..05`, `neck01..03`, `upperarm01/02`, `lowerarm01/02`, fingers `finger1` (thumb) to `finger5` (pinky), `metacarpal1..4` feeding fingers 2–5. Not Mixamo, Ready Player Me or VRM.
- Units are metres; glTF Y-up. The `root` node carries a 180° flip, so the character faces **+Z**; its left hand is at world **+X**, right hand at **−X**.
- **Bone length axis is local +Y** for every limb and finger bone; **finger curl is about local +X**. Rest pose is an A-pose (upper arms ≈ 37° below horizontal, slight elbow flex).
- Shoulder width ≈ 0.229 m; height ≈ 1.75 m.
- three.js strips `. : [ ]` from node names (`wrist.L` → `wristL`); `sanitizeBoneName()` in `rigMap.ts` handles it. Bone map: `frontend/src/features/avatar-test/rigMap.ts`; curl sign `+1` (the earlier `−1` produced backward claws).
- Machine-readable rest transforms: `rig_bones.json` (regenerate with the rig analysis script rather than keeping a long dump here).

## 5. Quality and validation gate

Before any UI copy changes from "not validated":

- **Automated:** no NaN tracks; angular velocity within physical bounds; clip duration within ±20% of the manifest; every bone name resolves in the GLB; morph targets limited to a whitelist.
- **Human:** a signer reviews each clip beside its reference; an ISL-fluent reviewer signs off vocabulary coverage before any public release.
- **Measure** retarget drift against a held-out take; if wrist error exceeds ≈ 10 cm equivalent, fix rest-pose calibration first. The thumb is the weak spot of any retarget — validate it most.

Risks: signer recordings are personal data — commit only retargeted bone tracks (no RGB, no face geometry), keep raw video local and delete on request; one-signer motion stays labelled as such; finger curl direction must follow the conventions above.

## 6. Growing beyond 14 signs

Recorded, multi-signer isolated-word video is the right source. The FDMSE-ISL extraction pipeline already saves per-video pose and hand landmarks for ≈ 2,000 words and could be converted to the `(T,75,3)` layout. **But** FDMSE-ISL is research-only and not redistributable: shipping clips derived from it needs the authors' explicit written permission ([MODELS_AND_DATA §3](MODELS_AND_DATA.md#3-fdmse-isl)). Until that is granted, use your own recordings or permissively licensed sources for shipped clips.
