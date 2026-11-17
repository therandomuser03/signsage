# Sign data audit — what exists locally for the 14-sign library

Date: 2026-11-17. 14 manifest signs: HELLO THANK_YOU SORRY PLEASE YES NO HELP WATER FOOD NAME YOU ME WHERE HOW.

## Datasets found in the repo

### A. `research/openhands_pilot/extracted_poses/*.npy` — MediaPipe Pose+Hands, isolated ISLRTC clips
- Layout: `(T, 75, 3)` float32. `[0:33]` MediaPipe Pose 33 (x,y image-normalized, z hip-relative depth), `[33:54]` left hand 21, `[54:75]` right hand 21 (MediaPipe Hands order). Hand slots are assigned by MediaPipe's handedness label, which for these **unmirrored** dictionary videos is unreliable (see note in `backend/app/ml/preprocessing/sign_features_v3.py`: "MediaPipe Left/Right handedness labels are IGNORED (camera is unmirrored)").
- Zeros = missing detection. Hand presence in every file listed below is 100% on both hands.
- Coverage: hello ×21, thank_you ×21, school ×20, market ×20, hospital ×20, friend ×20. (school/market/hospital/friend are not in the 14-sign library.)
- Source videos: `backend/data/isl/videos/{hello,thank_you}/…INCLUDE__….mp4` (ISLRTC dictionary, multiple signers, frontal).
- fps: source videos are 25/30 fps; T ≈ 55–97 frames per clip.
- This is the same layout produced by `research/openhands_pilot/extract_poses.py`.

### B. `backend/data/isl/videos/` — raw ISLRTC dictionary videos (not yet landmark-extracted)
- sorry ×2 (`sorry__ISLRTC_dictionary__00636__Sorry.mp4`, `…00637__Sorry_(Sign_2).mp4`)
- me ×1 (`me__ISLRTC_dictionary__00633__Myself.mp4`), come ×1 (not in library), plus subdirectories for hello/thank_you/friend/market/hospital/school.

### C. `data/videos/*.mp4` — local phone recordings of isolated signs (2026-11-14, low res, one signer)
- hello, help, me, no, water, where, yes, you (+ rest.mp4, not a sign). 272–490 px wide, 25–30 fps, 4.6–9.5 s.
- Landmarks extracted 2026-11-17 with `tools/extract_sign_landmarks.py` → `data/landmarks/*.npy` (same (T,75,3) layout as A).

### D. FDMSE-ISL 200-word training pipeline (`research/FDMSE/SignSage_FDMSE_ISL_200_Word_Training.ipynb`)
- Schema: MediaPipe **Holistic**, `(T, 543, 3)` float32 per video. Component order (verified in `research/evaluation/reports/352868924/landmark_contract.json`): POSE 0–33, FACE 33–501, **LEFT_HAND 501–522, RIGHT_HAND 522–543**; POSE_WORLD excluded from features. x,y image-normalized, z relative.
- Features actually fed to the model: `sign_features_v3` → (60, 122) hand-centric vector per clip (see `backend/app/ml/preprocessing/sign_features_v3.py`), x,y only, hands tracked by temporal nearest-wrist (labels ignored), active-hand mirrored right.
- The raw videos live **remotely** (rkmvu.ac.in portal, key-gated); they are NOT in this repo. Only extracted (T,75,3) poses for the 6 signs in A exist locally.
- Signer/split info: FDMSE splits by unique video UID (train/val/test), 200 target words, ≥6 train videos/class required.

### E. `staging_kaggle_results/s2t_v2/pose_cache/*.pose` — iSign sign-to-text, continuous signing
- MediaPipe PoseLandmarker flatbuffers (POSE_LANDMARKS, with FACE etc. in header; 543-pt Holistic-style). Sign-to-text corpus, **not** isolated signs; target words are free text, not the 14-sign library. Not usable directly for per-sign clips without alignment work. Not touched.

## Samples per library sign (after extraction)

| Sign | Usable samples | Source | Quality notes |
|------|---------------|--------|----------------|
| HELLO | 21 + 1 | A (INCLUDE) + C local | INCLUDE: both hands 100% frames. Local: L 59% / R 5% — poor |
| THANK_YOU | 21 | A (INCLUDE) | both hands 100% |
| SORRY | 2 | B (ISLRTC dictionary) | a: L 58% / R 59%; b: L 66% / R 60% — both partial |
| HELP | 1 | C local | L 42% / R 17% — poor |
| ME | 2 | C local + B (Myself) | me.mp4: L 11% / R 40%; myself: L 4% / R 37% — poor |
| NO | 1 | C local | L 79% / R 1% — effectively right-hand-only detection |
| WATER | 1 | C local | L 2% / R 53% — one hand |
| WHERE | 1 | C local | L 87% / R 78% — decent |
| YES | 1 | C local | L 75% / R 12% — one hand |
| YOU | 1 | C local | L 2% / R 37% — one hand |
| PLEASE | 0 | — | no data anywhere in repo |
| FOOD | 0 | — | none |
| NAME | 0 | — | none |
| HOW | 0 | — | none |

Handedness note: for unmirrored frontal video, a physical right hand appears on the image-left side; MediaPipe's "Left/Right" labels cannot be trusted. Physical side must be assigned by image x (and/or flipped if the source is mirrored). The avatar's right hand is world −X.

Depth note: all local data is 2.5D (image-normalized x,y + relative z). There is no metric 3D. Retargeting can be convincing for flexion (angles are mostly in-plane) and for large palm-orientation changes, but palm-facing-camera rotations and true 3D arcs are under-constrained.

## Consequences for the plan
- HELLO, THANK_YOU: retargetable with real multi-signer data today; expect decent quality with DTW-medoid selection.
- WHERE: one decent single-signer sample — usable as a single-sample clip (flagged needs-review).
- NO, WATER, YES, YOU, HELP, ME, SORRY: effectively 1–2 mediocre single-signer samples; clips will be noisy, one-handed where detection failed. Usable as *low-confidence* clips, must be flagged "needs review".
- PLEASE, FOOD, NAME, HOW: **no data at all.** Options: (a) hand-authored keyframes from the rig dump (explicitly allowed by the task as fallback), clearly labelled as authored, not recorded; (b) drop the clip and show "no clip yet". Never fabricate "recorded" provenance.

## Recommended
1. Retarget from A/B/C where they exist; keep per-sign provenance in manifest.json (source file, signer count, sample count, needs_review flag).
2. For PLEASE/FOOD/NAME/HOW: hand-author from ISL dictionary references (human signer needed for correctness) OR ship "no clip yet". Decision needed from owner (see Step 0 summary).
3. Before any "validated" label: a fluent ISL signer reviews every clip.
