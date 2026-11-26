# Changelog

All notable changes to SignSage. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); entries before 2026-11 are reconstructed from the retired status and audit reports, so dates are the dates those reports recorded.

## [Unreleased]

### Added
- **Live-path stress test** (`SignSage_03_LivePath_StressTest_v2.ipynb`): replays real saved test-clip landmarks through the serving code under 13 live-like conditions and saves `stress_test.csv`. Browser pose model switched from lite to full to match training (`landmarks.ts`).
- Motion floor for the FDMSE-ISL model written to its `metrics.json` (0.00181).
- **Serving integration for the FDMSE-ISL model** (`signsage-integration-patch.zip`): `sign_features_v4` serving recipe (resample to 48 frames, pose-derived face points, aspect correction), a new `fdmse_word_adapter` that loads the notebook-02 export (loader dispatch in `WordClassifierAdapter.load`), the frontend now sends the camera size and captures 3 s, 28 tests including a golden-vector parity test against the training code and an export-to-adapter prediction check. Motion floor cell added to notebook 02.
- **First full-vocabulary recogniser trained** (`baseline_v1`, BiGRU, FDMSE-ISL, 2,002 words): official test split (8 unseen signers) top-1 82.1%, top-5 95.1%; validation top-1 85.8%; exported with preprocessing, labels, metrics and model card. Offline result on studio video; not integrated and not promoted. See [MODELS_AND_DATA §7](docs/MODELS_AND_DATA.md#7-fdmse-isl-training-pipeline-v2).
- **FDMSE-ISL extraction finished** (Nov 2026): 40,033 of 40,033 videos, 801 chunks, all 2,002 classes, no corrupt chunks, 0 unusable clips; features built for the whole dataset. Details in [MODELS_AND_DATA §7](docs/MODELS_AND_DATA.md#7-fdmse-isl-training-pipeline-v2).
- **FDMSE-ISL extraction pipeline v2** (`SignSage_01_FDMSE_Extraction_v2.ipynb`): covers all 2,002 words; resumable 50-video chunks written atomically; streams videos from the portal or reads an archive you already have (no key needed); keeps raw landmarks; builds `sign_features_v3` features; locks settings across sessions and machines; `DRY_RUN` mode. See [MODELS_AND_DATA §7](docs/MODELS_AND_DATA.md#7-fdmse-isl-training-pipeline-v2).
- **Training notebook** (`SignSage_02_FDMSE_Training_v2.ipynb`): CPU-friendly resumable BiGRU training, test split used once, probe-aware confidence threshold, model export. Both notebooks can read a local dataset (zip file or unzipped folder) and write a progress log.
- Root files: `SECURITY.md`, `CONTRIBUTING.md`, `CREDITS.md`, `CHANGELOG.md`.

### Changed
- **Project direction:** from research prototype toward a real, user-facing product with real users, trained on the full dataset, with fingerspelling fallback and visible credits ([PRD](docs/PRD.md)).
- **Documentation consolidated** from 51 Markdown files (about 86,000 words) into 15 maintained documents (PRD, ARCHITECTURE, MODELS_AND_DATA, DESIGN_SYSTEM, CODE_STYLE, TESTING, DEPLOYMENT, SIGN_ANIMATION). Historical audit reports were retired; their conclusions live in the documents above and in this file. The old files remain in git history.

### Fixed
- **Two flaws of `sign_features_v3` with missed hands** (found by the live-path stress test, where a 10% hand-miss rate cost 16 accuracy points): the active hand was chosen from wrist travel measured on zero-filled arrays, and mirroring turned missed frames into phantom hands. `sign_features_v4` revision 2 (`robust=True`, the serving default) fixes both and is identical to v3 when no hand is missed; `sign_features_v3` itself is unchanged. Notebook 01 cell 12 builds robust features into `features_v4r2_pose_face`; the stress test compares both recipes.
- Privacy wording: the app also has an optional upload-a-video flow, so "raw video never leaves the browser" now applies to the live camera path only.
- The earlier `fdmse_word_adapter.py` (written for the 200-word model) had a different architecture and no input standardisation, face points or aspect handling; it is replaced.
- Reading chunk and feature files from Google Drive now retries and copies to local disk first: a transient Drive error (`Errno 103`, shown by numpy as "File is not a zip file") no longer aborts the feature build or training; an unreadable chunk is skipped with a warning and picked up on rerun.
- **Face landmarks were missing from the extracted dataset** (detected in about 0% of frames), which would have made the hand-to-face features depend on where the signer stands. Features are now `sign_features_v4`: face reference points derived from the pose model and isotropic coordinates, rebuilt from the saved landmarks without re-extraction. The training notebook now prints a detection-quality report before training and warns about this class of problem.
- The earlier 200-word Colab notebook kept the alphabetically first 200 words (all classes tied on its coverage score), so common words such as "hello" were never trained. v2 uses every class.
- `references.bib` contained Markdown code fences that break LaTeX; it is plain BibTeX now.

## 2026-11-17
### Added
- Avatar sign motion pipeline: MediaPipe landmarks → retargeted JSON clips on the fixed MakeHuman GLB. Clips for HELLO, THANK_YOU (multi-signer data) and WHERE (single sample); hand-authored approximations for PLEASE, FOOD, NAME, HOW. Verification report and a DEV-only clip viewer. ([SIGN_ANIMATION](docs/SIGN_ANIMATION.md))
- Landmark extraction for local single-signer recordings in `data/videos/`.
### Removed
- The generic articulation demo for glosses without a clip; they now show "No clip yet".
- Workspace cleanup: `mobile/`, `tools/`, `verified_checkpoint/`, ISLiM mobile exports and dataset zips removed; signer reference clips moved to `data/videos/`.

## 2026-11-15
### Changed
- Live Interpreter rebuilt around a dark `Interpretation` panel with explicit states; Browse Signs sheet replaced by a Help dialog.
- Sentence composer may reorder content words (every signed word still appears exactly once).
### Added
- Auto-Speak for completed interpretations; backend text-to-speech endpoint with automatic browser fallback; constrained Groq composer (`gpt-oss-20b`, one fallback model, then plain join).

## 2026-11-02
### Added
- Model decision record: no continuous model approved; Sid-V5 named the next research candidate; ISLiM and Sanket not approved. Documentation audit and index.

## 2026-11-01
### Added
- Isolated-word and ISLiM adapters with endpoints (evaluation blocked: no paired data).
### Removed
- Four local model files (≈ 2.5 GB): Sanket weights, ISLiM checkpoint B, two mobile exports.

## 2026-10-27
### Added
- Static image classification: alphabet CNN (87.1% test accuracy, A–Z) and a word CNN (19.0%, unpromoted) behind `POST /api/v1/classification/image`; Playwright E2E tests (7) on real fixtures.
- Dataset and vocabulary audits for temporal ISL data.

## 2026-10-26
### Added
- Isolated ISL recovery audit and a 6-word pilot (61.1% test accuracy on 18 samples; unpromoted).

## 2026-10-12 / 2026-10-13
### Added
- Checkpoint-free 5-shape heuristic recogniser; microphone input for Text → Sign; recognition expansion plan.
### Changed
- Auth/history review: sign-out now clears history and saved-signs lists; consent timestamps stamped at signup.
- Status, development, data/model and external-evaluation notes consolidated.
### Removed
- Unused neural text-to-sign stubs and `refinement.py`.

## 2026-10-02 / 2026-10-05
### Added
- Hostile-input rejection (NaN/±Inf), bounded WebSocket reconnect, `render.yaml` and checksum-verified `fetch_bundle.sh`, streaming inference with repeat-agreement.
### Verified
- End-to-end honesty machinery (capture → transport → validation → gating → explicit uncertainty); recognition itself not functional at that time.

## Phase 0 (earlier)
- Bounded run (2,000 / 300 / 300) and scaled run (10,000 / 1,500 / 1,500) of a sign-to-text transformer: severe mode collapse (WER ≈ 1.0, diversity ≈ 1%). Checkpoint `352868924` is kept only as a negative reference. External T5 candidate scored 0 / 1,887 exact match.
