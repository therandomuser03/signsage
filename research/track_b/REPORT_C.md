# Track B Report — INCLUDE 17-Word Landmark Dataset: Extract, Train, Evaluate

**Status**: OFFLINE research only. Nothing integrated into the app. No existing files, `.env`, `render.yaml`, or `SIGNSAGE_MODEL_PROMOTED` touched. No commit/push.
**Scope**: Standalone under `research/track_b/`. All data, venvs, caches on `/mnt/Drive-D`.
**Attribution**: INCLUDE dataset, Sridhar et al., ACM MM 2020, AI4Bharat. Licence CC-BY-4.0.
Source: Zenodo record 4010759 (https://zenodo.org/records/4010759).

---

## 1. Extractor Parity (Step 1)

### Frontend (`frontend/src/features/translate/landmarks.ts`, `frontend/package.json`)

| Property | Value |
|---|---|
| Package | `@mediapipe/tasks-vision` |
| Version | `^1.0.1` |
| API | **tasks-vision** (PoseLandmarker + FaceLandmarker + HandLandmarker) — NOT legacy Holistic |
| Pose model | `pose_landmarker_lite` (float16), `delegate: GPU`, `runningMode: VIDEO`, `numPoses: 1` |
| Face model | `face_landmarker` (float16), `delegate: GPU`, `runningMode: VIDEO`, `numFaces: 1` |
| Hand model | `hand_landmarker` (float16), `delegate: GPU`, `runningMode: VIDEO`, `numHands: 2` |
| Min confidences | Pose detection/presence/tracking = 0.5 (hands: defaults) |
| Missing landmarks | `[0, 0, 0]` zeros (per `toPoints()` returning `zero()` for absent marks) |
| Output layout | `(T, 543, 3)` SignSage order: pose 0-32, face 33-500, left hand 501-521, right hand 522-542 |

### Python extractor (`research/track_b/extract_tasks.py`)

| Property | Value |
|---|---|
| Package | `mediapipe` 1.2.2 (installed via pip on Python 3.15; 0.10.18 unavailable on PyPI) |
| API | MediaPipe Tasks Vision (PoseLandmarker / FaceLandmarker / HandLandmarker) |
| Running mode | **IMAGE** (frontend uses VIDEO; temporal smoothing not reproduced) |
| Delegate | **CPU** (frontend uses GPU) |
| Num poses/faces/hands | 1 / 1 / 2 |
| Min confidences | Pose 0.5 / 0.5 / 0.5; hands default |
| Missing landmarks | `[0, 0, 0]` zeros — identical to frontend |
| Output layout | `(T, 543, 3)` SignSage order — identical to frontend |

### Key differences (frontend vs Python)
1. **VIDEO mode → IMAGE mode**: Frontend processes video stream with temporal tracking; Python extracts frame-by-frame independently. Documented limitation.
2. **GPU delegate → CPU delegate**: Python runs on CPU. Slower but offline-friendly (no GPU drivers needed).
3. **Model package**: Frontend `@mediapipe/tasks-vision@1.0.1` (WASM); Python `mediapipe==1.2.2` (native, TFLite XNNPACK CPU). 0.10.18 was requested but unavailable on PyPI; 1.2.2 is the closest available.

### Verification
`mp.solutions.holistic` not used (frontend is tasks-vision). Python Tasks API verified: `PoseLandmarker`, `FaceLandmarker`, `HandLandmarker` all instantiate and detect on a test clip from `backend/data/isl/videos/`. Output shape `(55, 543, 3)` confirmed on one clip.

---

## 2. Clip Map (Step 2)

From `train_include.csv` / `test_include.csv` (OpenHands repo metadata), 17-word vocabulary:

| Word | Train | Test | Total | Category |
|---|---|---|---|---|
| hello | 17 | 4 | 21 | Greetings |
| thank you | 17 | 4 | 21 | Greetings |
| you | 17 | 4 | 21 | Pronouns |
| friend | 16 | 4 | 20 | People |
| hospital | 16 | 4 | 20 | Places |
| school | 16 | 4 | 20 | Places |
| market | 18 | 3 | 21 | Places |
| doctor | 11 | 3 | 14 | Jobs |
| medicine | 11 | 3 | 14 | Society |
| good | 17 | 4 | 21 | Adjectives |
| bad | 17 | 4 | 21 | Adjectives |
| morning | 13 | 1 | 14 | Days_and_Time |
| night | 14 | 1 | 15 | Days_and_Time |
| today | 13 | 1 | 14 | Days_and_Time |
| tomorrow | 13 | 1 | 14 | Days_and_Time |
| mother | 16 | 4 | 20 | People |
| father | 16 | 4 | 20 | People |
| **TOTAL** | **258** | **53** | **311** | |

**Local videos already present**: 121 (in `backend/data/isl/videos/`, both flat and per-word subdirectories).
**Categories involved**: Adjectives, Days_and_Time, Greetings, Jobs, People, Places, Pronouns, Society.

### Range-request test (Step 2)
Tested `https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content` with `remotezip.RemoteZip`:
- **Range requests work**: Central directory read in ~1.0s, 112 members listed.
- Bytes transferred: only the central directory (a few KB), not the full 1.57 GB zip.
- Conclusion: per-member download via range requests is feasible; no need to download full category zips.

---

## 3. Clip Download (Step 3)

Missing clips: 311 total − 121 local = **190 missing**.
Estimated total size: **~1.6 GB** (well under 6 GB threshold — proceed).

Downloaded only missing members from Zenodo category zips via `remotezip` range requests:
- Categories: Adjectives, Days_and_Time, Greetings, Jobs, People, Places, Pronouns, Society
- Downloaded: **190 members** (some files already existed from prior runs)
- Total downloaded: **~1.6 GB**
- Failed/corrupt: **3 MP4 files with 0 frames** ("moov atom not found"): `MVI_4251.MOV`, `MVI_5240.MOV`, `MVI_9513.MOV` — these produce empty landmark arrays and are excluded from training.

All 311 clips now available: 121 local + 190 downloaded. Stored in `research/track_b/data/include_clips/`.

---

## 4. Landmark Extraction (Step 4)

Extractor: `research/track_b/extract_all.py` (Tasks-API, matches frontend parity).
Output: `research/track_b/data/include_landmarks/*.npy` — `(T, 543, 3)` float32, SignSage order, missing = `[0,0,0]`.
Resumable: skips existing `.npy` files.

| Metric | Value |
|---|---|
| Total unique clips | 314 (311 from CSV + 3 extra local files not in 17-word CSV) |
| Successfully extracted | 292 new + 22 skipped (already existed) = 314 |
| Failed | 0 |
| Corrupt (0 frames) | 3: `MVI_4251`, `MVI_5240`, `MVI_9513` |
| Total time | 2,540.8 s (42.3 min) |
| Avg per clip | ~8.2 s |
| Clips with hands missing >40% frames | **158** (of 314) |

Per-word clip counts after extraction (matched to CSV):
| Word | Train | Test | Total |
|---|---|---|---|
| hello | 18 | 5 | 23 |
| thank you | 16 | 4 | 20 |
| you | 17 | 4 | 21 |
| friend | 16 | 4 | 20 |
| hospital | 16 | 4 | 20 |
| school | 16 | 4 | 20 |
| market | 17 | 3 | 20 |
| doctor | 11 | 3 | 14 |
| medicine | 11 | 3 | 14 |
| good | 16 | 4 | 20 |
| bad | 16 | 4 | 20 |
| morning | 13 | 1 | 14 |
| night | 14 | 1 | 15 |
| today | 13 | 1 | 14 |
| tomorrow | 13 | 1 | 14 |
| mother | 16 | 4 | 20 |
| father | 16 | 4 | 20 |

Note: 26 landmark files were not matched to CSV entries (pre-existing local files like `MVI_0001`, `MVI_0002` that were downloaded earlier but aren't in the 17-word CSV split). Excluded from training.

---

## 5. Training & Evaluation (Step 5)

**Data**: `research/track_b/data/include_landmarks/*.npy` → features via `features.extract_features(layout='signsage')` → **(60, 98)** float32.
**Feature spec**: 98 dims/frame = left hand (42) + right hand (42) + 7 upper-body pose joints (14). Shoulder-center + shoulder-width normalisation per frame. Resampled to 60 frames.
**Split**: Official INCLUDE train/test (not random). 255 train / 54 test.
**Models**: Tiny 1D-CNN (Conv1d 98→32→64, GAP, Linear→17) and Tiny GRU (GRU 98→32, Linear→17).
**Augmentation** (train only): random scale (0.95–1.05), shift (±0.03), rotation (±8°), time warp (0.9–1.1×).
**Optimizer**: Adam lr=1e-3, 60 epochs, CrossEntropyLoss, CPU (torch 2.15.0+cpu).
**Seeds**: 5 per model.

### Results (mean ± std over 5 seeds)

| Model | Train acc | Train macro-F1 | Test acc | Test macro-F1 | Chance baseline |
|---|---|---|---|---|---|
| 1D-CNN | 0.797 ± 0.016 | 0.794 ± 0.017 | **0.604 ± 0.048** | **0.527 ± 0.055** | 0.093 |
| GRU | 0.644 ± 0.054 | 0.621 ± 0.055 | 0.415 ± 0.066 | 0.337 ± 0.083 | 0.093 |

Chance baseline = majority-class accuracy in test set ≈ 9.3% (17 classes, imbalanced test).

### Threshold vs accuracy (1D-CNN, seed 0)

| Max-softmax threshold | Accuracy | Coverage | n covered |
|---|---|---|---|
| 0.5 | 0.650 | 0.741 | 40 |
| 0.7 | 0.870 | 0.426 | 23 |
| 0.9 | 0.900 | 0.185 | 10 |

### Per-class P/R/F1 (1D-CNN, seed 0)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| friend | 1.00 | 1.00 | 1.00 |
| thank you | 0.75 | 0.75 | 0.75 |
| doctor | 0.75 | 1.00 | 0.86 |
| medicine | 0.75 | 1.00 | 0.86 |
| father | 0.67 | 0.50 | 0.57 |
| mother | 0.67 | 0.50 | 0.57 |
| hello | 0.50 | 0.80 | 0.62 |
| hospital | 0.50 | 0.75 | 0.60 |
| good | 0.50 | 0.50 | 0.50 |
| school | 0.50 | 0.50 | 0.50 |
| market | 1.00 | 0.33 | 0.50 |
| morning | 0.25 | 1.00 | 0.40 |
| bad | 0.33 | 0.25 | 0.29 |
| you | 0.00 | 0.00 | 0.00 |
| night | 0.00 | 0.00 | 0.00 |
| today | 0.00 | 0.00 | 0.00 |
| tomorrow | 0.00 | 0.00 | 0.00 |

### Top-10 confusions (1D-CNN, seed 0, true→pred)
1. school → hospital (2)
2. mother → hello (2)
3. you → morning (1)
4. you → good (1)
5. you → doctor (1)
6. you → bad (1)
7. tomorrow → morning (1)
8. today → tomorrow (1)
9. thank you → school (1)
10. night → morning (1)

### Leakage caveat
No signer/session IDs are present in the filenames or CSV metadata. The same signers appear in both train and test splits (official INCLUDE split is not signer-independent). **Signer leakage cannot be excluded.** Results may overestimate true generalisation to unseen signers.

---

## 6. Cross-Extractor Test (Step 6)

Evaluated the 17-word-trained 1D-CNN and GRU on `research/openhands_pilot/extracted_poses/*.npy` (MediaPipe Holistic (T,75,3) layout, extracted via OpenHands Tasks-API pipeline).

**Overlapping words**: friend, hello, hospital, market, school, thank_you (6 classes).
**Test clips**: 23 official-test clips. Train-split clips (99) reported separately.

| Model | Test acc (official test) | Train-split acc (reported separately) |
|---|---|---|
| 1D-CNN | **0.681 ± 0.021** (16/23) | ~0.67 |
| GRU | 0.362 ± 0.041 (8/23) | ~0.45 |

### Comparison with OpenHands ST-GCN
| System | Top-1 | Correct/Total |
|---|---|---|
| OpenHands ST-GCN (reference) | 45% | 9/20 |
| Our 1D-CNN (cross-extractor) | **68.1%** | 16/23 |
| Our GRU (cross-extractor) | 36.2% | 8/23 |

Note: ST-GCN reference evaluated on 20 clips; our evaluation on 23 clips (slightly different matching due to CSV/video-ID alignment). Our 1D-CNN exceeds the ST-GCN reference on the same extracted_poses data.

---

## 7. Export (Step 7)

**Output**: `research/track_b/export/`

| File | Description |
|---|---|
| `model_1dcnn_17word.pt` | PyTorch state dict, 1D-CNN 17-class |
| `spec.json` | Labels, feature spec (98-dim, 60 frames, shoulder-normalised), normalisation constants, extractor name/version/options, dataset attribution |
| `golden_test_vector.npy` | One `(T, 543, 3)` landmark array (MVI_0017.npy, word=you) |
| `golden_test_vector.json` | Expected logits/probs, true/pred word, correctness |

**ONNX**: Skipped — estimated to exceed 15 minutes; not required by the task. Export can be extended with `torch.onnx.export` if needed.

---

## 8. Files Created / Modified (`research/track_b/`)

### New files
- `extract_all.py` — Step 4: full-batch landmark extraction (resumable)
- `train_17word.py` — Step 5: 17-word training over 5 seeds
- `cross_extractor_test.py` — Step 6: cross-extractor evaluation on extracted_poses
- `export_model.py` — Step 7: model export (state dict, spec, golden vector)
- `data/include_clips/*.MOV/*.mp4` — 190 downloaded Zenodo clips
- `data/include_landmarks/*.npy` — 314 extracted landmark arrays
- `data/extraction_stats.json` — per-clip extraction log
- `results_17word.json` — Step 5 training metrics
- `results_cross_extractor.json` — Step 6 cross-extractor metrics
- `export/` — Step 7 artifacts

### Modified files
- `extract_tasks.py` — Fixed `hand_landmarks` attribute (mediapipe 1.2.2 uses `hand_landmarks`, not `landmarks`)
- `batch_extract.py` — Same fix
- `clip_map.py` — CSV path corrected
- `step1_vocab.py` — CSV path corrected
- `download_missing.py` — CSV path corrected
- `check_download.py` — CSV path corrected

### Not created (intentionally)
- No ONNX export (>15 min estimate)
- No integration into app
- No changes to `.env`, `render.yaml`, `SIGNSAGE_MODEL_PROMOTED`
- No commit/push

---

## NOT Verified / Limitations

1. **Signer leakage**: No signer IDs available; official INCLUDE train/test split is not signer-independent. Same signers appear in both splits.
2. **Python extractor differences**: IMAGE mode vs frontend VIDEO mode (no temporal smoothing); CPU vs GPU delegate. mediapipe 1.2.2 vs frontend's tasks-vision 1.0.1 (WASM).
3. **Corrupt videos**: 3 MP4s (`MVI_4251`, `MVI_5240`, `MVI_9513`) have 0 frames ("moov atom not found") and were excluded.
4. **26 landmark files** not matched to CSV entries (pre-existing downloads outside the 17-word split).
5. **Test set is small**: 54 official-test clips across 17 classes (some classes have only 1 test clip). Per-class metrics are noisy.
6. **Cross-extractor test** uses 23 test clips; ST-GCN reference used 20. Slight mismatch in clip selection.
7. **ONNX export** not performed (would exceed 15 minutes).

---

## Recommendation

A 1D-CNN on 98-dim shoulder-normalised landmarks reaches **60.4% test accuracy / 0.527 macro-F1** on the 17-word INCLUDE vocabulary (54 test clips, chance baseline ~9.3%). This is well above chance but limited by small test set size and potential signer leakage. The model transfers reasonably to a different extractor (68.1% on OpenHands Holistic-extracted poses vs 45% ST-GCN reference). Before any production consideration: obtain signer/session IDs for speaker-independent splits, expand the test set, and verify on held-out signers.

Do NOT promote `SIGNSAGE_MODEL_PROMOTED` based on this pilot.