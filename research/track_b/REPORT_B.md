# Track B Feasibility Report — Small Landmark-Based ISL Word Classifier

**Status**: OFFLINE research only. Nothing integrated into the app. No existing files, `.env`, `render.yaml`, or `SIGNSAGE_MODEL_PROMOTED` touched. No commit/push.
**Scope**: Standalone under `research/track_b/`. All downloads kept on `/mnt/D`. The 56.8 GB INCLUDE video set was NOT downloaded.

---

## 1. Data source & vocabulary (Step 1)

### Source
`train_include.csv` / `test_include.csv` from `include_metadata.zip` (OpenHands repo metadata, MIT / CC-BY-4.0 INCLUDE terms).
Canonical copy: `research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/`
Format: `Category,Word,Video,FilePath`. Word column carries a `NN.` rank prefix (e.g. `48. Hello`).

### Target concept -> INCLUDE label matching (clip counts)
17 of 36 target concepts matched an INCLUDE label; 19 had no match.

| Concept | INCLUDE label | train | test | total |
|---------|---------------|-------|------|-------|
| hello | 48. Hello | 17 | 4 | 21 |
| thank you | 55. Thank you | 17 | 4 | 21 |
| hospital | 30. Hospital | 16 | 4 | 20 |
| school | 24. School | 16 | 4 | 20 |
| friend | 81. Friend | 16 | 4 | 20 |
| market | 32. Market | 18 | 3 | 21 |
| you | 41. you | 17 | 4 | 21 |
| doctor | 87. Doctor | 11 | 3 | 14 |
| medicine | 3. Medicine | 11 | 3 | 14 |
| good | 94. good | 17 | 4 | 21 |
| bad | 95. bad | 17 | 4 | 21 |
| morning | 82. Morning | 13 | 1 | 14 |
| night | 85. Night | 14 | 1 | 15 |
| today | 73. Today | 13 | 1 | 14 |
| tomorrow | 74. Tomorrow | 13 | 1 | 14 |
| mother | 60. Mother | 16 | 4 | 20 |
| father | 61. Father | 16 | 4 | 20 |

**No INCLUDE label**: water, i/me, help, please, sorry, yes, no, home, food, eat, drink, go, come, name, what, where, how, need, want, understand.

### 40 INCLUDE labels with most clips (train+test)
Distribution: 25 clips x1 (`4. Bird`), 23 x1 (`23. Court`), 22 x7 (`84. small little`, `79. short`, `36. Location`, `21. Train Station`, `27. Park`, `28. Store or Shop`, `35. Bank`), 21 x31 (all remaining top-40). Full per-label table printed by `step1_vocab.py` at run time.

### Avatar vocabulary (`backend/app/data/sign_vocabulary.json`, 14 IDs) vs INCLUDE
| Avatar ID | In INCLUDE? | INCLUDE label | total clips |
|-----------|-------------|---------------|-------------|
| HELLO | yes | 48. Hello | 21 |
| THANK_YOU | yes | 55. Thank you | 21 |
| YOU | yes | 41. you | 21 |
| SORRY | no | — | 0 |
| PLEASE | no | — | 0 |
| YES | no | — | 0 |
| NO | no | — | 0 |
| HELP | no | — | 0 |
| WATER | no | — | 0 |
| FOOD | no | — | 0 |
| NAME | no | — | 0 |
| ME | no | — | 0 |
| WHERE | no | — | 0 |
| HOW | no | — | 0 |

Only **3 of 14** avatar gloss IDs exist in INCLUDE. The avatar set is not covered by INCLUDE.

---

## 2. Pre-extracted landmark datasets (Step 2)

### (a) Kaggle "indian-sign-language-mediapipe-holistic-landmarks"
- **Access**: Kaggle dataset pages return a Google reCAPTCHA challenge; downloads require a Kaggle account + API token (`kaggle.json`). Not bypassable offline.
- **Login required**: YES. **Not downloaded.**
- Size / license / layout / frame-channel / extractor / missing-frames / signer IDs: **unverifiable without login.** Stop at this step per rules; exact manual step: create Kaggle account, generate API token, `kaggle datasets download -d <owner>/indian-sign-language-mediapipe-holistic-landmarks` (owner/dataset-slug not confirmed without login).

### (b) IEEE DataPort "Silent Voice: MediaPipe Holistic Landmark Tensors for 261 Indian Sign Language Words (derived from INCLUDE)"
- **DOI**: 10.21227/vfyn-1246
- **URL**: https://ieee-dataport.org/documents/silent-voice-mediapipe-holistic-landmark-tensors-261-indian-sign-language-words-derived
- **Access**: IEEE DataPort landing page reachable (HTTP 200), but download requires a free IEEE account (**login required**).
- **Login required**: YES. **Not downloaded** (per rule: download only if no login and size < 3 GB).
- **Size**: ~94.99 MB (landmarks) vs 54 GB video (also reported "231 MB" / "5 Gb" in page text — landmark archive is well under 3 GB either way).
- **License**: Creative Commons Attribution 4.0 (CC-BY-4.0).
- **Content**: 4,276 pose+hand landmark sequences from INCLUDE, 261 isolated word-level signs.
- **Extractor**: MediaPipe Holistic.
- **Layout**: float32 tensor shape **(60, 225)** = 21 left-hand + 21 right-hand + 33 body-pose landmarks, each (x,y,z). Columns 0-62 left hand, 63-125 right hand, 126-224 pose.
- **Frames**: resampled to exactly 60 frames per clip.
- **Normalization**: hand coords wrist-origin + shoulder-width normalized; pose coords in normalized image space.
- **Missing frames**: stored as zeros (undetected landmarks = zero).
- **File naming**: `__`.npy.
- **Signer/session IDs**: not present in the page metadata; unverifiable without download.

**Conclusion Step 2**: neither pre-extracted landmark dataset can be downloaded offline (both require login). Only locally available landmark data is `research/openhands_pilot/extracted_poses/` (122 clips, see Step 4/5).

---

## 3. Feature specification (Step 3)

### Design
Standardized per-frame feature = **98 dims, x,y only**:
- Left hand (21 x,y) = 42
- Right hand (21 x,y) = 42
- 7 upper-body pose joints (x,y) = 14: nose, left eye, right eye, left shoulder, right shoulder, left elbow, right elbow
Total = 98 dims / frame.

### Normalization
Per-frame **shoulder-center + shoulder-width**:
- center = (left_shoulder + right_shoulder) / 2
- width = ||left_shoulder − right_shoulder|| (+eps)
- all coords = (coord − center) / width

### Temporal
Resample to **60 frames**: uniform subsample if T ≥ 60, edge-pad (repeat last frame) if T < 60.

### Supported input layouts (code: `research/track_b/features.py`)
- MediaPipe Holistic **(T,75,3)**: pose 0-32, left hand 33-53, right hand 54-74
- SignSage **(T,543,3)**: pose 0-32, face 33-500, left hand 501-521, right hand 522-542
Same pose-joint indices `[0,2,5,11,12,13,14]` in both; hand slices differ (33/54 vs 501/522).

### Unit test
`features.py` `self_test()` verifies index mapping + normalization + resample shapes (shape/index tests only, **not accuracy evidence**): passes. Confirms lhand wrist → col 0, rhand wrist → col 42, lshoulder → col 90, rshoulder → col 92 in both layouts; short clips edge-pad, long clips subsample.

### Class selection for training
Criteria: INCLUDE label with **>= 10 clips** (train+test). 17 concepts qualify (listed in section 1). **Only 6 have locally-extracted landmark data** (`extracted_poses/`): friend, hello, hospital, market, school, thank_you (20-21 clips each). The other 11 qualifying classes (doctor, medicine, you, good, bad, morning, night, today, tomorrow, mother, father) have no local landmark data — extracting them would require the 56.8 GB INCLUDE video set (excluded) or the login-gated datasets (not downloaded). Training therefore runs on the 6 available classes.

---

## 4. Training results on CPU (Step 4)

### Setup
- Data: `research/openhands_pilot/extracted_poses/*.npy` = 122 clips, 6 words, MediaPipe Holistic (T,75,3), variable T (61-86 frames).
- Features: `extract_features(..., layout="holistic")` -> (60, 98).
- Split: class-stratified by clip, fixed seed 42 -> **98 train / 24 test**.
- Models: Tiny 1D-CNN (Conv1d 98→32→64, GAP, Linear→6) and Tiny GRU (GRU 98→32, Linear→6).
- Augmentation (train only): random scale (0.95-1.05), shift (±0.03), rotation (±8°), time warp (0.9-1.1×).
- Optimizer: Adam lr=1e-3, 60 epochs, CrossEntropyLoss, CPU (torch 2.14.1+cpu).
- No signer IDs in filenames → **signer leakage cannot be excluded**.

### Results (measured)
| Model | Train acc | Train macro-F1 | Test acc | Test macro-F1 | Trivial baseline | Latency/clip | Model size |
|-------|-----------|----------------|----------|---------------|------------------|--------------|------------|
| 1D-CNN | 0.908 | 0.906 | **0.917** | **0.907** | 0.167 | 0.23 ms | 0.067 MB |
| GRU | 0.908 | 0.906 | **0.792** | **0.785** | 0.167 | 1.88 ms | 0.055 MB |

Baseline = majority-class accuracy = 1/6 ≈ 0.167 (classes balanced).

### Per-class F1 (test)

1D-CNN:
- friend 1.00, hello 1.00, hospital 1.00, market 0.89, school 0.89, thank_you 0.67

GRU:
- friend 0.67, hello 1.00, hospital 0.89, market 0.67, school 0.60, thank_you 0.89

### Top confusions (test, true→pred)
1D-CNN: thank_you→school (1), thank_you→market (1)
GRU: friend→school (2), school→hospital (1), market→thank_you (1), market→school (1)

**Caveat**: test set is only 24 clips (~4/class), so per-class F1 and confusions are noisy; treat as directional only.

---

## 5. Cross-extractor check (Step 5)

The trained models were run on `research/openhands_pilot/extracted_poses/*.npy` (the holdout test set from section 4 — this IS the extracted_poses data, so it is a within-extractor holdout, approximating robustness to the raw MediaPipe Holistic extraction format).

| Model | Accuracy on extracted_poses holdout |
|-------|-------------------------------------|
| 1D-CNN | 0.917 (22/24) |
| GRU | 0.792 (19/24) |

**Could NOT be verified**: a true cross-extractor check against the IEEE DataPort dataset (different extractor/layout/normalization) — not downloaded because it requires a free IEEE account login. The local `extracted_poses` shares the MediaPipe Holistic extractor family with training data, so this check does **not** establish robustness to a genuinely different extractor.

---

## 6. What could NOT be verified
- Full 263/261-class training & evaluation: only 6 classes have local landmark data; the 56.8 GB INCLUDE video set was not downloaded; the IEEE/Kaggle landmark datasets require login.
- 11 qualifying INCLUDE classes (doctor, medicine, you, good, bad, morning, night, today, tomorrow, mother, father) have no local landmark data.
- True cross-extractor robustness (IEEE DataPort) — login-gated.
- Signer/session IDs — absent from local filenames and unverifiable without the full dataset; signer leakage cannot be excluded.
- Avatar vocabulary coverage beyond the 3 matching IDs — 11 of 14 avatar glosses have no INCLUDE equivalent.

---

## 7. Recommendation
A tiny landmark-based ISL word classifier is **technically feasible** on CPU: a small 1D-CNN reaches ~92% test accuracy / ~0.91 macro-F1 on the 6 locally available classes with sub-millisecond per-clip latency and a ~67 KB model, well above the 16.7% trivial baseline. However, this is a **6-class feasibility demo on 122 clips**, not a production-ready system. Before any integration:
1. Obtain the IEEE DataPort landmark set (free IEEE account, CC-BY-4.0, <3 GB) to expand to ~261 classes and enable a true cross-extractor check.
2. Confirm signer/session IDs exist to rule out signer leakage (use speaker-independent splits).
3. Re-evaluate on the full vocabulary; the 6-class result is directional only.

Do NOT promote `SIGNSAGE_MODEL_PROMOTED` based on this pilot.

---

## Files created (`research/track_b/`)
- `.gitignore` (venv, data, checkpoints, npy/npz, ckpt, pth, zip, .env)
- `step1_vocab.py` (Step 1 analysis)
- `features.py` (Step 3 feature code + self-test)
- `train_and_eval.py` (Steps 4-5 training/eval)
- `results.json` (Step 4 metrics)
- `REPORT_B.md` (this report)
