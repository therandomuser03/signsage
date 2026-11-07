# Track B — REPORT_D: S1 word-classifier evaluation (PRELIMINARY)

**Status label on every number below: `PRELIMINARY, OPTIMISTIC (same session, same position/lighting)` — 5-fold take CV on session S1 only. NOT a final result. S2 does not exist yet.**

Scope: `research/track_b/` only. No existing files edited, no `.env`/`render.yaml`/`SIGNSAGE_MODEL_PROMOTED` touched, no commit/push, nothing exported or integrated into the app. Offline only (CPU, vendored MediaPipe assets already on disk).

## 1. Data — step 1 (convert + validate)

Command (run from `research/track_b/`):

```
python recorder/convert_and_validate.py recordings --out data/s1_converted
```

- 347 recorder JSON clips converted → `data/s1_converted/manifest.csv` + one `.npy` per clip (all shapes `(T,543,3)` valid).
- Features check: `features.extract_features(..., layout='signsage')` → `(60, 98)` on every clip (98 dims × 60 frames, shoulder-center/width normalized, edge-pad/subsample to 60 frames).
- Session: S1 only. Takes per label: 12 each except `please` = 11.
- Labels present: 29 (no `transition` label exists in the recordings — none expected per task).

Per-label clip counts and share of clips with hands missing >40% of frames (left OR right hand all-zero in >40% of frames):

| label | clips | left >40% | right >40% | either hand >40% |
|-------|-------|-----------|------------|------------------|
| bad | 12 | 100% | 25% | 100% |
| doctor | 12 | 75% | 58% | 100% |
| father | 12 | 100% | 17% | 100% |
| food | 12 | 92% | 83% | 100% |
| friend | 12 | 58% | 92% | 100% |
| good | 12 | 100% | 17% | 100% |
| hello | 12 | 100% | 92% | 100% |
| help | 12 | 83% | 33% | 100% |
| hospital | 12 | 100% | 92% | 100% |
| how | 12 | 67% | 100% | 100% |
| market | 12 | 100% | 100% | 100% |
| me | 12 | 83% | 58% | 100% |
| medicine | 12 | 92% | 67% | 100% |
| morning | 12 | 100% | 67% | 100% |
| mother | 12 | 100% | 17% | 100% |
| name | 12 | 100% | 50% | 100% |
| night | 12 | 100% | 8% | 100% |
| no | 12 | 42% | 83% | 100% |
| please | 11 | 100% | 45% | 100% |
| rest | 12 | 100% | 17% | 100% |
| school | 12 | 75% | 92% | 100% |
| sorry | 12 | 100% | 50% | 100% |
| thank you | 12 | 100% | 75% | 100% |
| today | 12 | 100% | 75% | 100% |
| tomorrow | 12 | 100% | 25% | 100% |
| water | 12 | 83% | 67% | 100% |
| where | 12 | 83% | 92% | 100% |
| yes | 12 | 100% | 33% | 100% |
| you | 12 | 83% | 67% | 100% |

**Exclusion rule:** labels with fewer than 8 clips are excluded from training.
**Exclusion list: EMPTY** — the smallest label is `please` with 11 clips, so no label is excluded. All 29 labels are used in the FULL model.

Data-quality caveat (reported, not used as an exclusion criterion): **every one of the 347 clips is flagged >40% missing on at least one hand** — these are one-handed recordings where the non-active hand is out of frame (e.g. all 12 `you` takes have the right hand missing in 100% of frames). Clip lengths are 9–58 frames (short ones edge-padded to 60). This is S1-only data; whether it holds for S2 is unverified.

Core vs extra split:
- CORE words (10): hello, thank you, please, yes, no, help, water, you, me, where → 119 clips
- `rest` (no-sign): 12 clips
- CORE model classes: 11 (10 words + rest), 131 clips total
- EXTRA words (18): bad, doctor, father, food, friend, good, hospital, how, market, medicine, morning, mother, name, night, school, sorry, today, tomorrow → 216 clips
- FULL model classes: 29 (all labels + rest), 347 clips total

## 2. Setup — step 2

- Features: `research/track_b/features.py`, 98-dim (LH 42 + RH 42 + 7 pose joints 14), 60 frames, layout `signsage`.
- Models: 1D-CNN (primary) and tiny GRU (comparison only), same code family as `train_and_eval.py` (60 epochs, Adam 1e-3, batch 16, light frame-level augmentation). No hyperparameter tuning.
- Evaluation: 5-fold cross-validation **across takes** — each label's takes are split into 5 folds, fold k held out per label (no take appears in both train and test within a fold). 5 seeds; reported as mean ± std over seeds. All on CPU.

## 3. Results — (A) CORE model, 10 core words + rest (11 classes)

```
1D-CNN:  accuracy = 0.656 +/- 0.007   macro-F1 = 0.614 +/- 0.010   (5 seeds, 5-fold take CV)
GRU    :  accuracy = 0.645 +/- 0.028   macro-F1 = 0.614 +/- 0.027   (comparison)
```
Per-seed 1D-CNN CV accuracy: [0.650, 0.665, 0.665, 0.653, 0.647]

Pooled CV class table (majority vote over 5 seeds, n = 131 clips; overall acc 0.664, macro-F1 0.644):

| class | precision | recall | F1 | support | note |
|-------|-----------|--------|-----|---------|------|
| you | 0.00 | 0.00 | 0.00 | 12 | candidate drop |
| water | 0.29 | 0.17 | 0.21 | 12 | candidate drop |
| thank you | 0.56 | 0.42 | 0.48 | 12 | candidate drop |
| yes | 0.45 | 0.83 | 0.59 | 12 | candidate drop |
| no | 0.78 | 0.58 | 0.67 | 12 | candidate drop |
| please | 0.60 | 0.82 | 0.69 | 11 | candidate drop |
| me | 0.65 | 0.92 | 0.76 | 12 | candidate drop |
| help | 1.00 | 0.75 | 0.86 | 12 | |
| where | 0.91 | 0.83 | 0.87 | 12 | |
| hello | 0.92 | 1.00 | 0.96 | 12 | |
| rest | 1.00 | 1.00 | 1.00 | 12 | |

Baselines: uniform chance = 1/11 ≈ 0.091; majority-class = 0.092.

Top-10 confusion pairs (count, true → predicted):
1. you → yes (5)
2. water → you (5)
3. thank you → please (4)
4. you → water (3)
5. water → yes (3)
6. you → no (2)
7. water → me (2)
8. no → water (2)
9. no → yes (2)
10. please → thank you (2)

## 4. Results — (B) FULL model, all 28 non-rest labels + rest (29 classes)

```
1D-CNN:  accuracy = 0.733 +/- 0.015   macro-F1 = 0.711 +/- 0.018   (5 seeds, 5-fold take CV)
GRU    :  accuracy = 0.656 +/- 0.013   macro-F1 = 0.625 +/- 0.014   (comparison)
```
Per-seed 1D-CNN CV accuracy: [0.743, 0.712, 0.720, 0.750, 0.741]

Pooled CV class table (majority vote over 5 seeds, n = 347 clips; overall acc 0.752, macro-F1 0.743), sorted by F1 ascending so weak words are visible:

| class | precision | recall | F1 | support | note |
|-------|-----------|--------|-----|---------|------|
| you | 0.00 | 0.00 | 0.00 | 12 | |
| water | 0.20 | 0.08 | 0.12 | 12 | |
| thank you | 0.50 | 0.33 | 0.40 | 12 | |
| how | 0.40 | 0.50 | 0.44 | 12 | |
| yes | 0.38 | 0.67 | 0.48 | 12 | |
| me | 0.57 | 0.67 | 0.62 | 12 | |
| no | 0.70 | 0.58 | 0.64 | 12 | |
| mother | 0.64 | 0.75 | 0.69 | 12 | |
| market | 0.89 | 0.67 | 0.76 | 12 | |
| friend | 0.82 | 0.75 | 0.78 | 12 | |
| please | 0.75 | 0.82 | 0.78 | 11 | |
| sorry | 0.69 | 0.92 | 0.79 | 12 | |
| father | 0.77 | 0.83 | 0.80 | 12 | |
| food | 0.77 | 0.83 | 0.80 | 12 | |
| good | 0.77 | 0.83 | 0.80 | 12 | |
| help | 1.00 | 0.67 | 0.80 | 12 | |
| school | 0.77 | 0.83 | 0.80 | 12 | |
| today | 0.90 | 0.75 | 0.82 | 12 | |
| hospital | 0.83 | 0.83 | 0.83 | 12 | |
| name | 0.83 | 0.83 | 0.83 | 12 | |
| bad | 0.80 | 1.00 | 0.89 | 12 | |
| where | 1.00 | 0.83 | 0.91 | 12 | |
| medicine | 0.92 | 0.92 | 0.92 | 12 | |
| morning | 1.00 | 0.92 | 0.96 | 12 | |
| doctor | 0.92 | 1.00 | 0.96 | 12 | |
| hello | 0.92 | 1.00 | 0.96 | 12 | |
| tomorrow | 0.92 | 1.00 | 0.96 | 12 | |
| night | 1.00 | 1.00 | 1.00 | 12 | |
| rest | 1.00 | 1.00 | 1.00 | 12 | |

Baselines: uniform chance = 1/29 ≈ 0.034; majority-class = 0.035.

Top-10 confusion pairs (count, true → predicted):
1. you → yes (4)
2. how → yes (4)
3. water → you (3)
4. water → how (3)
5. thank you → please (3)
6. mother → father (3)
7. you → water (2)
8. you → no (2)
9. you → how (2)
10. yes → you (2)

Observations: `rest` is perfectly separated in both models (F1 = 1.00). The FULL model's higher accuracy vs CORE is largely driven by `rest` + highly separable extra words; the 10 core words are the weakest classes in FULL too (you 0.00, water 0.12, thank you 0.40, yes 0.48). GRU tracks 1D-CNN closely (comparison only, not preferred).

## 5. Unknown handling — step 5 (no `transition` class)

Method: out-of-fold predictions, seed 0, CORE 1D-CNN (each of the 131 clips predicted exactly once, from its held-out fold). A clip is "accepted" as a word when max softmax ≥ threshold; otherwise it is rejected as unknown. `rest` wrongly classified as a word = accepted rest clip predicted ≠ `rest`.

| threshold | accepted | coverage | accuracy | rest clips | rest → word | rest→word rate |
|-----------|----------|----------|----------|------------|-------------|----------------|
| argmax (no reject) | 131 | 100% | 0.649 | 12 | 0 | 0.0% |
| 0.5 | 94 | 71.8% | 0.745 | 12 | 0 | 0.0% |
| 0.7 | 75 | 57.3% | 0.867 | 12 | 0 | 0.0% |
| 0.9 | 55 | 42.0% | 0.945 | 12 | 0 | 0.0% |

- Coverage/accuracy trade-off is as expected: higher threshold → higher accuracy, lower coverage.
- `rest` clips are **never** wrongly classified as a word in this S1 CV (0/12 at every threshold) — `rest` is well separated from the word classes in-session.
- Caveat: threshold analysis is on a single seed (seed 0) OOF set, not averaged over 5 seeds.

## 6. Candidate drops for the core model — step 6

Words with CV F1 < 0.80 (CORE 1D-CNN, pooled): **you (0.00), water (0.21), thank you (0.48), yes (0.59), no (0.67), please (0.69), me (0.76)**.
Not drops: help (0.86), where (0.87), hello (0.96). `rest` is not a word and is not a drop candidate.

**No vocabulary is finalised.** This list is only input to a later S2-based decision. Notably `you` (F1 0.00) is likely a data-quality artifact: all 12 `you` takes have the right hand missing in every frame (right hand out of frame), and `you` is heavily confused with `yes`/`no`/`water`. Worth re-recording with both hands in frame before any drop decision.

## 7. What was NOT done / could NOT be verified

- **No final or export model trained** (step 7 respected). `export_model.py` was not run; nothing written to `models/` or `verified_checkpoint/`.
- **S2 does not exist** — no recordings for session S2 yet, so no cross-session generalization can be measured. All numbers above are same-session (S1) and therefore optimistic.
- **S2 protocol (step 8), when S2 clips exist:** train on S1 ONLY, test on S2 ONLY; S2 never used for training, model selection, or threshold tuning. That S2 result becomes the primary result; these S1 numbers are superseded.
- No `transition` label exists in the recordings; none was trained or evaluated.
- No held-out signer / position / lighting variation tested within S1 (single session).
- Threshold analysis on seed 0 only; not averaged over the 5 seeds.
- No hyperparameter tuning performed (epochs/lr/batch/augmentation inherited from `train_and_eval.py` conventions).
- Hardware mismatch: the recorder used MediaPipe GPU delegate; this evaluation ran on CPU. Feature shapes and ranges match `features.py` self_test, but GPU-vs-CPU landmark parity was not re-measured here.
- Edge-case behaviour when pose/shoulders are entirely missing (normalization divides by shoulder width with 1e-6 floor) was not separately tested.

## 8. Files created (this session)

- `research/track_b/eval_track_b.py` — training/evaluation script (new)
- `research/track_b/eval_track_b.log` — full run log (new)
- `research/track_b/results_track_b.json` — raw results dump (new)
- `research/track_b/data/s1_converted/` — manifest.csv + per-clip `.npy` (gitignored via `data/` and `*.npy`; not committed)
- `research/track_b/REPORT_D.md` — this file (new)

No existing files were edited. Nothing committed or pushed.

Command to confirm tree state:

```
git status --short
```

(see end of this report)

```
$ git status --short
?? research/track_b/eval_track_b.log
?? research/track_b/eval_track_b.py
?? research/track_b/recordings/
?? research/track_b/results_track_b.json
```

(`research/track_b/recordings/` was already untracked input data before this session; `data/s1_converted/` and `*.npy` are gitignored.)

**Bottom line:** S1 CV gives 1D-CNN CORE ≈ 0.66 accuracy / 0.61 macro-F1 and FULL ≈ 0.73 / 0.71, but every number is same-session S1 and labelled preliminary/optimistic. `rest` separates perfectly; `you`, `water`, `thank you`, `yes`, `no`, `please`, `me` are below F1 0.80 in the core model and are candidate drops only — no vocabulary decision is made until S2 testing.
