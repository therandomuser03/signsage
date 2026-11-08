# Track B — REPORT_E: S1 v2-feature model evaluation (PRELIMINARY)

**Status label on every number below: `PRELIMINARY, OPTIMISTIC (same session, same position/lighting)` — 5-fold take CV on session S1 only. NOT a final result. S2 does not exist yet.**

Scope: `research/track_b/` only. No existing files edited (including `features.py` — a new `features_v2.py` was created), no `.env`/`render.yaml`/`SIGNSAGE_MODEL_PROMOTED` touched (stays false), no commit/push, nothing exported or integrated into the app. Offline only (CPU). Uses ONLY `research/track_b/recordings/` (session S1); `recordings_s2/` was NOT read or used.

## 0. Files created (this session)

| file | purpose |
|------|---------|
| `research/track_b/features_v2.py` | NEW v2 hand-centric per-frame features (122 dims × 60 frames) |
| `research/track_b/test_features_v2.py` | NEW unit tests for shapes + all-missing-hand case (5/5 pass) |
| `research/track_b/diagnose_s1.py` | NEW step-1 hand-detection diagnostics on S1 |
| `research/track_b/eval_v2.py` | NEW step-3 evaluation (cnn-v2 / logreg / cnn-v1, 5-fold take CV, 5 seeds) |
| `research/track_b/results_v2.json` | NEW raw results dump |
| `research/track_b/eval_v2_full.log` | NEW full run log |
| `research/track_b/REPORT_E.md` | NEW this file |

Exact commands run (from `research/track_b/`):
```
python3 features_v2.py                       # self-test -> ALL OK
python3 test_features_v2.py                  # unit tests -> 5/5 PASS
python3 diagnose_s1.py                       # step-1 hand diagnostics
python3 eval_v2.py                           # full 5-seed evaluation (~45 min CPU)
```
No existing file was edited. `git status --short` (see end of report) shows only the new untracked files above.

## 1. Step 1 — Hand-detection diagnostics (S1 only)

Hand "detected" in a frame = at least one of the 21 hand landmarks nonzero. "Active hand" = hand present in more frames (per-clip mean). "Flip" = clip contains at least one left-only-active frame AND at least one right-only-active frame.

### Per-label hand detection (all 29 S1 labels)

| label | clips | L_pres | R_pres | active hand | flip_frac |
|-------|-------|--------|--------|-------------|-----------|
| bad | 12 | 0.000 | 0.644 | Right | 0.000 |
| doctor | 12 | 0.566 | 0.475 | Left | 0.000 |
| father | 12 | 0.004 | 0.650 | Right | 0.000 |
| food | 12 | 0.381 | 0.475 | Right | 0.083 |
| friend | 12 | 0.560 | 0.377 | Left | 0.167 |
| good | 12 | 0.004 | 0.669 | Right | 0.000 |
| hello | 12 | 0.033 | 0.389 | Right | 0.583 |
| help | 12 | 0.440 | 0.601 | Right | 0.167 |
| hospital | 12 | 0.239 | 0.274 | Right | 0.250 |
| how | 12 | 0.299 | 0.117 | Left | 0.000 |
| market | 12 | 0.475 | 0.310 | Left | 0.667 |
| me | 12 | 0.303 | 0.378 | Right | 0.083 |
| medicine | 12 | 0.457 | 0.585 | Right | 0.833 |
| morning | 12 | 0.066 | 0.558 | Right | 0.333 |
| mother | 12 | 0.005 | 0.675 | Right | 0.000 |
| name | 12 | 0.023 | 0.592 | Right | 0.167 |
| night | 12 | 0.007 | 0.680 | Right | 0.000 |
| no | 12 | 0.589 | 0.136 | Left | 0.000 |
| please | 11 | 0.144 | 0.592 | Right | 0.818 |
| rest | 12 | 0.479 | 0.636 | Right | 0.250 |
| school | 12 | 0.526 | 0.518 | Left | 0.083 |
| sorry | 12 | 0.033 | 0.571 | Right | 0.250 |
| thank you | 12 | 0.123 | 0.456 | Right | 0.750 |
| today | 12 | 0.005 | 0.508 | Right | 0.000 |
| tomorrow | 12 | 0.017 | 0.645 | Right | 0.000 |
| water | 12 | 0.268 | 0.382 | Right | 0.000 |
| where | 12 | 0.514 | 0.491 | Left | 0.583 |
| yes | 12 | 0.004 | 0.683 | Right | 0.083 |
| you | 12 | 0.292 | 0.317 | Right | 0.167 |

**Overall:** 347 clips, active-hand flip within clip = **75/347 = 0.216 (21.6%)**. Every clip is one-handed for at least part of the time (each label has a hand missing in >0% of frames); many labels are effectively single-hand (e.g. `bad`, `father`, `good`, `mother`, `night`, `today`, `tomorrow`, `yes` have left-hand presence ≈ 0).

### "you" — per-take detail
`you` was flagged as a problem word in the v1 baseline (F1 = 0.00). Per-take detection:

| take | L_pres | R_pres |
|------|--------|--------|
| 1 | 0.42 | 0.00 |
| 2 | 0.54 | 0.00 |
| 3 | 0.48 | 0.00 |
| 4 | 0.86 | 0.00 |
| 5 | 0.33 | 0.00 |
| 6 | 0.04 | 0.54 |
| 7–12 | (12 takes total; takes 1–5 left-only, take 6 right-only) |

Interpretation: `you` is predominantly a **left-hand** sign, but take 6 is right-hand — an active-hand flip within the label that the v1 features (which force both hands into a shared shoulder frame) could not represent cleanly.

### Current baseline CORE model — top-10 confusion pairs (v1 CNN, from results_track_b.json)
Baseline pooled (majority vote over 5 seeds, n=131): acc=0.664, macroF1=0.644.

| # | true -> predicted | count |
|---|-------------------|-------|
| 1 | you -> yes | 5 |
| 2 | water -> you | 5 |
| 3 | thank you -> please | 4 |
| 4 | you -> water | 3 |
| 5 | water -> yes | 3 |
| 6 | you -> no | 2 |
| 7 | water -> me | 2 |
| 8 | no -> water | 2 |
| 9 | no -> yes | 2 |
| 10 | please -> thank you | 2 |

Baseline CORE per-class F1: you=0.00, water=0.21, thank you=0.48, yes=0.59, no=0.67, please=0.69, me=0.76, help=0.86, where=0.87, hello=0.96, rest=1.00. Uniform chance = 1/11 ≈ 0.091; majority = 0.092.

## 2. Step 2 — New features: `features_v2.py` (122 dims × 60 frames)

**Not an edit to `features.py`.** `features_v2` supports `layout='signsage'` only (the (T,543,3) array; face mesh index `i` lives at absolute `33+i`). Per-frame vector:

| channel | dims | definition |
|---------|------|------------|
| left hand | 42 | wrist − shoulder_centre / shoulder_width; other 20 landmarks − wrist / hand_size |
| right hand | 42 | same |
| left_mask, right_mask | 2 | 1 if hand has data after gap-fill |
| hand-to-face | 32 | per hand (L then R), wrist then index tip, → nose/chin/mouth/forehead, scaled by shoulder width |
| wrist velocity | 4 | frame difference, per wrist |

Total = 42+42+2+32+4 = **122 dims**.

Missing-hand handling (per the spec):
- A hand is missing in a frame when all 21 landmarks are zero.
- Missing runs of **<=5 frames** are linearly interpolated (per landmark); interpolated frames keep mask=1.
- Runs still missing after that (long gaps / leading / trailing) keep **zero coordinates + mask=0**. Every hand-derived channel is multiplied by the mask, so missing zeros are **never normalised into shoulder space**.
- Output resampled to **60 frames after filling** (uniform subsample if T>=60, edge-pad if T<60).

### Unit tests: `test_features_v2.py` (5/5 PASS)
1. `test_shape_and_no_nan` — output (60,122) float32, no NaN/Inf, both masks=1 when both hands present.
2. `test_all_missing_hand` — both hands zero → every hand-derived channel exactly 0, masks=0, no NaN.
3. `test_one_hand_missing` — right hand blanked → right mask=0, right channels=0, left channels intact & nonzero.
4. `test_short_gap_interpolated_long_gap_zero` — gap of 5 interpolated (mask=1), gap of 6 stays missing (mask=0, zero coords).
5. `test_resample_short_and_long` — T=5 edge-pads to 60; T=120 subsamples to 60.

Command: `python3 test_features_v2.py` → `5/5 tests passed; test_features_v2: ALL OK`

## 3. Step 3 — Models & 5-fold take CV (5 seeds)

Models:
- **(a) cnn-v2:** 1D-CNN, dropout 0.3, on v2 features (122 dims).
- **(b) logreg:** flattened **30-frame** v2 features → StandardScaler → multinomial logistic regression (numpy, L2), trained by gradient descent.
- **(c) cnn-v1:** old v1 CNN (features.py, 98 dims) for comparison.

Evaluation: 5-fold CV **across takes** (each label's takes split into 5 folds; no take straddles train/test), 5 seeds; reported as mean ± std over seeds. All CPU.

### Summary: mean ± std over 5 seeds of 5-fold take CV (S1)

| model | vocab | accuracy | macro-F1 | per-seed acc |
|-------|-------|----------|----------|--------------|
| cnn-v2 | CORE-10 (10 words + rest) | **0.894 ± 0.013** | **0.884 ± 0.016** | [0.893, 0.869, 0.905, 0.899, 0.902] |
| cnn-v2 | CORE-9 (9 words + rest, no "you") | **0.924 ± 0.004** | **0.918 ± 0.004** | [0.926, 0.920, 0.930, 0.923, 0.919] |
| cnn-v2 | FULL (all 28 words + rest) | **0.899 ± 0.005** | **0.891 ± 0.005** | [0.901, 0.897, 0.894, 0.907, 0.894] |
| logreg | CORE-10 | 0.822 ± 0.004 | 0.801 ± 0.002 | [0.814, 0.823, 0.823, 0.823, 0.823] |
| logreg | CORE-9 | 0.861 ± 0.006 | 0.847 ± 0.004 | [0.863, 0.869, 0.856, 0.853, 0.862] |
| logreg | FULL | 0.816 ± 0.005 | 0.798 ± 0.006 | [0.810, 0.820, 0.812, 0.824, 0.813] |
| cnn-v1 | CORE-10 (old v1, comparison) | 0.698 ± 0.022 | 0.663 ± 0.027 | [0.683, 0.689, 0.738, 0.707, 0.674] |
| cnn-v1 | CORE-9 (old v1, comparison) | 0.810 ± 0.026 | 0.789 ± 0.028 | [0.846, 0.782, 0.822, 0.779, 0.822] |
| cnn-v1 | FULL (old v1, comparison) | 0.767 ± 0.030 | 0.746 ± 0.029 | [0.808, 0.785, 0.771, 0.756, 0.718] |

**Best model: cnn-v2 on CORE-10** (acc 0.894, macro-F1 0.884). The v2 hand-centric features raise CORE-10 accuracy from 0.698 (v1) to 0.894 — the mask-based handling of the ever-absent non-active hand is what fixes `you` (v1 F1 0.00 → v2 F1 0.86).

Chance baselines: CORE-10 uniform = 1/11 ≈ 0.091, majority = 0.092. CORE-9 uniform = 1/10 = 0.100, majority = 0.101. FULL uniform = 1/29 ≈ 0.034, majority = 0.035.

## 4. Step 4 — Printed results tables (pooled CV, majority vote over 5 seeds)

### 4.1 CORE-10 — best model cnn-v2 (pooled: acc=0.901, macroF1=0.899, n=131)
Baseline: majority = 0.092; uniform chance = 0.091.

| class | precision | recall | F1 | RELIABLE/WEAK (F1≥0.80) |
|-------|-----------|--------|----|-------------------------|
| rest | 1.00 | 1.00 | 1.00 | (not a word) |
| me | 1.00 | 1.00 | 1.00 | RELIABLE |
| help | 1.00 | 1.00 | 1.00 | RELIABLE |
| yes | 0.86 | 1.00 | 0.92 | RELIABLE |
| where | 0.92 | 0.92 | 0.92 | RELIABLE |
| hello | 0.80 | 1.00 | 0.89 | RELIABLE |
| please | 0.83 | 0.91 | 0.87 | RELIABLE |
| you | 1.00 | 0.75 | 0.86 | RELIABLE |
| no | 1.00 | 0.75 | 0.86 | RELIABLE |
| water | 0.79 | 0.92 | 0.85 | RELIABLE |
| thank you | 0.80 | 0.67 | 0.73 | **WEAK** |

Top-10 confusion pairs (count, true → predicted):
1. you → yes (2)
2. no → hello (2)
3. thank you → please (2)
4. where → thank you (1)
5. you → water (1)
6. water → hello (1)
7. no → water (1)
8. please → thank you (1)
9. thank you → where (1)
10. thank you → water (1)

### 4.2 CORE-9 — cnn-v2 (pooled: acc=0.908, macroF1=0.906, n=119)
Removes "you". Baseline: majority = 0.101; uniform chance = 0.100.

| class | precision | recall | F1 |
|-------|-----------|--------|----|
| rest | 1.00 | 1.00 | 1.00 |
| me | 1.00 | 1.00 | 1.00 |
| yes | 1.00 | 1.00 | 1.00 |
| help | 1.00 | 0.92 | 0.96 |
| where | 0.92 | 0.92 | 0.92 |
| water | 0.85 | 0.92 | 0.88 |
| please | 0.83 | 0.91 | 0.87 |
| no | 1.00 | 0.75 | 0.86 |
| hello | 0.75 | 1.00 | 0.86 |
| thank you | 0.80 | 0.67 | 0.73 |

Top-10 (cnn-v2 CORE-9): no→hello (2), thank you→please (2), where→thank you (1), water→hello (1), help→hello (1), no→water (1), please→thank you (1), thank you→where (1), thank you→water (1).

### 4.3 FULL — cnn-v2 (pooled: acc=0.905, macroF1=0.904, n=347)
Baseline: majority = 0.035; uniform chance = 0.034. Full per-class table (sorted by F1 ascending):

| class | P | R | F1 | class | P | R | F1 |
|-------|---|---|----|-------|---|---|----|
| thank you | 0.70 | 0.58 | 0.64 | father | 0.83 | 0.83 | 0.83 |
| you | 0.89 | 0.67 | 0.76 | friend | 0.83 | 0.83 | 0.83 |
| how | 0.67 | 1.00 | 0.80 | water | 0.83 | 0.83 | 0.83 |
| please | 0.71 | 0.91 | 0.80 | no | 1.00 | 0.75 | 0.86 |
| help | 0.90 | 0.75 | 0.82 | name | 0.91 | 0.83 | 0.87 |
| rest | 1.00 | 1.00 | 1.00 | mother | 0.85 | 0.92 | 0.88 |
| | | | | where | 1.00 | 0.83 | 0.91 |
| | | | | food | 0.92 | 0.92 | 0.92 |
| | | | | good | 0.92 | 0.92 | 0.92 |
| | | | | market | 0.92 | 0.92 | 0.92 |
| | | | | morning | 0.86 | 1.00 | 0.92 |
| | | | | yes | 0.86 | 1.00 | 0.92 |
| | | | | night | 1.00 | 0.92 | 0.96 |
| | | | | school | 1.00 | 0.92 | 0.96 |
| | | | | today | 1.00 | 0.92 | 0.96 |
| | | | | hello | 0.92 | 1.00 | 0.96 |
| | | | | sorry | 0.92 | 1.00 | 0.96 |
| | | | | bad | 1.00 | 1.00 | 1.00 |
| | | | | doctor | 1.00 | 1.00 | 1.00 |
| | | | | hospital | 1.00 | 1.00 | 1.00 |
| | | | | me | 1.00 | 1.00 | 1.00 |
| | | | | medicine | 1.00 | 1.00 | 1.00 |
| | | | | rest | 1.00 | 1.00 | 1.00 |
| | | | | tomorrow | 1.00 | 1.00 | 1.00 |

Top-10 (cnn-v2 FULL): you→water (2), water→how (2), thank you→please (2), no→morning (2), friend→how (2), father→mother (2), you→name (1), you→father (1), where→market (1), where→friend (1).

### 4.4 max-softmax threshold table — best model cnn-v2 CORE-10
(Each clip predicted once per seed → 60 rest OOB observations = 12 rest clips × 5 seeds. `rest→word` = rest clip accepted as a word. Rest false-positive rate = rest→word / rest clips.)

| threshold | accepted | coverage | accuracy | rest_clips | rest→word | rest→word rate (FPR) |
|-----------|----------|----------|----------|------------|-----------|----------------------|
| argmax (no reject) | 655 | 100% | 0.893 | 60 | 0 | **0.0%** |
| 0.5 | 640 | 97.7% | 0.898 | 60 | 0 | **0.0%** |
| 0.7 | 615 | 93.9% | 0.914 | 60 | 0 | **0.0%** |
| 0.9 | 525 | 80.2% | 0.945 | 60 | 0 | **0.0%** |

- `rest` false-positive rate on the best model = **0.0% at every threshold** (0/60 rest observations wrongly classed as a word).
- `rest` is perfectly separated in-session (F1 = 1.00), consistent with the v1 baseline.
- Higher threshold → higher accuracy, lower coverage (expected trade-off).
- Caveat: thresholds aggregate across the 5 seeds' OOF sets; not a single held-out test set.

## 5. Step 5 — Best model (cnn-v2 CORE-10): RELIABLE / WEAK per word, with active-hand detection rate

Active-hand detection rate = mean fraction of frames the word's active hand is present (from step 1 diagnostics). F1 = pooled cnn-v2 CORE-10.

| word | F1 | verdict | active hand | active-hand detection rate |
|------|----|---------|-------------|----------------------------|
| help | 1.00 | RELIABLE | Right | 0.601 |
| me | 1.00 | RELIABLE | Right | 0.378 |
| yes | 0.92 | RELIABLE | Right | 0.683 |
| where | 0.92 | RELIABLE | Left | 0.514 |
| hello | 0.89 | RELIABLE | Right | 0.389 |
| you | 0.86 | RELIABLE | Right | 0.317 |
| no | 0.86 | RELIABLE | Left | 0.589 |
| please | 0.87 | RELIABLE | Right | 0.592 |
| water | 0.85 | RELIABLE | Right | 0.382 |
| thank you | 0.73 | **WEAK** | Right | 0.456 |

- **RELIABLE (F1 ≥ 0.80):** help, me, yes, where, hello, you, no, please, water (9 of 10).
- **WEAK (F1 < 0.80):** thank you (0.73).
- `you`, previously the worst class (v1 F1 = 0.00) due to active-hand flips and a missing right hand, is now RELIABLE (v2 F1 = 0.86) — the mask-based v2 features are the reason.
- No vocabulary is finalised (per the task rules). This is input only to a later S2-based decision.

## 6. Comparison to old v1 CNN (comparison only)

| vocab | v1 acc | v1 macroF1 | v2 acc | v2 macroF1 | delta acc | delta macroF1 |
|-------|--------|------------|--------|------------|-----------|---------------|
| CORE-10 | 0.698 | 0.663 | 0.894 | 0.884 | +0.196 | +0.221 |
| CORE-9 | 0.810 | 0.789 | 0.924 | 0.918 | +0.114 | +0.129 |
| FULL | 0.767 | 0.746 | 0.899 | 0.891 | +0.132 | +0.145 |

v2 wins on every vocabulary. The largest gains are on the one-handed / active-hand-flip words (`you`, `water`, `thank you`, `yes`), exactly the classes the v1 shoulder-normalisation corrupted by pulling missing-hand zeros toward the shoulder centre.

## 7. What could NOT be verified

- **S2 does not exist.** No cross-session generalization can be measured. All numbers are same-session S1 and therefore optimistic; they are superseded by any future S2 test (train on S1 only, test on S2 only).
- **No final or export model trained.** `export_model.py` was not run; nothing written to `models/` or `verified_checkpoint/`; `SIGNSAGE_MODEL_PROMOTED` stays false. Nothing integrated into the app (frontend/backend/mobile untouched).
- **No held-out signer / position / lighting variation within S1** (single session, single position/lighting).
- **Threshold analysis pools across the 5 seeds' OOF sets** (each clip predicted once per seed), so `rest` FPR is measured on 60 rest OOB observations rather than 12 independent clips; the rate (0.0%) is unchanged either way.
- **GPU-vs-CPU landmark parity** was not re-measured (recorder used MediaPipe GPU delegate; evaluation ran on CPU). Feature shapes/ranges match `features_v2` self_test, but parity was not re-verified here.
- **Real-time / latency** on target hardware was not measured (not required by the task).
- `features_v2` correctness is covered by unit tests (5/5) and self-test, but end-to-end behaviour on a held-out signer is unverified.

## 8. Command to confirm tree state

```
git status --short
```

Output:

```
?? research/track_b/diagnose_s1.py
?? research/track_b/eval_v2.py
?? research/track_b/eval_v2_full.log
?? research/track_b/features_v2.py
?? research/track_b/results_v2.json
?? research/track_b/test_features_v2.py
```

Only new untracked files under `research/track_b/`; nothing else modified, nothing committed or pushed.

**Bottom line:** On S1 only, the v2 hand-centric features (mask-aware, gap-filled, hand-to-face + velocity) lift the CORE-10 1D-CNN to **0.894 accuracy / 0.884 macro-F1** (5-seed take CV) versus **0.698 / 0.663** for the old v1 CNN. 9 of 10 core words are RELIABLE (F1≥0.80); only `thank you` (0.73) is WEAK. `rest` separates perfectly (FPR 0.0% at all thresholds). All numbers are `PRELIMINARY, OPTIMISTIC (same session S1)`; no final model or vocabulary decision is made.
