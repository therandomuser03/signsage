# Track B — REPORT_H: Clean held-out test on session S3 (PRELIMINARY)

**Status label on every number below: `PRELIMINARY`. S3 is the clean held-out test — train on S1+S2 only, test on S3 only. S3 was never used for tuning, model selection, or threshold selection.**

Scope: `research/track_b/` only. No existing files edited, no `.env`/`render.yaml`/`SIGNSAGE_MODEL_PROMOTED` touched (stays false), no commit/push. Offline, CPU only. Nothing integrated into the app. Uses exactly the cnn-v3 recipe/hyperparameters from REPORT_G.

## 0. Files created (this session)

| file | purpose |
|------|---------|
| `research/track_b/verify_s3.py` | NEW step 1: S3 conversion + duplicate vs S1/S2 verification |
| `research/track_b/eval_s1s2train_s3test.py` | NEW step 2: train S1+S2 / test S3 evaluation |
| `research/track_b/results_s1s2train_s3test.json` | NEW raw S3 test results |
| `research/track_b/REPORT_H.md` | NEW this file |

Exact commands run (from `research/track_b/`):
```
python3 verify_s3.py
python3 eval_s1s2train_s3test.py
```

## 1. Step 1 — S3 conversion & duplicate verification

**S3 recordings -> data/s3_converted:** 50 clips converted, all shapes `(T,543,3)` valid, all with `session == "S3"`. **0 S3 clips duplicate S1** and **0 duplicate S2** (compared by label/take/timestamp). S3 is a clean held-out session.

Per-label S3 clip counts and hands-detected rate (mean fraction of frames where the hand is non-zero):

| label | clips | L_det | R_det | either_det |
|-------|-------|-------|-------|------------|
| hello | 5 | 0.009 | 0.425 | 0.425 |
| help | 5 | 0.411 | 0.645 | 0.645 |
| me | 5 | 0.009 | 0.672 | 0.672 |
| no | 5 | 0.000 | 0.691 | 0.691 |
| please | 5 | 0.000 | 0.609 | 0.609 |
| thank you | 5 | 0.000 | 0.542 | 0.542 |
| water | 5 | 0.000 | 0.582 | 0.582 |
| where | 5 | 0.394 | 0.417 | 0.530 |
| yes | 5 | 0.000 | 0.601 | 0.601 |
| you | 5 | 0.017 | 0.597 | 0.597 |

**CORE labels with NO S3 clips (NOT VERIFIED): NONE** — all 10 CORE words (hello, thank you, please, yes, no, help, water, you, me, where) are present in S3 (5 takes each).

**rest clips in S3: 0** → rest false-positive rate = **NOT VERIFIED** (cannot be measured without S3 rest clips).

## 2. Step 2 — Train cnn-v3 on S1+S2, test on S3 ONLY

Train: S1 (CORE-10 + rest, 131 clips) + S2 (CORE-10 words, 50 clips) = **181 clips, 11 classes**.
Test: S3 CORE-10 words only (**50 clips, no rest**). 5 seeds. cnn-v3 = features_v3 + 1D-CNN dropout 0.3, 60 epochs, Adam 1e-3, batch 16, training-only augmentation (hand-frame dropout 0–50%, scale ±10%, rotation ±10°, time-warp ±15%). Logreg baseline included for comparison.

### cnn-v3 (the REPORT_G recipe)

| metric | value |
|--------|-------|
| S3 accuracy (mean ± std, 5 seeds) | **0.856 ± 0.039** |
| S3 macro-F1 (mean ± std, 5 seeds) | **0.854 ± 0.042** |
| per-seed acc | [0.86, 0.88, 0.88, 0.78, 0.88] |
| per-seed macro-F1 | [0.861, 0.880, 0.880, 0.771, 0.880] |
| pooled (mean of seeds) acc / macro-F1 | 0.856 / 0.854 (n = 50) |
| chance baseline (uniform, 10 classes) | 0.100 |

Note: seed 3 is an outlier (0.78 acc / 0.771 macro-F1) while the other four seeds are 0.86–0.88; the std reflects this spread. All five seeds exceed the S2 dev performance (0.612 acc / 0.598 macro-F1).

### Logreg baseline (flattened 30-frame v3 → StandardScaler → multinomial logistic regression)

| metric | value |
|--------|-------|
| S3 accuracy (mean ± std) | **0.744 ± 0.008** |
| S3 macro-F1 (mean ± std) | **0.731 ± 0.010** |
| per-seed acc | [0.74, 0.74, 0.74, 0.74, 0.76] |

cnn-v3 clearly beats logreg on S3.

### S3 per-class precision/recall/F1 table (cnn-v3, seed-mean over 5 seeds)

| class | precision | recall | F1 | verdict |
|-------|-----------|--------|----|---------|
| me | 1.00 | 1.00 | 1.00 | RELIABLE |
| water | 1.00 | 1.00 | 1.00 | RELIABLE |
| help | 1.00 | 1.00 | 1.00 | RELIABLE |
| no | 1.00 | 1.00 | 1.00 | RELIABLE |
| hello | 0.91 | 1.00 | 0.94 | RELIABLE |
| yes | 0.83 | 1.00 | 0.91 | RELIABLE |
| you | 1.00 | 0.80 | 0.89 | RELIABLE |
| where | 1.00 | 0.72 | 0.83 | RELIABLE |
| thank you | 0.44 | 0.64 | 0.52 | WEAK |
| please | 0.50 | 0.40 | 0.44 | WEAK |

Verdict rule: RELIABLE F1 ≥ 0.80; BORDERLINE 0.60–0.80; WEAK < 0.60.
→ **8 RELIABLE (me, water, help, no, hello, yes, you, where), 0 BORDERLINE, 2 WEAK (thank you, please).**

### Top-10 confusion pairs (cnn-v3, S3, seed-0 representative, count true → predicted)

1. please → thank you (3)
2. you → yes (1)
3. where → thank you (1)
4. where → please (1)
5. thank you → please (1)

### Max-softmax threshold table (cnn-v3, S3)

| threshold | accepted | coverage | accuracy |
|-----------|----------|----------|----------|
| argmax (no reject) | 50 | 100% | 0.860 |
| 0.5 | 48 | 96% | 0.875 |
| 0.7 | 41 | 82% | 0.878 |
| 0.9 | 32 | 64% | 0.969 |

- `rest` false-positive rate: **NOT VERIFIED** (no rest clips in S3).
- Recommended operating point stays 0.7 (from REPORT_G, calibrated on S2 dev); on S3 it gives acc 0.878 at 82% coverage. Threshold not re-selected from S3.

## 3. Step 3 — Reliability marking (cnn-v3, S3)

| word | S3 F1 | verdict |
|------|-------|---------|
| me | 1.00 | RELIABLE |
| water | 1.00 | RELIABLE |
| help | 1.00 | RELIABLE |
| no | 1.00 | RELIABLE |
| hello | 0.94 | RELIABLE |
| yes | 0.91 | RELIABLE |
| you | 0.89 | RELIABLE |
| where | 0.83 | RELIABLE |
| thank you | 0.52 | WEAK |
| please | 0.44 | WEAK |

**8 RELIABLE, 0 BORDERLINE, 2 WEAK.** No iteration performed (results not poor enough to warrant diagnostics-only; and per the rules no tuning from S3 regardless). thank you and please remain weak — mostly confused with each other (3 of 5 please clips → thank you).

## 4. NOT VERIFIED list

- **rest false-positive rate** — NOT VERIFIED (0 rest clips in S3).
- All 10 CORE labels are present in S3, so no CORE label is NOT VERIFIED on that count.

## 5. Files created (this session)

- `research/track_b/verify_s3.py`
- `research/track_b/eval_s1s2train_s3test.py`
- `research/track_b/results_s1s2train_s3test.json`
- `research/track_b/REPORT_H.md`

**Bottom line:** Clean held-out S3 test gives cnn-v3 **0.856 ± 0.039 accuracy / 0.854 ± 0.042 macro-F1** (chance 0.100), a large improvement over the S2 dev result (0.612 / 0.598) and showing the v3 label-independent features generalise well to an unseen session. 8 of 10 CORE words RELIABLE; only thank you (0.52) and please (0.44) are WEAK. rest FPR NOT VERIFIED (no S3 rest clips). `SIGNSAGE_MODEL_PROMOTED` stays false; nothing integrated into the app.
