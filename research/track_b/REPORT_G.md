# Track B — REPORT_G: features_v3 (label-independent) + augmentation, train S1 / test S2 dev (PRELIMINARY)

**Status label on every number below: `PRELIMINARY, OPTIMISTIC`. S1 numbers are same-session 5-fold take CV (optimistic). S2 numbers are a DEV SET — S2 was already used to diagnose the handedness problem, so it is NOT a clean held-out test.**

Scope: `research/track_b/` only. No existing files edited, no `.env`/`render.yaml`/`SIGNSAGE_MODEL_PROMOTED` touched (stays false), no commit/push. Ignores `recordings_s3/` entirely. Offline, CPU only. Nothing integrated into the app.

## 0. Files created (this session)

| file | purpose |
|------|---------|
| `research/track_b/features_v3.py` | NEW v3 features: label-independent hand tracking, ACTIVE/SECONDARY slots, canonicalisation mirroring |
| `research/track_b/test_features_v3.py` | NEW unit tests (5/5 pass) |
| `research/track_b/sign_features_v3.py` | NEW standalone numpy-only features_v3 (identical output) |
| `research/track_b/test_sign_features_v2.py` | (existing, for v2) |
| `research/track_b/test_sign_features_v3.py` | NEW test proving sign_features_v3 == features_v3 |
| `research/track_b/eval_v3.py` | NEW training/evaluation (cnn-v3 + logreg, augmentation, S1 CV + S2 dev) |
| `research/track_b/results_v3.json` | NEW raw results |
| `research/track_b/eval_v3_full.log` | NEW full run log |
| `research/track_b/export_v3.py` | NEW export script |
| `research/track_b/export_v3_s1/` | NEW cnn-v3 trained on S1 ONLY (evaluated model) |
| `research/track_b/export_v3_final/` | NEW cnn-v3 trained on S1+S2, FINAL-UNEVALUATED |
| `research/track_b/REPORT_G.md` | NEW this file |

Exact commands run (from `research/track_b/`):
```
python3 test_features_v3.py          # 5/5 PASS
python3 test_sign_features_v3.py     # sign_features_v3 == features_v3, OK
QUICK=1 python3 eval_v3.py           # smoke test (1 seed)
python3 eval_v3.py                   # FULL 5-seed run (~16 min)
python3 export_v3.py
python3 test_sign_features_v3.py
```

## 1. features_v3 — design change (vs v2)

MediaPipe Left/Right handedness is unreliable because the camera is unmirrored (labels flip between sessions and within clips). v3 ignores them:

- **(a) Temporal tracking:** per frame, detected hands from both slots (labels ignored) are assigned to a stable physical identity — nearest wrist to the previous frame's wrist for that physical hand; if two hands appear in one frame, order by image x, then keep continuity.
- **(b) ACTIVE / SECONDARY slots:** ACTIVE hand per clip = hand with larger total wrist travel (tie: more frames present). The other is SECONDARY. Features use ACTIVE then SECONDARY instead of L/R (122 dims × 60 frames, same layout family as v2).
- **(c) Canonicalisation mirroring:** if the active hand's mean wrist x is image-left of the shoulder midpoint, mirror all x coordinates (hands + face) about the per-frame shoulder midpoint so every clip looks right-hand-active.
- Missing handling unchanged from v2: gap-fill ≤5 frames, zero coords + mask=0 otherwise, never normalise missing zeros into shoulder space, resample to 60 after filling.

## 2. Unit tests

`test_features_v3.py` (5/5 PASS):
1. shape (60,122), no NaN/Inf
2. all-missing-hand → all zeros, masks 0
3. swapping input L/R slots → IDENTICAL output (label independence)
4. mirrored copy → near-identical output (canonicalisation)
5. short/long resample to 60

`test_sign_features_v3.py` (PASS): standalone `sign_features_v3.py` matches `features_v3.py` exactly (atol 1e-5), including all-missing case.

## 3. Training-only augmentation (cnn-v3 batches only)

Applied per training batch to cnn-v3 (not to logreg, not at test time):
- random hand-frame dropout (mask=0, 0–50% of frames) per hand
- scale jitter ±10%
- rotation ±10 degrees
- time-warp ±15%

Hyperparameters otherwise exactly as REPORT_E cnn-v2: dropout 0.3, 60 epochs, Adam 1e-3, batch 16, no hyperparameter search.

## 4. Results

### 4.1 S1 5-fold CV across takes (CORE-10 + rest, 11 classes) — PRELIMINARY, OPTIMISTIC

| model | accuracy | macro-F1 | per-seed acc |
|-------|----------|----------|--------------|
| cnn-v3 | **0.848 ± 0.011** | **0.835 ± 0.016** | [0.845, 0.839, 0.861, 0.853, 0.839] |
| logreg (baseline) | 0.774 ± 0.005 | 0.740 ± 0.005 | [0.772, 0.779, 0.780, 0.776, 0.764] |

Pooled (majority vote over 5 seeds): cnn-v3 acc=0.863, macroF1=0.862; logreg acc=0.763, macroF1=0.765. Chance (uniform, 11) = 0.091; majority = 0.092.

### 4.2 Train S1 only / test S2 only — DEV SET (not a clean test)

cnn-v3, 5 seeds, S2 = 50 clips (10 CORE words, **no rest**):

| metric | value |
|--------|-------|
| accuracy (mean ± std) | **0.612 ± 0.050** |
| macro-F1 (mean ± std) | **0.598 ± 0.038** |
| pooled (majority vote) acc / macro-F1 | 0.612 / 0.598 (n = 50) |
| chance baseline (uniform, 10) | 0.100 |
| v2 cnn-v2 S2 acc (REPORT_F) | 0.592 |
| logreg S2 acc / macro-F1 | 0.524 ± 0.023 / 0.497 ± 0.026 |

v3 (0.612) is a slight gain over v2 (0.592) on S2, but both are poor — consistent with the small dev set and residual cross-session shift. Per the task rules (results poor → report diagnostics only, no iteration), no model changes were made.

### 4.3 S2 per-class F1 table (cnn-v3, seed-mean over 5 seeds)

| class | precision | recall | F1 | verdict |
|-------|-----------|--------|----|---------|
| me | 0.97 | 1.00 | 0.98 | RELIABLE |
| hello | 0.97 | 0.92 | 0.93 | RELIABLE |
| please | 0.68 | 1.00 | 0.81 | RELIABLE |
| help | 0.79 | 0.71 | 0.74 | BORDERLINE |
| you | 1.00 | 0.48 | 0.64 | BORDERLINE |
| water | 0.49 | 0.84 | 0.60 | BORDERLINE |
| no | 0.69 | 0.60 | 0.58 | WEAK |
| yes | 0.67 | 0.20 | 0.30 | WEAK |
| where | 0.21 | 0.36 | 0.27 | WEAK |
| thank you | 0.11 | 0.16 | 0.13 | WEAK |

Verdict rule: RELIABLE F1 ≥ 0.80; BORDERLINE 0.60–0.80; WEAK < 0.60.
→ **3 RELIABLE (me, hello, please), 3 BORDERLINE (help, you, water), 4 WEAK (no, yes, where, thank you).**

### 4.4 Top-10 confusion pairs (cnn-v3, S2 dev, seed-0 representative, count true → predicted)

1. thank you → where (5)
2. yes → water (3)
3. help → water (3)
4. where → please (2)
5. you → yes (1)
6. you → water (1)
7. you → help (1)
8. yes → thank you (1)

### 4.5 Max-softmax threshold table (cnn-v3, S2 dev)

| threshold | accepted | coverage | accuracy |
|-----------|----------|----------|----------|
| argmax (no reject) | 50 | 100% | 0.640 |
| 0.5 | 48 | 96% | 0.646 |
| 0.7 | 37 | 74% | 0.676 |
| 0.9 | 25 | 50% | 0.600 |

- Recommended operating point: **0.7** (acc 0.676 at 74% coverage).
- `rest` false-positive rate: **NOT VERIFIED** (no rest clips in S2).
- Stated as calibrated on S2 only; S2 is a dev set, not a clean test.

## 5. Hand-label check AFTER canonicalisation (mirrored fraction per label)

Fraction of clips that features_v3 mirrored (active hand made right-hand-active):

| label | S1 mirrored | S2 mirrored |
|-------|-------------|-------------|
| hello | 1.00 | 1.00 |
| help | 0.50 | 0.60 |
| me | 0.50 | 1.00 |
| no | 0.17 | 1.00 |
| please | 1.00 | 1.00 |
| thank you | 0.92 | 0.00 |
| water | 0.50 | 1.00 |
| where | 0.42 | 0.80 |
| yes | 1.00 | 1.00 |
| you | 0.50 | 1.00 |

Canonicalisation makes nearly all S2 clips right-hand-active (the session where handedness was most unstable), while S1 remains mixed. Despite this, cross-session accuracy stays poor (0.612), so handedness was only one contributor; the tiny dev set (50 clips) and residual session shift dominate.

## 6. Exports

| export | trained on | status |
|--------|-----------|--------|
| `research/track_b/export_v3_s1/` | S1 only | evaluated model (cnn-v3) |
| `research/track_b/export_v3_final/` | S1 + S2 | **FINAL-UNEVALUATED** (no held-out test; S2 consumed as dev set) |

Each export contains: `model.pt` (state dict), `meta.json` (labels, v3 feature spec, normalisation constants, extractor name/version/options, fps note, recommended confidence threshold **0.7** stated as calibrated on S2), `golden_test_vector.npy` ((T,543,3)), `golden_expected_logits.json` (verified to match the model exactly). CPU latency per clip ≈ 0.25–0.43 ms.

`SIGNSAGE_MODEL_PROMOTED` stays false. Nothing written to `models/` or `verified_checkpoint/`; nothing integrated into the app.

## 7. NOT VERIFIED list

- **rest false-positive rate** — NOT VERIFIED (0 rest clips in S2).
- **S2 is a dev set, not a clean held-out test** — it was used to diagnose the handedness problem; all S2 numbers are optimistic/soft.
- **export_v3_final (S1+S2)** — NOT VERIFIED (no held-out test). Labelled FINAL-UNEVALUATED. Do not promote.
- **Single cross-session pair only** — one S1→S2 split; no signer/position/lighting variation tested.
- **Generalisation beyond this S1/S2 pair** — untested. `recordings_s3/` deliberately ignored per task rules.

## 8. Recommendation

Do **not** promote or export-to-app any model from this session. Cross-session cnn-v3 accuracy (0.612 ± 0.050, macro-F1 0.598) is far below S1 CV (0.848 ± 0.011), and 4 of 10 CORE words are WEAK on the S2 dev set (thank you 0.13, where 0.27, yes 0.30, no 0.58). The v3 label-independent features + canonicalisation removed the handedness confound (nearly all S2 clips now right-hand-active) but accuracy remains poor, so the residual gap is not explained by handedness alone.

Before any vocabulary or promotion decision:
1. Obtain a **clean held-out test session** (not S2, which is consumed as the dev set) and re-verify.
2. Investigate the residual cross-session shift beyond handedness (small dev set size, other session factors).
3. Only then reconsider `SIGNSAGE_MODEL_PROMOTED` and `models/` export.
