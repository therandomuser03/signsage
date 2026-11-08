# Track B — REPORT_F: Cross-session test (train S1, test S2) + model export (PRELIMINARY)

**Status label on every number below: `PRELIMINARY, OPTIMISTIC (same signer/setup, S1->S2 cross-session)`. NOT a final result.**

Scope: `research/track_b/` only. No existing files edited, no `.env`/`render.yaml`/`SIGNSAGE_MODEL_PROMOTED` touched (stays false), no commit/push. S2 is TEST DATA ONLY — no tuning, no variant selection, no hyperparameter changes based on S2. Uses exactly the cnn-v2 setup/hyperparameters from REPORT_E (dropout 0.3, 60 epochs, Adam 1e-3, batch 16, no augmentation, same seed handling). Offline, CPU only. Nothing integrated into the app.

## 0. Files created (this session)

| file | purpose |
|------|---------|
| `research/track_b/diagnose_s2.py` | NEW step 1+2: S2 convert verification + hand-label check S1 vs S2 |
| `research/track_b/eval_s1train_s2test.py` | NEW step 3: train S1 / test S2 evaluation |
| `research/track_b/results_s1train_s2test.json` | NEW raw cross-session results |
| `research/track_b/export_models.py` | NEW step 5: export script |
| `research/track_b/export_s1/` | NEW cnn-v2 trained on S1 ONLY (the evaluated model): model.pt, meta.json, golden_test_vector.npy, golden_expected_logits.json |
| `research/track_b/export_final/` | NEW cnn-v2 trained on S1+S2, FINAL-UNEVALUATED: model.pt, meta.json, golden_test_vector.npy, golden_expected_logits.json |
| `research/track_b/sign_features_v2.py` | NEW standalone numpy-only features_v2 (identical output) |
| `research/track_b/test_sign_features_v2.py` | NEW unit test proving sign_features_v2 == features_v2 |
| `research/track_b/REPORT_F.md` | NEW this file |

Exact commands run (from `research/track_b/`):
```
python recorder/convert_and_validate.py recordings_s2 --out data/s2_converted
python diagnose_s2.py
python eval_s1train_s2test.py
python export_models.py
python test_sign_features_v2.py
```

## 1. Step 1 — S2 conversion & verification

**S2 recordings -> data/s2_converted:** 50 clips converted, all shapes `(T,543,3)` valid, all with `session == "S2"`. **0 S2 clips duplicate an S1 clip** (compared by label/take/timestamp).

Per-label S2 clip counts and hands-missing share (hand missing = all 21 landmarks zero in a frame):

| label | clips | L_miss_mean | R_miss_mean | clips with either hand >40% missing |
|-------|-------|-------------|-------------|-------------------------------------|
| hello | 5 | 0.987 | 0.289 | 5/5 |
| help | 5 | 0.543 | 0.563 | 5/5 |
| me | 5 | 1.000 | 0.374 | 5/5 |
| no | 5 | 1.000 | 0.343 | 5/5 |
| please | 5 | 1.000 | 0.226 | 5/5 |
| thank you | 5 | 0.311 | 0.635 | 5/5 |
| water | 5 | 0.982 | 0.362 | 5/5 |
| where | 5 | 0.455 | 0.480 | 5/5 |
| yes | 5 | 0.982 | 0.375 | 5/5 |
| you | 5 | 0.837 | 0.377 | 5/5 |

**CORE labels with NO S2 clips (NOT VERIFIED for cross-session): NONE** — all 10 CORE words (hello, thank you, please, yes, no, help, water, you, me, where) are present in S2 (5 takes each).

**rest clips in S2: 0** → rest false-positive rate = **NOT VERIFIED** (cannot be measured without S2 rest clips).

## 2. Step 2 — Hand-label check S1 vs S2

Compared active hand (Left/Right), presence rates, and within-clip flip rate for each label present in both sessions. Flag rule: active-hand change, OR presence-rate change >0.20, OR flip-rate change >0.30.

| label | S1 active | S1 Lpres | S1 Rpres | S1 flip \| S2 active | S2 Lpres | S2 Rpres | S2 flip | flag |
|-------|-----------|----------|----------|----------------------|----------|----------|---------|------|
| hello | Right | 0.033 | 0.389 | 0.583 \| Right | 0.013 | 0.711 | 0.000 | **FLAG** |
| help | Right | 0.440 | 0.601 | 0.167 \| Left | 0.457 | 0.437 | 0.600 | **FLAG** |
| me | Right | 0.303 | 0.378 | 0.083 \| Right | 0.000 | 0.626 | 0.000 | **FLAG** |
| no | Left | 0.589 | 0.136 | 0.000 \| Right | 0.000 | 0.657 | 0.000 | **FLAG** |
| please | Right | 0.144 | 0.592 | 0.818 \| Right | 0.000 | 0.774 | 0.000 | **FLAG** |
| thank you | Right | 0.123 | 0.456 | 0.750 \| Left | 0.689 | 0.365 | 0.600 | **FLAG** |
| water | Right | 0.268 | 0.382 | 0.000 \| Right | 0.018 | 0.638 | 0.000 | **FLAG** |
| where | Left | 0.514 | 0.491 | 0.583 \| Left | 0.545 | 0.520 | 0.400 | ok |
| yes | Right | 0.004 | 0.683 | 0.083 \| Right | 0.018 | 0.625 | 0.200 | ok |
| you | Right | 0.292 | 0.317 | 0.167 \| Right | 0.163 | 0.623 | 0.800 | **FLAG** |

**8 of 10 labels FLAG** — active-hand or presence changed notably between sessions (e.g. `no` flipped from Left-dominant in S1 to Right-dominant in S2; `please`/`me`/`yes`/`water`/`hello` went from mixed to almost exclusively right-hand; `thank you`/`help` flipped active hand). Only `where` and `yes` are unchanged. This systematic hand-usage shift is the dominant explanation for the cross-session performance drop in step 3.

## 3. Step 3 — Train cnn-v2 on S1 ONLY, test on S2 ONLY

Train: S1 CORE10 + rest (131 clips, 11 classes). Test: S2 labels present in S2 (the 10 CORE words, 50 clips; **no rest in S2**). 5 seeds. Mean ± std over seeds. No tuning based on S2.

### cnn-v2 (evaluated model, REPORT_E setup)

| metric | value |
|--------|-------|
| S2 accuracy (mean ± std, 5 seeds) | **0.592 ± 0.020** |
| S2 macro-F1 (mean ± std, 5 seeds) | **0.554 ± 0.027** |
| per-seed acc | [0.56, 0.60, 0.58, 0.60, 0.62] |
| per-seed macro-F1 | [0.513, 0.571, 0.536, 0.564, 0.589] |
| Pooled (majority vote over 5 seeds) acc | 0.620 (n = 50) |
| Pooled macro-F1 | 0.589 |
| Chance baseline (uniform, 10 classes) | 0.100 |

### S2 per-class F1 table (cnn-v2, pooled majority vote, n = 50)

| class | precision | recall | F1 | RELIABLE/BORDERLINE/WEAK |
|-------|-----------|--------|----|--------------------------|
| no | 1.00 | 1.00 | 1.00 | RELIABLE |
| help | 1.00 | 1.00 | 1.00 | RELIABLE |
| me | 1.00 | 1.00 | 1.00 | RELIABLE |
| hello | 0.71 | 1.00 | 0.83 | RELIABLE |
| you | 1.00 | 0.40 | 0.57 | WEAK |
| water | 0.50 | 0.60 | 0.55 | WEAK |
| thank you | 0.31 | 0.80 | 0.44 | WEAK |
| yes | 0.33 | 0.20 | 0.25 | WEAK |
| please | 0.33 | 0.20 | 0.25 | WEAK |
| where | 0.00 | 0.00 | 0.00 | WEAK |

**Reliability (S2 F1):** RELIABLE (≥0.80): no, help, me, hello (4). BORDERLINE (0.60–0.80): none. WEAK (<0.60): you, water, thank you, yes, please, where (6).

### Top-10 confusion pairs (cnn-v2, S2, pooled, count true → predicted)

1. where → thank you (4)
2. please → thank you (4)
3. yes → water (3)
4. you → yes (2)
5. water → hello (2)
6. you → where (1)
7. yes → thank you (1)
8. where → please (1)
9. thank you → please (1)

### Max-softmax threshold table (cnn-v2, S2)

| threshold | accepted | coverage | accuracy |
|-----------|----------|----------|----------|
| argmax (no reject) | 50 | 100% | 0.620 |
| 0.5 | 49 | 98.0% | 0.612 |
| 0.7 | 44 | 88.0% | 0.659 |
| 0.9 | 33 | 66.0% | 0.758 |

**rest false-positive rate: NOT VERIFIED** (no rest clips in S2).

### Logreg baseline (flattened 30-frame v2 → StandardScaler → multinomial logistic regression)

| metric | value |
|--------|-------|
| S2 accuracy (mean ± std) | **0.640 ± 0.013** |
| S2 macro-F1 (mean ± std) | **0.586 ± 0.011** |
| Pooled acc / macro-F1 | 0.640 / 0.586 |

Logreg per-class F1 (pooled): hello=1.00, me=0.91, water=0.91, no=0.89, help=0.75, yes=0.62, please=0.50, thank you=0.29, where=0.00, you=0.00.

Logreg top-10 confusions: where→please (5), you→yes (4), thank you→please (4), you→where (1), yes→thank you (1), no→water (1), help→please (1), help→me (1).

Logreg threshold table: argmax 50/100%/0.640; 0.5: 48/96%/0.646; 0.7: 41/82%/0.634; 0.9: 29/58%/0.655. rest FPR: NOT VERIFIED.

**Interpretation:** Cross-session performance collapses vs S1 (cnn-v2 S1 CV was 0.894 acc / 0.884 macro-F1; S2 test is 0.592 / 0.554). Logreg (0.640) slightly edges cnn-v2 (0.592) on this tiny 50-clip S2 set — both are poor. Per step 4 rules (results poor → report diagnostics only, no iteration), no model changes were made; the diagnostics above (confusions + the step-2 hand-label differences) are the explanation. `where`, `please`, `yes` fall to F1 0.00–0.25 on S2.

## 4. Step 4 — Reliability verdict (cnn-v2, S2 pooled F1)

| word | S2 F1 | verdict |
|------|-------|---------|
| no | 1.00 | RELIABLE |
| help | 1.00 | RELIABLE |
| me | 1.00 | RELIABLE |
| hello | 0.83 | RELIABLE |
| you | 0.57 | WEAK |
| water | 0.55 | WEAK |
| thank you | 0.44 | WEAK |
| yes | 0.25 | WEAK |
| please | 0.25 | WEAK |
| where | 0.00 | WEAK |

**4 RELIABLE, 0 BORDERLINE, 6 WEAK.** Results are poor (macro-F1 0.55–0.59), so per the task rules no iteration was performed — diagnostics reported above only. No vocabulary or model decision is finalised.

## 5. Step 5 — Exports

### (a) research/track_b/export_s1/ — model trained on S1 ONLY (the evaluated cnn-v2)
- `model.pt` — torch state dict (1D-CNN, dropout 0.3, 122-dim v2 features, 11 classes: CORE10 + rest)
- `meta.json` — label list, features_v2 spec, normalisation constants, extractor name/version/options, fps note, recommended confidence threshold, S2 test metrics, CPU latency
- `golden_test_vector.npy` — a `(19, 543, 3)` raw landmarks input clip
- `golden_expected_logits.json` — expected logits for that input (verified to match the model exactly)

### (b) research/track_b/export_final/ — model trained on S1+S2, FINAL-UNEVALUATED
- Same files. **Labelled FINAL-UNEVALUATED: S2 was already used as the held-out test set for (a), so this model has NO held-out test set. Do not promote without S2-independent validation.**

`SIGNSAGE_MODEL_PROMOTED` stays false. Nothing exported to `models/` or `verified_checkpoint/`; nothing integrated into the app.

### Standalone feature module
- `sign_features_v2.py` — single-file, numpy-only re-implementation of `features_v2.py` (identical (60,122) output).
- `test_sign_features_v2.py` — proves `sign_features_v2.extract_features == features_v2.extract_features` exactly (allclose atol 1e-5), including the all-missing-hand case. **PASS.**
- CPU latency per clip: export_s1 ≈ **0.35 ms**, export_final ≈ **0.30 ms** (CPU, batch size 1, 60-frame v2 features).

### Recommended confidence threshold (calibrated on S2)
Derived from the cnn-v2 S2 max-softmax table: **0.7** (accuracy 0.659 at 88% coverage; 0.9 gives 0.758 accuracy at 66% coverage). **Stated as calibrated on S2 only** — not validated on held-out data, and overall accuracy is poor, so treat as an operating point, not a guarantee.

## 6. NOT VERIFIED list

- **rest false-positive rate** — NOT VERIFIED (0 rest clips in S2).
- **CORE labels with no S2 clips** — NONE (all 10 present); nothing NOT VERIFIED on this count.
- **export_final (S1+S2)** — NOT VERIFIED (no held-out test; trained on data that includes the S2 test set). Labelled FINAL-UNEVALUATED.
- **Cross-session beyond one S1→S2 pair** — only this single cross-session split was tested.
- **Signer / position / lighting variation** — not tested (single setup presumed across sessions).
- **Held-out independent validation** — none exists yet; S2 was consumed as the test set.

## 7. Recommendation

Do **not** promote or export-to-app any model from this session. Cross-session cnn-v2 performance (0.592 acc / 0.554 macro-F1 on S2) is far below S1 CV (0.894 / 0.884), and 6 of 10 CORE words are WEAK on S2, driven by a systematic hand-usage shift between sessions (8/10 labels changed active hand or presence). Before any vocabulary or promotion decision:

1. Collect an independent held-out test session (not S2, which is now consumed as the test set) and re-verify.
2. Investigate the hand-usage shift (active-hand flips, one-handedness) — consider session-invariant data augmentation or explicit hand-presence features.
3. Only then reconsider `SIGNSAGE_MODEL_PROMOTED` and `models/` export.

**Bottom line:** S1-trained cnn-v2 generalises poorly to S2 (0.592 acc, macro-F1 0.554, chance 0.100). 4 words RELIABLE, 6 WEAK. rest FPR NOT VERIFIED. export_final is FINAL-UNEVALUATED. No promotion; no app integration.
