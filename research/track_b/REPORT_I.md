# Track B — REPORT_I: Final word model for demo vocabulary (8 words + rest) (PRELIMINARY)

**Status label: `PRELIMINARY`. The demo vocabulary was chosen AFTER seeing S3 results, so step 1 numbers are slightly optimistic. The FINAL model (trained on S1+S2+S3) is labelled FINAL-UNEVALUATED — no held-out data exists for it.**

Scope: `research/track_b/` only. No existing files edited, no `.env`/`render.yaml`/`SIGNSAGE_MODEL_PROMOTED` touched (stays false), no commit/push. Offline, CPU only. Nothing integrated into the app. Uses exactly the cnn-v3 recipe/hyperparameters from REPORT_G/REPORT_H. No tuning, no variants, no search.

## 0. Files created (this session)

| file | purpose |
|------|---------|
| `research/track_b/eval_final8.py` | NEW step 1+2: 8-class eval + 10-class comparison |
| `research/track_b/results_final8.json` | NEW raw results |
| `research/track_b/export_final8.py` | NEW step 3+4: FINAL model + export |
| `research/track_b/export_final8/` | NEW FINAL-UNEVALUATED export: model.pt, meta.json, golden_test_vector.npy, golden_expected_logits.json |
| `research/track_b/test_sign_features_v3.py` | (existing) re-run: equivalence test PASS |
| `research/track_b/REPORT_I.md` | NEW this file |

Exact commands run (from `research/track_b/`):
```
python3 eval_final8.py          # step 1+2 (~78s)
python3 export_final8.py        # step 3+4 (~4 min)
python3 test_sign_features_v3.py  # equivalence test
```

## 1. Vocabulary

**Demo vocabulary (8 words + rest):**
`hello, me, you, yes, no, help, water, where` + `rest`

**Excluded (weak on S3 in the 10-class model):**
`thank you` (F1 0.52), `please` (F1 0.44). Their clips are dropped completely from training and testing.

Label on step 1 numbers: **"VOCABULARY CHOSEN AFTER SEEING S3: slightly optimistic"**. Threshold kept at 0.7 (not re-selected from S3).

## 2. Step 1 — Train cnn-v3 on S1+S2 (8 words + rest), test on S3 (8 words only), 5 seeds

Train: S1 (8 words × 12 = 96 + rest 12 = 108 clips) + S2 (8 words × 5 = 40 clips) = **148 clips, 9 classes**.
Test: S3 (8 words × 5 = **40 clips**, no rest).

| metric | value |
|--------|-------|
| accuracy (mean ± std, 5 seeds) | **0.935 ± 0.012** |
| macro-F1 (mean ± std, 5 seeds) | **0.933 ± 0.013** |
| per-seed acc | [0.95, 0.925, 0.925, 0.925, 0.95] |
| per-seed macro-F1 | [0.949, 0.923, 0.923, 0.923, 0.948] |
| pooled (mean of seeds) acc / macro-F1 | 0.935 / 0.933 (n = 40) |
| chance baseline (uniform, 8 classes) | 0.125 |

All 8 words are RELIABLE (F1 ≥ 0.85).

### Per-class precision/recall/F1 table (cnn-v3, S3, seed-mean over 5 seeds)

| class | precision | recall | F1 | verdict |
|-------|-----------|--------|----|---------|
| water | 1.00 | 1.00 | 1.00 | RELIABLE |
| hello | 1.00 | 1.00 | 1.00 | RELIABLE |
| me | 0.97 | 1.00 | 0.98 | RELIABLE |
| help | 1.00 | 0.96 | 0.98 | RELIABLE |
| no | 0.83 | 1.00 | 0.90 | RELIABLE |
| yes | 0.81 | 1.00 | 0.89 | RELIABLE |
| you | 1.00 | 0.76 | 0.86 | RELIABLE |
| where | 1.00 | 0.76 | 0.85 | RELIABLE |

### Top-10 confusion pairs (cnn-v3, S3, seed-0 representative, count true → predicted)

1. you → yes (1)
2. help → me (1)

Only 2 errors across all 40 S3 clips in seed 0.

### Max-softmax threshold table (cnn-v3, S3); threshold kept at 0.7

| threshold | accepted | coverage | accuracy |
|-----------|----------|----------|----------|
| argmax (no reject) | 40 | 100% | 0.950 |
| 0.5 | 39 | 97.5% | 0.949 |
| 0.7 | 36 | 90% | 0.972 |
| 0.9 | 33 | 82.5% | 0.970 |

- Recommended operating point: **0.7** (acc 0.972 at 90% coverage), kept from REPORT_G (calibrated on S2 dev), **not re-selected from S3**.
- `rest` false-positive rate: **NOT VERIFIED** (no rest clips in S3).

## 3. Step 2 — Comparison: earlier 10-class model on the same 8 S3 words

Train cnn-v3 on S1+S2 with all 10 words + rest (181 clips, 11 classes), test on S3's 8 words, counting predictions of thank you / please / rest as errors.

| metric | value |
|--------|-------|
| accuracy (mean ± std, 5 seeds) | **0.940 ± 0.012** |
| per-seed acc | [0.925, 0.95, 0.95, 0.925, 0.95] |

The 10-class model comparison is slightly higher (0.940 vs 0.935) — within seed variance; both are strong. The 8-class model is the final vocabulary model.

## 4. Step 3 — FINAL model (FINAL-UNEVALUATED)

Train cnn-v3 on **S1+S2+S3 (8 words + rest)** with **fixed seed 42**:
- S1: 108 clips, S2: 40 clips, S3: 40 clips → **188 clips, 9 classes**

Label: **FINAL-UNEVALUATED** — no held-out data exists for it (S3 was consumed as the test set in step 1). Expected quality estimated by step 1 (acc 0.935, macro-F1 0.933).

## 5. Step 4 — Export to research/track_b/export_final8/

| file | contents |
|------|----------|
| `model.pt` | torch state dict (cnn-v3, 9 classes) |
| `meta.json` | labels incl. rest, features_v3 spec, normalisation constants, extractor name/version/options, fps note, threshold 0.7, model_version "word_v3_8w_s123", excluded words, signer statement |
| `golden_test_vector.npy` | a (T,543,3) raw landmarks clip (from S3) |
| `golden_expected_logits.json` | expected logits for that input (9 values, label order) |

Verification:
- Golden vector reproduces expected logits when loaded from the export folder: **True (maxdiff = 0.00e+00)**.
- Predicted label on golden vector: `hello`.
- `sign_features_v3.py` equivalence test (standalone numpy-only == features_v3.py): **PASS** (test_sign_features_v3.py).
- CPU latency per clip: **0.41 ms** (CPU, batch size 1).

`meta.json` key fields:
- `model_version`: `word_v3_8w_s123`
- `status`: `FINAL-UNEVALUATED`
- `label_list`: `["hello","me","you","yes","no","help","water","where","rest"]`
- `excluded_words`: `["thank you","please"]`
- `signer_note`: `adapted to one signer; not signer-independent`
- `recommended_confidence_threshold`: `0.7`
- `extractor`: `@mediapipe/tasks-vision HolisticExtractor`, version `1.0.1`, VIDEO mode, GPU delegate, same options as the SignSage frontend

`SIGNSAGE_MODEL_PROMOTED` stays false. Nothing written to `models/` or `verified_checkpoint/`; nothing integrated into the app.

## 6. NOT VERIFIED list

- **rest false-positive rate** — NOT VERIFIED (0 rest clips in S3, and rest is excluded from the demo vocabulary anyway).
- **FINAL-UNEVALUATED model** — no held-out test; quality estimated only by step 1 (train S1+S2, test S3).
- **Signer independence** — `adapted to one signer; not signer-independent`; no multi-signer data exists.
- **Single cross-session pair only** — S1+S2 train, S3 test; no other sessions tested. `recordings_s3/` was used (this is the held-out test per this task); `recordings_s2/` was consumed as a dev set in earlier tasks.

## 7. Recommendation

The 8-word + rest vocabulary is strong on the clean S3 held-out test (0.935 acc, 0.933 macro-F1, all 8 words RELIABLE). However:
- The vocabulary was chosen after seeing S3, so these numbers are **slightly optimistic**.
- The FINAL model (S1+S2+S3) is **FINAL-UNEVALUATED** with no held-out data.
- `rest` FPR is **NOT VERIFIED** (no rest clips in S3).
- The model is **adapted to one signer; not signer-independent**.

Before promoting (`SIGNSAGE_MODEL_PROMOTED`), obtain an independent held-out session with rest clips and multi-signer data, and verify rest FPR and cross-signer generalisation. Do not integrate into the app from this session.
