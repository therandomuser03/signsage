# REPORT_K — Temporal-crop robustness of the final word model (cnn-v3, word_v3_8w_s4ref_s123)

**Date:** 2026-11-15 · **Scope:** S1/S2/S3/S4 clips of the 8 demo words + `rest` (197 clips) · **Outcome:** the retrain gate did NOT trigger — no retraining, no `export_final8_crop/`, no backend change.

## Headline caveat (read this first)

S1, S2, S3 and S4 are all in this model's training data (S1+S2+S3 SignSage captures, S4 external reference videos). Everything below is a **robustness test on in-distribution data — it is NOT a generalisation estimate**. A 100% score here proves the model does not fall apart on its own training data under temporal crops; it says nothing about unseen signers, unseen environments, or genuinely hard captures.

Acceptance threshold: softmax max prob ≥ 0.7 (the production word model's threshold). Accepted rate is reported separately from top-1 accuracy. "macro-F1" below is over all 9 classes; S2 and S3 contain no `rest` clips, so their all-9 macro-F1 floor is 0.889 by construction — macro-F1 over present classes is 1.000 for every variant and session.

## 1. Materials built (new files only)

- `crop_utils.py` — hand-active span = first/last frame with any nonzero in the hand blocks (indices 501–521 and 522–542, any component; x/y/z). Crop = span expanded by **±400 ms** using each clip's `effective_fps` from its conversion manifest (frames are not individually timestamped, so ms→frames uses `round`/`floor`/`ceil` of `0.4 × fps`), clamped to the clip; clips with no hands (`rest`) are kept whole; window expanded symmetrically to a **minimum of 12 frames** (no clips were shorter than 12 total).
- `data/cropped/` — the 197 9-class clips (108 S1 + 40 S2 + 40 S3 + 9 S4), names `*__crop400.npy`, combined `manifest.csv` with `source_npy`, `crop_start_frame`/`crop_end_frame`. Frames 2902→2330 (S1), 889→746 (S2), 970→782 (S3), 1605→1523 (S4); 80–95% of original per session.
- `data/cropped_jitter/` — same clips with **seeded ±100 ms jitter on each boundary** (per-clip `default_rng(20261115 + sha1(clip) % 2³²)`, independent of processing order), names `*__crop400j.npy`. 90/197 clips shift window by ≥1 frame (at 4.8–23 fps, 100 ms ≈ 0.5–2.3 frames, so most jitters round away). One clip in S2 hits the 12-frame floor.
- `eval_crop_current.py` — evaluation harness (model, metrics, JSON). `results_crop_current.json` — raw numbers.

## 2. Measured results (current final model, `export_final8_s4/model.pt`, unmodified)

Overall (pooled 197 clips):

| Variant | n | top-1 | accepted @0.7 | acc \| accepted | mean maxprob | macro-F1 (all 9) | macro-F1 (present) |
|---|---|---|---|---|---|---|---|
| uncropped | 197 | **1.000** | 0.990 | 1.000 | 0.9827 | 1.000 | 1.000 |
| cropped (400 ms) | 197 | **1.000** | 0.990 | 1.000 | 0.9834 | 1.000 | 1.000 |
| jittered (±100 ms) | 197 | **1.000** | 0.990 | 1.000 | 0.9834 | 1.000 | 1.000 |

Per session (top-1 / accepted):

| Session | uncropped | cropped | jittered |
|---|---|---|---|
| S1 (n=108) | 1.000 / 0.991 | 1.000 / 0.991 | 1.000 / 0.991 |
| S2 (n=40) | 1.000 / 0.975 | 1.000 / 0.975 | 1.000 / 0.975 |
| S3 (n=40) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| S4 (n=9) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Cropped-set confusion pairs (true → pred): **none — 197/197 correct, off-diagonal sum 0.** Same for jittered.

Below-threshold clips (both correctly predicted): `S1 no__S1__take02…` maxprob 0.553 (pred `no`), `S2 yes__S2__take02…` maxprob 0.584 (pred `yes`). Accepted rate 2/197 = 0.990; identical across variants.

Sanity check that the pipeline sees different inputs: one clip, full vs crop (28f→22f) — features differ with max abs diff 1.93, so decisions are *not* an artifact of identical preprocessing; the model is simply saturated (near-one-hot) on its own training data.

## 3. Decision gate

Spec: if cropped top-1 is **> 3 percentage points** below uncropped, retrain with the same cnn-v3 recipe + a crop augmentation (crop to hand-active span + random pad 0–600 ms, prob 0.5, training-only), compared honestly (S1+S2(+S4) → S3, 5 seeds), then possibly export `export_final8_crop/`.

Measured drop: **0.0 percentage points** (1.000 → 1.000), overall and per session; accepted rate also identical. **The retrain gate did not trigger.** Per the spec, I did not retrain, did not create `export_final8_crop/`, and wrote nothing to the backend. `SIGNSAGE_MODEL_PROMOTED`, `models/word_v3/`, `backend/word_model/`, and all existing files are untouched.

## 4. Caveats and honest limits

- **In-distribution saturation.** 100% on training data means this experiment can only falsify gross brittleness; it cannot measure unseen-signer robustness (S4 is external videos, ~1 clip/word, but S1–S3 are the production signer) or real-world capture variation.
- **100 ms jitter ≈ 0.5–2.3 frames** at these fps, so 107/197 clips are nearly identical to their uncropped counterpart; jitter results carry little independent signal beyond the crop test.
- **`rest` clips** (12 S1 + 1 S4) have no hands, so their "crop" is the full clip — the tightest-crop stress case in this dataset is a real sign clip.
- **Crop ≠ user error.** Auto-mode produces exactly this span (hand-active ± 400 ms) — this test validates that capture path in-distribution. It does not validate sign lengths outside this data, occlusions, or two-hand switching.
- **Accepted-rate figure unchanged (0.990)** — the two sub-0.7 clips are the same as in the uncropped variant; the threshold was not re-tuned.
- Crop copies are mechanical transforms of the same recordings; all sessions remain the model's training data — repeated "eval on S3" here is not held-out evidence.

## 5. Recommendation

Do **not** retrain or replace the production word model on the basis of this result — there is no in-distribution crop penalty to fix. If temporal-crop insurance is still desired, run the (not-executed) crop-augmentation experiment as a separate task — train S1+S2 (+S4) → S3, 5 seeds, current recipe vs crop-augmentation — and only adopt if it is not worse on uncropped S3 within 1 std and better on cropped S3. Out of scope here; the spec gate says no.

For the deployment question this implies: the word model's behaviour on tighter crops is not the weak point; the weak points remain (a) everything outside S1–S4 distribution, which no local experiment can measure, and (b) the two sub-0.7 correct clips, which are threshold-calibration questions, not architecture or augmentation questions.

## 6. Reproduce

```
python build_crops.py              # rebuild data/cropped{,_jitter}/ + manifests
python eval_crop_current.py        # re-measure, overwrites results_crop_current.json
```

Files added by this task: `crop_utils.py`, `build_crops.py`, `eval_crop_current.py`, `results_crop_current.json`, `REPORT_K.md`, `data/cropped/` (197 `.npy` + `manifest.csv`), `data/cropped_jitter/` (197 `.npy` + `manifest.csv`). No existing file modified; backend, frontend, `.env`, `render.yaml`, `SIGNSAGE_MODEL_PROMOTED`, `backend/word_model/`, `models/word_v3/` untouched; nothing committed, staged, pushed or stashed.
