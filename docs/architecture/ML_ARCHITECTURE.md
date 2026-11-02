# ML Architecture

Source: `docs/ARCHITECTURE/DATA_AND_MODEL.md`, `docs/development/DEVELOPMENT.md`, `backend/app/ml/`, `research/evaluation/`.

## Input / representation
- MediaPipe Holistic: `(T, 543, 3)` per sequence. Components: POSE 33 (0-32), FACE 468 (33-500), LEFT_HAND 21 (501-521), RIGHT_HAND 21 (522-542). Frontend `landmarks.ts` confirms; backend `landmark_contract.json` pins.
- Normalization: `pose_normalize_v1` (hip-midpoint center, scale by median body-33 std, `nan_to_num→0`). Must match training export exactly (B4 blocker).
- Sequence handling: max 160 frames; rolling window 80 / stride 20; `SequenceTooLongError` at 640 cap.

## Adapter contracts (verified load, not promoted inference)
| Adapter | Input contract | Status | Evidence |
|---------|---------------|--------|----------|
| `ImageClassifierAdapter` (`alphabet`) | `(3, 64, 64)` RGB image | VERIFIED (87.1% test accuracy) | `test.npz`, endpoint `/api/v1/classification/image?task=alphabet` |
| `ImageClassifierAdapter` (`word`) | `(3, 64, 64)` RGB image | UNPROMOTED (19.0% test accuracy) | `research/image_classification/artifacts/words/` |
| `IsolatedWordAdapter` | `(T, 543, 3)` → 225 features (pose+hands), resampled 64 frames | EXPERIMENTAL (61.1% accuracy, 6 words, unpromoted) | `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §4 |
| `SignToTextAdapter` | `(T, 543, 3)` → `(160, 1629)` → model input | BLOCKED (adapter works; no promoted model feeds it) | `backend/app/ml/adapters/sign_to_text_adapter.py`; `SIGNSAGE_MODEL_PROMOTED=false` |
| `i_sign_t5_adapter` (`research/evaluation/islim_inference_verification/`) | `(B, T, 94, 4)` float32, 128 frames | NOT INTEGRATED (`docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §1) | External adapter; 543→94 conversion partial (face order unverified) |

## Tokenizer tracks (architectural, not bug)
- Serving: SP-16k (`PAD0/UNK1/BOS2/EOS3`). Bundle: `tokenizer/signsage_en_sp.model` + `.vocab`.
- T5 track (`google-t5/t5-small`, 32k vocab): separate adapter/export required; `t5_poc_gate` synthetic gate: `WIRING-FAIL`. `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §1: no compatible Sanket inference source located.
- Tokenizer incompatibility is preserved by design; do not cross-load.

## Checkpoint evaluation (verified, not fabricated)
- 352868924 (`signsage_transformer_full.pt`): NO-GO. WER 1.0046, BLEU 0.2429 (below trivial baseline 0.3964), ROUGE-L 0.089, EM 0.0, diversity 0.0099 (120/12153 unique), top-prediction 44.84% (`"the police said that the government has been arrested."`). `docs/MASTER.md` §4; `backend/models/README.md`; `research/manifests/models/signsage-s2t_v352868924.yaml`.
- Isolated-word pilot (`backend/models/isl_bundle/`): 122 INCLUDE clips from `isl-isolated-40words`, CC-BY-4.0, 6-word vocab (`friend`, `hello`, `hospital`, `market`, `school`, `thank you`). Test accuracy 61.1%, macro recall 61.1%, per-class recall: `friend` 0.0%, `hello` 100%, `hospital` 66.7%, `market` 100%, `school` 33.3%, `thank you` 66.7%. Not signer-independent; below promotion thresholds (80/80/60). `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §4; `docs/ARCHITECTURE/DATA_AND_MODEL.md` §4.
- ISLiM (`research/pretrained_models/islim/`): Checkpoints `ISLiM_4Loss_best_for_10kSpamoOF.pt` (859MB, epoch 500, BLEU-4 5.44, val_loss not present in metadata, val_results bleu1=13.89/bleu2=8.26/bleu3=6.34/bleu4=5.44/rouge_l=13.38/n_valid=1001) and `ISLiM_full_posestitch_pre_processed.pt` (859MB, epoch 103, val_loss=3.1433, corpus_bleu=0.0/sent_bleu_avg=7.71/ps_b1=22.54/ps_b2=13.25/ps_b3=8.8/sent_bleu_p50=4.06/sent_bleu_zero=1277/1001). Both load (`torch.load(..., weights_only=False)`, 209/208 state_dict entries, 124.3M params). No `keypoints.pt` paired input locally; only synthetic smoke tests executed (NOT verified translations). Adapter contract: 94 landmarks (not adapter's 543). `docs/reports/ALTERNATIVE_ISL_MODEL_AUDIT.md`; `docs/evaluation/islim_inference_verification/ISLIM_INFERENCE_VERIFICATION_REPORT.md`.
- Sanket (`research/pretrained_models/sanket/`): `model_best.pth` (1.4GB, 541 keys, MIT license). Dual-stream (TimeSformer video + 195/258 pose features + T5-base decoder). INCLUDE dataset (~4284 videos, 250 categories). BLEU-1 50.6% (not comparable to ISLiM BLEU-4). Not adapter-compatible (video input required). `docs/reports/ALTERNATIVE_ISL_MODEL_AUDIT.md`; `research/pretrained_models/sanket/MODEL_AUDIT.md`.
- Sid-V5 (`Sid-V5/isl-translator`): ST-GCN + Transformer + CTC, 543-point MediaPipe input compatible. `checkpoints/best.pt` NOT directly downloadable (`curl` 404); requires `git clone` + `git lfs` or release inspection. Reported WER 30.77% (continuous gloss recognition, not sentence translation quality). Separate `IndicTrans2` stage required for gloss→English. MIT repo license. Not integrated; not promoted. `docs/reports/ALTERNATIVE_ISL_MODEL_AUDIT.md`.
- External `t5-small-motion`: `docs/evaluation/EXTERNAL_EVALUATION.md` — 1887 real iSign samples, exact match 0/1887, mode collapse 21.3% (`"isn't that amazing!"` dominant), unique ratio 0.1579 (298/1887). NO-GO (`SIGNSAGE_MODEL_PROMOTED` unchanged). `metrics.json` verified.
- T5 POC: `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §1; `t5_poc_gate/gate_report.json` = `WIRING-FAIL`.

## Model readiness gates (promotion only on evidence)
Per `docs/ARCHITECTURE/DATA_AND_MODEL.md` §5; `docs/reports/SIGN_RECOGNITION_EXPANSION_PLAN.md` §7; `docs/development/READINESS.md`.
- Quality: exact-match ≥70% AND ≥20pp above trivial baseline; per-class recall ≥60% (pilot) / ≥50% (expanded); diversity unique_fraction ≥0.5; overfit gate (64 train samples ≥95% exact); collapse gate (`unique_fraction < 0.05` after epoch 2 → STOP); `SIGNSAGE_MODEL_PROMOTED` stays `false` until a valid `evaluation_results.json` shows `gate.GO`.
- Structural: `export_bundle()` produces `checkpoint/` + `tokenizer/` + `landmark_contract.json` + `sign_to_text_deployment.json`; `model_check.py` exits 0 (load + zero-motion reject); adapter tests pass.
- License: iSign weights derive from CC BY-NC-SA 4.0 data; commercial deployment requires separate license review (`docs/MASTER.md` §6; `docs/reports/ALTERNATIVE_ISL_MODEL_AUDIT.md` §E).
