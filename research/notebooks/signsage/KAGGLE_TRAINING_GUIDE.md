# SignSage Dual-Model Notebook — Kaggle Training Guide

Scope: `research/notebooks/signsage/signsage_dual_model_training.ipynb`
Status: EXPERIMENTAL. Not production-ready. `SIGNSAGE_MODEL_PROMOTED=false` preserved.

## 1. Datasets (attach manually via Kaggle "Add Data")

| Model | Dataset | Format | Manifest columns needed | Pose root |
|---|---|---|---|---|
| A (Isolated recognition) | CISLR / Amrita SLR / INCLUDE (if licensed) | `.npy`, `.npz`, `.pose`, raw `.mp4` | `path`, `label`, `video_id`, `signer` (optional `video_ref`) | `ISOLATED_DATA_ROOT` points at dataset folder |
| B (Translation) | GloSL-ISL (`annotations.csv`) | `.pose` files in `poses/` | `pose_file`, `text`, `video_ref`, `split`, `signer` | `TRANSLATION_POSE_ROOT` points at dataset root |
| B (Translation) | iSign (gated HF) | `.pose` named `<uid>.pose` | `uid`, `text` | `TRANSLATION_POSE_ROOT` points at pose dir; format=`isign` |
| B (Translation) | ISL-CSLTR (Mendeley) | raw `.mp4` + sentence mapping | canonical: `pose_path`, `text`, `video_id`, `split`, `signer` | convert first via `convert_isl_csltr_to_canonical_manifest()` |

No automated multi-GB downloads. Attach manually.

## 2. Configure (only edit the Configuration cell)

```
TRAINING_MODE = "pilot" | "full"    # 0 = unlimited; must be explicit
CONFIRM_DATASET_RIGHTS = True
ALLOW_RESEARCH_ONLY_DATASETS = False  # set True only for verified research-only data
TRAIN_RECOGNITION = False
TRAIN_TRANSLATION = False
ISOLATED_MANIFEST = None              # set to Path("/kaggle/input/.../manifest.csv")
TRANSLATION_MANIFEST = None            # set to attached annotations CSV
```

Invalid `TRAINING_MODE` now raises `ValueError` instead of silently falling through to unlimited.

## 3. Run independently

- Model A only: `TRAIN_RECOGNITION = True`, `TRAIN_TRANSLATION = False`
- Model B only: `TRAIN_RECOGNITION = False`, `TRAIN_TRANSLATION = True`
- Both: set both `True`. A preflight failure in one does not stop the other.

## 4. Pilot vs Full

- Pilot (`TRAINING_MODE="pilot"`, default `MAX_SAMPLES_PER_SPLIT=256`): deterministic cap per split; tagged `scope: "pilot_subset"`.
- Full (`TRAINING_MODE="full"`, `MAX_SAMPLES_PER_SPLIT=0`): no cap; tagged `scope: "full_dataset"`.
- `limit_split` enforces minimum `MIN_SPLIT_SIZE=2`; if cap produces smaller, raises `ValueError`.

## 5. Environment check (new cell `environment-report`)

Runs at start of session. Reports Python, torch, CUDA, disk, RAM, Kaggle `/kaggle/input` contents, dependency readiness. Required packages (`torch`, `numpy`) must be present; optional (`sentencepiece`, `pose-format`, `mediapipe`, `opencv-python-headless`) are reported missing without blocking pipelines that don't need them.

## 6. Dataset discovery (new cell `dataset-discovery`)

Scans `/kaggle/input` (and `/kaggle/working`), lists `.csv`, `.pose`, `.npy`, `.npz`, `.mp4`, reports manifest fields/row counts/empty annotations, checks for duplicate pose-file stems. Does not assume dataset folder names.

## 7. Recover after Kaggle interruption

Both pipelines save `training_state.pt` after each epoch. Resume with `RESUME_RECOGNITION=True` / `RESUME_TRANSLATION=True`. Resume verifies dataset fingerprint, model config, label mapping / tokenizer hash match; mismatches raise `ValueError`.

## 8. Checkpoints and reports saved at

```
/kaggle/working/outputs/recognition/
  model.pt, labels.json, metadata.json (format: signsage_isolated_words_v1)
  evaluation_results.json, training_history.json, model_config.json, preprocessing_config.json
/kaggle/working/outputs/translation/
  checkpoint/signsage_transformer_full.pt
  tokenizer/signsage_en_sp.model + tokenizer_metadata.json
  evaluation_report.json, training_history.json, model_config.json
  landmark_contract.json, sign_to_text_deployment.json
```

Exports match backend contracts (`landmark_contract.json`, `sign_to_text_deployment.json`). `SIGNSAGE_MODEL_PROMOTED` unchanged (`false`).

## 9. Metrics (do not confuse training loss with success)

- A: accuracy, macro precision/recall/F1, per-class recall, confusion matrix, unseen-class tracking.
- B: WER, BLEU-4, ROUGE-L, prediction diversity, top prediction frequency, collapse warning. Low loss does not prove translation learned.

Both models: experimental only; promotion requires additional evaluation gates, signer-independent verification, and promotion-eligible checks — none of which are met by a single notebook run.

## 10. Unresolved limitations

- Real Kaggle dataset paths not verified in this environment (no live `/kaggle/input` dataset present).
- Real `.pose` file parsing requires `pose-format`; optional dependency not installed in current session.
- Raw video extraction requires `mediapipe` + `opencv-python-headless`; optional.
- Full training requires attached real dataset with verified license terms.
- Translation tokenizer (`sentencepiece`) works; full dataset evaluation requires real GloSL/iSign/ISL-CSLTR attachment.
