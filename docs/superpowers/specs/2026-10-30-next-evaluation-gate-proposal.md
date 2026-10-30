# SignSage — Next Evaluation Gate Proposal

Status: proposal only. No experiment launched, no weights/deployment/promotion changes.

## Goal
Run a bounded, reproducible sign→text evaluation after the diagnostic overfit gate passes, recording enough provenance to justify (or deny) a promotion decision. Default verdict: **NO-GO** (`SIGNSAGE_MODEL_PROMOTED=false` must remain unchanged until evidence overrides it).

## Reproducible configuration (to be recorded in the run report)
| Item | Source / how captured |
|------|-----------------------|
| Dataset subset identity & split manifest | `kaggle_input_v4/s2t_small6/auto_subset.json` (`subset`, `subset_seed`) + `pose_cache/manifest.json` (version/count/uid_to_shard), validated identically to `kaggle_diagnostic.validate_input` (CSV-derived expected UIDs, never self-compared). |
| Split rule | Documented `video_id` 80/10/10 split (`audit_dataset.video_split`), so val/test contain videos never seen in training (genuinely unseen videos). |
| Random seed | `--seed` (fixed, logged in `training_history.json` and `gate_report.json`). |
| Tokenizer fingerprint | `train_sign_to_text._sp_fingerprint(sp, model_path)` (SHA-16 of model bytes + vocab size + special-token IDs), recorded per checkpoint (`ckpt_tokenizer_fp`) and in the export bundle `model_config`. |
| Source revision | **Not currently captured.** Proposed: add `git rev-parse HEAD` (and `git status --porcelain`) to the run report / export bundle so every run is tied to an exact source revision and dirty-state is disclosed. |
| Training/validation loss by epoch | Already recorded: `training_history.json` entries `{epoch, train_loss, val_loss, val_nll, ...}`. |
| Exact-match accuracy | `evaluation_metrics.compute_metrics` → `exact_match` (normalized ref vs hyp), plus per-epoch diagnostic `diagnostic_metrics`. |
| WER / BLEU / ROUGE-L | `compute_metrics` → `WER`, `BLEU` (sacrebleu, effective order), `BLEU-1..4`, `chrF`, `TER`, `ROUGE-L`. All scales documented in `METRIC_SCALES`. |
| Unique-prediction fraction | `evaluation_metrics.diversity_metrics` → `unique_fraction` (also `top_prediction_freq`, `distinct_1`, `distinct_2`, `collapsed`). |
| Prediction examples for manual review | Already recorded: `training_history.json` `validation_samples` (16 `{uid, ref, hyp, conf}` per epoch) + `gate_report.json` `failures` (ref/hyp/ids/lengths/EOS per missed sample). |
| Gradient norm & clipping statistics | Already logged per epoch: `grad_norm_mean`, `grad_norm_max`, `grad_clip_frac`, `param_norm` (from `nn.utils.clip_grad_norm_` with `GRAD_CLIP`). |
| Non-finite value checks | Loss guarded (`run_epoch` raises `RuntimeError("Non-finite training loss.")` when `not torch.isfinite(loss)`); `compute_metrics` rejects NaN/Inf metrics. **Gap:** no explicit non-finite guard on the recorded gradient norm — proposed addition: reject/flag a run when any `grad_norm` is non-finite. |
| Checkpoint provenance | `export_bundle` writes `model_config` (run_id, subset, subset_seed, vocab, architecture, `is_pilot`, `training_scope`) plus checkpoint `tokenizer_fingerprint`, `initialization_config`, `training_args`. |
| Explicit NO-GO default | `evaluation_metrics.evaluate_gate` returns `"GO": False` unless ALL criteria hold (beats trivial WER/BLEU/chRF, exact_match >= trivial baseline, diversity not collapsed, nonempty >= 90%). Default is NO-GO; promotion requires a passed gate. |

## Unseen videos vs unseen signers
- **Unseen videos: supported.** The `video_id` split puts whole videos into exactly one of train/val/test, so evaluation runs on videos the model never trained on. Verified in `audit_dataset.video_split` (disjoint-set assertions) and `kaggle_diagnostic.validate_input` (independent UID identity from CSV/subset).
- **Unseen signers: NOT established.** `iSign_v1.1.csv` has only `uid` and `text` columns — `audit_dataset._signer_column` finds no `signer_id`/`signerid`/`signer` column, so `video_split` reports `signer_column: None` and a `signer_note` that independence is "assumed from video-level split only; confirm on the training host." Per the repo's confirmed evidence and STATUS_AND_ROADMAP, signer independence is **not measured** and must **not** be claimed. To evaluate unseen signers, signer metadata must be added to the CSV (or obtained from the training host) and `require_signer_independent` enforced before any signer-independence claim.

## Experiment command (proposed, not launched)
After the diagnostic fix (`research/experiments/kaggle_diagnostic.py` stages `tokenizer,overfit,train`), the next gate would run the diagnostic to a fresh output dir, then run full `train_sign_to_text.py` with stages `tokenizer,prepare,overfit,train,evaluate,export` on a fixed seed/subset, writing `training_history.json`, `gate_report.json`, and the deployment bundle. This is explicitly **not** launched in this task (no GPU quota, no full training).

## What is verified vs not run
- Verified by tests: diagnostic command construction now includes the `tokenizer` stage; `train_tokenizer` creates `signsage_en_sp.model` from scratch when missing (12/12 `test_kaggle_diagnostic.py`, 4/4 `test_tokenizer_fingerprint_guard.py`).
- Not re-run end-to-end: the Kaggle diagnostic subprocess / full training (would require GPU and the mounted dataset; pilot overfit result 96.9% / 98.4% at step 100 stands as prior evidence only — an overfit pass is not proof of generalization).