# SignSage — training, export and inference

Everything below was checked against the source in this repository. Where something was **not**
run (no Kaggle session, GPU or real iSign data were available while writing this) it says so.

## 0. Read this first

| Question | Answer |
|---|---|
| Does the pipeline work end to end? | Plumbing: **yes, verified on synthetic data** (`--smoke` → tokenizer → gate → train → evaluate → export → real backend adapter loads it and decodes). **Not run on real data.** |
| Will the Kaggle notebook train on all 123,779 samples? | **No, and it will not silently pretend to.** Measured in your 21-Oct run: streaming = ~1.0 s/sample ≈ 32 h per epoch (~263 h for 8 epochs, `FEASIBILITY: FAIL`); a full local cache is ~95–143 GiB. The notebook trains a labelled **PILOT** subset (2,500 per split in that run). |
| Will the pilot model translate sign language? | **Do not expect it to.** The pilot proves the pipeline. The earlier *full-dataset* run (`352868924`, ~99.7k train samples) was **NO-GO**: WER 1.0046, 0.99 % unique outputs (collapse) — see `backend/models/README.md`. A 2,500-sample pilot has far less data than that run. |
| Can I put the result in front of users? | **No.** The backend now refuses a pilot or smoke bundle when `SIGNSAGE_MODEL_PROMOTED=true`. Promotion needs a full-dataset model whose `evaluation_results.json` shows `gate.GO = true`, plus your own review. |
| Does it accept camera video? | **No.** It is a pose→text model: input is MediaPipe Holistic landmarks `(T, 543, 3)`. The existing frontend (`frontend/src/features/translate/landmarks.ts`) already produces them. |

## 1. Scopes — never conflate them

| Stage | What it is | A pass proves | A pass does **not** prove |
|---|---|---|---|
| Smoke (`--smoke`) | CPU, synthetic poses | script plumbing | any model quality |
| Overfit gate | memorise 64 reachable *train* samples, ≥ 95 % exact | model + optimiser + tokenizer can learn | generalisation |
| **Pilot** (the notebook) | train + held-out test on a per-split subset (`auto_subset.json`) | pipeline validated on a small subset | that it is the full-dataset model |
| Full dataset | 98,485 / 13,014 / 12,280 train / val / test (21-Oct audit; 123,779 usable) | — | — (not feasible on Kaggle, §7) |
| Evaluate / export | held-out **test** split of the *same* scope → deployment bundle | quality of *that scope only* | production readiness |

The exported bundle records `training_scope` (`pilot_subset` / `full_dataset` / `smoke_synthetic`), the
real row counts used and available, and the seed (`sign_to_text_deployment.json → training_config.data_counts`,
`evaluation_results.json → scope / rows_evaluated`, checkpoint `model_config`).

## 2. What you must provide

| Item | Where | Notes |
|---|---|---|
| Kaggle accelerator | notebook sidebar → Session options → Accelerator = **GPU T4 x2** (or P100) | The `.ipynb` metadata does **not** select it. The RUN SCOPE cell stops if no GPU is visible. The script uses one GPU (`cuda:0`). |
| Internet | ON | pose archive is read over HTTP Range requests |
| Secret `HF_TOKEN` | Kaggle → Add-ons → Secrets | needed for `--pose-source remote` (not for a local zip) |
| Secret `GITHUB_TOKEN` | same | read access to the private repo (checkout cell) |
| iSign CSV (`iSign_v1.1.csv`) | attached Kaggle dataset (+ `ISIGN_DATASET_SLUG`) **or** downloaded via `HF_TOKEN` | the provisioning cell resolves it into `CSV_PATH` |
| Prior session output (optional) | attach as input; `PRIOR_OUTPUT_SLUG` if ambiguous | the restore cell verifies it before reuse |

## 3. One-time: get the patched code into git and pin it

The notebook verifies the **content** of the checked-out training script
(`EXPECTED_TRAIN_SCRIPT_SHA256`) as well as the commit, and stops on any mismatch.

```bash
git apply signsage_changes.patch          # or copy the changed files over your tree
sha256sum research/experiments/train_sign_to_text.py
#   must equal EXPECTED_TRAIN_SCRIPT_SHA256 in notebook cell "checkout" (printed in the change report)
git add -A && git commit -m "SignSage: fail-closed cache/gate, bounded retries, atomic checkpoints, scope labelling"
git push
git rev-parse HEAD                        # full 40-char hash  ->  PINNED_SHA in the checkout cell
```

Until `PINNED_SHA` points at a commit containing exactly that script, the checkout cell raises
`PINNED_SHA … contains a different train_sign_to_text.py`. That is deliberate.

## 4. Running the notebook (cell order)

Run top to bottom. Check these messages before moving on:

| Cell | Healthy output | If not |
|---|---|---|
| checkout | `repo commit: <PINNED_SHA>`, `training script sha256: …` (no error) | see §3 |
| restore | `FRESH RUN` or `pose cache: VERIFIED …` | fix slug / re-attach prior output |
| provisioning, preflight, pose-source preflight | `HF_TOKEN available…`, preflight JSON written | token / network |
| **RUN SCOPE + ACCELERATOR** | `accelerator (nvidia-smi): ['Tesla T4, …']`, `cuda available: True`, `RUN_SCOPE = PILOT` | enable the GPU, restart session |
| audit, **measure** | `AUTO-SELECTED SUBSET …` (or a FIT verdict) | — |
| certify pilot | certification passes for that subset | — |
| **PREPARE** | `pose cache complete`, exact `N/N` | re-run (resumable); never proceeds with `subset<=0` |
| **OVERFIT GATE ONLY** | `PRE-CHECKS PASSED …`, then `OVERFIT GATE (standalone) PASSED …` | the message names the failing check; nothing was relaxed |
| **V2 TRAINING** | strict cache gates pass; log shows `pose cache: N/N samples cached` and `training pose source: LOCAL CACHE` | `LOCAL POSE CACHE INCOMPLETE` ⇒ prepare/training used different `--seed/--subset`; re-run PREPARE |
| evaluate + export | bundle written | read the CLASSIFICATION line |
| PERSIST → **Save Version** | sizes printed, no secret-named files | — |
| **FINAL VERIFICATION** | `EXPORT VERIFIED (structure/consistency)`, scope banner, held-out pilot metrics | raises if any stage failed or artifacts disagree |

Failure behaviour: a failed prepare/training/evaluation/export is recorded, the PERSIST cell still runs (so caches and
checkpoints are kept), and the FINAL VERIFICATION cell then **raises** so Kaggle marks the run failed. Whether Kaggle keeps
`/kaggle/working` for a failed *version* is **not verified** — Save Version from the interactive session before closing.

**Time/quota — what is and is not known.** From the 21-Oct run: prepare of 7,500 samples ≈ 25 min (`--jobs 8`), certify
≈ 36 min, gate (when streaming) ≈ 34 min. Pilot **training time was never measured** (that run stopped at the gate); the
measure cell prints its own projection. No finish-time guarantee is made.

## 5. Resuming after an interrupted session

Recovery checkpoints are written **atomically** once per epoch (`checkpoints/signsage_transformer_recovery.pt`), so a killed
session cannot leave a truncated file; at most the current epoch is lost.

1. Save Version (output persists `/kaggle/working`), start a new session, attach that output as an input dataset.
2. Run checkout → restore → provisioning (and PREPARE if you want its cheap re-verification).
3. In the V2 TRAINING cell set `RESUME_TRAINING = True` and run it. A fresh start (`False`) **refuses** to overwrite an
   existing recovery checkpoint; `True` refuses if there is nothing to resume.

## 6. Output artifacts (`/kaggle/working/s2t_small6/`)

```
signsage_deployment_bundle/              <- what the backend loads
  checkpoint/signsage_transformer_full.pt      weights + model_config (incl. training_scope) + vocab
  tokenizer/signsage_en_sp.model               SentencePiece model
  landmark_contract.json                       input/normalisation contract
  sign_to_text_deployment.json                 architecture, decoding, training_config.data_counts
  evaluation_results.json                      test_metrics, trivial_baseline, diversity, gate.GO, scope, rows_evaluated
signsage-s2t_v<run_id>.yaml              model manifest (status: PILOT … NOT FOR PRODUCTION)
bundle_verification.json                 sha256 of checkpoint/tokenizer/contract + scope (consistency only)
gate_report.json                         overfit gate: passed, status, quality_verdict, final.exact_constrained
checkpoints/…_recovery.pt, …_full.pt     per-epoch recovery / best
pose_cache/, tokenizer/                  reusable caches (verified before reuse)
subprocess_stdout_stderr.log, overfit_gate_stdout_stderr.log     redacted logs
auto_subset.json, feasibility.json       scope + feasibility record
```

## 7. Full-dataset training (not on Kaggle)

Why not: ~32 h/epoch streaming, ~95–143 GiB cache, and an epoch longer than a session can never write its per-epoch
recovery checkpoint. The notebook's `RUN_SCOPE` cell raises if you change it from `PILOT`.

The code supports a **local pose ZIP** (`--pose-source <file>.zip` → `ZipPoseSource`; `HF_TOKEN` is only required for
`--pose-source remote`). **Not run end to end by the author of this guide** — treat as a recipe to validate:

```bash
# 1. On a machine with a GPU and ample disk (roughly 150-200 GB free; the archive is ~100-150 GiB class):
#    download the four byte-range parts of ONE zip from the Hugging Face dataset Exploration-Lab/iSign
#    (iSign-poses_v1.1_part_aa ... _ad) and iSign_v1.1.csv, then join the parts:
cat iSign-poses_v1.1_part_a? > iSign-poses_v1.1.zip

# 2. Honest feasibility on THAT machine first (writes feasibility.json; no training):
python research/training/train_sign_to_text.py --csv iSign_v1.1.csv --pose-source iSign-poses_v1.1.zip \
    --out runs/full --stages measure --measure-samples 2000 --seed 42

# 3. Train (same seed everywhere; resumable; per-epoch atomic recovery checkpoints):
python research/training/train_sign_to_text.py --csv iSign_v1.1.csv --pose-source iSign-poses_v1.1.zip \
    --out runs/full --stages audit,tokenizer,overfit,train,evaluate,export \
    --data-mode stream --seed 42 --epochs 8 --batch 32 --max-hours 24 --resume
```

The training stage refuses a feasibility `FAIL` unless you pass `--force` (loudly logged). The bundle from this run is
labelled `full_dataset`; it is promotion-*eligible* only if `evaluation_results.json → gate.GO` is true.

## 8. Using a bundle in SignSage

The backend already contains the full inference path — nothing to rewrite:

* Input: `POST /api/v1/translation/sign-to-text` with `{"frames":[{"landmarks":[[x,y,z] × 543]}, …], "min_motion", "min_confidence"}`
  (2–640 frames), or the live WebSocket `/ws/translate`. The frontend already builds these frames.
* Model: `app/ml/adapters/sign_to_text_adapter.py` (`SignToTextAdapter.load(bundle).transcribe(landmarks)`), which applies the
  same preprocessing as training, rejects non-canonical input, and withholds output on `insufficient_motion` / low confidence.

```bash
cd backend
export SIGNSAGE_MODEL_BUNDLE_DIR=/abs/path/to/signsage_deployment_bundle
export SIGNSAGE_ALLOW_UNPROMOTED_DEV=true      # REQUIRED for a pilot bundle; promoted mode refuses it
python -m scripts.model_check                   # existing self-test against the configured bundle
python -m scripts.predict_landmarks --bundle "$SIGNSAGE_MODEL_BUNDLE_DIR" --input landmarks.npy
```

`predict_landmarks` (new) prints one JSON object (`text, accepted, reason, confidence, motion_energy, model_scope, warning`);
exit 0 = ran, 2 = malformed input (wrong shape/dtype, corrupt or unsupported file), 1 = bundle could not be loaded.
`confidence` is a decoding-probability proxy, **not** a calibrated accuracy.

Keep `SIGNSAGE_MODEL_PROMOTED=false` until a full-dataset bundle clears the gate and you have reviewed it. Commit the
manifest YAML under `research/manifests/models/` and keep bundles outside `backend/models/` (see its README).

## 9. What was changed and how it was checked

| Change | Why (evidence) |
|---|---|
| `RemoteConcatFile._probe`: bounded retry (5 attempts, Retry-After capped at 60 s), typed `RemoteTransportError` | 21-Oct traceback: unretried `_probe` reset inside a DataLoader worker |
| `--data-mode cache` refuses an incomplete cache for overfit/train/evaluate/export | silent fallback to remote streaming (471/7,500 cached) |
| Atomic checkpoint writes + readable resume error | a killed session could truncate the recovery file |
| Gate report: `status` (`passed`/`failed`/`interrupted`), `quality_verdict`, `pose_source` | an interrupted gate previously looked like a quality failure |
| Bundle/eval record `training_scope`, real counts, seed; smoke labelled | `data_counts` was read but never set; smoke bundles looked like normal ones |
| Backend: adapter exposes `training_scope`; registry refuses pilot/smoke when promoted | stops a pilot model reaching users |
| New `scripts/predict_landmarks.py` | file-based inference entry point with clean exit codes |
| Notebook: gate cell, RUN SCOPE + GPU check, pin-by-content, one seed, no duplicate evaluate/export, guarded resume, targeted redaction, FINAL VERIFICATION | see change report |

Tests: `python -m pytest research/experiments/test_pipeline_hardening.py research/experiments/test_notebook_final_structure.py -q`
and, from `backend/`, `PYTHONPATH=../research/experiments python -m pytest tests -q --ignore=tests/ml`.
Known pre-existing failures: `test_split_pipeline` (needs the iSign CSV), `test_notebook_restore_slug_strictness_live`
(fixture lacks a full prior-output), and the 17 `test_v2_data_workflow` functions that are written for their own runner
(`python research/experiments/test_v2_data_workflow.py`), not pytest fixtures.
