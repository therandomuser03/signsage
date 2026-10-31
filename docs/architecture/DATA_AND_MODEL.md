# SignSage Data & Models

Facts about the dataset, the data contract, what the models may claim, and the
gates a checkpoint must pass. (Consolidated 2026-10-13 from `DATA_MODEL.md`,
`DATASET_STORAGE.md`, `SUPPORTED_ISL_VOCABULARY.md`, `MODEL_READINESS.md`.)

## 1. iSign dataset facts

Source: Joshi et al., *iSign* (ACL Findings 2024), gated `Exploration-Lab/iSign`,
license **CC BY-NC-SA 4.0**. Paper claims 118,228 pairs; the audited v1.1 archive
has **127,237 CSV rows and 127,237 `.pose` files** (one per UID/row, not per
`video_id`). Confirmed columns: `uid`, `text`; `video_id`/`seq` derive from `uid`
by right-split; `n_words` for quality filtering; no per-segment timing columns.

- Split rule: by `video_id` (leakage control). ~1% junk text (`blank`, `[Music]`,
  digits) — filter on `n_words`, not just empties. Only a small subset was
  certified-signer-verified: treat the rest as data-quality risk.
- Rules: never commit dataset, pose archives, or HF caches; never hardcode
  tokens; never silently change a confirmed split/representation.

## 2. Representations

Application/research Sign→Text: first 543 image-space landmarks
(POSE 33 + FACE 468 + LEFT_HAND 21 + RIGHT_HAND 21; trailing world-pose block
excluded), 543×3 = **1629** floats/frame. Exact order, normalization
(`pose_normalize_v1`), temporal convention, and message shapes are pinned in
`reference/end-to-end-contract.md` — order and versions, not just counts.

## 3. Dataset storage & integrity (old laptop = data node only)

Layout: `/signsage-data/isign_2024/` (annotations/videos/labels), `checksums/`
(`*_sha256.txt`, verified with `sha256sum -c`), `metadata/dataset_manifest.json`
(name/version/checksum; bump version on composition change; backend reads it via
`SIGNSAGE_DATASET_VERSION`). Materialized subsets under
`signed_subsets/<run>/` are immutable + carry `subset_manifest.json`.
Artifacts (`.pt`, logs, configs) live under `/signsage-artifacts/` (checkpoints,
logs, evaluations as versioned JSON with wer/em/diversity/samples).

- The 14 `sign_vocabulary.json` IDs are EN→ISL **generation** labels, never
  recognition labels. Helper scripts (`verify_dataset.sh`, `check_manifest.py`)
  stay off-git on the laptop/dev machine.
- Offline policy: laptop unreachable / checksum fail / nothing promoted →
  `MODEL_UNAVAILABLE`, never fabricated text. Kaggle can materialize subsets
  independently (`ssh signsage-laptop.local` over Tailscale when reachable).
- **Never promote checkpoint 352868924** (verified NO-GO: WER 1.0046, EM 0.0,
  diversity 0.99%).

## 4. Supported vocabulary: evidence-bound (isolated-word pilot)

Rule: SUPPORTED requires present-in-training-data + present-in-evaluation +
recognized end-to-end. The isolated-word pilot bundle
(`backend/models/isl_bundle/`, trained on the INCLUDE subset of
vidit031/isl-isolated-40words, revision `255ce7fd`) has measured held-out
results and is servable in dev mode via `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`,
but does NOT meet the promotion bar (80% accuracy / 80% macro recall /
60% per-class recall) and is not signer-independent.

**Supported recognition vocabulary (pilot, not promoted):** friend, hello,
hospital, market, school, thank you — 6 ISL words from real video data.

Measured pilot evaluation (test split, 18 samples):
- Accuracy: 61.1%, macro recall: 61.1%
- Per-class recall: friend 0.0%, hello 100%, hospital 66.7%, market 100%,
  school 33.3%, thank you 66.7%
- Status: NO-GO, promotion_eligible: false, signer_independent: false
- Inference latency: ~2 ms/sample on CPU

The original transformer checkpoint 352868924 remains NO-GO (WER 1.0046,
EM 0.0, diversity 0.99%) and cannot be recovered for ISL recognition — it
was trained on synthetic iSign text→pose data (news headlines mapped to
generated poses), not real ISL video.

- 352868924 eval: EM 0.0 — no training-domain phrase recognized. T5 POC: no
  weights. ISLRTC PDF + alphabet chart: learning references, never labels —
  do not enumerate them as capabilities. The 14 avatar IDs are the wrong
  direction (generation, not recognition). Inputs are always dynamic sequences
  (`(T,543,3)`, T≥2); no static-sign classifier exists for recognition.
- UI honesty: no "supported signs" language beyond the pilot vocabulary above,
  no "try signing X" prompts, no alphabet chart in the recognition flow until
  a SUPPORTED row gains evidence.

## 5. Model readiness: can anything serve? No.

| Checkpoint | Measured | Verdict |
|---|---|---|
| 352868924 `signsage_transformer_full.pt` | WER 1.0046, BLEU 0.24, ROUGE-L 0.089, EM 0.0, diversity 0.99%, top-pred 44.8% (`research/evaluation/reports/352868924/`, `research/manifests/models/signsage-s2t_v352868924.yaml`) | BROKEN translator / WORKING negative reference. Loads, deterministically reproduces collapse ("the police said that"). Stays unpromoted. Cannot be recovered for ISL recognition — trained on synthetic iSign text→pose data, not real ISL video. |
| Isolated-word pilot (`backend/models/isl_bundle/`) | INCLUDE 6 words, 122 samples; test accuracy 61.1%, macro recall 61.1%, per-class recall: friend 0.0%, hello 100%, hospital 66.7%, market 100%, school 33.3%, thank you 66.7%; inference ~2 ms/sample CPU | EXPERIMENTAL PILOT. Servable in dev via `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`. NOT promoted — below 80/80/60 thresholds, not signer-independent. |
| ISLiM (`ISLiM_4Loss_best_for_10kSpamoOF.pt`, `ISLiM_full_posestitch_pre_processed.pt`) | **Not integrated**, **Blocked** (94-landmark input vs adapter 543-landmark contract), **Experimental**, **Verified** (load only), **Report but not independently verified** (BLEU-4 5.44 vs paper 5.85). iSign dataset: research-only (`research/pretrained_models/islim/MODEL_AUDIT.md`). | Not promoted. `SIGNSAGE_MODEL_PROMOTED=false`. |
| Sanket (`model_best.pth`) | **Not integrated**, **Blocked** (dual-stream video+pose vs adapter pose-only), **Experimental**, **Verified** (load with `weights_only=True`, 541 keys), dataset license **unverified**, metric **not independently verified** (BLEU-1 50.6% not comparable to BLEU-4). MIT model license (`research/pretrained_models/sanket/MODEL_AUDIT.md`). | Not promoted. `SIGNSAGE_MODEL_PROMOTED=false`. |
| SignTransformer V2 (planned) | No training yet (`research/training/READINESS.md`: pre-norm d256/4L, SP-16k, ≤12ep) | EXPERIMENTAL |
| T5-small POC (61,605,888 params) | Synthetic-gate EM 0.0→0.094 vs 0.5 → `WIRING-FAIL` (`research/evaluation/t5_poc_gate/gate_report.json`); unloadable via `SignToTextAdapter` (32k vs 16k vocab) | EXPERIMENTAL, research-only |
| External `t5-small-motion` | 0/1887 EM (see `EXTERNAL_EVALUATION.md`) | NO-GO, never promoted |

Tokenizer incompatibility is architectural, not a bug: serving track is SP-16k
(`PAD0/UNK1/BOS2/EOS3`); T5 track needs `google-t5/t5-small` 32k for pretrained
knowledge. A T5 model would need its own adapter/contract/export — none exist.

Kept-from-post-mortem gates: WER/BLEU-1..4/ROUGE-L/EM + trivial mode-copy
baseline (`beats_trivial_bleu4`) + diversity (`collapsed<0.05` STOP) +
sensitivity (swap/zero/shuffle → `pose_conditioned`) + qualitative examples +
export/load verification (`scripts/model_check.py` zero-motion-must-reject).
T5 track adds a binding 64-sample overfit gate (EM ≥ 0.95 or hard STOP).
Loss decrease, fluent text, model load, or an overfit-gate pass alone never
constitute success.

Promotion requires, in order: (a) real weights beating trivial + diversity_ok on
held-out data; (b) `export_bundle` → `checkpoint/ + tokenizer/ +
landmark_contract.json + sign_to_text_deployment.json`; (c) `model_check.py`
pass; (d) `BUNDLE_DIR` + `PROMOTED=true` + green adapter tests. None exist today;
honest backend state is `MODEL_UNAVAILABLE`. Procedure details live in
`backend/models/README.md` (weights never in git; `SIGNSAGE_ALLOW_UNPROMOTED_DEV`
is dev-only, never on Render). License gate (§6) joins the quality evidence.

## 6. License gate (review before any public/commercial use)

Code is MIT, but iSign-trained **weights derive from CC BY-NC-SA 4.0 data** and
do not inherit MIT terms. Before any public/commercial checkpoint deployment,
record: (1) whether NC/SA permits the use or a separate agreement is needed;
(2) whether serving predictions (incl. Render bundle download) counts as
commercial use; (3) attribution/ShareAlike obligations for bundle + adaptations;
(4) whether the dataset-derived tokenizer needs the same treatment. Local
research (train/eval/dev inference) is not blocked; presenting a checkpoint as
commercially cleared without this review is.

## 7. Pretrained research candidates (ISLiM, Sanket) — current status
Both are **Not integrated**, **Not promoted**, and must not be promoted until adapter contracts, licensing, and evaluation evidence are independently verified. See `research/pretrained_models/PRETRAINED_MODEL_COMPARISON.md`. Existing iSign T5-small-motion (`models/iSign_t5_small_motion/`) remains a separate production model path and is preserved unchanged. `SIGNSAGE_MODEL_PROMOTED=false` remains unchanged.
