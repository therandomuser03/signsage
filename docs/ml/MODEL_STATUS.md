# Model Status

**Last reviewed:** 2026-11-02  
**Production promotion:** `SIGNSAGE_MODEL_PROMOTED=false`

This document is the concise decision record. Detailed evidence remains in the linked audit reports.

## Current decision

**No continuous ISL-to-English model is approved or promoted.** Sid-V5 is the next research candidate to investigate, not a selected production model.

| Model / feature | Current status | Decision |
|---|---|---|
| SignSage Transformer checkpoint `352868924` | NO-GO; reported collapse and poor evaluation | Rejected for serving |
| ISLiM | Research-only; execution was verified, but paired local evaluation was unavailable and its input contract differs from the current 543-point pipeline | Not approved for continuous translation |
| Sanket | Pretrained checkpoint was verified, but its training task is isolated-sign/category translation using INCLUDE, not continuous sentence translation | Removed from local model assets; not a candidate for the required task |
| Sid-V5 | ST-GCN + Transformer + CTC; repository reports 30.77% WER for continuous gloss recognition; `checkpoints/best.pt` was not downloadable from the tested raw GitHub URL (HTTP 404) | **Next candidate to investigate; checkpoint and full translation path unverified** |
| ISH-NEWS | Sentence-level ISL-to-English research with EfficientNetB7 + Transformer; pretrained weights not verified as publicly released | Research lead only |
| Multilingual Pose-Based Transformer | Paper reference found; usable checkpoint/task details not established in the audit | Not adopted |
| Static alphabet image classifier | Verified feature; 26 A–Z classes, reported 87.1% test accuracy in project evidence | Retain for the image-classification demo |
| Static word image classifier | Reported 19.0% test accuracy | Do not present as reliable |
| Isolated-word video recognizer | Six-word pilot, reported 61.1% test accuracy; unpromoted and not signer-independent | Experimental only |
| English-to-ISL gloss/avatar path | Deterministic mapping over a bounded 14-ID vocabulary | Demonstrable within its documented vocabulary and asset limits |

## Sid-V5: what is and is not established

- **Architecture claim:** ST-GCN + Transformer + CTC, with a 543-point MediaPipe input described by its repository.
- **Primary output:** a gloss sequence. CTC gloss recognition is not direct English sentence translation.
- **Reported metric:** 30.77% WER is a continuous recognition result, not an English translation metric.
- **Checkpoint access:** the tested raw GitHub URL returned HTTP 404. Git LFS, release assets, or author-provided access have not yet been verified.
- **English output:** requires a separate gloss-to-English stage. IndicTrans2 is a proposed route, not a verified ISL gloss translator.
- **Integration:** requires a separate adapter/pipeline, graph preprocessing, CTC decoding, and evaluation. It is not a drop-in replacement for the existing ISLiM adapter.
- **Gate:** do not integrate until checkpoint provenance, exact architecture, licensing, reproducible gloss recognition, and end-to-end English translation quality are verified.

## Local asset cleanup result

According to [MODEL_ASSET_CLEANUP_REPORT.md](../reports/MODEL_ASSET_CLEANUP_REPORT.md), four files totaling 2,696,106,774 bytes (reported as 2.51 GB) were deleted. These included Sanket `model_best.pth`, ISLiM checkpoint B, and two unused ISLiM mobile export files.

The ISLiM checkpoint A was **retained**, because the report identifies it as the default dependency of the existing `/translation/islim/translate` and `/translation/islim/translate-video` endpoints. Its presence does not mean the model is approved or promoted. The report also records a stale `ALT_CKPT` reference to the deleted checkpoint B; do not use that alternate path.

The alphabet and word image-classifier bundles were retained. The word classifier remains low-accuracy and unpromoted.

## Promotion requirements

Do not set `SIGNSAGE_MODEL_PROMOTED=true` until all applicable gates pass:

1. Checkpoint source, version, configuration, and license are verified.
2. Model architecture and preprocessing match the checkpoint.
3. Evaluation uses real paired inputs and references from an appropriate dataset.
4. Translation quality, diversity, failure behavior, and baseline comparisons are reported.
5. Signer-independent evaluation is performed or its absence is explicitly accepted for a narrowly scoped prototype.
6. Adapter, API, and frontend integration tests pass.
7. A human release decision explicitly approves promotion.

See [ALTERNATIVE_ISL_MODEL_AUDIT.md](../reports/ALTERNATIVE_ISL_MODEL_AUDIT.md), [FINAL_DEMONSTRATION_READINESS_REPORT.md](../reports/FINAL_DEMONSTRATION_READINESS_REPORT.md), and [MODEL_ASSET_CLEANUP_REPORT.md](../reports/MODEL_ASSET_CLEANUP_REPORT.md).
