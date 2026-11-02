# Dataset Catalog

This is a navigation summary, not a substitute for the detailed dataset audits. Dataset names, metrics, and licenses are not interchangeable across tasks.

## Continuous / sentence-level ISL

### iSign v1.1

- Used in project research as a reference for continuous ISL-to-English and pose-based tasks.
- The CSV evidence described in the project audits contains `uid` and `text`, but no verified signer identifier column.
- The large `.pose` archive was not fully downloaded in the audited workspace.
- A video-level split does not establish signer-independent evaluation.
- The project audits identify gaps in temporal annotation and in the intended six-word pilot vocabulary mapping.
- License recorded by the project: CC BY-NC-SA 4.0. Review the source terms before redistribution or commercial use.

**Current conclusion:** useful research reference, but the local dataset and annotation evidence do not currently support a validated continuous-translation demonstration.

Detailed evidence: [TEMPORAL_ISL_DATASET_AUDIT.md](../datasets/TEMPORAL_ISL_DATASET_AUDIT.md), [VOCABULARY_ALIGNMENT_REVIEW.md](../datasets/VOCABULARY_ALIGNMENT_REVIEW.md), and [DATA_AND_MODEL.md](../architecture/DATA_AND_MODEL.md).

## Isolated-sign video datasets

### `isl-isolated-40words`

- Aggregate described in the audit as 642 clips across 40 glosses and multiple source datasets.
- Only 122 INCLUDE clips were downloaded and decoded for the reported six-word pilot.
- The aggregate has mixed provenance and a non-uniform/unclear aggregate license. Do not infer that every source inherits the INCLUDE license.
- The six-word pilot is isolated-word recognition, not continuous sentence translation.
- The reported pilot split was recording-ID-disjoint, not signer-independent.

Detailed evidence: [ISOLATED_ISL_RECOVERY_AUDIT.md](../evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md).

## Static image datasets

### RealSign alphabet data

- Used for the separate 26-class static alphabet image classifier.
- Project documentation records CC0-1.0 for this dataset and 87.1% held-out test accuracy.
- This is image classification, not temporal sign recognition or sentence translation.

### Mendeley word-image data

- Used for a separate static word-image classifier.
- Project documentation records 20 classes and 19.0% test accuracy.
- This result is not suitable for a reliable product claim.

The source audit is referenced in [STATUS_AND_ROADMAP.md](../STATUS_AND_ROADMAP.md) and [PRODUCT_OVERVIEW.md](../product/PRODUCT_OVERVIEW.md). The source audit itself is outside this docs-only ZIP and must be checked in the repository.

## Dataset handling rules

1. Keep source provenance, original labels, license, and preprocessing metadata with each dataset.
2. Do not merge isolated signs, continuous signing, static images, and fingerspelling into one task without an explicit task definition.
3. Keep train/validation/test splits reproducible and prevent source-video leakage.
4. Verify signer independence where possible; if signer IDs are absent, state that limitation.
5. Do not redistribute source data or derived weights until dataset and model licenses are reviewed.
6. Do not claim a dataset is locally available merely because its metadata or a download link exists.
