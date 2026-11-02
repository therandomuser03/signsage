# Documentation Restructure Review Report

**Review date:** 2026-11-02  
**Scope:** the submitted `docs.zip` only.  
**Important:** this review did not inspect or modify the full repository source code, root README, backend, frontend, datasets, or model binaries.

## Findings in the submitted archive

- The archive contains 38 Markdown files, one PNG architecture diagram, one PNG ISL alphabet reference, and one PDF reference book (plus directory entries).
- A central `docs/README.md` index was missing.
- The existing `MASTER.md` still described an older documentation layout and linked to paths that are no longer the clearest current entry points.
- `DOCUMENTATION_AUDIT.md` described a 56-file repository-wide inventory, but its action totals did not add up to 56 and several proposed destination documents were absent from this ZIP.
- The original restructure report contained repeated filler text (including repeated "caveman full" and "restructuring verified" phrases) and asserted that documents had been created or merged when the submitted archive did not contain all of those destinations.
- `DEVELOPMENT_SETUP.md`, `PRIVACY_AND_LIMITATIONS.md`, and `DATASET_CATALOG.md` contained excessive repeated cross-reference/proof language that obscured the actual instructions.
- `STATUS_AND_ROADMAP.md` began with a very long duplicated audit summary before its actual title.
- The submitted docs contain both current summaries and detailed historical reports. Historical evidence should remain available, but it should not be mistaken for current operating instructions.

## Changes made in this reviewed copy

1. Added `docs/README.md` as the central documentation index and authority guide.
2. Rewrote `docs/ml/MODEL_STATUS.md` as a concise current decision record:
   - no continuous model is promoted;
   - Sid-V5 is an investigation candidate, not an adopted backend model;
   - 30.77% WER is gloss recognition, not English translation quality;
   - Sanket was isolated-sign and is not suitable for the continuous task;
   - the cleanup report says ISLiM checkpoint A remains because existing endpoints depend on it.
3. Rewrote `docs/ml/DATASET_CATALOG.md` as a short catalog pointing to detailed audits, preserving task and license distinctions.
4. Rewrote `docs/development/DEVELOPMENT_SETUP.md` to remove repetitive evidence prose and retain actionable setup commands and cautions.
5. Rewrote `docs/product/PRIVACY_AND_LIMITATIONS.md` to state concrete privacy commitments and product limitations without self-referential confirmation chains.
6. Added `docs/product/DEMONSTRATION_GUIDE.md` with a constrained, evidence-based demo sequence and explicit non-claims.
7. Replaced this report with a factual account of the submitted archive and the changes in this reviewed copy.

## Important issues not silently changed

- `docs/reports/MODEL_ASSET_CLEANUP_REPORT.md` reports that Sanket `model_best.pth`, ISLiM checkpoint B, and two ISLiM mobile export files were deleted, while ISLiM checkpoint A remains due to an active endpoint dependency. This reviewed copy records that distinction; it does not verify the local filesystem.
- The cleanup report documents a stale `ALT_CKPT` reference in `islim_adapter.py` to the deleted checkpoint B. This needs a source-code follow-up; it was not changed because the source repository was not included.
- Sid-V5 checkpoint access remains unverified. The 30.77% WER is a gloss-recognition metric, not direct English translation quality.
- Some docs refer to repository files outside this ZIP. Their existence and links cannot be validated from this archive alone.
- The original `DOCUMENTATION_AUDIT.md` inventory describes the wider repository, not just `docs/`. It requires a fresh repository-wide inventory before it can be treated as a current audit.

## Recommended next source-repository checks

1. Update the opening index/table in `MASTER.md` and the duplicated preamble in `STATUS_AND_ROADMAP.md` to point to the current docs layout.
2. Search the repository for references to the deleted ISLiM checkpoint B and decide whether to remove or guard the stale `ALT_CKPT` path.
3. Re-run a repository-wide Markdown link check, including root README and files outside `docs/`.
4. Confirm that the source files and runtime behavior still match the API, privacy, and setup documentation.
5. Keep historical reports in `reports/`; do not delete them merely to reduce the visible file count.

## Verification boundary

The revised ZIP was generated from the submitted archive. No application tests, API calls, model inference, source-code checks, or Git operations were performed as part of this documentation-only review.
