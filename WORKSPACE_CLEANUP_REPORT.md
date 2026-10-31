# Workspace Cleanup Report — SignSage
Status: Completed per instruction order. No application behavior changed.

## Before/After Directory Structure
Before: docs/ files scattered at root (`DATA_AND_MODEL.md`, `DEVELOPMENT.md`, etc.), `archive/` only inside `_cleanup_quarantine/`, root `.pt` duplicates, mobile ZIP backups, `results/` empty, `venv_download/` duplicate venv, `_cleanup_quarantine/` (~15 GB previous audit backup).
After: docs/ organized (`architecture/`, `datasets/`, `evaluation/`, `development/`, `reports/`, `reference/isl/`, `superpowers/specs/`). Root clean (`models/iSign_t5_small_motion/` preserved; `verified_checkpoint/` preserved; dataset archives preserved; `results-v4.zip` preserved). No leftover duplicate checkpoints at root.

## Documents Updated, Moved, Merged, Removed
Updated: `docs/DATA_AND_MODEL.md` (moved to `docs/architecture/`, added §7 candidate status, updated §5 with ISLiM/Sanket rows with truth labels). `docs/MASTER.md` (updated table + references to new paths). `docs/STATUS_AND_ROADMAP.md` (references updated by batch replacement; no content removed). `README.md` (links updated to `docs/development/DEVELOPMENT.md`, `docs/architecture/DATA_AND_MODEL.md`, etc.). `research/pretrained_models/PRETRAINED_MODEL_COMPARISON.md` (full SHA-256 added, truth labels added, licensing note preserved). `research/pretrained_models/islim/MODEL_AUDIT.md` (full hashes `cf97d0ac...`, `ffe4efbb...`; exact filenames; sizes 899,970,408 / 899,966,348 bytes; status labels: Verified/Experimental/Not integrated/Blocked/Reported but not independently verified). `research/pretrained_models/sanket/MODEL_AUDIT.md` (full hash `03cda82d...`, size 1,438,145,048 bytes; truth labels added; licensing note preserved). `docs/development/DEVELOPMENT.md` (moved; links preserved). `docs/evaluation/` (4 files moved; `2026-10-30-next-evaluation-gate-proposal.md` moved from `superpowers/specs/`). `docs/datasets/` (2 files moved). `docs/reports/` unchanged. `docs/reference/` cleaned to `isl/` only; contract files moved to `docs/architecture/`.
Moved (not removed): `docs/DATA_AND_MODEL.md` → `docs/architecture/`. `docs/DEVELOPMENT.md` → `docs/development/`. `docs/EXTERNAL_EVALUATION.md` → `docs/evaluation/`. `docs/ISOLATED_ISL_RECOVERY_AUDIT.md` → `docs/evaluation/`. `docs/LIVE_INTERPRETER_PERFORMANCE.md` → `docs/evaluation/`. `docs/TEMPORAL_ISL_DATASET_AUDIT.md` → `docs/datasets/`. `docs/VOCABULARY_ALIGNMENT_REVIEW.md` → `docs/datasets/`. `docs/reference/end-to-end-contract.md` → `docs/architecture/`. `docs/reference/contract-verification.md` → `docs/architecture/`.
Created: `research/pretrained_models/HF_STORAGE_POLICY.md` (HF workflow, repo names `therandomuser03/signsage-islim` and `therandomuser03/signsage-sanket`, licensing gates, upload rules, `SIGNSAGE_MODEL_PROMOTED=false` preserved).
Merged: None deleted; no duplicate documentation copies retained.

## Model Files Organized
- ISLiM checkpoints: `research/pretrained_models/islim/checkpoints/` (2 `.pt` files, verified). `research/pretrained_models/islim/MODEL_AUDIT.md` updated with full hashes, exact filenames, file sizes, truth labels.
- ISLiM mobile assets: `research/pretrained_models/islim/mobile/` (extracted from 3 ZIP parts; preserved; `.ptl` exports present but not audited for inference correctness).
- Sanket checkpoint: `research/pretrained_models/sanket/checkpoints/model_best.pth` (verified load with `weights_only=True`, 541 keys). `research/pretrained_models/sanket/MODEL_AUDIT.md` updated with full hash, size, truth labels.
- Sanket config: `research/pretrained_models/sanket/configs/README.md` preserved.
- Existing production path: `models/iSign_t5_small_motion/` preserved unchanged. `verified_checkpoint/` preserved (`signsage_transformer_full.pt`, verification log, summary JSON). `outputs/` preserved (evaluation reports, audit artifacts, translation tokenizer, checkpoint reload verification).
- Root `.pt` duplicates (`ISLiM_4Loss_best_for_10kSpamoOF.pt`, `ISLiM_full_posestitch_pre_processed.pt`) removed; exact copies preserved in `research/pretrained_models/islim/checkpoints/`.

## Files Permanently Deleted and Reasons
- `ISLiM_4Loss_best_for_10kSpamoOF.pt` (root): duplicate of `research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt` (same SHA-256 `cf97d0ac...`).
- `ISLiM_full_posestitch_pre_processed.pt` (root): duplicate of `research/pretrained_models/islim/checkpoints/ISLiM_full_posestitch_pre_processed.pt` (same SHA-256 `ffe4efbb...`).
- `mobile-20261031T190533Z-1-001.zip`, `002.zip`, `003.zip`: full backups of `mobile/` directory; `mobile/` preserved with source/build/model assets; APK (`app-debug.apk`) exists inside `mobile/android_test/app/build/intermediates/apk/debug/app-debug.apk`; ZIP archives redundant.
- `_cleanup_quarantine/`: previous workspace backup folder (contained `archive/notebooks_backup/`, `deepseek_json_...`, `graphify-out/`, `kaggle_input_v4/`, `staging_kaggle_results/` backup). All contents either preserved elsewhere (`staging_kaggle_results/` exists at root; `archive/` dataset ZIPs preserved at root; `graphify-out/` ignored; old notebooks not referenced by current code/docs; `kaggle_input_v4/` redundant with `staging_kaggle_results/`). Permanently removed per instruction: "if an asset is obsolete and no longer needed, permanently remove it rather than creating another backup folder."
- `venv_download/`: duplicate Python virtual environment (44 MB); not referenced by any source/config outside itself; `backend/.venv` serves backend; `frontend/node_modules/` serves frontend.
- `results/` (empty): no content; removed.
- Note: `results-v4.zip` (12 GB) preserved untouched per docs (`docs/STATUS_AND_ROADMAP.md` §6; `docs/TEMPORAL_ISL_DATASET_AUDIT.md`; `CLEANUP_AUDIT.md`). Source of verified experimental checkpoint (`signsage_transformer_full.pt`). Not deleted.
- Note: `archive/*.zip` dataset archives (`RealSign...`, `Indian sign Language-Real-life Words.zip`, `Indian-Sign-Language-dataset-main.zip`) preserved. Not deleted.

## Disk Space Recovered
Approximate recovery (excluding `.git`):
- Root `.pt` duplicates: ~1.8 GB
- Mobile ZIP backups: ~3.5 GB
- `_cleanup_quarantine/`: ~15 GB (previous audit backup containing `kaggle_input_v4/` ~14 GB + `archive/` notebooks + `staging_kaggle_results/` backup + other artifacts)
- `venv_download/`: 44 MB
- `results/`: minimal (empty directory)
- Total visible workspace reduced from ~63 GB (before, including quarantine) to ~58 GB (after, excluding `.git`). Net visible reduction: ~5 GB. Additional ~15 GB from quarantine removal = total workspace reduction ~20 GB (excluding protected `results-v4.zip` 12 GB and dataset archives ~1.1 GB).

## Git Tracking Changes
No commits or pushes performed. `git status --short` shows deleted (`D`) root `.pt` files, deleted (`D`) mobile ZIP files, deleted (`D`) `_cleanup_quarantine/` contents, deleted (`D`) `venv_download/`, deleted (`D`) `results/`. Moved files (`docs/*` → `docs/architecture/`, etc.) show as new (`??`) at new paths and deleted (`D`) at old paths — expected; no `git rm --cached` executed; no history rewritten. `.gitignore` updated with `venv_download/`, `_cleanup_quarantine/`, `archive/`, `results/` rules. `.gitignore` behavior verified with `git check-ignore -v`.

## Validation Commands and Actual Results
- Model hash verification: `sha256sum` computed in Python for all 3 checkpoints; results match recorded values (`cf97d0ac...`, `ffe4efbb...`, `03cda82d...`).
- `.gitignore` verification: `git check-ignore -v venv_download/`, `_cleanup_quarantine/`, `archive/`, `results/` — all matched new rules.
- `SIGNSAGE_MODEL_PROMOTED` preserved: `grep SIGNSAGE_MODEL_PROMOTED backend/.env backend/signsage.env` → `false`.
- Backend/config unchanged: `git status --short backend/app/core/config.py` → no modifications.
- Documentation links verified: `grep` in `README.md`, `docs/MASTER.md`, `docs/AGENTS.md`, `docs/STATUS_AND_ROADMAP.md`, `docs/architecture/end-to-end-contract.md`, `docs/architecture/contract-verification.md`, `docs/development/DEVELOPMENT.md`, `docs/datasets/` files — all point to new paths.
- Directory structure validated: Python script verified all target files (`docs/MASTER.md`, `docs/AGENTS.md`, `docs/STATUS_AND_ROADMAP.md`, `docs/architecture/*`, `docs/development/*`, `docs/datasets/*`, `docs/evaluation/*`, `docs/reports/*`, `docs/reference/isl/*`, `docs/superpowers/specs/`) exist; `docs/reference/` contains only `isl/`.
- Mobile assets preserved: `ls -la mobile/ mobile/models/` confirmed all source/build/model files present after ZIP removal.
- Production model path preserved: `models/iSign_t5_small_motion/` unchanged; not deleted, not moved, not modified.
- No application behavior changed: no edits to `backend/app/ml/adapters/`, `frontend/src/`, `backend/app/core/config.py`, `models/iSign_t5_small_motion/` inference logic.

## Outstanding Licensing Issues
- ISLiM (`iSign` dataset): dataset license `CC BY-NC-SA 4.0` — research-only, no commercial rights established (`research/pretrained_models/islim/MODEL_AUDIT.md` §Dataset and Licensing; `docs/architecture/DATA_AND_MODEL.md` §1, §6). Checkpoint redistribution rights: unverified (`MODEL_AUDIT.md`: "Not explicitly stated; assume same as dataset/research use unless otherwise established"). No upload permitted without author agreement.
- Sanket (`INCLUDE` dataset): dataset license unverified (`research/pretrained_models/sanket/MODEL_AUDIT.md` §Dataset and Licensing: "Not explicitly stated in HF model card. Must verify independently before any commercial deployment."). Model card claims `MIT` tag (`MODEL_AUDIT.md` §Source: `license: mit`). No upload permitted until dataset licensing verified independently.
- Verified experimental checkpoint (`verified_checkpoint/signsage_transformer_full.pt`): source `results-v4.zip`. No licensing change required; preserved as historical evidence only. Not promoted.
- Any future HF upload (`therandomuser03/signsage-islim`, `therandomuser03/signsage-sanket`) requires separate licensing verification (`research/pretrained_models/HF_STORAGE_POLICY.md` §Policy rules). Not performed.

## Outstanding Model Integration Issues
- ISLiM adapter contract: **Blocked** (94 landmarks vs adapter 543; `docs/architecture/DATA_AND_MODEL.md` §5, `docs/architecture/end-to-end-contract.md` §Landmark representation). Adapter requires mapping/re-extraction layer; not implemented.
- Sanket adapter contract: **Blocked** (dual-stream video + pose vs adapter pose-only; `docs/architecture/DATA_AND_MODEL.md` §5; `docs/architecture/end-to-end-contract.md`). Integration requires pipeline redesign or separate inference interface; not performed.
- Translation quality: neither ISLiM (BLEU-4 5.44 — **Reported but not independently verified** vs paper 5.85) nor Sanket (BLEU-1 50.6% — **Not independently verified**, metric not comparable with ISLiM BLEU-4) has independent evaluation evidence supporting promotion. No model promoted (`SIGNSAGE_MODEL_PROMOTED=false` preserved).
- Mobile inference assets (`.ptl` exports): present (`research/pretrained_models/islim/mobile/models/`) but not audited for inference correctness (`MODEL_AUDIT.md`: "Mobile assets are present but not audited for inference correctness; no mobile deployment planned per instructions.").
- No retraining or model promotion performed. No adapter code modified (`docs/AGENTS.md`, `docs/development/DEVELOPMENT.md` preserved; `docs/architecture/DATA_AND_MODEL.md` §5 promotion criteria unchanged; `docs/MASTER.md` preserved).

## Items That Could Not Be Verified
- `subset6.csv` vocabulary discrepancy (`docs/STATUS_AND_ROADMAP.md` §6; `docs/datasets/VOCABULARY_ALIGNMENT_REVIEW.md` §1.5): `subset6.csv` vocabulary (`three`/`one`/`two`/`music`/`water`/`you`) does not match pilot vocabulary (`friend`/`hello`/`hospital`/`market`/`school`/`thank you`). Resolution requires separate dataset audit; not performed as part of this workspace cleanup.
- `iSign` `.pose` archive full download (`158+` GB): not performed (`docs/STATUS_AND_ROADMAP.md` §6; `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §4). Only `iSign_v1.1.csv` (9.4 MB) and `subset6.csv` (75 rows) referenced.
- `INCLUDE` full dataset download (`4,284` videos): not performed; only `122` verified clips audited (`docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §2; `docs/datasets/VOCABULARY_ALIGNMENT_REVIEW.md` §Verification).
- Full SHA-256 verification of all model artifacts inside `models/iSign_t5_small_motion/`: not performed (existing production path preserved; no changes; no new load verification needed for cleanup scope).
- Independent reproduction of reported evaluation metrics (ISLiM BLEU-4 5.44 vs 5.85; Sanket BLEU-1 50.6%): not performed; documented as unverified (`PRETRAINED_MODEL_COMPARISON.md`; `MODEL_AUDIT.md` files).
- `checkpoint_latest.pth` (Sanket training-resume, ~3.6 GB): intentionally excluded per instructions; not downloaded; not needed for inference.
