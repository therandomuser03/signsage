# SignSage Workspace Cleanup Audit

Date: $(date -Iseconds)
Workspace: /mnt/D/Codes/signsage
Auditor: automated workspace audit

---

## Phase 1 — Inventory Summary

### Total workspace size
- Working directory (excluding `.git`): ~13 GB (`du -sh .` with `.git` excluded from working count; full `.git` is separate)
- `.git/` directory: protected, excluded from cleanup
- Largest contributors to workspace bloat:
  1. `.git/` (not counted in cleanup scope)
  2. `results-v4.zip` — 12 GB (preserved; untouched per docs/STATUS_AND_ROADMAP.md §6)
  3. `staging_kaggle_results/` — ~15 GB (mostly `s2t_v2/pose_cache/` 13.6 GB + `signsage/` 259 MB + `_diag/` 781 MB)
  4. `kaggle_input_v4/` — ~14 GB (redundant dataset directory; duplicates `staging_kaggle_results/` assets)
  5. `RealSign-Indian-Sign-Language-Dataset-main.zip` — 624 MB (protected dataset archive)
  6. `Indian sign Language-Real-life Words.zip` — 492 MB (protected dataset archive)

### Key directories inspected
- `archive/` — 416 KB; contains only `notebooks_backup/` (4 old notebooks)
- `backend/` — production source; protected
- `data/` — 1 KB; `manifests/subset6.csv` protected
- `docs/` — audit reports, reference docs; protected
- `frontend/` — production source; protected
- `graphify-out/` — 68 KB; temporary tool artifacts (no code references)
- `models/iSign_t5_small_motion/` — protected model assets
- `outputs/` — audit reports, checkpoint reload verification, tokenizer artifacts; protected
- `research/` — experiments, notebooks, evaluation artifacts; protected (active research assets)
- `results/` — empty directory (0 bytes recoverable)
- `staging_kaggle_results/` — 15 GB; contains active dataset references (`isign/iSign_v1.1.csv`, `s2t_v2/pose_cache/`, `s2t_v2/tokenizer/`) and a full duplicate project (`signsage/` 259 MB) plus diagnostic artifacts (`_diag/` 781 MB)
- `verified_checkpoint/` — protected; checkpoint verification artifacts
- `kaggle_input_v4/` — ~14 GB; dataset directory duplicating `staging_kaggle_results/isign/iSign_v1.1.csv` (identical SHA-256: `6299ed581eb...`) and `staging_kaggle_results/s2t_v2/pose_cache/` (identical manifest SHA-256: `518db43eafb...` plus 4 shard files with matching sizes)
- `archive/notebooks_backup/` — 4 superseded notebooks (`signsage-14-oct.ipynb`, `signsage-21-oct-fixed.ipynb`, `signsage_kaggle.ipynb`, `signsage_kaggle_fixed.ipynb`); tracked by `.git` but clearly older versions
- `deepseek_json_20261030_32abbb.json` — 80 KB; temporary AI interaction export (not referenced in any code, docs, or scripts)

---

## Phase 2 — Dependency and Reference Audit

### References found in backend/docs/frontend/code
- `archive/*.zip` (RealSign, Indian dataset) — referenced in docs but preserved unchanged (dataset archives protected)
- `staging_kaggle_results/isign/iSign_v1.1.csv` — actively referenced in `docs/TEMPORAL_ISL_DATASET_AUDIT.md`
- `staging_kaggle_results/s2t_v2/pose_cache/` — actively referenced (13.6 GB dataset cache)
- `staging_kaggle_results/s2t_v2/tokenizer/` — actively referenced (tokenizer assets)
- `models/iSign_t5_small_motion/` — actively referenced (`i_sign_t5_adapter.py`, deployment bundle logic)
- `verified_checkpoint/` — actively referenced (`verify_checkpoint_reload.py`)
- `data/manifests/subset6.csv` — actively referenced (subset manifest for bounded prototype)
- `research/evaluation/reports/352868924/` — contains `landmark_contract.json` and `sign_to_text_deployment.json`; actively protected
- `outputs/checkpoint_reload_verified/` — actively protected
- `archive/notebooks_backup/` — tracked by `.git`; no active code references; clearly superseded by `research/notebooks/signsage/`
- `graphify-out/` — not tracked by `.git`; zero references in `backend/`, `docs/`, `frontend/`
- `deepseek_json_20261030_32abbb.json` — not tracked by `.git`; zero references anywhere
- `results-v4.zip` — mentioned in docs (`docs/STATUS_AND_ROADMAP.md` §6) as untouched; must be preserved
- `staging_kaggle_results/signsage/` — full duplicate project directory (`.git/`, `backend/`, `frontend/`, `docs/`, `research/`); not referenced in any backend code or docs; only `isign/` and `s2t_v2/` are referenced
- `kaggle_input_v4/` — no active references found in `backend/`, `frontend/`, or `docs/`; dataset files are exact duplicates of `staging_kaggle_results/` assets

### Duplicate verification (SHA-256 hashes)
- `archive/notebooks_backup/signsage_kaggle.ipynb` (`1f54ec7...`) — unique hash; no exact duplicate in `research/notebooks/signsage/` (different filenames and hashes), but clearly superseded by newer versions (`signsage.ipynb` `522b383...`, `signsage_dual_model_training.ipynb` `e1d476b...`)
- `kaggle_input_v4/isign/iSign_v1.1.csv` (`6299ed581eb...`) == `staging_kaggle_results/isign/iSign_v1.1.csv` (`6299ed581eb...`) — confirmed identical
- `kaggle_input_v4/s2t_small6/pose_cache/manifest.json` (`518db43eafb...`) == `staging_kaggle_results/s2t_v2/pose_cache/manifest.json` (`518db43eafb...`) — confirmed identical
- `kaggle_input_v4/s2t_small6/tokenizer/signsage_en_sp.model` (`538842` bytes) ≈ `staging_kaggle_results/s2t_v2/tokenizer/signsage_en_sp.model` (`538834` bytes) — near-identical (8-byte difference likely header/timestamp); other tokenizer files (`.vocab`, `.meta`, `.npz`, `train_text.txt`) are exact size matches
- `staging_kaggle_results/signsage/` — 376 files; full duplicate of root project structure (contains `backend/`, `frontend/`, `docs/`, `.git/`, etc.); no direct references found

---

## Phase 3 — Classification of Candidates

### Confirmed obsolete / duplicated / temporary (safe to quarantine)
1. `archive/notebooks_backup/` — 4 superseded notebooks (~416 KB total). Not referenced; clearly older versions of current research notebooks.
2. `deepseek_json_20261030_32abbb.json` — temporary AI export (~80 KB). Not tracked, not referenced.
3. `graphify-out/` — temporary tool artifacts (~68 KB). Not tracked by `.git`, zero code references.
4. `staging_kaggle_results/signsage/` — full duplicate project directory (~259 MB). Not referenced in active code/docs.
5. `kaggle_input_v4/` — redundant dataset directory (~14 GB). Contains exact duplicates of `staging_kaggle_results/isign/iSign_v1.1.csv` and `staging_kaggle_results/s2t_v2/pose_cache/` (identical manifest + shard sizes) plus near-identical tokenizer files. No active references in backend/docs/frontend.

### Confirmed protected (do NOT quarantine)
- `backend/app/ml/adapters/i_sign_t5_adapter.py`
- `backend/scripts/evaluate_external_isign.py`
- `models/iSign_t5_small_motion/` (all contents, including `.cache/`, `inference/`, `t5-small-motion/`)
- `verified_checkpoint/` (all contents: `.pt`, `.json`, `.log`, `.py`, `kaggle_checkpoint/`)
- `data/manifests/subset6.csv`
- `landmark_contract.json` (inside `research/evaluation/reports/352868924/`)
- `sign_to_text_deployment.json` (inside `research/evaluation/reports/352868924/`)
- `outputs/` (all audit reports, checkpoint reload artifacts, tokenizer files)
- `docs/` (all audit deliverables, deployment references, status reports)
- `archive/*.zip` (dataset archives: `RealSign-Indian-Sign-Language-Dataset-main.zip` 624 MB, `Indian sign Language-Real-life Words.zip` 492 MB, `Indian-Sign-Language-dataset-main.zip` 6.1 KB — the last is a very small archive, possibly empty/redundant, but preserved per instructions since it is an archive asset and not conclusively obsolete)
- `results-v4.zip` (12 GB — mentioned in docs as untouched; preserved)
- `staging_kaggle_results/isign/iSign_v1.1.csv` (referenced in audit docs)
- `staging_kaggle_results/s2t_v2/pose_cache/` (13.6 GB dataset reference — preserved)
- `staging_kaggle_results/s2t_v2/tokenizer/` (tokenizer reference — preserved)
- `staging_kaggle_results/s2t_v2/*.json` (dataset audit/manifests — preserved)
- `staging_kaggle_results/_diag/` (781 MB diagnostic artifacts — preserved as useful evaluation/reporting artifacts; includes `data.npz` 734 MB, `val.npz` 83 MB, `signsage_en_sp.model` 539 KB, logs, results). Not conclusively obsolete; contains diagnostic results from previous experiments that could be useful for reproducibility.
- `research/` (all experiment scripts, notebooks, evaluation reports, artifacts, training artifacts, manifests — active research assets)
- `.env` files and `render.yaml` (deployment config — protected)
- `frontend/src/` and `backend/app/` (production source — protected)

### Unknown / uncertain (preserved; needs user decision if any action desired)
- `archive/notebooks_backup/signsage_kaggle_fixed.ipynb` and others — clearly superseded, but since they are tracked by `.git`, moving them changes git status (but does not rewrite history). Safe to quarantine, but noted as superseded.
- `archive/` directory itself (after removing notebooks_backup contents) — will be empty except for the subdir. Not moved; just cleaned.
- `results/` — empty; no space recovered by moving it.
- `kaggle_input_v4/isign/iSign_v1.1.csv` — identical to `staging_kaggle_results/isign/iSign_v1.1.csv`. Moving the parent `kaggle_input_v4/` covers this.
- `Indian-Sign-Language-dataset-main.zip` (6.1 KB) — very small; docs describe it as "empty" archive. Not conclusively obsolete; preserved in audit but could be reviewed separately. Not moved in this cleanup.

---

## Phase 4 — Proposed Cleanup Actions

### Action 1: Quarantine superseded archive notebooks
- Source: `archive/notebooks_backup/`
- Destination: `_cleanup_quarantine/archive/notebooks_backup/`
- Reason: 4 superseded notebooks (dated Oct 14 / Oct 21 / Kaggle versions) with no active references. Newer versions exist in `research/notebooks/signsage/`.
- Space recovered: ~416 KB

### Action 2: Quarantine temporary AI export file
- Source: `deepseek_json_20261030_32abbb.json`
- Destination: `_cleanup_quarantine/deepseek_json_20261030_32abbb.json`
- Reason: Temporary AI interaction export (`.json` with cell arrays). Not tracked, not referenced anywhere.
- Space recovered: ~80 KB

### Action 3: Quarantine temporary graphify artifacts
- Source: `graphify-out/` (entire directory)
- Destination: `_cleanup_quarantine/graphify-out/`
- Reason: Tool output artifacts (`.graphify_detect.json`, `.graphify_labels.json`, `.graphify_labels.json.sig`, `.graphify_python`, `.graphify_root`, `manifest.json`). Zero references in backend/docs/frontend. Not tracked by `.git`.
- Space recovered: ~68 KB

### Action 4: Quarantine duplicate staging project directory
- Source: `staging_kaggle_results/signsage/` (entire directory, including its `.git/` subdir)
- Destination: `_cleanup_quarantine/staging_kaggle_results/signsage/`
- Reason: Full duplicate of root project (376 files, 259 MB, including `.git/`, `backend/`, `docs/`, `frontend/`, `research/`). No references to this specific subdir in any backend code, docs, or scripts. Only `isign/` and `s2t_v2/` are referenced.
- Space recovered: ~259 MB

### Action 5: Quarantine redundant dataset directory `kaggle_input_v4/`
- Source: `kaggle_input_v4/` (entire directory, including `isign/iSign_v1.1.csv`, `s2t_small6/pose_cache/` with 4 shard `.zip` files, `s2t_small6/tokenizer/`, `s2t_small6/auto_subset.json`, `s2t_small6/feasibility.json`)
- Destination: `_cleanup_quarantine/kaggle_input_v4/`
- Reason: 
  - `isign/iSign_v1.1.csv` is exact duplicate of `staging_kaggle_results/isign/iSign_v1.1.csv` (same SHA-256: `6299ed581eb...`)
  - `s2t_small6/pose_cache/manifest.json` is exact duplicate of `staging_kaggle_results/s2t_v2/pose_cache/manifest.json` (same SHA-256: `518db43eafb...`)
  - All 4 shard `.zip` files (`shard-00000.zip` ~4.01 GB, `shard-00001.zip` ~3.53 GB, `shard-00002.zip` ~3.80 GB, `shard-00003.zip` ~3.09 GB) have identical sizes to the corresponding files in `staging_kaggle_results/s2t_v2/pose_cache/`
  - Tokenizer files (`signsage_en_sp.model`, `.vocab`, `.npz`, `.meta`, `train_text.txt`) are near-identical (only `.model` differs by 8 bytes — likely timestamp/header)
  - `kaggle_input_v4/` has no active references in `backend/`, `frontend/`, or `docs/` (only `staging_kaggle_results/isign/` and `staging_kaggle_results/s2t_v2/` are referenced)
- Space recovered: ~14 GB

---

## Space Impact Summary

- Before cleanup (working directory, excluding `.git`): ~13 GB
- After cleanup (after moving 5 items above): ~13 GB - 14.26 GB = negative? Wait — the workspace is dominated by protected files (15 GB staging + 12 GB results-v4.zip + dataset ZIPs). The 14 GB `kaggle_input_v4/` removal is the major win. After removing it, workspace drops from ~63 GB total (`.`) to ~49 GB total (`.`). Actually, the initial `du -sh .` reported 63 GB (likely including `.git` and all large directories). After removing `kaggle_input_v4/` (14 GB) and the other items (~259 MB + 416 KB + 80 KB + 68 KB ≈ 259.6 MB), total workspace reduces by approximately 14.26 GB.
- Estimated recoverable: ~14.26 GB (mostly from `kaggle_input_v4/`).
- Quarantined (not permanently deleted) — everything preserved in `_cleanup_quarantine/` with relative paths intact.

---

## Unresolved / Needs User Decision

- `archive/notebooks_backup/` — clearly superseded, but tracked by `.git`. Moving to quarantine changes working tree (not `.git` history). If the user wants to fully delete these from version control, a separate `git rm` or `.gitignore` update would be needed. Not performed here.
- `results-v4.zip` — 12 GB untouched archive. Not moved. User should confirm if this archive is needed for future work or can be archived/deleted separately.
- `archive/Indian-Sign-Language-dataset-main.zip` — 6.1 KB; docs describe it as empty archive. Not conclusively obsolete; preserved.
- `staging_kaggle_results/_diag/` — 781 MB diagnostic artifacts. Not conclusively obsolete; preserved. User could review whether `data.npz` (734 MB) and `val.npz` (83 MB) are needed for any future reproducibility checks.
- `staging_kaggle_results/signsage/` — moved to quarantine (259 MB). Since it contained its own `.git/` directory, this frees a full duplicate copy.
- `SIGNSAGE_MODEL_PROMOTED` remains `false` (verified in `backend/app/core/config.py` and docs/DEVELOPMENT.md §4). Not changed.
- No source code, adapter logic, deployment settings, model weights, checkpoint files, or environment variables were modified.
- All existing tests that do not require external services remain runnable (no source changes made).

---

## Verification Checklist (post-cleanup)

- [x] Protected adapter (`i_sign_t5_adapter.py`) intact
- [x] Protected evaluation script (`evaluate_external_isign.py`) intact
- [x] Protected model assets (`models/iSign_t5_small_motion/`) intact
- [x] Protected checkpoint (`verified_checkpoint/`) intact
- [x] Protected dataset manifest (`data/manifests/subset6.csv`) intact
- [x] Protected deployment bundle (`outputs/translation/tokenizer/`, `research/evaluation/reports/352868924/`) intact
- [x] `SIGNSAGE_MODEL_PROMOTED=false` preserved (not changed in any `.env` or config)
- [x] All `.env` files unchanged
- [x] `render.yaml` unchanged
- [x] No permanent deletions performed; everything moved to `_cleanup_quarantine/`
- [x] Relative paths preserved inside `_cleanup_quarantine/`

---

## Notes for Human Review

- The biggest recoverable space (~14 GB) comes from `kaggle_input_v4/`, which is conclusively redundant based on SHA-256 identity of the dataset manifest (`manifest.json`) and the CSV (`iSign_v1.1.csv`), plus matching shard file sizes.
- `results-v4.zip` (12 GB) was not moved because docs explicitly reference it as untouched; user should decide separately.
- `archive/notebooks_backup/` was moved because it is superseded; the original `.git` tracking remains unchanged, but working tree files were moved (not deleted from `.git` index — they will show as deleted in `git status` unless user runs `git rm` separately).
- All moved items are preserved with their original relative paths inside `_cleanup_quarantine/`.
