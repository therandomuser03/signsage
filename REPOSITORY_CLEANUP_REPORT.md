# SignSage Repository Cleanup Report

**Date:** 2026-10-12 · **Branch:** `master` · **Baseline:** `c2c597f`
**Scope:** docs consolidation + confirmed-obsolete removal. No code moves, no
hyperparameter/architecture changes, no training runs.

## 1. Baseline (safe)

- `git status` before work: clean except two untracked operator files
  (`scan_signsage_structure.py`, `signsage_project_structure.txt`) — **kept, untouched**.
- One pre-existing stash (`WIP on master: f63f82b`) — **untouched**.
- 343 tracked files. Ignored large artifacts left on disk, never committed:
  `results.zip` (12G), `staging_kaggle_results/` (14G), `research/artifacts/`
  (empty bundle skeleton, no weights), `backend/.env`, `frontend/.env.local`.
- No secret values were read or printed (only `.env.example` files).

## 2. Entry points (verified, unchanged)

- Backend: `uvicorn app.main:app` (`backend/app/main.py`); routes under
  `/api/v1/*`, WS at unversioned `/ws/translate`; tests via `pytest` in `backend/`.
- Frontend: `npm run dev` / `npm run build`; router in `frontend/src/app/routes.tsx`.
- Training (authoritative chain): `research/training/READINESS.md` (why) →
  `research/training/train_sign_to_text.py` (how) →
  `research/training/signsage_kaggle_fixed.ipynb` (current runbook; the older
  `signsage_kaggle.ipynb` is retained as history) → `audit_dataset.py` (clean/split lib).
- Bundle resolution sole owner: `backend/app/infrastructure/storage/model_storage.py`
  via `SIGNSAGE_MODEL_BUNDLE_DIR`. Default points at the ignored
  `research/artifacts/352868924/` skeleton (empty: no `.pt`/tokenizer on disk);
  Render fetches via `backend/scripts/fetch_bundle.sh`. No-gate behavior unchanged.

## 3. Documentation consolidation (30 → 24 files)

| Removed file | Evidence | Merged into |
|---|---|---|
| `docs/AI_AGENT_GUIDE.md` (236 w) | Zero unique content vs `AGENTS.md`; only refs were root `README.md` ×2 | `docs/AGENTS.md` §12 (workflow + never-infer-quality rules); README links updated |
| `docs/README.md` (612 w) | ~70% duplicate of `MASTER.md`; no inbound refs | `docs/MASTER.md` §§8–9 (claims policy + thesis framing) |
| `docs/EXTERNAL_ISIGN_EVALUATION_SET.md` (480 w) | Classes/tables verbatim in QUALITY §§3–4; repetitions guidance already in QUALITY §4 | `docs/EXTERNAL_ISIGN_QUALITY_EVALUATION.md` (§4 one-line addition + §15 provenance row) |
| `docs/EXTERNAL_ISIGN_ADAPTER.md` | Stale spec; code (`i_sign_t5_adapter.py`) is truth | `docs/EXTERNAL_ISIGN_CONTRACT_VERIFICATION.md` Appendix A |
| `docs/EXTERNAL_ISIGN_T5_AUDIT.md` | Duplicates VERIFICATION §§2–6; unique = shapes/HW/plan history | Same file, Appendix B (marked dated; verdicts superseded by VALIDATION/QUALITY NO-GO) |
| `docs/DOCUMENTATION_REVIEW.md` | Stale 2026-09-29 meta-review; numbered-copy ban already in `AGENTS.md` §10; no refs | Deleted (rule preserved) |
| `frontend/src/assets/react.svg`, `vite.svg` | No imports in `src/` or `index.html` (grep-verified Vite boilerplate) | Deleted; `hero.png` kept (see §6) |

Reference updates: `research/EXTERNAL_ISIGN_T5_KAGGLE_EVALUATION.ipynb` cell 5 now
cites VERIFICATION Appendix A (JSON re-validated); QUALITY §15 table updated.
`docs/PROJECT_STRUCTURE.md` rewritten to the actual `git ls-files` tree (old
version described non-existent `pages/`, `docs/architecture/`, `ADR-*`,
`troubleshooting/`, `infrastructure/` paths).

Kept separate deliberately: `API/DEPLOYMENT/SECURITY/TESTING` (distinct concerns),
`DATA_MODEL/DATASET_STORAGE/SUPPORTED_ISL_VOCABULARY` (clean split),
`STATUS/MODEL_READINESS/PRODUCTION_BLOCKERS/APPLICATION_AUDIT/END_TO_END_*`
(evidence records, not drift), `ARCHITECTURE/DESIGN/DECISIONS/PRD/PHASES`.

## 4. Structure decision

Existing `backend/ frontend/ research/ models/ docs/ tools/` + `render.yaml` is
sound; **no file/directory moves** — moves would churn imports, CI, and notebook
paths for zero ownership gain. Only docs were merged/deleted as above.

## 5. ML pipeline protection

- Untouched: all of `research/training/`, both runbook notebooks, `t5_pose_poc/`,
  all `evaluation/` + `manifests/` + `experiments/` records, `models/` mirror,
  backend `ml/` + registry + `model_storage.py`, all tests, `render.yaml`.
- All 5 notebooks re-validated with `json.load` (incl. edited Kaggle eval notebook).
- Duplicate notebooks compared: `_fixed` is the current runbook, plain `signsage_kaggle.ipynb`
  is the older variant — **both retained** per brief §7.

## 6. Remaining candidates (left untouched, for owner review)

- `staging_kaggle_results/signsage/` (259M nested snapshot with its own `.git`,
  `TASKS.md`): gitignored staging data — needs an owner decision (extract info vs delete).
- `frontend/src/assets/hero.png` (12.8K, no current imports): likely unused but kept
  as project asset pending design confirmation.
- `frontend/src/components/ui/` (~70 shadcn files): widely imported; no per-file
  dead-code pass attempted.
- `tools/signsage_full_audit.py`: unreferenced (nothing imports it) but harmless and
  potentially useful; kept.
- `research/artifacts/352868924/` skeleton: backend bundle check fails until an
  operator supplies a bundle — expected, documented behavior.

## 7. Validation

- `python3 -m compileall app` (backend): **PASS**.
- `python3 -m pytest -q` (backend): **120 passed, 32 failed — all 32 failures are
  pre-existing environment gaps** (`async def not natively supported`: `pytest-asyncio`
  not installed; `requirements.txt` ships only `pytest`, the plugin is a dev extra
  in `pyproject.toml`). No backend code was touched; failures are unrelated to this cleanup.
- Frontend `npx tsc -b`: **PASS (exit 0)**. `npm run build`: **PASS (13.5s)**.
- Markdown relative-link check (`README.md` + `docs/*.md`): **ALL_LINKS_OK**.
- Deleted-path reference sweep: only intentional provenance notes in the merged docs.
- `git status`: 8 deletions + 6 modifications staged/unstaged, reviewable via
  `git diff HEAD --stat`; the two untracked operator files remain untracked.
- NOT run: full training, GPU inference, live two-server validation, Supabase-backed flows.

## 8. Acceptance

- [x] Easier to navigate (accurate structure doc, one authority per topic).
- [x] Less duplication (30→24 docs; external group 6→3).
- [x] Obsolete files removed with evidence; risky/valuable items kept + listed above.
- [x] Backend/frontend/model/training workflows retain all files and references.
- [x] No secrets exposed; no pre-existing changes discarded; diff is reviewable.
