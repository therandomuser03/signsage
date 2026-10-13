# SignSage Second-Pass Cleanup Report

**Date:** 2026-10-12 · **Branch:** `master` · **Follows:** `REPOSITORY_CLEANUP_REPORT.md`
(first pass: docs 30→24, staged, uncommitted — preserved and extended here).
**Method:** import-graph + reference evidence for every deletion; aggressive on proven
junk, conservative on uncertainty. No training runs, no architecture/hyperparameter changes.

## 1. Before / after

| Metric | HEAD (pre-cleanup) | After | Δ |
|---|---|---|---|
| Tracked files | 343 | 274 | **−69** |
| Markdown docs (`docs/*.md`) | 30 | 24 | −6 (first pass; this pass removed none under `docs/`) |
| Frontend files (tracked) | 155 | 99 | −56 |
| Backend `.py` files | 104 | 96 | −8 |
| Notebooks (`.ipynb`) | 5 | 5 | 0 (all preserved) |
| Tracked blob size | 19,745,968 B | 19,400,761 B | −~0.3 MB (bulk is `pose_encoder.pt` 1.1 MB + notebooks, all kept) |
| Frontend deps (`package.json`) | 40 + 18 dev | 22 + 19 dev | −18 removed, +1 `@types/three` made explicit |

## 2. Removed files (with evidence)

### Backend — 8 dead modules (zero importers in `app/`+`tests/`+`scripts/`, zero tests)

| File | Evidence |
|---|---|
| `app/domain/sign_language/__init__.py`, `contracts.py`, `entities.py` | `contracts.py` defined `EXPECTED_LANDMARK_COUNT`, `entities.py` `LandmarkFrame/Sequence` — no importer anywhere; whole dir removed |
| `app/ml/inference/text_to_sign.py` | Docstring-only placeholder, 0 importers |
| `app/ml/adapters/text_to_sign_adapter.py` | `TextToSignAdapter` raised `NotImplementedError`, 0 importers |
| `app/ml/models/text_to_sign.py` | `TextToPoseModel` raised, 0 importers; shipped EN→ISL is `avatar_gloss` route+service |
| `app/ml/postprocessing/refinement.py` | `refine_for_display`, 0 callers |
| `app/infrastructure/external/__init__.py` | 244-byte docstring placeholder ("would live here"); real Groq/Ollama calls live in `services/avatar_gloss_service.py` via `httpx`; 0 importers |

Docstring pointers to the deleted path updated in `api/routes/avatar_gloss.py`,
`schemas/avatar_gloss.py`, `services/avatar_gloss_service.py`;
`docs/APPLICATION_AUDIT.md` rows 1.16/1.17 updated to REMOVED. Stale `.pyc` orphans deleted.

### Frontend — 53 files + 1 stale link

- 51 unreachable `src` files proven by reachability closure from `src/main.tsx`
  (both `@/`-alias and relative imports, incl. CSS side-effects): 49 unused shadcn
  `ui/*` components, `toggle.tsx` + `input-group.tsx` (only importers were dead files),
  `item.tsx` (0 importers), `magicui/striped-pattern.tsx` (0 importers).
  Live `field.tsx`/`empty.tsx`/`checkbox`/`dialog`/etc. kept — multiline-import
  false negative in the first script draft was caught and corrected before deleting.
- `src/assets/hero.png` (0 references; only CSS/comment mentions of "hero").
- `index.html`: removed `<link href="/src/style.css">` — target never existed (404).
- Kept: all routes, services, hooks (`use-mobile` ← live `sidebar`), `lib/utils` (`cn` ← 21 files),
  all effect components actually used (DotGrid, GhostFibers, Silk, SpecularButton, …).

### Docs — 2 files

| File | Evidence |
|---|---|
| `frontend/README.md` | Pure Vite template, 0 SignSage content, 0 inbound refs — deleted outright |
| `research/datasets/README.md` | 3-line placeholder, content subsumed by `DATA_MODEL.md` §5; sole inbound cite (`APPLICATION_AUDIT.md:42`) repointed to `DATA_MODEL.md` §5 |

All other merge hypotheses were tested and **rejected on evidence** (PHASES/PRODUCTION_BLOCKERS/
SUPPORTED_VOCAB/backend-models-README/text-to-sign-README/GESTURETALK/DATASET_STORAGE each
carry unique, load-bearing content). `SUPPORTED_ISL_VOCABULARY.md` gained its first inbound
link (root README); TASKS.md nuggets (lint baseline ~39 errors, outstanding manual browser E2E)
preserved in `TESTING.md` §9.

### Dependencies

- Frontend: removed 18 with zero references in `src/`+configs+css+html —
  `@fontsource-variable/geist`, `@hookform/resolvers`, `@react-three/drei`,
  `@tanstack/react-query`, `date-fns`, `tw-animate-css`, `zustand` (direct), plus
  `cmdk`, `embla-carousel-react`, `input-otp`, `react-day-picker`,
  `react-resizable-panels`, `react-tweet`, `recharts`, `svg-dotted-map`, `cobe`,
  `next-themes`, `@shadcn/react` (orphaned by the ui purge, re-verified post-delete).
  Added explicit `@types/three` devDep — it had been arriving only as a hoisted
  transitive dep of the removed `drei`; its absence broke `tsc` (caught by verification).
- Kept deliberately: `dotenv`, `ws`, `@tailwindcss/vite`, `shadcn` (tooling/uncertain),
  `motion`/`@base-ui/react`/`class-variance-authority`/`cn` (live), `groq` (see §4).
- Backend `requirements.txt`: added `transformers>=5.17,<6` (imported by
  `scripts/evaluate_external_isign.py:35` + smoke test, previously undeclared) and
  `pytest-asyncio>=0.23,<1` (all async tests require it; matches `pyproject` dev extra).
  Fixed broken `from backend.app…` import in `evaluate_external_isign.py:38`.
- Backend `config.py`: removed unread `host`/`port`/`max_sequence_seconds`
  (port comes from `$PORT` per `Dockerfile:32`; `extra="ignore"` makes removal
  startup-safe with existing `.env` files); `.env.example` updated to match.
  `backend/.env` (user file) untouched.

## 3. Deliberately preserved

- **Nested snapshot `staging_kaggle_results/signsage/` (259M, ignored): PROPOSED for removal,
  NOT executed.** Proven fully redundant: HEAD `d40ff1d` is an ancestor of main HEAD (same
  origin, no unique branches/stash); worktree `backend/` byte-identical, `frontend/`/`docs/`/
  `research/` older-or-equal (predates both cleanup passes; lacks `signsage_kaggle_fixed.ipynb`,
  has 132B placeholder `pose_encoder.pt` vs main's 1.1MB weights); `TASKS.md` byte-identical
  to versions in main history (`cfaaff4`, `9730f67`), its two unique nuggets preserved in
  `TESTING.md` §9. Safe removal command for the owner:
  `rm -rf staging_kaggle_results/signsage` (untracked data — no git recovery; owner's call).
- `results.zip` (12G), `staging_kaggle_results/{isign,s2t_v2}/` (14G Kaggle outputs),
  `research/artifacts/` skeleton, `models/` (+1.1MB weights + HF `.cache`), all 5 notebooks,
  both Kaggle runbooks, `tools/signsage_full_audit.py`, `test`-only `i_sign_t5_adapter.py`
  and `value_objects.py`, all tests, `.gitignore` (verified adequate), `backend/.env`,
  `frontend/.env.local`, stash, untracked operator scan files.
- Empty `__init__.py` package markers kept (packaging, zero cost).

## 4. Open notes

- `groq` PyPI package has zero backend imports (runtime uses `httpx`) but one guarded
  optional consumer (`signsage.ipynb` cell 55, `try/except`, documented never-required) — kept.
- `requirements.txt` ↔ `pyproject.toml` version-floor drift left as-is (both valid).
- `npm run lint` (~39 pre-existing errors) and manual browser E2E remain for a dedicated pass.

## 5. Validation

- Backend (project `backend/.venv`, has pytest-asyncio 1.4.0): **152/152 passed**
  (43s unit+integration+contract, 161s external-T5 smoke) — re-run AFTER all deletions.
  (System-python run shows 32 async failures from missing plugin — environment gap, not code.)
- `compileall` on `app`+`scripts`: PASS. `from app.main import app`: 5 routes, PASS.
- Frontend `npx tsc -b`: PASS (caught+fixed the `@types/three` gap). `npm run build`: exit 0, 12.9s.
- All 5 notebooks `json.load`: valid (81/15/15/15/19 cells).
- Markdown links (`*.md`, `docs/`, `research/training/`, `t5_pose_poc/`): ALL_OK.
- Deleted-path sweep: only intentional provenance notes in merged docs + updated AUDIT rows.
- `git status`: all changes staged, reviewable via `git diff HEAD --stat`; nothing committed/pushed;
  stash, untracked files, and user env files untouched.

## 6. Documentation consolidation pass (2026-10-13, docs-only, no code touched)

All 24 `docs/*.md` files read in full. Consolidated to **6 top-level + 2 reference**
(8 files incl. nested, was 24/30):

| New / kept file | Absorbed sources (unique content preserved) |
|---|---|
| `MASTER.md` (rewritten, ~230 lines) | Old MASTER (map, authority, claims, thesis) + ARCHITECTURE (mermaid, boundaries, risks) + DESIGN (visual spec, fail-loudly, responsive) + DECISIONS (16-row table, promotion/DTW rules) + PRD (scope, R1–R6, success criteria) + PHASES (run table, phase status, Phase-4 list, deferred) + PROJECT_STRUCTURE (accurate tree + artifact notes) |
| `DEVELOPMENT.md` (new) | API (canonical-protocol block, streaming semantics; self-declared-stale JSON examples trimmed) + DEPLOYMENT (topology, Render/bundle flow; stale `SIGNSAGE_PORT` line fixed) + SECURITY (incl. §5a license-inheritance gate) + TESTING (incl. §9 verification debt) + setup/workflow commands |
| `DATA_AND_MODEL.md` (new) | DATA_MODEL (dataset facts, representation, rules) + DATASET_STORAGE (layout, checksums, manifests, offline policy; trimmed overlap with DATA_MODEL §5) + SUPPORTED_ISL_VOCABULARY (empty-set verdict, exclusions, honesty rule) + MODEL_READINESS (inventory, tokenizer incompatibility, gates, promotion requirements) + license gate |
| `STATUS_AND_ROADMAP.md` (new) | STATUS (vocabulary, system table, passes, promotion rule, gaps) + APPLICATION_AUDIT (live-path table, model path, hallucination audit, test taxonomy; §2 trimmed to pointer) + PRODUCTION_BLOCKERS (B1–B6 + summaries A–E + order) + END_TO_END_VALIDATION (full scenario matrix + NOT FUNCTIONAL verdict) |
| `EXTERNAL_EVALUATION.md` (new) | CONTRACT_VERIFICATION condensed (contract table, checkpoint rec) + MODEL_VALIDATION condensed (load recipe, smoke, CPU numbers, status table) + QUALITY_EVALUATION (vocab classes, protocol, metrics, hallucination rule + examples, acceptance table, limitations, rubric, completed 0/1887 NO-GO) + Appendices A (adapter) + B (plan history) |
| `reference/contract-verification.md` (moved, trimmed) | Full line-by-line code-quote evidence §§1–12,14–18; §13 draft pseudocode replaced with pointer to implementation (single-spec rule) |
| `reference/end-to-end-contract.md` (moved, verbatim) | Exact-values pin, unchanged |
| `AGENTS.md` (kept, updated) | Read-first list + truth hierarchy + doc-update rules repointed to new names |

Removed after incorporation: the other 20 top-level docs (list in §2 of the
consolidation diff). Deliberately kept separate: `AGENTS.md` (tooling-required
repo instructions) and `reference/` (exact-value pins that would bloat the
authorities). `docs/SignSage_Project_Pipeline_Architecture.png` and
`reference/isl/` retained unreferenced-but-harmless (possible landing use).

Cross-reference updates: root README docs map, AGENTS.md, Kaggle eval notebook
(cell 5 + embedded output note), `evaluation_report.md:75`, `metrics.json:167`,
`backend/models/README.md:31` (stale `docs/ai-ml/model-contract.md` → reference).

Validation (docs-only pass; no source file changed, so the 152/152 suite,
`tsc`, and build results stand): Markdown links ALL_OK (root + docs +
reference + research + backend/models); deleted-path sweep clean except
intentional provenance notes; notebook JSON re-validated after pointer edits;
`git diff HEAD --stat`: 116 files, +989/−13,369.

