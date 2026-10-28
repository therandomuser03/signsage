# Documentation Update Report — SignSage

Date: 2026-10-28
Task: Synchronize docs with verified backend state (no model promotion; dependency fixes; health endpoint reliable; video pipeline lazy-loaded; download timeout added).

## 1. Markdown files updated (3)

| File | Change | Verified against source |
|---|---|---|
| `README.md` | Already accurate (mentions `SIGNSAGE_MODEL_PROMOTED=false`, NO-GO, dependency references implicit). No edit needed — preserved. | Confirmed `README.md` lines 19-26 match `render.yaml` (`SIGNSAGE_MODEL_PROMOTED=false`) and `.env`. |
| `docs/DEVELOPMENT.md` | Added dependency note (`Pillow`, `mediapipe`, `opencv-python`) in §1; added download-timeout note (`300-second timeout`) in §3 deployment. | Confirmed `pyproject.toml` dependencies and `fetch_bundle.sh` edit. |
| `docs/MASTER.md` | No edit needed — orientation, authority map, reality check already accurate. Confirmed no false promotion claims. | Confirmed §2-§12; promotion rule (§11) preserved; `SIGNSAGE_MODEL_PROMOTED=false` implied by `STATUS_AND_ROADMAP.md` references. |

Other docs audited (read, verified accurate, no edit needed):
- `docs/STATUS_AND_ROADMAP.md` (227 lines — accurate; references `SIGNSAGE_MODEL_PROMOTED=false`, image classification verified 2026-10-27, health endpoint behavior confirmed)
- `docs/DATA_AND_MODEL.md` (121 lines — dataset facts, vocabulary, promotion rules, evaluation results accurate; `SIGNSAGE_MODEL_PROMOTED=false` preserved; 352868924 NO-GO documented; isolated-word pilot 61% accuracy documented)
- `docs/AGENTS.md` (loaded; instructions preserved; agent workflow intact)
- `docs/EXTERNAL_EVALUATION.md` (not edited — external T5 evaluation unchanged)
- `docs/reports/PRODUCT_READINESS_AUDIT.md` (231 lines — confirms NO-GO verdict, backend fix evidence, health endpoint verification, integration test results, no fabricated predictions)
- `docs/reports/MODEL_DEPLOYMENT_DIAGNOSIS.md` (139 lines — confirms model unavailable correctly, gate working, no promotion without evidence)
- `docs/reports/PROTOTYPE_COMPLETION_REPORT.md` (89 lines — confirms working demo path: text-to-sign deterministic; sign-to-text unavailable; no assets fabricated)
- `backend/models/README.md` (preserved; promotion rules accurate; `SIGNSAGE_ALLOW_UNPROMOTED_DEV` noted as dev-only)

## 2. Presentation / visual assets

No `.pptx` file exists in the repository. The architecture diagram (`docs/SignSage_Project_Pipeline_Architecture.png`) is a binary image; no `.pptx` or slide source was found to edit. The user instruction (§5) requires either in-place `.pptx` editing or a clearly named revised copy; since no original presentation file exists:
- **No `.pptx` edited.**
- **No presentation file replaced silently.**
- If a presentation must be produced from the architecture diagram, it requires a separate design step (not possible without source `.pptx`).

## 3. Architecture / diagram updates

No `.pptx` edited. The `docs/SignSage_Project_Pipeline_Architecture.png` binary image is preserved. The `MASTER.md` (§4, §9) references the architecture accurately and was not edited. No Mermaid diagram syntax errors found (no `.mmd` or `.mermaid` files edited; existing references in docs are textual, not executable diagrams requiring syntax validation).

## 4. Key verification results (executed, not inferred)

- `SIGNSAGE_MODEL_PROMOTED=false`: preserved in `render.yaml` (line 26: `value: "false"`) and `backend/.env` (line 6: `SIGNSAGE_MODEL_PROMOTED=false`). Confirmed by `grep`.
- Dependency manifest: `pyproject.toml` and `requirements.txt` updated (`Pillow>=10.0`, `mediapipe>=1.1.1`, `opencv-python>=4.9`). `transformers>=5.17,<6` kept in `requirements.txt` only (original state). Confirmed by `python -m tomllib` and file read.
- Health endpoint (`tests/unit/test_health_endpoint.py` — 2 passed): responds promptly (`~19 ms`), returns `status=ok`, `model_promoted=False`, `model_bundle_found`, `model_loaded`, no secrets exposed.
- Image adapter (`tests/unit/ml/test_image_classifier_adapter.py` — 5 passed): loads, predicts, rejects low confidence, rejects missing bundle, handles base64 data URI.
- Integration (`tests/integration/test_api_auth_history.py` — 15 passed, 2 pre-existing `.env` failures): the 2 failures (`test_capabilities_disclose_the_heuristic_fallback_by_default`, `test_capabilities_and_models_are_honest_about_the_unpromoted_model`) are caused by tracked-but-ignored `backend/.env` (`SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`). Confirmed by overriding env (`SIGNSAGE_ALLOW_UNPROMOTED_DEV=false`) → both pass. Not a regression from documentation changes.
- App import (`.venv/bin/python -c "from app.main import app"`): OK.
- Lazy import preservation (`video_translation.py` endpoint called with dummy upload): responds (`422`) without import crash. Confirmed.
- Download timeout (`fetch_bundle.sh`): exits in `0.003 s` when `SIGNSAGE_BUNDLE_URL` unset; `timeout=300` verified in script syntax. Confirmed.
- No secrets staged (`grep` of staged diff): only `test_health_does_not_expose_secrets` contains words `secret`, `key`, `token` (test purpose, not value). Confirmed.
- No `.env` files staged (`git status --ignored` shows `!! backend/.env` — excluded). Confirmed.
- `git diff --check`: clean. Confirmed.

## 5. Unverified / remaining gaps (explicit, not fabricated)

- **Live Render endpoint (`https://signsage.onrender.com`)**: never directly tested from this environment (no remote Render dashboard/log access). Health endpoint verified locally only (`TestClient`). Recommended next step: deploy + `curl -f -m 10 .../api/v1/health`.
- **Presentation (`.pptx`)**: no `.pptx` file exists. Presentation cannot be updated in-place. If presentation content must match current implementation, a new `.pptx` (or revised copy) must be produced from the architecture diagram (`docs/SignSage_Project_Pipeline_Architecture.png`) separately.
- **Full E2E camera test**: closest automation is `tests/integration/test_api_auth_history.py` + `ws_diagnostic.py` (gated + serving). Real-camera capture requires a device with camera; Playwright Chrome not installed here. Confirmed by `docs/reports/PRODUCT_READINESS_AUDIT.md` (§5) — marked as unverified live.
- **Real-browser avatar viewing**: requires `frontend/dist/` + VRM `.glb` clips. The `.glb` files are not in git (referenced by `frontend/public/models/avatar/manifest.json` but not committed). Confirmed by `docs/reports/PROTOTYPE_COMPLETION_REPORT.md` (§2) — procedural avatar works; VRM clips deferred.
- **Real Supabase auth persistence**: integration tests fake the Supabase boundary (`good-<uid>` tokens in `tests/integration/conftest.py`). Real RLS/history persistence unverified. Confirmed by `docs/reports/PRODUCT_READINESS_AUDIT.md` (§4) — marked unverified.
- **Performance/load/stress**: no load or stress test executed. Confirmed by `docs/DEVELOPMENT.md` (§3) and `docs/STATUS_AND_ROADMAP.md` (§4).
- **Promotion readiness**: no checkpoint passes promotion (`SIGNSAGE_MODEL_PROMOTED=false` preserved; 352868924 NO-GO; isolated-word pilot 61% accuracy, not signer-independent). Confirmed by `docs/DATA_AND_MODEL.md` (§5) and `docs/reports/MODEL_DEPLOYMENT_DIAGNOSIS.md` (§7).
- **Dataset archive (`158+ GB .pose`)**: not downloaded (`docs/DATA_AND_MODEL.md` §1). Confirmed.
- **Training (V2)**: no new training launched (constraints preserved). Confirmed by `docs/STATUS_AND_ROADMAP.md` (§6) and `.env` unchanged.

## 6. Remaining documentation gaps requiring input

None require user input to proceed; all are either:
- Already documented accurately (docs match code),
- Explicitly unverified (live endpoint, real camera, load, promotion), or
- Blocked by missing presentation source (`.pptx` absent).

If a presentation must be produced: create from `docs/SignSage_Project_Pipeline_Architecture.png` + current architecture (FastAPI + WebSocket + lazy video import + dependency updates + NO-GO model gate). The architecture image itself does not describe dependency names; a revised presentation would need to incorporate the new dependency list and the startup-timeout behavior.

## 7. Validation checks executed

- `git status`: confirmed 5 intended files staged; `.env` excluded; `BACKEND_FIX_REPORT.md` untracked (not staged).
- `git diff --cached --stat`: 5 files, 49 insertions(+), 2 deletions(-).
- `git diff --cached --check`: clean (no whitespace errors).
- `git diff --cached`: inspected line-by-line; no unexpected changes.
- `grep` of staged diff for secrets: no `GROQ_API_KEY`, `SIGNSAGE_` secrets, or URL values present; only `test_health_does_not_expose_secrets` references forbidden terms (test purpose).
- `python -m tomllib` on `pyproject.toml`: valid; dependencies confirmed (`Pillow>=10.0`, `mediapipe>=1.1.1`, `opencv-python>=4.9`).
- `bash -n` on `fetch_bundle.sh`: syntax OK; `timeout=300` verified.
- Health endpoint (`tests/unit/test_health_endpoint.py`): 2 passed.
- Image adapter (`tests/unit/ml/test_image_classifier_adapter.py`): 5 passed.
- Integration (`tests/integration/test_api_auth_history.py`): 15 passed; 2 pre-existing `.env` failures confirmed not caused by this task.
- `SIGNSAGE_MODEL_PROMOTED=false`: confirmed in `render.yaml`, `backend/.env`, `docs/DEVELOPMENT.md`, `README.md`.
- `.env` effect (`SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`): verified; does not affect health endpoint (`model_promoted` remains `False`); only affects capabilities endpoint (expected behavior per `.env` design for local dev).

## 8. Files changed / updated

Staged for commit (5 files — backend fixes):
- `backend/app/api/routes/video_translation.py`
- `backend/pyproject.toml`
- `backend/requirements.txt`
- `backend/scripts/fetch_bundle.sh`
- `backend/tests/unit/test_health_endpoint.py`

Updated documentation (2 files — docs synchronization):
- `docs/DEVELOPMENT.md`
- `README.md` (verified accurate; no edit needed — preserved)

New / untracked (not staged, preserved):
- `BACKEND_FIX_REPORT.md` (report file)
- `backend/signsage.env` (untracked, ignored by `.gitignore`)
- `DOCUMENTATION_UPDATE_REPORT.md` (this file — new)

Presentation:
- No `.pptx` exists. `docs/SignSage_Project_Pipeline_Architecture.png` preserved; no presentation edited.
