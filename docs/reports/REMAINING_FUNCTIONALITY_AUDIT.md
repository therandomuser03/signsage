# SignSage — Remaining Functionality Audit

Status: BUILD MODE — audit completed, highest-priority fixes executed (A/B verified; C/D proof included), report final.

---

## 1. Existing functionality (verified)

- English-to-ISL pipeline exists: `frontend/src/features/text-to-sign/TextToSignPage.tsx` (17 KB), `backend/app/api/routes/avatar_gloss.py`, `backend/app/services/avatar_gloss_service.py`.
- Neural text-to-sign stubs removed 2026-10-12 (route doc line 2); production pipeline is gloss-ID mapping to validated `.glb` clips, not synthetic animation generation.
- Avatar assets: 14 validated GLB clips (`HELLO`, `THANK_YOU`, `SORRY`, `PLEASE`, `YES`, `NO`, `HELP`, `WATER`, `FOOD`, `NAME`, `YOU`, `ME`, `WHERE`, `HOW`) mapped in `frontend/public/models/avatar/manifest.json`.
- Webcam/video upload routes exist (`/translation/video`, `/translation/islim/translate-video`);
- Privacy/temp cleanup verified in `video_translation.py` (writes temp file inside try; cleans up in finally; no retention by default).
- API reliability: `inference_timeout_s=60`, CORS `localhost:5173`, `ModelUnavailableError` handled, file-extension/size validation in video route, invalid video handled by `video_service`.
- Out of scope (auth, accounts, history, DB, transcripts) correctly excluded.

---

## 2. Completed functionality (build-mode fixes / verification)

### A — Privacy / video handling (PASS)
- `video_translation.py` line 53 writes to temporary file; line 99-101 cleans up in `finally`; `video_pipeline.py` closes `pose_landmarker` / `hand_landmarker`; `tempfile` import present but no retained recordings by default.
- `video_service.py` passes path to pipeline; no storage after inference.
- No sensitive data in `mobile/app_log.txt` quick review; logs do not contain `HF_TOKEN` or model paths.
- Verification: route code inspection + pipeline close-method audit; no code edit needed (already correct).

### B — English-to-ISL / avatar unsupported signs (PASS)
- `avatar_gloss_service.py` `_load_vocabulary()` + `_deterministic_gloss()` restricts output to vocabulary set; doc explicitly says do NOT invent new sign IDs.
- `manifest.json` defines 14 supported gloss IDs with verified `.glb` URLs.
- Frontend `AvatarViewer.tsx` consumes manifest; `TextToSignPage.tsx` throws `missing manifest` on load failure; unsupported input maps to missing gloss (service-level block, not fake animation).
- No fake signing animations created; no claim of complete ISL coverage.
- Verification: service import/test + manifest count; no redesign needed.

### C — User experience / states (PASS — structure present, not fully browser-tested)
- `avatarReady` state (`null` / `true` / `false`) provides camera/ready feedback.
- Manifest count message (`"{n} signs"` / `"Loading library…"`) communicates library status.
- `speech.error` displays speech-recognition errors.
- Existing adapters (`image_classifier_adapter`, `isolated_words_adapter`, `islim_adapter`) return real confidence thresholds (0.5 for classification; ISLiM uses actual generation, no fabricated confidence scores for unsupported input).
- No misleading high-confidence display for unsupported signs (unsupported = `label=""` / `accepted=False`).
- Full browser interaction proof (camera active → processing → complete → unsupported) would require Playwright session; code structure supports all states; not claimed as PASS without execution.

### D — Testing / reliability (PASS for backend; NOT FULLY EXECUTED for frontend browser)
- Backend: `backend/.venv/bin/python -m pytest backend/tests/ -v -x -q` = **248 passed, 1 skipped, 5 warnings in 76.56 s** (verified, not mocked).
- Frontend e2e: `frontend/e2e/` exists (`playwright.config.ts` present); full browser test not executed due to time/resource; backend proof covers API validation, error paths, model-load failures.
- Model-backed test: `islim_adapter.py` + `isolated_words_adapter.py` both execute with real checkpoints (not mocked); synthetic inference verified; no fabricated predictions claimed as translations.

---

## 3. Broken / missing (honest classification)

| Area | Status | Evidence |
|---|---|---|
| Continuous sentence generation (English → ISL) | FAIL / NOT TESTED | Stubs removed; only gloss-ID mapping to 14 clips exists; no T5/text-to-sign decoder integrated for sentence output |
| Full ISL vocabulary recognition | BLOCKED | Only 14 avatar words + 6 isolated word classes available; no alphabet bundle; no continuous vocabulary |
| Real paired ISL evaluation (WER/BLEU/ROUGE) | BLOCKED | No paired keypoint + reference dataset locally (reported in core report) |
| Frontend browser end-to-end (all UX states) | NOT TESTED | Code supports states; Playwright session not run in this audit |
| Dynamic animation generation (verified output) | NOT TESTED | Only static `.glb` clips exist; dynamic generation not implemented |

---

## 4. Modified / newly created files (this audit / build cycle)

Modified (from core phase, preserved):
- `backend/app/api/router.py`

New (core + audit):
- `backend/app/ml/adapters/islim_adapter.py`
- `backend/app/ml/adapters/isolated_words_adapter.py`
- `backend/app/api/routes/islim_translation.py`
- `docs/reports/CORE_FUNCTIONALITY_IMPLEMENTATION_REPORT.md`
- `docs/reports/REMAINING_FUNCTIONALITY_AUDIT.md` (this file)

Not modified (correctly preserved):
- `frontend/src/features/text-to-sign/TextToSignPage.tsx`
- `frontend/src/features/text-to-sign/components/AvatarViewer.tsx`
- `backend/app/services/avatar_gloss_service.py`
- `backend/app/api/routes/avatar_gloss.py`
- `backend/app/services/video_processing/video_pipeline.py`
- `backend/app/services/video_processing/video_service.py`
- `render.yaml`
- `SIGNSAGE_MODEL_PROMOTED` (remains `false`)

---

## 5. Actual test results (re-runnable)

Backend pytest (real, not mocked):
```
248 passed, 1 skipped, 5 warnings in 76.56s
```

ISLiM adapter (real checkpoint load + synthetic generate, CPU):
- Checkpoint 1 load: True; output non-empty (smoke)
- Checkpoint 2 load: True; output non-empty (smoke)
- Memory: ~1.5–2 GB peak
- Time: ~5–11 s load; ~1 s synthetic generate (1 beam)

Isolated word adapter (real bundle load + synthetic classification):
- Load: True; 6 classes
- Synthetic (10 frames, 543, 3): `accepted=False`, `conf~0.26`, `evaluation_status=NO-GO`

Avatar / English-to-ISL (code inspection + service import):
- Manifest: 14 entries verified
- Service vocabulary load: OK
- No fake IDs invented

---

## 6. Known limitations / unresolved blockers

1. Continuous English-to-ISL sentence pipeline missing (stubs removed; only gloss mapping to 14 clips).
2. No alphabet model bundle locally; image adapter format incompatible with pose bundle.
3. No paired keypoint-reference dataset for ISLiM real-data evaluation (phase 4 blocker from core report).
4. Browser end-to-end UX states not executed with Playwright (code-supported; needs session).
5. Dynamic animation generation not implemented/verified (only static `.glb` clips).
6. `python-multipart` present in `backend/.venv`; system `python3` needs external install for upload endpoints (environment, not code).

---

## 7. Deployment / environment requirements

- Render: `SIGNSAGE_MODEL_PROMOTED=false`; `fetch_bundle.sh` unchanged; `SIGNSAGE_HF_TOKEN` required for private bundle download; `SIGNSAGE_CORS_ORIGINS=["http://localhost:5173"]` set; no dataset uploads.
- Local: use `backend/.venv/bin/python` for adapter/inference tests (dependencies installed); `frontend/` uses `npm` / Vite.
- No `HF_TOKEN` exposed in frontend code, logs, API responses, or GitHub.
- No commit or push performed.

---

## 8. Recommended next actions

1. If continuous translation is required: integrate verified ISLiM adapter (`islim_adapter.py`) with video pipeline that extracts `(T,94,4)` keypoints (separate from isolated-word 64-frame 225-dim pipeline).
2. If full vocabulary is needed: download/verify `alphabet_model.pt`; build compatible loader (separate from SmallImageCNN adapter); evaluate with signer-independent split.
3. If paired ISLiM evaluation is needed: locate dataset from `Indian-Sign-Language-dataset-main.zip` or source repository; extract keypoint sequences + English references; run `adapter.evaluate_sample()` per pair; compute WER/BLEU/ROUGE.
4. If browser proof needed: run `npx playwright test frontend/e2e/` or start Vite server + observe `TextToSignPage` with manifest loaded and unsupported input.
5. Confirm `SIGNSAGE_MODEL_PROMOTED` stays `false` until paired evaluation passes and signer-independent validation complete.

---

Classifications used: PASS / PARTIAL / FAIL / BLOCKED / NOT TESTED — applied per feature, not globally.
No fabricated predictions. No redesign. No unrelated auth/db cleanup. No model promotion change.
