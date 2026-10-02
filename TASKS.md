# SignSage — Integration + Deployment Tasks

Locked decisions: dev-only unpromoted mode · bundle download-at-start on Render ·
bounded-backoff reconnect · no Prisma/redesign/fakes · preserve `/ws/translate` + 543×3 contract.

Do tasks in order. Each task lists files, what to do, and how to verify.
Rule: never fake a translation; surface `MODEL_UNAVAILABLE` / `accepted=false` honestly.

## 0. Pre-flight (verify before changing anything)

- [x] `cd backend && python -m compileall app && pytest -q`
- [x] `cd frontend && npm install && npm run build`
- [x] Confirm WS path is unversioned `/ws/translate`:
  `backend/app/api/router.py:12`, `backend/app/api/websocket/translation.py:32`,
  `frontend/src/app/config.ts:3-5`, `frontend/src/services/websocket/protocol.ts:1`
- [x] Confirm model gate: `SIGNSAGE_MODEL_PROMOTED=false` → WS closes with
  `MODEL_UNAVAILABLE` (`backend/app/ml/models/registry.py:41-47`)

## 1. Backend finite-value validation

Files: `backend/app/schemas/translation.py` (shared by `websocket.py:38`).

- [x] Extend `validate_landmark_rows` to reject NaN / ±Inf (and non-list rows)
      with a clear `ValueError`; keep the existing length==3 check.
- [x] Add tests: NaN frame, Inf frame, ragged row, wrong count (542/544) →
      `INVALID_MESSAGE`; oversized message → `MESSAGE_TOO_LARGE`.
- [x] Verify: `cd backend && pytest -q tests/integration/test_websocket_auth_and_limits.py tests/contract/`

## 2. Dev-only unpromoted mode (local real inference, gate intact)

Files: `backend/app/core/config.py`, `backend/app/ml/models/registry.py`,
`backend/app/core/dependencies.py`, `backend/app/api/websocket/translation.py`,
`backend/app/main.py`, `backend/.env.example`.

- [x] Add `SIGNSAGE_ALLOW_UNPROMOTED_DEV: bool = False` to `Settings`.
- [x] Thread `allow_unpromoted` from settings through
      `build_translation_service` → `get_sign_to_text_pipeline`
      (WS route + REST dependency + lifespan warm-up).
- [x] Enforce locality: honor the flag only for local/dev use
      (e.g. require explicit opt-in and refuse when running against the
      production bundle path unless operator overrides knowingly).
      Never silently flip `SIGNSAGE_MODEL_PROMOTED`.
- [x] Document in `.env.example` + `backend/models/README.md`:
      real checkpoint still NO-GO quality (WER ~1.0); dev use only.
- [x] Verify: gated (default) → `MODEL_UNAVAILABLE` on connect;
      dev flag + supplied bundle → `translation.text` with real
      `accepted/reason/confidence`.

## 3. Frontend bounded-backoff reconnect

Files: `frontend/src/services/websocket/connection.ts`,
`frontend/src/services/websocket/client.ts`,
`frontend/src/features/translate/TranslatePage.tsx`.

- [x] Plumb `maxReconnectAttempts` / `reconnectBaseDelayMs` through
      `TranslationSocket` → `Connection` (default e.g. 5 attempts, 500ms base,
      exponential backoff already in `Connection.maybeReconnect`).
- [x] Show "Reconnecting…" state (existing `connecting` status) instead of
      fake "Live"; stop sending frames while closed; single socket only.
- [x] Cancel reconnect + close socket + stop camera tracks + `extractor.close()`
      on Stop/unmount (extend existing `teardown`).
- [x] Verify: kill backend mid-session → reconnecting UI, no frame sends,
      no duplicate sockets; restart backend → session resumes after
      manual/auto reconnect; leave page → everything torn down.

## 4. Env examples + doc corrections (no secrets)

Files: `frontend/.env.example`, `backend/.env.example`, `README.md`.

- [x] Rewrite `frontend/.env.example` with real vars:
      `VITE_SIGNSAGE_API_URL`, `VITE_SIGNSAGE_WS_URL`,
      `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`
      (dev defaults noted in `frontend/src/app/config.ts:7-13`).
      Delete the stale `DATABASE_URL` placeholder.
- [x] Fix README: WS default is `/ws/translate` (not `/api/v1/ws/translate`);
      reconnect is manual until Task 3 lands; keep NO-GO disclosure.
- [x] Verify: `grep -rn "localhost\|127.0.0.1\|ws://" frontend/src`
      → only `config.ts:7` default + comments remain.

## 5. Render readiness (bundle download-at-start)

Files: `backend/Dockerfile`, new `render.yaml`, new `backend/scripts/fetch_bundle.sh`,
`backend/app/infrastructure/storage/model_storage.py` (unchanged — sole path owner).

- [x] Add `render.yaml`: web service, build `pip install .` (CPU torch),
      start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`,
      health check `/api/v1/health`, env placeholders
      (`SIGNSAGE_MODEL_PROMOTED=false`, `SIGNSAGE_CORS_ORIGINS`,
      `SIGNSAGE_BUNDLE_URL`, `SIGNSAGE_BUNDLE_SHA256`).
- [x] Add `fetch_bundle.sh`: download `$SIGNSAGE_BUNDLE_URL` to
      `/models/signsage_deployment_bundle`, verify SHA256, fail loudly
      on missing/checksum mismatch. Wire into start command.
      Do NOT bake weights into the image.
- [x] Note: `Settings.port` reads `SIGNSAGE_PORT`; Render's `$PORT` is
      honored by the Docker CMD shell — keep that, document it.
- [x] Verify: clean build → `/api/v1/health` reports
      `{status:"ok", model_promoted, model_bundle_found, model_loaded}`;
      missing bundle → `found:false`, clear log, no crash.

## 6. WS transport diagnostic (synthetic, labelled)

New: `backend/scripts/ws_diagnostic.py` (dev-only tool).

- [x] Connect → send synthetic 543×3 ramp frames (code comment:
      TRANSPORT/CONTRACT TEST — not sign recognition) →
      `sentence.boundary` → assert structured `translation.text` or
      honest `MODEL_UNAVAILABLE` when gated.
- [x] Cover: invalid JSON, bad shape, unknown type, empty boundary,
      oversize message → matching `error{code,message}`.
- [x] Verify: `python backend/scripts/ws_diagnostic.py ws://localhost:8000/ws/translate`
      passes in both gated and dev-unpromoted modes.

## 7. Observability (sampled, no PII)

Files: `backend/app/api/websocket/translation.py`, `backend/app/main.py`.

- [x] Log `[WEBSOCKET]` connect/disconnect/errors, `[MODEL]` load outcome,
      `[INFERENCE]` per `sentence.boundary` only (result + ms).
      No per-frame logs, no tokens, no landmark data.
- [x] Verify: run diagnostic → logs show session lifecycle + inference
      latency without frame spam.

## 8. Final verification (definition of done)

- [x] `cd backend && pytest -q` green (118); `cd frontend && npm run build` green.
  `npm run lint` still reports 39 pre-existing react-hooks/style errors
  (40 at baseline; none on edited lines) — not introduced here, left for a
  dedicated lint pass.
- [ ] Local E2E in a real browser: camera → overlay dots → `framesSent`
  increments at ~10fps → Stop sends `sentence.boundary` → backend reply
  renders. NOT verifiable headless — still to do by hand.
  Verified instead: `ws_diagnostic.py` over real sockets against a live
  server, both gated (`MODEL_UNAVAILABLE`) and serving (tiny real bundle +
  dev flag: 25 synthetic frames → real inference → honest
  `accepted=false/insufficient_motion`).
- [x] Refresh/repeat; disconnect cleanup confirmed (server logs show
  connect/disconnect; diagnostic closes cleanly).
- [x] Write final report per brief §31 (status table, files changed,
      fixes, blockers — esp. bundle absence + NO-GO quality, commands,
      LOCAL vs PRODUCTION env names, what was actually tested).

## Known blockers (do not hide)

- `research/artifacts/` contains empty shells only (gitignored); the ~112MB
  deployment bundle (checkpoint + SentencePiece + contracts) is not in this
  checkout — operator must supply `SIGNSAGE_BUNDLE_URL` (+ SHA256).
- Checkpoint 352868924 is evaluated NO-GO (WER ~1.0, ~1% output diversity);
  pipeline work is technical integration, not accuracy validation.
- No `DTW/demo-prototype` fallback exists in code despite README wording —
  do not reintroduce one as fake intelligence.
