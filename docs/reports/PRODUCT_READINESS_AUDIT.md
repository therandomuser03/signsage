# SignSage — Product Readiness Audit (source + executed checks are authority)

Date: 2026-10-12. Docs consulted as guidance only: `docs/MASTER.md`, `DEVELOPMENT.md`,
`DATA_AND_MODEL.md`, `STATUS_AND_ROADMAP.md`, `EXTERNAL_EVALUATION.md`.
`SECOND_PASS_CLEANUP_REPORT.md` / `REPOSITORY_CLEANUP_REPORT.md` are **not present
on disk** (staged adds, then deleted in working tree — see Git row). All status
below comes from files on disk + commands run in this session.

Constraint compliance: no training started, no gates/auth disabled, no mass
reorganization, no artifacts deleted, no fake predictions, no new DB/framework/service,
no secrets exposed (`.env` values never read — only key names / booleans derived
from code), no commit/push/reset.

## Subsystem table

| Subsystem | Status | Evidence (file + executed check) | Blocker | Next action |
|---|---|---|---|---|
| Git / repo state | Staged cleanup, uncommitted | `git status`: branch `master`; staged: cleanup deletions (docs 24→6, dead UI components, neural text-to-sign stubs, req additions) + small config/comment edits; unstaged: 2 report files deleted in working tree; untracked: `scan_signsage_structure.py` (preserved, untouched) | Reports cited by task are absent on disk; staged tree is uncommitted (must stay so per constraints) | Do not commit; leave tree as-is; this audit replaces reliance on the missing reports |
| Backend entry / startup | Implemented, verified start | `backend/app/main.py` (`uvicorn app.main:app`, lifespan warms model, failure logged not fatal); `import app.main` → ok; TestClient boot → ok | None | None |
| REST health / capabilities / models | Implemented, verified live | TestClient 2026-10-12: `GET /health` 200 `promoted:false bundle_found:false loaded:false`; `GET /capabilities` 200 `sign_to_text_available:false reason:model_not_promoted avatar_vocab:14 landmark:543`; `GET /models` 200 `promoted:false loaded:false metrics:{}` | None — honesty machinery works | None |
| REST sign-to-text | Implemented, honestly gated | `backend/app/api/routes/translation.py`; TestClient `POST /sign-to-text` → **503 `MODEL_NOT_PROMOTED`** (not 500) | No promotable weights (see Model row) — endpoint correctly refuses | Keep gate; do not bypass |
| WebSocket `/ws/translate` | Implemented, verified gated path | `backend/app/api/websocket/translation.py` ↔ `schemas/websocket.py` ↔ `frontend/.../websocket/protocol.ts` (543×3, interim/final, error codes match); TestClient WS connect → first frame `{"type":"error","code":"MODEL_UNAVAILABLE",...}` + close | Same model blocker; live-recognition path untestable until promotion | Keep; re-run `ws_diagnostic.py` only after a promoted bundle exists |
| Auth / session | Implemented, partially verified | `core/security.py` (Supabase `GET /auth/v1/user` + TTL cache) + `api/routes/auth.py` (`GET /auth/me`); TestClient `GET /auth/me` no-token → 401 `UNAUTHORIZED`; `auth_enabled=True`, `auth_required=False` derived from settings (values not read); `frontend/services/supabase/{client,auth}.ts` (guest-safe, auto-refresh) | **Live Supabase unverified**: integration tests fake the Supabase boundary (`good-<uid>` tokens, `tests/integration/conftest.py`); no real register/login/logout run in this env | Report as unverified; do not claim auth works end-to-end |
| DB / history / migrations | Implemented, partially verified | `infrastructure/database/migrations/001_translation_sessions.sql` (RLS own-rows, no UPDATE) + `supabase.py` (user-token forwarding) + `history_service.py` (best-effort `try_record`); `history_backend=supabase` in local settings; tests use `memory` backend — 41 auth/history/WS/contract tests pass | **Live Postgres unverified** (no real RLS round-trip run here); Supabase outage path relies on code inspection + mocked tests | Run real history CRUD only where live credentials/service exist; otherwise leave as unverified |
| Webcam / landmarks | Implemented, code-verified only | `frontend/.../translate/landmarks.ts` (`HolisticExtractor`, 543 = 33+468+21+21, order POSE→FACE→LHAND→RHAND matches `backend/.../landmark_contract` + `TOTAL_LANDMARKS=543`); `TranslatePage.tsx` (permission denied, detector failure, backend offline, model-unavailable, empty-frame skip, pause/resume/stop/clear, interim vs final) | **No real-camera run in this env** (no webcam); closest automation is gated WS + serving path | Mark webcam recognition untested with real input; do not claim it works |
| Model loading / bundle paths | Correctly refusing, verified | `infrastructure/storage/model_storage.py` (`resolve_bundle_dir` requires `checkpoint/ tokenizer/ landmark_contract.json sign_to_text_deployment.json`); `ml/models/registry.py` (promotion gate); settings: `bundle_dir=../research/artifacts/352868924/...`, `promoted=False`, `allow_unpromoted=False`; `research/artifacts/.../checkpoint/` + `tokenizer/` are **empty dirs**, contracts absent → `ModelBundleNotFoundError` + `ModelNotPromotedError` both confirmed live | **No usable checkpoint exists anywhere**: in-house `.pt` absent (gitignored skeleton only); `staging_kaggle_results/` has pose_cache shards + tokenizer but no checkpoint; `models/iSign_t5_small_motion/` has only 1.1 MB `pose_encoder.pt` (T5 body weights absent) | Smallest viable demo = current honest refusal + avatar-gloss direction (below). No training per constraints; next training (V2) is operator's separate step, not this task |
| Inference / preprocessing / tokenizer | Pipeline code verified, weights absent | `ml/{adapters/sign_to_text_adapter, inference/sign_to_text, postprocessing/{decoding,confidence}, preprocessing/{landmarks,normalization,sequence}}`; synthetic check: random 10×543×3 motion ≈ 1.16, static zeros = 0.0; external-T5 adapter smoke `test_external_isign_t5_smoke.py` → **1 passed**; full suite → **152 passed** (82 s) | No real input → real prediction possible today; diversity-collapse guard history (epoch-2 stop) is consistent with NO-GO verdict — checkpoint must stay unpromoted | Do not invent predictions; re-evaluate only after a future promoted bundle |
| Frontend build / routes / API clients | Implemented, verified build | `frontend/src/{main.tsx, app/{routes,shell,config}, services/{api,websocket,supabase}, features/{translate, text-to-sign, history, ...}}`; `npm run build` (`tsc -b` + vite) → **PASS** 8.9 s, 2982 modules; `endpoints.ts` paths match backend OpenAPI (contract test passes); `config.ts` derives WS from API URL | `npm run lint` has ~39 pre-existing errors (dedicated pass, out of scope) | None for demo; fix lint separately |
| Text-to-sign / avatar gloss | Implemented, verified deterministic path | Backend `services/avatar_gloss_service.py` (Groq→Ollama→deterministic, vocab-filtered, never invents IDs) + `app/data/sign_vocabulary.json` (14 IDs = keys of `frontend/public/models/avatar/manifest.json`, contract-tested); live TestClient: `POST /text-to-sign {"text":"hello thank you water"}` → 200 `["HELLO","THANK_YOU","WATER"] source:deterministic`; `GET /vocabulary` → 14 IDs; frontend `TextToSignPage` + `glossMapper` (dictionary + assisted, unknown words dropped, missing VRM handled with honest empty state) | VRM `.glb` clips + `signsage.vrm` not bundled (avatar panel honestly shows unavailable; gloss preview still works); Groq/Ollama stages fell through gracefully in this env (upstream 404s → deterministic) | Ship deterministic + assisted-gloss preview as the working demo direction; no avatar-generation pipeline |
| Tests / templates / deploy | Present, executed where local | `backend/tests/` (contract/integration/unit/ml); `pytest -q` → **152 passed**; `frontend` build PASS; `backend/.env.example` + `frontend/.env.example` present (real `.env` files exist; values never read); `render.yaml` + `backend/scripts/{fetch_bundle.sh, model_check.py, ws_diagnostic.py, health_check.py}` + `Dockerfile` present and consistent (no baked weights, honest no-bundle start) | Live deploy / target-device latency / load claims have no basis | Smoke tests prove protocol + gating only — never quality |

## What this means for the demo

- **Working today (verified this session):** app boots; health/capabilities/models honest;
  REST + WS refuse without a model with predictable `MODEL_NOT_PROMOTED` / `MODEL_UNAVAILABLE`
  (never 500, never fake text); avatar-gloss deterministic mapping works end-to-end
  (backend + frontend shapes match); frontend builds; camera/permission/offline/model-down
  states are handled in code; history/auth boundaries are code-correct with mocked integration.
- **Not working / not claimed:** sign recognition itself — supported vocabulary is ∅,
  no loadable checkpoint exists, and no real-input prediction was (or can be) produced.
  Auth/DB live round-trips and real-camera capture are **unverified in this environment**.
- **Smallest viable demonstration:** (1) Translate page showing live camera preview +
  landmark overlay + honest `MODEL_UNAVAILABLE` refusal; (2) Text→Sign page generating
  real gloss sequences from the 14-sign library (deterministic now, assisted where
  configured) with unknown words dropped, never guessed. No external LLM needed for this.

## Fixes applied after this audit (kept minimal, see final report)

1. `CapabilitiesPage.tsx` honesty copy: Sign→Text badge `Live` → research-preview
   wording (was claiming "Everything below exists… Nothing listed is aspirational"
   while recognition has ∅ supported vocabulary and the backend correctly refuses).
2. `glossMapper.ts` stale comment: `backend/app/services/gloss_service.py` →
   `avatar_gloss_service.py` (file never existed under the old name).

---

# Final integration pass (2026-10-12, second session — real servers, no mocks)

Prior fixes preserved (unstaged diff for both files confirmed intact; untracked
`PRODUCT_READINESS_AUDIT.md` + `scan_signsage_structure.py` untouched; staged
cleanup left as-is, uncommitted). No training, no gate bypass, no commits.

## Reproducible local run instructions

Backend (from `backend/`, existing `.venv`, Python 3.13.5, port 8000):

```bash
cd backend
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# dev variant per docs/DEVELOPMENT.md: uvicorn app.main:app --reload
```

Setup only if needed: `python3 -m venv .venv && .venv/bin/pip install -r
requirements.txt`; `cp .env.example .env` (never overwrite an existing `.env`).
Manifests: `backend/pyproject.toml` (requires-python >=3.11) + `backend/requirements.txt`.

Frontend (from `frontend/`, Node v26.5.0, port 5173):

```bash
cd frontend && npm install && npm run dev        # :5173
npm run build                                     # tsc -b + vite → dist/
```

`frontend/src/app/config.ts` defaults the API to `http://localhost:8000` and
derives `ws://…/ws/translate`; this matches the backend CORS default, so plain
local startup needs no env edits. Stop with normal SIGTERM/SIGINT (kill PIDs).

## User flows verified against REAL servers this pass (not TestClient, not mocks)

| Flow | Method | Result |
|---|---|---|
| Backend connectivity | `GET /api/v1/health` on live :8000 | 200 `promoted:false bundle_found:false loaded:false` |
| Capability honesty | `GET /api/v1/capabilities` live | 200 `sign_to_text_available:false reason:model_not_promoted`, vocab 14, landmarks 543 |
| Model status | `GET /api/v1/models` live | 200 unpromoted, unloaded, empty metrics |
| Text-to-Sign | `POST /api/v1/text-to-sign` live | 200 `["HELLO","THANK_YOU","WATER"] source:deterministic` |
| Vocabulary | `GET /api/v1/text-to-sign/vocabulary` live | 200, 14 IDs |
| Sign-to-Text unavailable state | `POST /api/v1/translation/sign-to-text` live | 503 `MODEL_NOT_PROMOTED` (predictable, never 500/fake) |
| Sign-to-Text WS unavailable state | real `ws://127.0.0.1:8000/ws/translate` via `websockets` lib | first frame `MODEL_UNAVAILABLE`, then clean close 1000 |
| Protected routes | `GET /api/v1/auth/me`, `GET /api/v1/history` live, no token | both 401 `UNAUTHORIZED` — guests blocked, no data leaked |
| API docs / CORS | `GET /docs` 200; `GET /health` with `Origin: http://localhost:5173` 200 | docs serve; vite origin accepted |
| Frontend serving | `npm run dev` + curl `/`, `/app/translate`, `/capabilities`, `/app/text-to-sign` | all 200 |
| Real-browser render (headless Chromium, `--dump-dom`) | `/` renders app DOM; `/capabilities` contains "What SignSage does" + "Research preview"; `/app/translate` contains "Live Interpreter" + "Start interpreting" + honest initial model status | served build includes prior honesty fixes; no JS crash (only expected headless-WebGL warnings on stderr) |
| Server logs | backend + vite logs reviewed | only expected lines: our 401 probes, Groq/Ollama→deterministic fallthrough, MODEL refusal warnings; zero tracebacks, zero vite errors |

## Explicitly mock-only or static (NOT claimed as live)

- **Auth login/logout/session + history persistence:** integration tests fake the
  Supabase boundary (`good-<uid>` tokens in `tests/integration/conftest.py`) with
  `history_backend=memory`; `test_supabase_repository.py` uses `httpx.MockTransport`.
  Real Supabase HTTP was never touched — no live register/login/RLS round-trip exists
  in this environment. Auth *enforcement shape* (401s) is proven live; identity
  *issuance and persistence* are not.
- **Camera capture / landmark detection:** no camera device here. `TranslatePage`
  renders and its permission-denied / detector-failure / offline / model-down copy is
  present in source and the served bundle, but no real frame was captured.
- **Recognition accuracy:** nothing to exercise — no servable weights (next section).

## Saved-artifact inspection (filenames + metadata only; 12 GB archive NOT extracted)

- `results.zip` (12 GB): name listing only (415 entries). Contains
  `s2t_v2/checkpoints/signsage_transformer_full.pt` (~203 MB) +
  `signsage_transformer_recovery.pt` (~203 MB) **but** `s2t_v2/COLLAPSED.txt` reads:
  `epoch 2: unique_fraction=0.0195 < 0.05; val_loss=22.8017; top='…' (70.3%)` — the
  diversity-collapse guard fired at epoch 2, exactly as previously reported. The archive
  holds tokenizer pieces (`signsage_en_sp.model/.vocab`, `tokens_meta.json`,
  `tokens.npz`) but **no** `signsage_deployment_bundle/` (no `landmark_contract.json` /
  `sign_to_text_deployment.json`). Verdict: a collapsed checkpoint without a deployment
  contract is not a servable pair — extracting 400 MB of weights would prove nothing
  beyond what the guard already verdicts, so nothing was extracted or copied.
- `staging_kaggle_results/s2t_v2/`: `gate_report.json` → `passed: False` (threshold 0.95;
  constrained exact 0.921875); `measure_report.json` (2000-sample cache survey),
  `pose_preflight.json` (127238 pose members, probe frames `[160, 1629]`),
  `dataset_audit.json` (127237 rows → 123779 usable, 3797 videos). Planning/audit
  metadata only — no weights, no bundle.
- `models/iSign_t5_small_motion/`: only 1.1 MB `pose_encoder.pt` + `config.json`; T5 body
  weights absent. `research/artifacts/352868924/…/checkpoint/` + `tokenizer/` remain empty.
- Conclusion: **no compatible checkpoint/tokenizer/contract triple exists anywhere**.
  Smallest viable alternative stands: honest refusal + Text-to-Sign demo (below). No gate
  was bypassed and no recognition result was fabricated.

## Defects found this pass

None. Every exercised flow behaved as designed (including graceful Groq/Ollama→
deterministic fallthrough on the live server). Per the no-speculation rule no code was
changed in this pass — the tree contains only the two prior honesty fixes plus this audit.

## Demo-readiness verdict: CONDITIONAL GO (constrained demo)

- **Go:** Text-to-Sign gloss generation (deterministic, vocab-filtered) end-to-end over
  real HTTP; full app shell served and rendered in a real browser; all unavailable-model,
  auth-required, offline, and camera-denied states honest and exercised where locally possible.
- **No-go (unchanged, by design):** sign recognition — ∅ supported vocabulary, no
  promotable weights, WS/REST correctly refuse. Authenticated history and real camera
  capture are wired but **unproven live**; prove them with real credentials/hardware before
  any demo that depends on them.
- Credible 5-minute demo: (1) Translate page — camera preview + landmark overlay +
  `MODEL_UNAVAILABLE` refusal on sentence stop; (2) Text→Sign — `hello thank you water`
  → `HELLO THANK_YOU WATER` with unmatched words dropped. No external API needed.

---

# Reconnect/status fix (2026-10-12, third session — frontend only, backend untouched)

**Root cause:** `Connection.maybeReconnect()` treated the server's *normal* close after
`MODEL_UNAVAILABLE` as a transient failure, firing up to 5 auto-reconnects
(`TranslatePage.tsx` sets `maxReconnectAttempts: 5`) that each received the same verdict
— flashing Connecting/Reconnecting… then landing on Disconnected. Status text derived
only from transport state, so a healthy server looked like a network failure; the
mount-time REST `backendOnline`/`capabilities` fetch could additionally stick at
"Backend Offline" / "Model status unknown" while the panel already knew the truth,
and the activity pill read "listening" with no recognition occurring.
Exact sequence: backend accept → `build_translation_service` raises (not promoted) →
sends `{"type":"error","code":"MODEL_UNAVAILABLE"}` → normal close 1000
(`backend/app/api/websocket/translation.py`); frontend `handle()` → `onclose` → retry.

**Fix (terminal model state, retries preserved for real failures):**
- `services/websocket/connection.ts`: new `suppressReconnect()` (terminal verdicts stop
  retries; only `connect()` re-arms). `maybeReconnect()` honors it.
- `services/websocket/client.ts`: `modelUnavailable` flag — set + suppression on the
  exact `MODEL_UNAVAILABLE` code (other errors still retry); reset on `connect()`.
- `features/translate/sessionStatus.ts` (new, pure/tested): `connectionLabel()` →
  "Model unavailable" for the terminal state (transport keeps its own wording),
  `activityLabel()` → "unavailable" instead of false "listening".
- `TranslatePage.tsx`: badges use the helpers (amber dot for model state); any live WS
  contact sets `backendOnline=true` (a server that answers is not offline); manual
  Start/Reconnect clears the terminal state and refreshes health/capabilities once
  (user-initiated only, no polling). Camera loop, Pause/Stop/Clear untouched — preview
  and overlay keep running while recognition is disabled.
- `tsconfig.app.json`: `types` gains `"node"` (@types/node was already a devDep) so the
  new `node:test` suites typecheck; `package.json`: `npm test` (tsx --test, no new deps).
- Backend, promotion gate, protocol, and artifacts unchanged; no translations fabricated.

**Tests:** `npm test` → **13/13 pass** (MODEL_UNAVAILABLE+close ⇒ 1 connection, no loop;
transient close ⇒ retry preserved; SESSION_ERROR ⇒ retry; manual connect() ⇒ reset +
re-arm; malformed frames ignored; label/activity matrix). `npm run build` (tsc -b + vite)
**PASS**; new "Model unavailable" strings confirmed in `dist/`. Live check with real
backend over real WS (`tsx` script driving the actual `TranslationSocket`): **1 connection
in 6 s** (old code: 6), `modelUnavailable=true`, error delivered — LIVE-VERIFY-PASS.
Backend suite not re-run (no backend/protocol change; prior 152-pass stands).

**Remaining limitation:** page-level auto-flow (camera → session → badge) still needs a
human with a camera; label rendering is unit-tested and the served bundle verified,
but the full click-through was not run headless. Sign recognition itself remains gated
by design — this fix only makes the unavailability honest.

---

# Status-indicator polish (2026-10-12, fourth session — frontend only)

**Problem:** after the reconnect fix, an active gated session showed two "Model
unavailable" pills (header model indicator, red + socket badge, amber) — one state
rendered from two sources — while idle copy ("Start interpreting") still promised
recognition the gate forbids.

**Fix:** `sessionStatus.ts` now fuses one authoritative `ModelState`
(terminal server verdict wins over the capability snapshot) with a single `modelLabel`;
`transportLabel` is purely transport and the pill hides via `shouldShowTransport` whenever
it would duplicate the verdict; session surfaces read `sessionBadgeLabel` → "Preview
only" instead of a misleading "Disconnected"; the idle action uses `startActionLabel`
→ Retry / Start camera preview / Start interpreting / Start session, with an honest
gate explanation beside it. Camera loop, Pause/Stop/Reconnect/Clear, retry suppression,
backend, gates, and auth untouched.

**Tests:** `npm test` → **15/15 pass** (fused-state priority, transport visibility matrix,
preview-only wording, start-label matrix, prior reconnect/activity suites). `npm run build`
(tsc -b + vite) **PASS**; new strings confirmed in `dist/`; idle `/app/translate` render
checked in headless Chromium (unknown-model idle copy truthful, no crash). Backend suite
not re-run (no backend change; prior 152-pass stands).

**Remaining limitation:** full camera → session → badge click-through still needs a human
with a camera; sign recognition remains gated by design.
