# SignSage Development

Interfaces, deployment, security, testing, and the developer workflow.
(Consolidated 2026-10-13 from `API.md`, `DEPLOYMENT.md`, `SECURITY.md`, `TESTING.md`.)

## 1. Setup & workflow

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate \
  && pip install -r requirements.txt && cp .env.example .env
uvicorn app.main:app --reload          # :8000, Swagger at /docs
pytest                                  # full suite from backend/ (uses .venv)
```

```bash
cd frontend && npm install && npm run dev        # :5173
npm run build                                     # → frontend/dist/
```

Env templates (`backend/.env.example`, `frontend/.env.example`) hold no real
secrets. Supabase auth is optional (translator works signed-out). After a
non-trivial change run the smallest relevant check (backend → `pytest`,
frontend → `tsc -b` + build, API → contract tests, research → notebook/eval
checks) and update `STATUS_AND_ROADMAP.md` + this doc / `DATA_AND_MODEL.md` if
behavior changed; record architectural changes in `MASTER.md` ("Key decisions").
Never create numbered copies (`API(11).md`); never claim a check passed
un-run. `npm run lint` carries ~39 pre-existing errors (dedicated pass, not here).

## 2. Interfaces

Core loop is WebSocket; diagnostics/product ops are REST. Canonical source is
`backend/app/schemas/websocket.py` (checked by `test_frontend_contract.py`,
mirrored in `frontend/src/services/websocket/protocol.ts`) — where prose
disagrees with that schema, the schema wins.

- WS `/ws/translate` (unversioned): `landmarks.frame{landmarks: 543×[x,y,z],
  all finite}` → interim `translation.text{final:false}` under repeat-agreement
  (trailing 80-frame window every 20; first accepted prediction immediately,
  updates only after repeating on two windows; rejections emit nothing) →
  `sentence.boundary{save_history?}` → `translation.text{final:true}` or
  `error{MODEL_UNAVAILABLE|INVALID_MESSAGE|…}` + close. NaN/±Inf → `INVALID_MESSAGE`.
  Real reply: `translation.text{text, accepted, reason, confidence, result_type,
  history_id}`. No `session_id` envelope, no `text.input`, no `translation.pose`.
- REST: `GET /api/v1/health|/models|/capabilities`, `POST
  /api/v1/translation/sign-to-text` (backend live, frontend uncalled — WS is the
  only translation transport), avatar-gloss + history endpoints per OpenAPI.
- Auth: Supabase tokens; history isolated per user. Exact values (landmark
  order, dims 543/1629/3258, tokenizer ids, verdict vocabulary) are pinned in
  `reference/end-to-end-contract.md`.
- Interpretation rule: a valid response proves protocol correctness only — never
  accuracy, generalization, pose quality, or production readiness.

## 3. Deployment

Topology: browser (React/Vite + MediaPipe) → WS/HTTPS → FastAPI
(translation/auth/history/capabilities) → Supabase + optional Groq/Ollama.

- Frontend → Vercel (`frontend/`, `npm run build`, `dist/`; only public env).
- Backend → Render (`render.yaml`; context `backend/`; `pip install .`;
  start `bash scripts/fetch_bundle.sh && exec uvicorn app.main:app --host
  0.0.0.0 --port $PORT`; health `/api/v1/health`). Weights are NOT baked in:
  `fetch_bundle.sh` downloads `SIGNSAGE_BUNDLE_URL` into `/models/...` (2GB
  disk), verifies optional SHA256 and the `checkpoint/ + tokenizer/ +
  landmark_contract.json + sign_to_text_deployment.json` entries; no URL → still
  starts, `/health` reports `model_bundle_found:false`, translation refuses
  `MODEL_UNAVAILABLE`. `SIGNSAGE_ALLOW_UNPROMOTED_DEV` is absent from Render.
- Free/small instances may not suit PyTorch/checkpoint/WS workloads — starting
  ≠ production-grade. Verified only after: frontend loads, auth works, health
  responds, WS connects, payloads accepted, errors surface, history isolated,
  model status honest. Smoke tests never validate model quality.

## 4. Security, privacy & safety

- **Landmarks, not video.** MediaPipe runs in-browser; no camera-upload endpoint
  exists in the translation path; backend stores no video; landmark sequences
  are not a permanent biometric store. History is per-user (RLS on).
- Never commit `GROQ_API_KEY`, `HF_TOKEN` (research token stays a runtime
  secret, never printed to notebook output), Supabase service-role creds, or
  any private credential — env/platform secret store only.
- Dataset is CC BY-NC-SA 4.0: never commit it, pose files, or HF caches. Model
  license gate: iSign-trained weights do NOT inherit the code's MIT terms —
  see `DATA_AND_MODEL.md` §6 before any public/commercial deployment.
- Safety: never present SignSage as a certified-interpreter replacement
  (legal/medical/emergency). Fluent-looking output can be wrong (documented
  collapse). UI must separate unavailable/uncertain/experimental from success,
  never show experimental pose output as validated signing, never swap
  MODEL_UNAVAILABLE for a fake translation. Future two-device work needs a
  fresh review (room entropy, expiry, auth, isolation, reconnect, rate limits).

## 5. Testing

Prioritize failure detection and research validity over coverage numbers.

- Software: imports, route registration, WS contract, auth, history isolation,
  capability/model status, error handling; frontend `tsc` + build + contract
  parity; contract tests pin paths, message types, fields, error codes,
  landmark dims, avatar vocab.
- Research: Sign→Text needs WER/BLEU/ROUGE-L + diversity vs a baseline;
  Text→Pose needs normalized MSE + DTW **+ motion energy** (DTW/MSE alone hide
  near-static output); data runs verify rows/pose correspondence, readability,
  splits, junk-text. Layers: unit → integration → live E2E → quality eval →
  target-device; passing one never substitutes the next.
- Record: Phase-0 showed Sign→Text collapse (~1% diversity) and Text→Pose
  ~6.7% motion energy — diagnostic, not deployment, evidence.
- Outstanding: real-browser camera E2E by hand (closest automation:
  `ws_diagnostic.py` gated + serving); no basis for load/stress or
  broad-signer claims.
