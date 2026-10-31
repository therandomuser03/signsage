# SignSage

**Bidirectional Indian Sign Language (ISL) ↔ English — an M.Tech research prototype.**

> Not a production accessibility product. Not certified for medical, legal, emergency, or
> other high-stakes use. See [`docs/development/DEVELOPMENT.md`](docs/development/DEVELOPMENT.md) (§4) and
> `frontend`'s Privacy page.

This README was rewritten as part of a pre-demo audit/fix pass. If you read an older
copy of this file describing "frontend has no live camera/MediaPipe/WebSocket
integration" — that's now out of date. The frontend **does** have a working camera →
MediaPipe → WebSocket → backend pipeline; see **Known limitations** below for what
still doesn't work well.

---

## Research Status

**Phase 0 model evaluation: NO-GO for general translation**

The research checkpoints are not sufficiently reliable for open-vocabulary
ISL translation, so live translation is gated off by default
(`SIGNSAGE_MODEL_PROMOTED=false` → the backend answers `MODEL_UNAVAILABLE`
instead of serving NO-GO-quality output as if it were reliable). The research
models remain available for experimentation via the dev-only
`SIGNSAGE_ALLOW_UNPROMOTED_DEV` flag (see `backend/models/README.md`).

Full numbers for the verified NO-GO run are recorded in
[`research/evaluation/reports/352868924/evaluation_results.json`](research/evaluation/reports/352868924/evaluation_results.json)
(run manifest
[`research/manifests/models/signsage-s2t_v352868924.yaml`](research/manifests/models/signsage-s2t_v352868924.yaml);
see also [`docs/architecture/DATA_AND_MODEL.md`](docs/architecture/DATA_AND_MODEL.md) (§5)).
The historical path `research/outputs/phase0/RESULTS.md` referenced by older
documentation is TBD/missing in this checkout and is not used here.
The run label `20260916T144755Z` below is a historical reference (TBD mapping
to the verified run above).
No checked-in `backend/signsage_knowledge_manifest.json` exists; the running
backend reports its bundle/manifest status via `/api/v1/models` only when an
operator-supplied bundle is configured.

- **Sign→Text**: both the GRU baseline and the Transformer mode-collapsed during
  evaluation (0.6% / 3.9% unique outputs on the test set). WER is ~1.0–1.03 for every
  model, including the trivial "always predict the same sentence" baseline — i.e. none
  of them reliably beat guessing.
- **Text→Pose**: the generative GRU model's output is near-static (~7.9% of
  ground-truth motion energy).
- Because of this, **live Sign→Text refuses to serve while unpromoted**:
  low-motion or low-confidence output is returned as an honest rejection
  (`accepted: false` with a reason), never as a fabricated translation, and
  nothing is persisted to history unless the result is presentable.

This is disclosed in the app itself (Research & Benchmarks page, the red NO-GO banner,
and inline notes in the translator) rather than hidden behind favorable-looking numbers.

## How it works

The following is the current high-level runtime flow. Detailed API, data-model,
and ML contracts are documented separately in `docs/`.

```text
Browser: getUserMedia → MediaPipe Holistic (WASM) → 543 landmarks/frame (1629 floats)
  → raw coordinates (backend normalizes) → ~10 fps → WebSocket (landmarks.frame, sentence.boundary)
Backend: FastAPI → per-connection buffer → Sign→Text Transformer
  → acceptance gate (motion + confidence) → translation.text (or honest rejection)
```

```text
Browser: text input → REST (avatar gloss) → gloss ID sequence
Backend: deterministic word lookup, optionally LLM-assisted mapping (Groq → Ollama → lookup)
  → pre-authored avatar clips only; unknown words are dropped, never guessed
Browser: single VRM avatar plays the clips back-to-back on Canvas/WebGL
```

Raw camera video **never** leaves the browser — only extracted landmark coordinates are
sent to the backend for inference. This does mean a live backend connection is required;
there's no offline/on-device inference mode in this build.

## Repository layout

At a high level:

```text
backend/    FastAPI + PyTorch inference server
frontend/   Vite + React UI: camera capture, MediaPipe, WebSocket client, pose canvas
research/   Training notebooks + Phase 0 experiment record and research artifacts
docs/       Architecture, data-model, decisions, security, testing, and development notes
```

For the repository layout, documentation map, and agent workflow, see
[`docs/MASTER.md`](docs/MASTER.md) and [`docs/AGENTS.md`](docs/AGENTS.md).

---

## Documentation

Start with [`docs/MASTER.md`](docs/MASTER.md) (orientation + map), then the
authoritative document for your question:

- [`docs/STATUS_AND_ROADMAP.md`](docs/STATUS_AND_ROADMAP.md) — current state, audit evidence, blockers, validation, work order.
- [`docs/architecture/DATA_AND_MODEL.md`](docs/architecture/DATA_AND_MODEL.md) — dataset facts, storage ops, vocabulary verdict, model readiness & gates.
- [`docs/development/DEVELOPMENT.md`](docs/development/DEVELOPMENT.md) — setup, interfaces, deployment, security, testing.
- [`docs/evaluation/EXTERNAL_EVALUATION.md`](docs/evaluation/EXTERNAL_EVALUATION.md) — external T5 candidate evidence.
- [`docs/AGENTS.md`](docs/AGENTS.md) — workflow and rules for AI coding agents.
- [`docs/reference/`](docs/reference/) — exact-values pins (`end-to-end-contract.md`, `contract-verification.md`).

When README details and implementation differ, verify the current code and the
canonical documentation rather than relying on an older snapshot.

---

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # defaults work out of the box for a local demo
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. Swagger docs at `/docs`. Health/model info at
`/api/v1/models` (this is where you can see the live NO-GO status and which checkpoints
are loaded).

Run the backend test suite from `backend/`:

```bash
pytest
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local     # optional: only needed to change the backend URL, or for Supabase auth
npm run dev
```

Frontend runs at `http://127.0.0.1:5173` and connects to the backend WebSocket at
`ws://127.0.0.1:8000/ws/translate` by default (`VITE_SIGNSAGE_WS_URL` in
`.env.local` overrides this; `VITE_SIGNSAGE_API_URL` overrides the REST base
and the WS URL is derived from it when unset). Supabase auth is optional — the translator works without
signing in; auth vars are only needed if you want account/session features.

To build for deployment: `npm run build` (outputs to `frontend/dist/`).

### Environment variables

See `backend/.env.example` and `frontend/.env.example` for the current environment-variable list with
comments. Neither file contains real secrets — **if you had an earlier copy of this
project with a committed `backend/.env` or `frontend/.env.local`, treat any Groq API
key and Postgres/Supabase database password in it as compromised and rotate them.**
Supabase auth is entirely optional for the demo.

## Demo flow

1. Start the backend, then the frontend, then open the app in a browser and allow
   camera permission when prompted.
2. **ISL → English**: open Sign → Text under `/app`, press Start, and sign
   toward the camera (a landmark overlay confirms detection; a frame counter
   shows what is being streamed). Press Stop to end the sentence: the buffered
   sequence is sent for inference and the result appears with its acceptance
   state, result type, and backend-reported confidence. With the default
   `SIGNSAGE_MODEL_PROMOTED=false` the backend honestly reports the model as
   unavailable instead of translating; set `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`
   locally (dev only, never production) to exercise the real inference path
   against a configured checkpoint.
3. **English → ISL**: open Text → Sign and type text. Known dictionary words
   map to pre-authored avatar clips; unknown words are dropped, never guessed.
   Backend-assisted gloss needs the gloss service; failures fall through to
   the deterministic lookup.
4. If the backend drops mid-session, the frontend retries the connection with
   bounded backoff (5 attempts) and shows Reconnecting…; leaving the page
   closes the socket and stops the camera.

## Known limitations (not hidden, not "fixed" by hiding them)

- **Open-vocabulary translation does not work reliably in either direction.**
  Sign→Text is gated off until a checkpoint passes the deployment gate; this
  is a research prototype, not a general ISL translator.
- **Text→Sign covers a fixed dictionary only** (see
  `backend/app/data/sign_vocabulary.json`): unknown words are dropped, and
  the avatar plays pre-authored clips — no generative pose model is served.
- No offline/on-device inference mode — the backend must be reachable over the network.
- This is a Phase 0 research checkpoint, not a validated clinical, legal, or safety
  tool, and not a substitute for a certified human ISL interpreter.
