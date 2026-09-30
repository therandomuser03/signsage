# SignSage

**Bidirectional Indian Sign Language (ISL) ↔ English — an M.Tech research prototype.**

> Not a production accessibility product. Not certified for medical, legal, emergency, or
> other high-stakes use. See [`docs/SECURITY.md`](docs/SECURITY.md) and
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
ISL translation. The application therefore uses a bounded demo fallback for
a small set of supported phrases while preserving the research models for
experimentation.

Full numbers: [`research/outputs/phase0/RESULTS.md`](research/outputs/phase0/RESULTS.md)
(run `20260916T144755Z`, also mirrored into
[`backend/signsage_knowledge_manifest.json`](backend/signsage_knowledge_manifest.json),
which is what the running backend actually serves via `/api/v1/models`).

- **Sign→Text**: both the GRU baseline and the Transformer mode-collapsed during
  evaluation (0.6% / 3.9% unique outputs on the test set). WER is ~1.0–1.03 for every
  model, including the trivial "always predict the same sentence" baseline — i.e. none
  of them reliably beat guessing.
- **Text→Pose**: the generative GRU model's output is near-static (~7.9% of
  ground-truth motion energy).
- Because of this, **both directions use a bounded, honestly-labelled demo fallback**
  for a fixed set of ~10 phrases (Hello, Thank you, Yes, No, Help, Please, Sorry, Good
  morning, How are you, I need help), backed by real recorded clips in
  `backend/demo_prototypes/`. Outside that set, the general research models run and the
  UI says so.

This is disclosed in the app itself (Research & Benchmarks page, the red NO-GO banner,
and inline notes in the translator) rather than hidden behind favorable-looking numbers.

## How it works

The following is the current high-level runtime flow. Detailed API, data-model,
and ML contracts are documented separately in `docs/`.

```text
Browser: getUserMedia → MediaPipe Holistic (WASM) → 543 landmarks/frame (1629 floats)
  → hip-center + scale normalize → buffer → WebSocket (landmarks.frame, sentence.boundary)
Backend: FastAPI → buffered sequence → Sign→Text Transformer
  → if output looks unreliable, DTW-match against 10 recorded demo phrases → translation.text
```

```text
Browser: text input → WebSocket (text.input)
Backend: match against 10 supported phrases (+ aliases like "hi"→"Hello")
  → match found: real recorded hand-motion clip → translation.pose (is_fallback=true)
  → no match: gloss → Text→Pose GRU (near-static, disclosed) → translation.pose
Browser: renders hands (always) + body pose skeleton (only for the GRU path) on Canvas 2D
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

For the detailed repository tree, see [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

For the documentation map and agent workflow, see [`docs/MASTER.md`](docs/MASTER.md)
and [`docs/AI_AGENT_GUIDE.md`](docs/AI_AGENT_GUIDE.md).

---

## Documentation

Start with:

- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) — detailed repository structure.
- [`docs/MASTER.md`](docs/MASTER.md) — canonical project orientation and source-of-truth hierarchy.
- [`docs/STATUS.md`](docs/STATUS.md) — current implementation, verification, and research status.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system architecture and runtime boundaries.
- [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) — landmark/data representation and research data rules.
- [`docs/API.md`](docs/API.md) — HTTP/WebSocket interface documentation.
- [`docs/SECURITY.md`](docs/SECURITY.md) — privacy, secrets, dataset, and safety boundaries.
- [`docs/TESTING.md`](docs/TESTING.md) — software verification and research evaluation.
- [`docs/AI_AGENT_GUIDE.md`](docs/AI_AGENT_GUIDE.md) — workflow and rules for AI coding agents.

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
`ws://127.0.0.1:8000/api/v1/ws/translate` by default (`VITE_SIGNSAGE_WS_URL` in
`.env.local` overrides this). Supabase auth is optional — the translator works without
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
   camera + microphone permission when prompted.
2. **ISL → English**: open the Translator, switch to Sign mode, and sign one of the
   supported phrases (see the on-screen phrase list) in front of your camera, pausing
   briefly at the end. A caption appears; the app labels whether it came from the
   bounded demo match or the research model.
3. **English → ISL**: switch to Text mode and type one of the same supported phrases
   (or a close rephrasing, e.g. "hi", "thanks"). A recorded hand-motion clip plays on
   the pose canvas, labelled as a recorded demo clip. Typing something outside that set
   falls through to the generative model, which is visibly near-static — the UI says
   so rather than hiding it.
4. If you kill and restart the backend mid-session, the frontend now reconnects
   automatically instead of requiring a page reload.

## Known limitations (not hidden, not "fixed" by hiding them)

- **Open-vocabulary translation does not work reliably in either direction.** Only the
  ~10 bounded demo phrases are dependable; this is a research prototype, not a general
  ISL translator.
- **The bounded demo clips only have real captured hand motion**, not full-body pose
  (a pre-existing data-capture quirk — see `docs/DECISIONS.md` / the code comments in
  `backend/app/services/pose_preprocessing.py`). The pose canvas therefore renders
  hands for demo clips rather than a fabricated full-body skeleton.
- **The no-face ablation Transformer checkpoint is loaded but not served** by the live
  Sign→Text flow; its outputs are unverified in this build.
- No offline/on-device inference mode — the backend must be reachable over the network.
- This is a Phase 0 research checkpoint, not a validated clinical, legal, or safety
  tool, and not a substitute for a certified human ISL interpreter.
