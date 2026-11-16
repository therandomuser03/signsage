# SignSage

**Bidirectional Indian Sign Language (ISL) ↔ English — an M.Tech research prototype.**

Raw camera video **never** leaves the browser: only landmark coordinates are sent to the backend. This is a research prototype, not a certified accessibility product — not for medical, legal, emergency or other high-stakes use.

> **Current model status:** `SIGNSAGE_MODEL_PROMOTED=false`. There is no production Sign→Text model. The backend refuses to serve NO-GO-quality output (`/api/v1/capabilities` says so explicitly). See [`docs/STATUS_AND_ROADMAP.md`](docs/STATUS_AND_ROADMAP.md).

## What works today

- Browser camera → MediaPipe Holistic (WASM) → 543 landmarks/frame → WebSocket → FastAPI backend.
- **Static alphabet image classification** (26 A–Z; RealSign bundle, ~87% test accuracy).
- **English → ISL** through a fixed avatar-gloss vocabulary (**14 IDs**, deterministic mapping; unknown words are dropped, never guessed).
- **ISL word recognition**: `word_v3` (9 words, one signer, **unpromoted** experimental artifact).
- **Authentication & history** via Supabase (optional; signed-out mode works).
- **Text-to-speech**: browser Web Speech fallback; Groq Orpheus when configured.
- Health/capabilities endpoints that honestly report model state.

## What does not work / is not claimed

- **Open-vocabulary sentence-level Sign→Text translation.** The transformer mode-collapsed in evaluation (WER ≈ 1.0); the promotion gate stays closed by design.
- No general offline inference — the backend must be reachable over the network.
- The FDMSE-ISL 200-word recognizer (this repository's latest work) is **experimental and not promoted** pending evaluation.

## Architecture

```text
Browser  getUserMedia → MediaPipe Holistic → landmarks → ~10 fps → WebSocket /ws/translate
Backend  FastAPI → buffer → inference pipeline → acceptance gate → translation (or honest rejection)
                                              ↘ history (Supabase)
Browser  text input → REST /api/v1/text-to-sign → gloss IDs → avatar clips (WebGL)
```

**Signers pipeline:** camera → landmarks only — no video leaves the device.
**English side:** deterministic word lookup + optional Groq-assisted gloss mapper (fallback to lookup on error or missing key).

## Tech stack

| Layer | Stack |
|---|---|
| Frontend | React 19 + Vite + Tailwind 4 + shadcn/ui, Three.js (avatar), TypeScript |
| Backend | FastAPI + PyTorch (MediaPipe, isolated-word CNN) |
| Auth/History | Supabase (Postgres + JWT + RLS) — optional |
| TTS | Browser Web Speech API / Groq Orpheus |
| STT | Browser Web Speech / Groq Whisper |
| Hosting | Vercel (frontend) · Render (backend) |

## Repository structure

```text
signsage/
├── frontend/        React + Vite UI (camera capture, WebSocket client, avatar)
├── backend/         FastAPI inference server (word CNN, image CNN, routes, auth)
├── research/        Training notebooks, experiment records, datasets, artifacts
├── docs/            Master plan, status reports, architecture contracts, guides
├── outputs/         Research outputs & generated reports
├── models/          Model metadata (weights are gitignored; fetched at runtime)
└── render.yaml      Render blueprint for backend + model disk
```

Ignored by git (not committed): `graphify-out/` (knowledge-graph artifacts), `staging_kaggle_results/` (Kaggle scratch), `videos/`, dataset zips, `*.pt`/`*.pth`, `node_modules/`, `.venv/`.

**Suggested cleanup** (not done, to avoid losing work): `mobile/` (~4 GB ISLiM build artifacts), `outputs/` (~300 MB) and `verified_checkpoint/` (~490 MB) could be archived to an external drive or a private artifact store. `docs/references.bib` and `research/FDMSE/` are new; both are intentionally tracked.

## Getting started

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # defaults work for local demo
uvicorn app.main:app --reload   # http://127.0.0.1:8000 — /docs for Swagger
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local      # only needed to change backend URLs or enable Supabase
npm run dev                     # http://127.0.0.1:5173
```

### Test

```bash
cd backend && pytest            # backend suite (~292 tests)
cd frontend && npm test         # frontend unit suite
```

## Deployment

| Piece | Platform | Notes |
|---|---|---|
| Frontend | **Vercel** (`frontend/`) | Static build to `dist/`; env: backend URL |
| Backend | **Render** (`render.yaml`) | Docker; `/models` persistent disk; fetches bundle at start via `scripts/fetch_bundle.sh` |
| Auth/History | **Supabase** | Postgres + JWT + per-user RLS; optional |
| TTS | **Groq** | `SIGNSAGE_GROQ_TTS_API_KEY` (server-side); browser fallback |
| STT | **Groq Whisper** via `/api/v1/speech/transcribe` | Server-side; browser speech fallback |

## Documentation

Start with **[`docs/MASTER.md`](docs/MASTER.md)** (orientation and doc map), then:

- [`docs/STATUS_AND_ROADMAP.md`](docs/STATUS_AND_ROADMAP.md) — verified current state and blockers
- [`docs/development/DEVELOPMENT.md`](docs/development/DEVELOPMENT.md) — setup, interfaces, deployment
- [`docs/architecture/DATA_AND_MODEL.md`](docs/architecture/DATA_AND_MODEL.md) — dataset and model facts
- [`README_COLAB_TRAINING.md`](docs/README_COLAB_TRAINING.md) — training the FDMSE-ISL 200-word experimental model on Colab

If this README and the code disagree, the code wins — please open an issue.

## Credits

This project builds on earlier work that demonstrated ASL ↔ English translation.
**V1 (ASL-based) contributors:**

- [Anwesa Sinha](https://www.linkedin.com/in/anwesa-sinha/)
- [Draksha](https://www.linkedin.com/in/draksha-aa3164227/)
- [Satyaki Dawn](https://www.linkedin.com/in/satyaki-dawn-116420230/)
- [Subhadip Das](https://www.linkedin.com/in/subhadip-das-7a150319b/)

The current ISL version is a re-implementation and research extension of that
original ASL prototype. Full attribution and license terms for those components
remain in each subdirectory's own LICENSE files.

## License

MIT — see [LICENSE](LICENSE).
