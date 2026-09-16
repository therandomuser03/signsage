# SignSage — Run Instructions

Start the functional shared-device app. No dataset or training needed — checkpoints already in `backend/weights/` + `backend/vocab/`.

## Prerequisites

- Python 3.13, Node 20+
- Chrome/Edge (camera + Web Speech API)
- Two terminals

## 1. Backend (FastAPI + PyTorch, port 8000)

```bash
cd backend
python -m venv .venv
# linux/mac
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env   # keep defaults; set GROQ key only if you want LLM gloss
# verify weights present: ls weights/*.pt vocab/*.json

# smoke (should print Sign→Text + pose shape — output is mode-collapsed, that's expected per research/outputs/phase0/RESULTS.md)
.venv/bin/python scripts/verify_backend.py
# or: python -m pytest -q  -> 40 passed

# start
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000/docs  -> try GET /api/v1/health, /api/v1/models, /api/v1/capabilities
# WS at ws://127.0.0.1:8000/api/v1/ws/translate (no REST for translation by design, see docs/API.md)
```

Landmarks are `normalized` by default (`SIGNSAGE_LANDMARK_INPUT_MODE=normalized` — browser does `lib/landmarks.ts:68` normalize). Use `raw` only if client sends raw MediaPipe coordinates.

## 2. Frontend (Vite + React, port 5173)

```bash
cd frontend
npm install
cp .env.example .env.local   # default WS is ws://127.0.0.1:8000/api/v1/ws/translate
# optional offline model cache:
# echo 'VITE_MEDIAPIPE_MODEL_URL=/holistic_landmarker.task' >> .env.local

npm run dev    # http://localhost:5173
# build check: npm run build && npm run test  (vitest 3 passed; e2e via npx playwright test)
```

Open `http://localhost:5173`:
- Landing → Sign to Text → Start camera → sign → pause ~650ms or press Send → caption appears (with `latency_ms`; expect repetitive output per MODEL_CARD.md)
- Text to Sign → type or press Speak (mic, `en-IN`) → Translate → skeleton canvas @25fps + gloss (pose is near-static, flagged as research limit)
- Conversation transcript below collects both directions; CheckCircle2 footer = "landmarks only, video stays on device"

Requires `localhost` or HTTPS for camera/mic; allow permissions when prompted. One tab = one `TranslationSession` (`backend/app/services/session.py`).

## Quick verify both are talking

- Backend log shows `SignSage models loaded on cpu/cuda`
- Frontend badge shows `Connected` (green). If `Offline` → backend not on 8000 or `VITE_SIGNSAGE_WS_URL` wrong.
- Send empty `sentence.boundary` → expect `error SESSION_ERROR` in transcript (fail-loudly, per DESIGN.md)

## Env knobs (only if needed)

```env
# backend/.env
SIGNSAGE_ENABLE_GROQ_GLOSS=true
SIGNSAGE_GROQ_API_KEY=...
SIGNSAGE_ENABLE_OLLAMA_GLOSS=true  # needs ollama run llama3.2:3b

# frontend/.env.local
VITE_SIGNSAGE_WS_URL=wss://your-host/api/v1/ws/translate
```

## Model limitations (not a bug)

Current `*.pt` checkpoints are plumbing-verified, not translation-verified — see `research/outputs/phase0/RESULTS.md` + `backend/MODEL_CARD.md`. A successful WS round-trip = pipeline ran, not accurate ISL.

## Troubleshooting

- `pytest: command not found` → use `.venv/bin/python -m pytest`
- Camera black → check `chrome://settings/content/camera`, must be localhost/HTTPS
- `Voice unsupported` → use Chrome/Edge; Firefox has no Web Speech API
- CORS → add your frontend origin to `SIGNSAGE_CORS_ORIGINS` in `backend/.env`
