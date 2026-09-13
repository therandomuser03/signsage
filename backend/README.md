# SignSage Backend

FastAPI + PyTorch inference backend for the SignSage single-device research prototype.

## What this contains

- Sign→Text Transformer inference
- Text→ISL-gloss conversion
- Text→Pose GRU inference
- WebSocket streaming contract
- REST diagnostic endpoints
- checkpoint loading
- exact notebook landmark normalization
- no-face ablation model
- Groq → Ollama → deterministic gloss fallback
- validation, error responses, smoke tests, and a benchmark script

The backend accepts **landmarks, not raw camera video**.

## Directory

```text
backend/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   └── services/
├── docs/API.md
├── tests/
├── scripts/
├── weights/
├── vocab/
├── .env.example
├── MODEL_CARD.md
├── pyproject.toml
├── requirements.txt
└── signsage_knowledge_manifest.json
```

## Setup

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Start

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Landmark input mode

The official contract says each frame carries training-compatible normalized landmarks.

Default:

```env
SIGNSAGE_LANDMARK_INPUT_MODE=normalized
```

If a future client sends raw MediaPipe image-space coordinates instead:

```env
SIGNSAGE_LANDMARK_INPUT_MODE=raw
```

Do not normalize twice.

## WebSocket

```text
ws://127.0.0.1:8000/api/v1/ws/translate
```

Supported client messages:

- `landmarks.frame`
- `sentence.boundary`
- `text.input`

Supported server messages:

- `translation.text`
- `translation.pose`
- `error`

See `docs/API.md`.

## REST diagnostics

`GET /api/v1/health`, `GET /api/v1/models`, and `GET /api/v1/capabilities`. There is no
REST endpoint for translation itself -- Sign→Text and Text→Pose are only reachable over
the WebSocket above (see `docs/API.md` and `DECISIONS.md` for why). `/capabilities`
exists specifically so an integrator can query limits and enabled features (no-face
model, Groq/Ollama gloss) without needing a WebSocket connection first.

## Session / turn state

One WebSocket connection is one session, matching the single-device MVP: a shared
device, two people taking turns. Per-connection state (buffered landmark frames, turn
counters) lives in `app/services/session.py::TranslationSession`, which is a plain,
FastAPI-independent object -- see `tests/test_session.py`. There is deliberately no
cross-connection session registry, room code, or persistence; that would be scaffolding
for the explicitly-deferred two-device relay feature (see ARCHITECTURE.md).

## Testing

```bash
pytest
```

Model/inference verification:

```bash
python scripts/verify_backend.py
```

Benchmark:

```bash
python scripts/benchmark.py --runs 10
```

The benchmark measures backend/model execution only. It does not measure browser
MediaPipe extraction, network transfer, or UI rendering.

## Groq

Optional:

```env
SIGNSAGE_ENABLE_GROQ_GLOSS=true
SIGNSAGE_GROQ_API_KEY=your_key
```

The selected model is `openai/gpt-oss-20b`, matching the project's recorded decision.

## Ollama

Optional local fallback:

```env
SIGNSAGE_ENABLE_OLLAMA_GLOSS=true
SIGNSAGE_OLLAMA_MODEL=llama3.2:3b
```

If Groq is unavailable and Ollama is unavailable, the backend falls back to the
deterministic notebook-style naive gloss.

## Model limitations

Read `MODEL_CARD.md`. The current research checkpoints are mechanically usable but
were not shown by the bounded evaluation to provide reliable translation. The backend
does not hide this limitation.
