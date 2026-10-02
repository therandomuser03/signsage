# SignSage — Deployment

**Status:** Deployment architecture/documentation; production verification is separate.

## 1. Target topology

```text
Browser
  ├── React/Vite frontend
  └── MediaPipe
        │
        ▼
   WebSocket / HTTPS
        │
        ▼
FastAPI backend
  ├── translation
  ├── authentication
  ├── history
  └── capabilities
        │
        ├── Supabase
        └── optional Groq / local Ollama
```

## 2. Frontend

The documented hosting target is Vercel.

Typical configuration:

```text
Root: frontend/
Build: npm run build
Output: dist/
```

Frontend environment configuration must contain only public/client-safe values.

WebSocket and Supabase URLs should be configured through environment variables rather than hardcoded.

## 3. Backend

The documented hosting target is Render (see `render.yaml` at the repo root).

```text
Runtime: docker (build context: backend/, see backend/Dockerfile)
Install: pip install .  (CPU-only torch first, then the app)
Start:   bash scripts/fetch_bundle.sh && exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
Health:  GET /api/v1/health
```

The application entry point is `app.main:app`. The server binds `0.0.0.0` and
uses Render's injected `$PORT` (the `SIGNSAGE_PORT` setting is the local-dev
equivalent only).

Model weights are NOT baked into the image. At start,
`backend/scripts/fetch_bundle.sh` downloads `SIGNSAGE_BUNDLE_URL` (a
`signsage_deployment_bundle.zip`) into `SIGNSAGE_MODEL_BUNDLE_DIR`
(`/models/signsage_deployment_bundle`, a 2 GB persistent disk), verifies the
optional `SIGNSAGE_BUNDLE_SHA256` checksum, and requires the entries
`app/infrastructure/storage/model_storage.py` checks for
(`checkpoint/`, `tokenizer/`, `landmark_contract.json`,
`sign_to_text_deployment.json`). Without a bundle URL the server still starts
and `/health` honestly reports `model_bundle_found: false`; translation
endpoints then refuse with `MODEL_UNAVAILABLE` instead of faking readiness.

`SIGNSAGE_ALLOW_UNPROMOTED_DEV` is deliberately absent from `render.yaml`:
production never serves an unpromoted checkpoint.

## 4. Important hosting limitation

A free/small backend instance may be unsuitable for:

- PyTorch model loading,
- large checkpoints,
- sustained WebSocket workloads,
- concurrent inference.

Do not describe a free hosting deployment as production-grade merely because it starts successfully.

## 5. Environment secrets

Backend-only secrets include:

- Supabase service credentials where required,
- Groq API key,
- model/research access tokens where applicable.

Never expose server secrets through frontend environment variables.

## 6. Browser privacy

MediaPipe runs in the browser. The camera stream should remain local to the browser while numerical landmarks are sent to the backend.

## 7. Deployment verification

A deployment is considered verified only after checking:

1. frontend loads,
2. authentication works,
3. backend health endpoint responds,
4. WebSocket connects,
5. landmark payloads are accepted,
6. errors surface correctly,
7. history isolation works,
8. the actual model status is correctly reported.

A successful deployment smoke test does not validate model quality.

## 8. Current model caveat

The documented research Sign→Text checkpoint is experimental and has known collapse/failure behavior. Deployment of the software infrastructure must not be presented as deployment of a validated general-purpose ISL translator.
