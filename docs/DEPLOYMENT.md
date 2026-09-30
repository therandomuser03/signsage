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

The documented hosting target is Render.

Typical configuration:

```text
Root: backend/
Install: pip install -r requirements.txt
Start: uvicorn <application>:app --host 0.0.0.0 --port $PORT
```

The exact module path must follow the repository's actual backend entry point.

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
