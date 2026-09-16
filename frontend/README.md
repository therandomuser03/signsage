# SignSage Frontend

Shared-device conversational UI — Vite + React + TypeScript. See `PLAN.md` for pipeline status.

## What this is

- **Sign→Text:** camera → MediaPipe Holistic WASM (`@mediapipe/tasks-vision`) → 543 landmarks (pose 33+face 468+hands 21+21) flattened 1629-dim → browser `normalizeLandmarkSequence` (`lib/landmarks.ts` port of notebook) → motion-pause heuristic (<0.012 / 650ms / ≥15 frames) + manual Send → WebSocket `landmarks.frame` + `sentence.boundary` → `translation.text` caption. Privacy: raw video never leaves device.
- **Text→Pose:** typed text (Speech-to-Text via Web Speech API in Phase B) → `text.input` → gloss (Groq→Ollama→naive) → 60-frame pose → `PoseCanvas` 33-point skeleton @25fps. Honesty banners per `research/outputs/phase0/RESULTS.md`.
- **Transport:** `hooks/use-translation-socket.ts` — one WS per tab (`VITE_SIGNSAGE_WS_URL`), `session_id` per-message.
- **Verified:** `npm run build` + `npm run test` (vitest) green; `PLAN.md` Phase A.

## Run

```bash
npm install
npm run dev        # Vite at http://localhost:5173
```

Backend must be running (`backend/README.md` → `uvicorn app.main:app --reload` at :8000) or set:

```bash
# frontend/.env.local
VITE_SIGNSAGE_WS_URL=ws://127.0.0.1:8000/api/v1/ws/translate
VITE_MEDIAPIPE_MODEL_URL=https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

## Build / Test

```bash
npm run build
npm run test         # vitest
npx playwright test  # e2e (landing + workspace)
```

## Key files

| File | Role |
|---|---|
| `src/App.tsx` | Landing + `SignSageWorkspace`/`SignToTextPanel`/`TextToSignPanel`, hash routing |
| `src/lib/landmarks.ts` | `flattenHolisticLandmarks` + `normalizeLandmarkSequence` + `poseMotion` |
| `src/hooks/use-translation-socket.ts` | WS connect/send |
| `src/components/pose-canvas.tsx` | 33-point skeleton canvas |
| `src/hooks/useSpeechRecognition.ts` | Phase B — browser STT (pending) |

## Caveat

Current checkpoints mode-collapse / near-static per `research/outputs/phase0/RESULTS.md` and `backend/MODEL_CARD.md`. A successful WS round-trip = pipeline ran, not translation verified.
