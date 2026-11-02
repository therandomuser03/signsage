# System Architecture

Verified authority: `docs/MASTER.md`, `docs/STATUS_AND_ROADMAP.md`, `docs/development/DEVELOPMENT.md`, `docs/architecture/end-to-end-contract.md`.

## Components
- Frontend: React + TypeScript + Vite (`frontend/src/`). Browser-side MediaPipe Holistic extraction (`landmarks.ts`), WebSocket `/ws/translate`, REST endpoints (`endpoints.ts`).
- Backend: FastAPI (`backend/app/`), Python 3.13, `.venv`. WebSocket translation (`api/websocket/translation.py`), REST health/capabilities/models (`api/routes/`).
- ML pipeline: adapter registry (`app/ml/adapters/`), preprocessing (`normalization.py`, `landmarks.py`), inference (`continuous_sign.py`, `decoder.py`).
- Auth: Supabase JWT (`auth/v1/user`), RLS (`migrations/001_translation_sessions.sql`), history isolation per user.
- Deployment: `render.yaml`, Docker (`Dockerfile`, `fetch_bundle.sh`), no baked weights.

## Key contracts (pinned in `docs/architecture/end-to-end-contract.md`)
- Landmarks: 543 per frame (POSE 33 + FACE 468 + LEFT_HAND 21 + RIGHT_HAND 21). Order: POSE(0-32) → FACE(33-500) → LEFT_HAND(501-521) → RIGHT_HAND(522-542). Coordinates [x,y,z] only (no confidence in adapter contract).
- Normalization: `pose_normalize_v1` (hip-midpoint center, scale = median body-33 std, `nan_to_num→0`). Byte-identical between training and serving.
- Temporal: max 160 frames, rolling window 80 / stride 20, hard cap 640 (`SequenceTooLongError`). Streaming interims (`final:false`) → boundary (`final:true`).
- Transport: WS unversioned `/ws/translate`. REST `POST /api/v1/translation/sign-to-text` (live, uncalled by frontend — blocker B6). Status `GET /api/v1/health`, capabilities `GET /api/v1/capabilities`.
- Bundle contract: `checkpoint/`, `tokenizer/`, `landmark_contract.json`, `sign_to_text_deployment.json`. Missing in current checkout (bundle dir points to empty `research/artifacts/352868924/`).

## Integrity
- No single-authority contradiction: code + tests determine implementation (`AGENTS.md` rule 1); `STATUS_AND_ROADMAP.md` determines state; specialist docs determine contracts.
- `SIGNSAGE_MODEL_PROMOTED=false`: promotion requires (a) real weights beating trivial + diversity_ok; (b) `export_bundle`; (c) `model_check.py` pass; (d) `BUNDLE_DIR` + `PROMOTED=true`. None exist today.
