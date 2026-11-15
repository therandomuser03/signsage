# API Reference

Verified contracts: `docs/architecture/end-to-end-contract.md` (authoritative pin); `backend/app/schemas/websocket.py`; `backend/app/schemas/translation.py`.

## REST
- `GET /api/v1/health` → `{status, model_promoted, model_bundle_found, model_loaded}`. Honest refusal when `promoted=false`.
- `GET /api/v1/capabilities` → `{sign_to_text_available, unavailable_reason, text_to_sign_mode: "avatar_gloss", model_bundle_found, ...}`.
- `GET /api/v1/models` → `{promoted: false, loaded: false, metrics: {}, ...}`.
- `POST /api/v1/translation/sign-to-text` → `SignToTextRequest/Response`. 503 `MODEL_NOT_PROMOTED` when gate closed (correct behavior — not 500, not fabricated output).
- `POST /api/v1/classification/image?task=alphabet` → `label` (A-Z), `confidence`, `accepted`, `reason`, `model_version=signsage-image-alphabet`.
- `POST /api/v1/classification/image?task=word` → 20-class word; unpromoted.
- `POST /api/v1/speech/speak` → `{text, voice?}` → `audio/wav` (Groq Orpheus, `canopylabs/orpheus-v1-english`, ≤200 chars, voices autumn/diana/hannah/austin/daniel/troy). Rate-limited per client (`SIGNSAGE_RATE_LIMIT_TTS`, default 9/min). 503 `TTS_UNAVAILABLE` when unconfigured, over the character limit, rate-limited, or the provider fails → frontend falls back to browser Web Speech voices. Requires one-time terms acceptance for the model in the Groq console (org admin).
- `POST /api/v1/text-to-sign` → gloss IDs (`HELLO`, `THANK_YOU`, etc.) from 14-sign vocabulary. Deterministic mapping; never invents IDs.
- `GET /api/v1/auth/me` → user info or 401 (`UNAUTHORIZED`).
- `GET /api/v1/history` → user-isolated, session-based translation history; requires auth; 401 without. Supports `limit`, `offset`, `direction`, `q`, `sort`. Sessions embed their completed sentences.
- `GET /api/v1/history/{id}` → one session with its sentences, chronologically ordered; 404 for another user's id.
- `DELETE /api/v1/history/{id}` → delete one session (sentences cascade); 404 for another user's id.
- `POST /api/v1/history/bulk-delete` → `{"ids": [...]}` deletes the caller's own sessions among the ids.
- `DELETE /api/v1/history` → delete ALL of the caller's sessions (and sentences).
- `POST /api/v1/translation/word` → isolated-word recognition (gated by `SIGNSAGE_WORD_MODE_ENABLED` + `word_model/`); returns `word, confidence, accepted, top3, threshold, model_version, status`.
- `POST /api/v1/translation/sentence` → phrase capture → ordered glosses → English sentence; `method` is `llm` (Groq, constrained to the signed words) or `fallback` (rule-based/comma-join; the LLM can never add content words — output is guard-checked, unverifiable output falls back to comma-join).
- `POST /api/v1/translation/compose` → same composition from a client-held word list; `save_history` + `client_session_key`/`client_key` group sentences under one session (idempotent writes).

## WebSocket (`/ws/translate` — unversioned)
- Auth first message: `{"type":"auth"}` with JWT from Supabase.
- Landmarks: `{"type":"landmarks","frame":{"landmarks":[[x,y,z]×543], ...}}`. Missing → `[0,0,0]`.
- Interim: `{"type":"translation","text":"...","final":false}` (deduped by repeat-agreement). Only emitted after sufficient motion accepted.
- Boundary: `{"type":"sentence","boundary":"...","text":"...","final":true}`.
- Errors: `{"type":"error","code":"MODEL_UNAVAILABLE",...}` (gate doing its job) or `INVALID_MESSAGE` / `SEQUENCE_TOO_LONG`.
- Reconnect: bounded 5 attempts with backoff (`docs/development/DEVELOPMENT.md`).

## Contract values (pinned)
| Property | Value | Source |
|----------|-------|--------|
| Total landmarks | 543 | `landmarks.ts`, `normalization.py`, `end-to-end-contract.md` |
| Per-landmark coords | 3 (x, y, z) — no confidence in adapter path | `landmarks.ts:14-20` |
| Component order | POSE(0-32) → FACE(33-500) → LEFT_HAND(501-521) → RIGHT_HAND(522-542) | `end-to-end-contract.md` §1 |
| Per-frame feature | 543 × 3 = 1629 floats | Contract pin |
| Sequence cap | 160 frames; rolling 80 / stride 20; hard cap 640 | `config.py`, `end-to-end-contract.md` §2 |
| Normalization | `pose_normalize_v1` (hip-midpoint center, scale = median std body-33, `nan_to_num→0`) | `normalization.py:21-61`, `DATA_AND_MODEL.md` §2 |
| Preprocessing identity | Byte-identical between training export and serving adapter (`DATA_AND_MODEL.md` §2) | Contract + adapter |
| Message max bytes | Enforced in `schemas/websocket.py` | `end-to-end-contract.md` §5 |
| Tokenizer track | SP-16k (`tokenizer/signsage_en_sp.model` + `.vocab`) | `DATA_AND_MODEL.md` §3 |
| T5 track | `google-t5/t5-small` 32k; requires separate adapter/export (`t5_poc_gate` FAIL) | `DATA_AND_MODEL.md` §3 |

## Safety / authorization boundaries
- No raw camera upload path (only landmarks over WS; `AGENTS.md` rule 8).
- Auth boundary: JWT verified at `core/security.py`; RLS enforced on `translation_sessions`; user isolation verified by `test_api_auth_history.py` (mocked); live multi-account proof BLOCKED (needs test account + RLS live round-trip).
- `SIGNSAGE_ALLOW_UNPROMOTED_DEV`: local dev escape only (`.env`); `render.yaml` omits; `docs/development/RENDER_MODEL_DEPLOYMENT.md` explicitly says never enable on production.
