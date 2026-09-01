SignSage — API / Communication Contract

## Status

No standalone backend is implemented yet. This is the planned contract for the MVP
(single-device) architecture in ARCHITECTURE.md — treat it as a design target, update
it the moment the real implementation diverges.

An earlier prototype of this contract lived inside the Phase-0 notebook (Section 18,
gated behind `RUN_LIVE_DEMO = False`). It has since been **removed from the notebook
entirely** (see DECISIONS.md) — no FastAPI or WebSocket code remains in
`notebooks/signsage.ipynb`. Its divergences from the contract below
are recorded here deliberately, as the checklist the real backend must not repeat: it
implemented `landmarks.frame`, `text.input`, `translation.text`, and `translation.pose`
over `ws://…/ws/{session_id}`, but carried `session_id` in the URL path rather than in
each message; had no `sentence.boundary` message (it auto-sent after 160 buffered
frames); omitted `latency_ms` from `translation.text`; omitted `gloss` and `fps` from
`translation.pose`; and did not implement the `error` type at all, so failures surfaced
as a dropped socket rather than the "fail loudly" behaviour DESIGN.md requires.

## Transport

WebSocket, one connection per active session (one browser tab = one session, since the
MVP is single-device). No REST endpoints planned for the core translation loop — a
persistent socket avoids per-frame HTTP overhead for streaming landmarks.

## Client -> Server messages

### `landmarks.frame`
Sent continuously while the camera is active and a sentence is being signed.
```json
{
  "type": "landmarks.frame",
  "session_id": "string",
  "timestamp_ms": 0,
  "landmarks": [0.0]
}
```
`landmarks` is a flat array of 1629 floats — one frame of MediaPipe Holistic output for
543 landmarks (pose 33 + face 468 + left hand 21 + right hand 21), in that order, as
`[x, y, z]` triples, zero-padded for any component the frame is missing. **It must carry
the same normalization the model was trained on** (hip-centered, body-scale divided — see
DATA_MODEL.md); the removed notebook prototype sent raw coordinates and was wrong on this
point (see ARCHITECTURE.md risk 1). Whether normalization happens client-side or on
receipt is open, but it cannot be skipped.

### `sentence.boundary`
Sent when the client's motion-pause heuristic fires, or the user hits manual "Send."
```json
{ "type": "sentence.boundary", "session_id": "string", "trigger": "motion_pause" }
```
`trigger` is `"motion_pause"` or `"manual"` — worth logging during development to tune
the heuristic (see ARCHITECTURE.md tricky-integration-points).

### `text.input`
Sent when the hearing-side user types or speaks a sentence to translate to ISL.
```json
{ "type": "text.input", "session_id": "string", "text": "string" }
```

## Server -> Client messages

### `translation.text`
Result of the Sign->Text pipeline for one completed sentence.
```json
{ "type": "translation.text", "session_id": "string", "text": "string", "latency_ms": 0 }
```

### `translation.pose`
Result of the Text->Pose pipeline — a sequence of frames to animate.
```json
{
  "type": "translation.pose",
  "session_id": "string",
  "gloss": "string",
  "frames": [[0.0]],
  "fps": 25
}
```

### `error`
```json
{ "type": "error", "session_id": "string", "message": "string", "code": "string" }
```
Used for the "fail loudly" principle in DESIGN.md — e.g. low landmark confidence, model
timeout, LLM API failure.

## External APIs

- **Groq** (primary LLM, text->gloss): model `openai/gpt-oss-20b`, `reasoning_effort="low"`.
  API key via environment variable (`GROQ_API_KEY`), never hardcoded or committed. Any
  exception falls back to the deterministic naive gloss (lowercase + stopword removal) so
  a deprecation degrades instead of crashing. See SECURITY.md.
- **Ollama** (offline fallback): local, no API key, intended for when Groq is unavailable
  or for cost-free local demos. **Not implemented yet** — only Groq-or-naive-heuristic
  exists in code today (see DECISIONS.md for the intended model).
