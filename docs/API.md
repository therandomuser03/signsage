SignSage — API / Communication Contract

## Status

No standalone backend is implemented yet — this remains true even though Phase 0 has now
produced a `GO-CANDIDATE` (see `research/outputs/phase0/RESULTS.md`). This is the planned
contract for the MVP (single-device) architecture in ARCHITECTURE.md — treat it as a
design target, update it the moment the real implementation diverges.

An earlier prototype of this contract lived inside the Phase-0 notebook (Section 18,
gated behind `RUN_LIVE_DEMO = False`). It has since been **removed from the notebook
entirely** (see DECISIONS.md) — no FastAPI or WebSocket code remains in
`research/notebooks/signsage.ipynb`. Its divergences from the contract below
are recorded here deliberately, as the checklist the real backend must not repeat: it
implemented `landmarks.frame`, `text.input`, `translation.text`, and `translation.pose`
over `ws://…/ws/{session_id}`, but carried `session_id` in the URL path rather than in
each message; had no `sentence.boundary` message (it auto-sent after 160 buffered
frames); omitted `latency_ms` from `translation.text`; omitted `gloss` and `fps` from
`translation.pose`; and did not implement the `error` type at all, so failures surfaced
as a dropped socket rather than the "fail loudly" behaviour DESIGN.md requires.

**Before wiring this contract up to a real model, read `research/outputs/phase0/RESULTS.md`.**
The current Sign→Text checkpoints repeat a small number of generic phrases regardless of
input (mode collapse), and the current Text→Pose checkpoint outputs a near-static pose
(motion collapse). `translation.text` and `translation.pose` will both work mechanically
against these checkpoints — they will just not produce useful output yet. Don't treat a
successful WebSocket round-trip as evidence the translation itself is correct.

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
the same normalization the model was trained on** (hip-centered, body-scale divided —
see DATA_MODEL.md and the `normalize_pose_sequence()` function in the notebook, which is
the literal source of truth for this transform, not this description of it) — the
removed notebook prototype sent raw coordinates and was wrong on this point (see
ARCHITECTURE.md risk 1). Whether normalization happens client-side or on receipt is
open, but it cannot be skipped, and it cannot be reimplemented from scratch without
risking subtle drift from the training-time version — extract it as shared code (see
PHASES.md Phase 0.5) rather than re-deriving it from this doc's prose description.

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
  a deprecation degrades instead of crashing. See SECURITY.md. Note: Phase 0's reported
  results use the naive heuristic gloss only (`USE_LLM_GLOSS = False`) — the Groq path is
  implemented but has not itself been part of any evaluated run.
- **Ollama** (offline fallback): local, no API key, intended for when Groq is unavailable
  or for cost-free local demos. **Not implemented yet** — only Groq-or-naive-heuristic
  exists in code today (see DECISIONS.md for the intended model).
