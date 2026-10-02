# SignSage — API & Communication Contract

**Status:** Implemented contract; keep synchronized with backend tests.

## 1. Transport

The core translation loop uses WebSocket.

Diagnostics and product functionality use REST.

### Diagnostics REST

```text
GET /api/v1/health
GET /api/v1/models
GET /api/v1/capabilities
```

These endpoints do not perform the streaming translation itself.

## 2. Client → server WebSocket messages

> Canonical source: `backend/app/schemas/websocket.py` (checked by
> `backend/tests/contract/test_frontend_contract.py` and mirrored in
> `frontend/src/services/websocket/protocol.ts`). Where this section
> disagrees with that schema, the schema wins. Known stale fields below:
> the live protocol has no `session_id`/`timestamp_ms` envelope, no
> `trigger` on `sentence.boundary` (only optional `save_history`), no
> `text.input` message, and no `translation.pose` / `latency_ms` reply —
> the real reply is `translation.text{text, accepted, reason, confidence,
> result_type, history_id}`. Every coordinate must be a finite number
> (NaN/±Inf rejected as `INVALID_MESSAGE`). Route: `POST`-less
> WebSocket at `/ws/translate` (unversioned).
>
> Streaming: while frames stream, the backend runs interim inference on the
> trailing `SIGNSAGE_STREAM_WINDOW_FRAMES` (default 80) every
> `SIGNSAGE_STREAM_STRIDE_FRAMES` (default 20) and emits stabilizing
> `translation.text` with `final: false` under repeat-agreement (first accepted
> prediction immediately, updates only after repeating on two consecutive
> windows; rejected windows emit nothing). `sentence.boundary` finalizes with
> `final: true` and consumes the buffer. Interims are never persisted to
> history.

### `landmarks.frame`

```json
{
  "type": "landmarks.frame",
  "session_id": "string",
  "timestamp_ms": 0,
  "landmarks": [0.0]
}
```

The landmark vector represents the browser-generated numerical landmark state expected by the backend contract.

### `sentence.boundary`

```json
{
  "type": "sentence.boundary",
  "session_id": "string",
  "trigger": "motion_pause"
}
```

`trigger` may represent motion-pause or manual submission.

### `text.input`

```json
{
  "type": "text.input",
  "session_id": "string",
  "text": "string"
}
```

## 3. Server → client messages

### `translation.text`

```json
{
  "type": "translation.text",
  "session_id": "string",
  "text": "string",
  "latency_ms": 0
}
```

### `translation.pose`

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
{
  "type": "error",
  "session_id": "string",
  "message": "string",
  "code": "string"
}
```

## 4. REST product surface

The current integration documentation also describes:

- text→sign request handling,
- history listing/deletion/clear operations,
- authenticated access,
- capability/model-status reporting.

The exact request/response schemas should remain synchronized with backend contract tests and the generated OpenAPI schema.

## 5. Authentication

Supabase access tokens are used for authenticated product operations. The WebSocket path can receive authentication information according to the frontend/backend integration contract.

## 6. Critical interpretation

A valid API response proves protocol correctness only.

It does **not** prove:

- semantic translation accuracy,
- signer generalization,
- useful pose generation,
- or production readiness.
