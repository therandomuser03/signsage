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
