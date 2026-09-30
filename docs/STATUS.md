# SignSage — Current Status

**Status date:** 2026-09-29

## Status vocabulary

| Status | Meaning |
|---|---|
| Implemented | Code exists |
| Verified | A documented test/check has passed |
| Integrated | Components communicate through the intended contract |
| Experimental | Works technically but lacks sufficient validation |
| Blocked | Cannot be considered complete because a prerequisite is missing |
| Deferred | Intentionally postponed |
| Planned | Future work |

## System status

| Area | Status | Notes |
|---|---|---|
| Frontend | Implemented / integrated | React + TypeScript application and translation UI |
| MediaPipe | Implemented | Browser-side landmark extraction |
| WebSocket | Implemented / verified | Landmark and result message contract |
| Backend | Implemented / verified | FastAPI services and diagnostics |
| Authentication | Implemented | Supabase-based auth integration |
| History | Implemented | Signed-in history with user isolation |
| Capabilities | Implemented | Reports model availability/status |
| English → gloss | Implemented | Groq → Ollama → deterministic fallback |
| Text → pose UI | Implemented infrastructure | Rendering exists; research quality remains experimental |
| Sign → Text model | Experimental / not promoted | Research checkpoint has documented mode collapse |
| General ISL accuracy | Unverified | No current evidence supports a broad accuracy claim |
| End-to-end latency | Partially verified | Model-only measurements exist; full target-device benchmark remains separate |
| Avatar assets | Experimental/dependent on assets | Real ISL animation content must be validated independently |
| Two-device relay | Deferred | Single-device architecture remains the defined scope |
| High-stakes use | Out of scope | Not a certified interpreter replacement |

## Model promotion rule

A checkpoint must not be called the default production model merely because:

- it loads,
- inference does not crash,
- the WebSocket succeeds,
- training loss decreases,
- BLEU/ROUGE-L improve,
- DTW/MSE looks favorable.

Promotion requires an explicit evaluation record covering translation quality, output diversity, failure modes, and the relevant baseline.

## Known research gaps

- Signer-independent generalization has not been established.
- Sign→Text mode collapse remains unresolved in the documented experiments.
- Text→Pose motion fidelity remains weak.
- End-to-end latency on defined target hardware needs a repeatable measurement.
