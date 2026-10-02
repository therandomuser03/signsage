# SignSage — Current Status

**Status date:** 2026-10-02

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

## Integration pass (2026-10-02)

Software-pipeline work only; no change to the NO-GO research verdict or the
promotion rule above:

- Landmark validation rejects non-finite coordinates (NaN/±Inf) at the
  contract boundary (`INVALID_MESSAGE`); covered by WebSocket hostile-input tests.
- `SIGNSAGE_ALLOW_UNPROMOTED_DEV` (default false, never on Render) lets a
  local backend serve an unpromoted checkpoint for end-to-end exercising.
  `/models` still reports `promoted: false`; availability ≠ quality.
- Frontend auto-reconnects with bounded exponential backoff (5 attempts) and
  shows Reconnecting… vs Connecting; socket/camera/teardown cleanup preserved.
- Frontend builds green again after migrating Radix `asChild` call sites to
  the installed Base UI `render` prop and removing two dead, unimported
  components (`file-tree`, `scroll-area`); no UI redesign.
- `backend/scripts/ws_diagnostic.py` verifies the live WS transport against a
  running server using labelled synthetic landmarks (transport test, not an
  accuracy test). Verified: gated refusal path and full serving path through
  real model load + inference + acceptance gate.
- `render.yaml` + `backend/scripts/fetch_bundle.sh`: Docker deploy with
  download-at-start bundle provisioning (checksum-verified, fail-loud).

## Model recovery + streaming pass (2026-10-02)

Root cause of 352868924 (not a code bug — masking/shift/loss/split all
verified; tiny-overfit hit 100%): severe under-training of a small post-norm
Transformer (3 epochs, val loss still falling) on 100k diverse pairs, so the
decoder stayed in the prior regime and greedy sampled one mode. Details and
the corrected v2 config (pre-norm, d256×4L, label smoothing, ≤12 epochs,
per-epoch diversity collapse gate, hard overfit gate) are in
`research/training/READINESS.md`; reproducible job in
`research/training/train_sign_to_text.py` (+ thin Kaggle notebook
`signsage_kaggle.ipynb`); dataset audit in `research/training/audit_dataset.py`.
v2 training itself is NOT run here — the Kaggle GPU run is the operator's
explicit next step, so general ISL accuracy stays Unverified.

- Streaming inference is implemented and verified with the experimental
  checkpoint: rolling 80-frame window / 20-frame stride, repeat-agreement
  stabilization, `final: false` interims vs `final: true` boundary results,
  frontend partial-vs-final display. Measured CPU cost ~50–70 ms per interim;
  interim path covered by `test_rolling_interim_then_final`.
- Backend `SignTransformer` accepts `norm_first` (default False: old bundles
  byte-identical); v2 bundles set it True via `model_config`.

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
