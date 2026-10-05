# SignSage End-to-End Validation (2026-10-05, no code changed)

Method: two live servers — (A) stock backend (gate closed, the production
truth) and (B) the same server with a throwaway tiny bundle (`/tmp`,
uncommitted, untrained weights) to exercise transport, gating, and honesty
paths. Server B proves plumbing and gate behavior ONLY; its "translations"
are untrained-model noise and are recorded as such, never as recognition.
No camera/browser exists on this machine: capture-side items are
code-verified against the reviewed sources, marked [CODE] vs [LIVE].

Safety rule applied throughout: a wrong confident sentence = FAILURE;
explicit uncertain/unsupported = the correct answer when unrecognized.

## Scenario matrix

| # | Input | Expected | Actual | Verdict | Reason |
|---|-------|----------|--------|---------|--------|
| 1 | Supported static sign | Recognized with text | No supported vocabulary exists (no trained weights anywhere: no `best.pt`, `backend/models/` empty, only NO-GO 352868924). Real server answers `MODEL_UNAVAILABLE`. | FAIL (blocked) | Nothing to recognize with; correctly refused rather than faked |
| 2 | Supported dynamic sign | Recognized with text | Same as T1. | FAIL (blocked) | Same as T1 |
| 3 | Multiple consecutive signs | Each recognized / segmented | Same as T1; segmentation path (`sentence.boundary` → final) proven working on harness in prior turn. | FAIL (blocked) | Same as T1 |
| 4 | No person in frame (all-zero 20f) [LIVE-B] | Rejected, `text ""` | `accepted=false reason=insufficient_motion text='' conf=0.085` | PASS | Motion gate catches empty input |
| 4b | Static pose, no motion [LIVE-B] | Rejected | `accepted=false reason=insufficient_motion text=''` | PASS | Same gate |
| 5 | Random full-body movement [LIVE-B] | Uncertain (ideally) | `accepted=true reason=ok text='fm yourislvr' conf=0.084` | FAIL (harness) | Uncalibrated confidence + `min_confidence=0.0` default cannot discriminate; untrained weights decode noise as text. **This is exactly why the gate stays closed (blocker B3).** Not a code defect — fix = trained weights + calibrated floor |
| 5b | Hands-only random motion [LIVE-B] | Uncertain (ideally) | `accepted=true text='fm yourislvr' conf=0.084` | FAIL (harness) | Same as T5 |
| 6 | Smooth structured non-sign wave [LIVE-B] | Uncertain (ideally) | `accepted=true text='fm yourislvr' conf=0.085` | FAIL (harness) | Same as T5; structure without meaning still decodes |
| 7 | Partial sign (5 frames) [LIVE-B] | Processed without crash; ideally uncertain | `accepted=true text='fm yourislvr' conf=0.082`, http 200, no crash | PASS (robustness) / FAIL (discrimination, harness) | Pipeline handles short sequences (pad + mask verified earlier); verdict quality same caveat as T5 |
| 8 | Poor detection (90% zeros) [LIVE-B] | Uncertain (ideally) | `accepted=true text='fm yourislvr' conf=0.085` | FAIL (harness) | Same as T5 |
| 9 | Multiple people [CODE] | Defined behavior | `numPoses:1/numFaces:1`, first detection only (`landmarks.ts:91,100,124-125`); 2nd person ignored by design; backend takes one 543×3/frame | PASS | Deterministic by construction, documented limit |
| 10 | Camera denied/unavailable [CODE] | Clear error state | `getUserMedia` absence/denial → error phase + message (`TranslatePage.tsx:207-227`); mid-session unplug NOT handled (no `track.onended`; frozen frame, no error) | PASS with limitation | Denial path honest; unplug gap noted, not fixed (no defect in tested paths) |
| 11 | Backend dead (server killed) [LIVE] | Loud failure, no hang | WS connect → `ConnectionRefusedError` immediately; client has bounded reconnect (5 attempts, backoff, no send-queue: `connection.ts`) | PASS | Fails loudly, never queues stale frames |
| 12 | Model unavailable, server up [LIVE-A] | Honest refusal everywhere | WS → `{"type":"error","code":"MODEL_UNAVAILABLE"}` then close; REST → 503 `MODEL_NOT_PROMOTED`; capabilities `available:false reason:model_not_promoted`; frontend renders distinct "Translation unavailable" panel (new, verified by build + contract tests; no browser to click through) | PASS | The production truth today, end to end |
| 13 | Low-confidence inference [LIVE-B] | Unsupported, `text ""` | `min_confidence=1.0` → `accepted=false reason=low_model_confidence text=''` (REST-proven; mechanism, not a calibrated default — none is claimed) | PASS | Floor mechanism works; default stays 0.0 until calibrated on real validation data |
| R | Repeat stability [LIVE-B] | Identical verdicts | Same 20f input ×3 → byte-identical responses; adapter-level test additionally proves eval mode + zero parameter mutation | PASS | No per-request state, no drift |

## Safety-requirement verdict

On the production configuration (server A), the requirement holds
trivially and absolutely: **no model → no text, only `MODEL_UNAVAILABLE` /
`accepted:false` with reasons**. No fluent sentence can be produced.
On the harness (server B), an unvalidated model with the default floor DOES
emit confident gibberish for random input (T5/T5b/T6/T8) — recorded as FAIL
(harness) because that configuration must never serve users, which it does
not: the promotion gate, the absent weights, and the 503s above are the
enforcement. The frontend never fabricates text in any path (no mocks found;
rejected/empty results render "Not confidently recognized", never `text`).

## Classification: NOT FUNCTIONAL

SignSage cannot recognize any sign today — there is no supported vocabulary
at all, so it is not functional for even a restricted vocabulary, and
nothing here supports calling it production-ready. What IS functional is the
honesty machinery around the missing model: capture → transport → validation
→ gating → explicit uncertainty all behave as specified. The single path from
NOT FUNCTIONAL to restricted-vocabulary functional is: train V2 → beat the
NO-GO bar with diversity → calibrate a real confidence floor → export →
promote. Until then the correct production output remains `MODEL_UNAVAILABLE`.
