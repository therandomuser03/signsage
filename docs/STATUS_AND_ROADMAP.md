# SignSage Status & Roadmap

Where the project stands, what was verified, what blocks production, and the
exact order of work. (Consolidated 2026-10-13 from `STATUS.md`,
`APPLICATION_AUDIT.md`, `PRODUCTION_BLOCKERS.md`, `END_TO_END_VALIDATION.md`.
Status date of record: 2026-10-02; validation run: 2026-10-05.)

## 1. Vocabulary & system status

| Status | Meaning |
|---|---|
| Implemented / Verified / Integrated | Code exists / a documented check passed / components talk through the contract |
| Experimental | Works technically, lacks validation |
| Blocked / Deferred / Planned / Out of scope | As named |

| Area | Status |
|---|---|
| Frontend, MediaPipe, WebSocket, backend, auth, history, capabilities, English→gloss | Implemented (WS verified) |
| Text→pose UI | Infrastructure implemented; quality experimental |
| Sign→Text model | Experimental pilot: isolated-word recognizer (6 INCLUDE words, 61% test accuracy), NOT promoted; original transformer (352868924) remains NO-GO |
| General ISL accuracy | Limited to 6 ISL words (friend, hello, hospital, market, school, thank you); no evidence supports broader claims |
| End-to-end latency | Partially verified (model-only; no target-device benchmark) |
| Two-device relay | Deferred (single-device scope) |
| High-stakes use | Out of scope (not a certified replacement) |

Promotion rule: a checkpoint is production only on an explicit evaluation record
covering quality, diversity, failure modes, and baselines — never merely because
it loads, runs, trains, or improves one metric. Known gaps: signer-independent
generalization, Sign→Text collapse, Text→Pose motion fidelity, target-hardware
latency measurement.

## 2. Integration passes (2026-10-02; software only, NO-GO verdict unchanged)

- Hostile-input rejection (NaN/±Inf → `INVALID_MESSAGE`); `ALLOW_UNPROMOTED_DEV`
  (default false, never on Render) for local end-to-end exercising.
- Bounded WS reconnect (5 attempts, backoff); `ws_diagnostic.py` proves gated
  refusal + full serving path on synthetic landmarks (transport, not accuracy).
- `render.yaml` + `fetch_bundle.sh` Docker deploy with checksum-verified
  download-at-start bundle provisioning.
- Streaming inference verified on the experimental checkpoint: 80/20 rolling
  window, repeat-agreement stabilization, `final:false` interims vs `final:true`
  boundary; ~50–70ms CPU per interim. `SignTransformer(norm_first=False`
  default) keeps old bundles byte-identical.
- 352868924 root cause: under-trained small post-norm Transformer, not a code
  bug — v2 config (pre-norm d256×4L, label smoothing, ≤12ep, diversity +
  overfit gates) specified in `research/training/READINESS.md`; the Kaggle GPU
  run is the operator's next step.

## 3. Application audit (read-only; ethics rule: never invent a translation)

Live path (webcam → English), all WORKING unless noted: routing/TranslatePage
default; 3× tasks-vision VIDEO landmark extraction (543×[x,y,z], missing→zero,
selfie-swap accepted limit); stateless frontend → backend-owned 640-cap window
(80/20 interims, `sentence.boundary` final); WS transport (unversioned
`/ws/translate`); REST sign-to-text backend-live but frontend-uncalled;
`pose_normalize_v1` preprocessing; load-time contracts (counts/dims checked —
**order/range values never compared**, latent B2); SignTransformer + greedy
decode (emits *something* always — honesty depends on the gate); acceptance
verdicts (`insufficient_motion`/`low_model_confidence`/`empty_output`/`ok`)
with `min_confidence=0.0` (confidence leg OFF — latent B3).

Contract checkpoints: dims 543/1629/3258 CONFIRMED consistent; tokenizer tracks
(SP-16k vs T5-32k) intentionally incompatible, never cross-loaded. Model path:
dataset MISSING by design (see `DATA_AND_MODEL.md` §1); V2 code EXPERIMENTAL;
T5 POC EXPERIMENTAL (synthetic-gate WIRING-FAIL, correctly blocking);
352868924 BROKEN-translator/WORKING-negative-reference, unpromoted. Removed
2026-10-12: neural text-to-sign stubs + `refinement.py` (were unused).

Hallucination audit: unconstrained behavior is gone from serving (unpromoted →
`MODEL_UNAVAILABLE`, no text). Latent: a future promoted model at
`min_confidence=0.0` would accept fluent-wrong output — promotion must calibrate
it (B3). Avatar-gloss LLMs are vocab-constrained, never invent IDs.

Tests: contract parity, trust-boundary rejects, bundle load + static-reject +
tamper rejects, promotion refusal, domain invariants, auth boundary, gloss-LLM
fallback, `model_check` zero-reject, `ws_diagnostic` transport separation — all
asserting, all passing at audit time. Gaps: order values, norm-equivalence vs
training export, calibrated threshold, T5-rejection path.

## 4. Production blockers (ordered; B1 is the blocker)

- **B1. No promotable model.** The isolated-word pilot (61% test accuracy, zeros in
  per-class recall, not signer-independent) does not meet the promotion bar
  (80% accuracy / 80% macro recall / 60% per-class recall). The original
  352868924 transformer remains NO-GO (WER 1.0046, EM 0.0) and cannot be
  recovered for ISL recognition — it was trained on synthetic iSign
  text→pose data, not real ISL video. Fix = train on more real ISL data →
  beat thresholds → export → verify → promote.
- **B2. Landmark order trusted, never verified.** Fix = compare
  order/range values at bundle load + swapped-order rejection test.
- **B3. Confidence gate ships OFF.** Fix with any promotion = calibrate on
  held-out data (or add unsupported verdict) + wrong-sign-must-reject test.
- **B4. No numeric norm-equivalence check.** Fix = golden-vector test shared by
  training export and backend.
- **B5. T5↔backend incompatibility implicit.** Fix only if T5 is ever pursued
  (dedicated adapter/contract/export; do not build speculatively).
- **B6. Frontend REST translation path dead.** Fix = wire with gates or delete;
  never claim it works.

Working today: capture → MediaPipe 543×3 → overlay → throttled WS → validation →
windowing → preprocessing → contract-checked adapter → isolated-word inference →
verdict rendering; honest `MODEL_UNAVAILABLE` when gated; isolated-word pilot
servable via `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true` (6 ISL words, 61% accuracy,
NOT promoted); capabilities/health; avatar-gloss with constrained fallback;
green suite (235 tests). Broken: no promotable ISL recognition weights exist
(gate doing its job). Path to restricted-vocabulary functional: train on more
real ISL data → beat promotion thresholds (80/80/60) → export → promote.

## 5. End-to-end validation (2026-10-05, no code changed)

Two live servers: (A) stock gated backend (production truth), (B) same server +
throwaway tiny untrained bundle (plumbing/gating only — its "translations" are
noise, recorded as such). Capture-side items code-verified ([CODE] vs [LIVE]).
Rule: wrong confident sentence = FAILURE; explicit uncertainty = correct.

| # | Input | Result |
|---|---|---|
| 1–3 | Supported static/dynamic/consecutive signs | FAIL (blocked) — nothing to recognize with; correctly refused (`MODEL_UNAVAILABLE`) |
| 4/4b | Empty frame / static pose [LIVE-B] | PASS — `accepted=false/insufficient_motion` |
| 5/5b/6/8 | Random/structured/poor-detection motion [LIVE-B] | FAIL (harness) — `accepted=true` gibberish at `min_confidence=0.0`; exactly why the gate stays closed (B3), not a code defect |
| 7 | 5-frame partial [LIVE-B] | PASS robustness / FAIL discrimination (same caveat) |
| 9 | Multiple people [CODE] | PASS — first detection only, by design |
| 10 | Camera denied [CODE] | PASS with limitation (mid-session unplug unhandled — noted, unfixed) |
| 11 | Backend dead [LIVE] | PASS — immediate loud failure, bounded reconnect, no stale queue |
| 12 | Model unavailable, server up [LIVE-A] | PASS — WS `MODEL_UNAVAILABLE`+close, REST 503, capabilities `false`, distinct UI panel |
| 13 | Forced floor [LIVE-B] | PASS — mechanism works; no calibrated default claimed |
| R | Repeat stability [LIVE-B] | PASS — byte-identical, no drift |

Classification: **NOT FUNCTIONAL** for recognition (no supported vocabulary at
all). Functional: the honesty machinery — capture → transport → validation →
gating → explicit uncertainty. Path to restricted-vocabulary functional: train
V2 → beat NO-GO bar with diversity → calibrate floor → export → promote.
