# SignSage Production Blockers (ordered)

## B1. No promotable model exists (the blocker; everything else is secondary)
Old checkpoint 352868924 is verified NO-GO (WER 1.0046, EM 0.0, diversity 0.99%) and correctly gated off. V2 training not started. T5 track has no 5k weights and never will serve through the current adapter. Fix = train (V2 canonical track first), beat trivial baseline with diversity_ok, export, verify, then promote. No code change promotes a bad model; the gate must stay.

## B2. Landmark order is trusted, never verified
Pipeline assumes POSE(33, hips 23/24)+FACE(468)+LH(21)+RH(21) end to end, but `landmark_contract.py` and `schemas/*` check only counts, never order/range values. A reordered upstream (MediaPipe version, tasks-vision file, export header) passes all checks and silently mistranslates. Fix = compare `component_order`/`component_ranges` values at bundle load + add a contract test with a swapped-order fixture (must reject).

## B3. Confidence gate ships OFF (`min_confidence=0.0`)
Structure is honest (`insufficient_motion`/`low_model_confidence`/`empty_output`/`ok`), but with the default threshold any moving input that decodes non-empty is `accepted=True` — including a future model's fluent mistakes. There is no unknown-class and the confidence score is documented uncalibrated. Fix before/with any promotion = calibrate a threshold on held-out data (or add an explicit unsupported verdict) and set it in config; add a wrong-sign-must-reject test.

## B4. No numeric normalization-equivalence check
`pose_normalize_v1` is pinned by version string, but serve-time code vs training-export numerics are never differentially tested. A silent divergence degrades every translation without tripping any gate. Fix = golden-vector test (fixed 543×3 input → expected normalized output) shared by training export and backend.

## B5. T5↔backend incompatibility is implicit
Nothing in-repo states at load time that a T5 `best.pt` is unloadable here; rejection relies on incidental shape errors. Fix = only if T5 is ever pursued: dedicated adapter/contract/export (currently correctly absent — do not build speculatively).

## B6. Frontend REST path is dead code; WS is the only live translation transport
`ApiClient.signToText` has no callers; interim/final translation flows only over `/ws/translate`. Not a user-facing break today, but any "REST translation works" claim would be false. Fix = either wire it (with the same gates) or delete it; do not claim it.

---

# Required summary

## A. Current working functionality
Camera → MediaPipe tasks-vision 543×3 → canvas overlay → throttled WS send → 543-shape validation → session windowing (80/20, cap 640) → preprocessing → contract-checked adapter → greedy decode → acceptance verdict → `accepted/reason/result_type/confidence` rendered; honest `MODEL_UNAVAILABLE` when unpromoted; capabilities/health; avatar-gloss (other direction) with vocab-constrained fallback; full test suite green (contract, trust-boundary, adapter behavior, protocol, auth).

## B. Current broken functionality
Sign→Text translation itself: the only loadable weights are verified NO-GO and correctly blocked, so every live session ends in `MODEL_UNAVAILABLE`. No regression — this is the gate doing its job.

## C. Exact reason there is no reliable ISL translation today
There is no trained, evaluated, promoted model. The pipeline from camera to English is plumbed and honest, but the one component that must map poses to words — promotable weights — does not exist. Secondary latent risks (B2 order-trust, B3 threshold-off, B4 norm-equivalence) would matter the day weights exist; they do not cause today's outage.

## D. Smallest viable architecture that genuinely works
Keep everything: the current frontend capture, WS transport, backend preprocessing/contracts/adapter/gates, and the SignTransformer serving track are the architecture. Add only: (1) real V2 training run on the laptop-stored iSign data → evaluated bundle; (2) B2 order-value check; (3) B3 calibrated threshold; (4) B4 golden-vector test. No new model family, no T5 serving branch, no transport redesign, no unconstrained generation — the previous fluent-fiction behavior must never return.

## E. Exact implementation order
1. B4 golden-vector test + B2 order-value check (cheap, protect the training run's meaning).
2. Run V2 training per `research/training/READINESS.md` (old laptop data → Kaggle GPU → bundle).
3. Evaluate against the NO-GO bar (beats-trivial + diversity_ok + sensitivity + `model_check` zero-reject).
4. Calibrate and set `min_confidence` (B3) with a wrong-sign-must-reject test.
5. Promote (`BUNDLE_DIR` + `PROMOTED=true`, `render.yaml` SHA) and verify WS end-to-end via `ws_diagnostic.py` + capabilities.
6. Then, and only then, decide REST-wire-vs-delete (B6). T5 stays research-only unless SignTransformer demonstrably plateaus.
