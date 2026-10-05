# SignSage Application Audit (read-only, no code changed)

Scope: webcam → English text (live path) plus dataset → training → checkpoint → adapter (model path).
Ethics rule enforced throughout: never invent a translation; unclear input must yield uncertainty/unsupported, not fluent fiction.

Status vocabulary: WORKING / PARTIALLY WORKING / BROKEN / UNUSED / EXPERIMENTAL / UNSAFE / MISSING.

## 1. Live path (webcam → English)

| # | Component | File(s) | Does / expects / produces | Used? | Status |
|---|-----------|---------|---------------------------|-------|--------|
| 1.1 | App routing, TranslatePage default | `frontend/src/app/routes.tsx:9,50-57`, `frontend/src/features/translate/TranslatePage.tsx` (648 lines) | `getUserMedia` → `<video>` → rAF loop → landmark detect → canvas overlay → throttled WS send. | Yes, default route | WORKING |
| 1.2 | Landmark extraction (3× tasks-vision VIDEO detectors) | `frontend/src/features/translate/landmarks.ts:14-20,56-124` | Expects `HTMLVideoElement`; produces `HolisticFrame{landmarks:543×[x,y,z]}` in order POSE(33)+FACE(468)+LH(21)+RH(21), raw normalized x/y, missing→`[0,0,0]`. VIDEO mode needs strictly increasing timestamps (`video.currentTime` guard). | Yes, sole extractor | WORKING (network needed on first run for wasm/task files; mirrored-selfie L/R swap accepted as known limit) |
| 1.3 | Temporal buffering | Frontend: none (stateless per-frame). Backend owns window: `backend/app/services/session_service.py:18-55`, `backend/app/core/config.py:61,70-71` | Frontend sends every 100 ms (`SEND_INTERVAL_MS`, `TranslatePage.tsx:36,151-164`), only when `frame.detected` and socket open. Backend buffers up to 640 frames, interim inference every window=80/stride=20, final on `sentence.boundary` (`backend/app/api/websocket/translation.py:89-151`). | Yes | WORKING (split ownership is intentional; frontend has no sequence logic) |
| 1.4 | Transport: WebSocket (primary) | `frontend/src/services/websocket/{protocol.ts:5-10,client.ts:57-75,connection.ts:62-73}`, `frontend/src/services/api/endpoints.ts:15`, `backend/app/api/websocket/translation.py:33-37`, `backend/app/schemas/websocket.py:1-50` | `landmarks.frame{543×3}` + `sentence.boundary` + `auth`; backend validates 543 rows × finite xyz. Unversioned `/ws/translate`. | Yes | WORKING |
| 1.5 | Transport: REST sign-to-text | `frontend/src/services/api/{endpoints.ts:10,client.ts:46-58}`, `backend/app/api/routes/translation.py:19-45` | `SignToTextRequest{frames…}` → `SignToTextResponse{text,accepted,reason,confidence,motion_energy,result_type}`. | Backend live; **frontend has zero callers** (`signToText` only in client/endpoints + contract test) — status display only (`health`, `capabilities` on mount) | PARTIALLY WORKING (backend path live, frontend never calls it) |
| 1.6 | Preprocessing (backend authority) | `backend/app/ml/preprocessing/{landmarks.py:12-41,normalization.py:21-61,sequence.py:12-44}` | Expects `(T,543,3)`; hip-midpoint center + body-spread scale (`pose_normalize_v1`) → `(F,1629)`; `resize_sequence` linspace downsample/zero-pad to 160; `prepare_live_landmarks` → `((160,1629), true_len)`; motion energy. Callers must NOT precompute deltas (model cats delta internally to 3258). | Yes | WORKING, with one silent-risk noted below |
| 1.7 | Contracts | `backend/app/ml/contracts/{landmark_contract.py:28-83,model_contract.py:21-94,inference_contract.py:28-60}` | Validate counts (543/3/1629/3258), norm version, token-id distinctness, decoding block; cross-check bundle vs code. | Yes, at load | PARTIALLY WORKING — counts/dims checked, but **component order/ranges values never compared** (reordered bundle with same counts passes) |
| 1.8 | Model: SignTransformer | `backend/app/ml/models/sign_to_text.py:14-115` | From-scratch Transformer, `FrameProjection` cats `[x1629, delta1629]` → 3258-D internally. `input_dim` param = post-cat 3258. | Yes, production owner | WORKING code, but only NO-GO weights exist (see §3) |
| 1.9 | Inference: greedy decode | `backend/app/ml/inference/sign_to_text.py:16-68` | Real autoregressive from BOS; bans pad/bos/unk, min-length EOS gate, no-repeat-ngram, repetition penalty. Always emits until EOS/max tokens. | Yes | WORKING code / UNSAFE system property (always emits *something*; honesty depends on downstream gate) |
| 1.10 | Adapter (sole service entry) | `backend/app/ml/adapters/sign_to_text_adapter.py:37-138` | `transcribe((T,543,3))` → prepare → motion → greedy → decode → confidence → acceptance gate → `TranscriptionResult(text or "" if rejected)`. Load validates checkpoint dims + tokenizer ids. | Yes | WORKING |
| 1.11 | Decoding | `backend/app/ml/postprocessing/decoding.py:9-40` | Strips BOS/PAD, breaks at EOS, SentencePiece decode, lower/split-join normalize. No semantic filter. | Yes | WORKING (passes model output through; fact-check is the gate's job, not decode's) |
| 1.12 | Confidence / acceptance gate | `backend/app/ml/postprocessing/confidence.py:14-32`, defaults `backend/app/core/config.py:52,58`, `adapters/sign_to_text_adapter.py:47` | `compute_confidence` = exp(mean logp), docstring says "diagnostic, not calibrated". `passes_acceptance_gate`: `insufficient_motion` / `low_model_confidence` / `empty_output` / `ok`. **Default `min_confidence=0.0` → confidence leg is OFF; only motion + non-empty gate.** | Yes | PARTIALLY WORKING — structure honest, threshold unsafe by default (see §4) |
| 1.13 | Registry / promotion gate | `backend/app/ml/models/registry.py:26-71`, `backend/app/core/config.py:39,47`, `backend/app/main.py:27-37` | Refuses serving unless `model_promoted=true` (or explicit dev override `allow_unpromoted_dev`, which logs a warning). Contract/checkpoint errors → `ModelUnavailableError`. | Yes | WORKING (this is what keeps the NO-GO model off the wire) |
| 1.14 | Endpoints honesty | `backend/app/api/websocket/translation.py:35-45`, `backend/app/api/routes/capabilities.py:33-55`, `backend/app/schemas/translation.py`, `domain/translation/policies.py:7-22` | Unpromoted/missing bundle → `MODEL_UNAVAILABLE` + close; capabilities expose `sign_to_text_available/reason`; failed/empty never persisted as presentable. | Yes | WORKING |
| 1.15 | Frontend rendering of verdicts | `TranslatePage.tsx:356,490,638` | Shows `accepted` vs `reason`, `result_type`, confidence %. Reasons (`insufficient_motion`, etc.) surface to the user. | Yes | WORKING |
| 1.16 | Text→Sign neural path | `backend/app/ml/models/text_to_sign.py`, `adapters/text_to_sign_adapter.py:15-19`, `inference/text_to_sign.py` | `raise NotImplementedError` by design (pose-collapse risk documented). Shipped EN→ISL is `api/routes/avatar_gloss.py` + `services/avatar_gloss_service.py` (Groq→Ollama→deterministic, fixed vocab, unmatched words reported never guessed). | Neural path dead by design; gloss path live | UNUSED (neural) / WORKING (avatar gloss, other direction) |
| 1.17 | `refinement.py` (capitalize+punctuate) | `backend/app/ml/postprocessing/refinement.py:13-20` | After-acceptance display polish. Zero callers in `backend/app`. | No | UNUSED (harmless) |

## 2. Contract-match checkpoints (live path)

- Dimensions: 543×3 = 1629 raw, 3258 post-cat — consistent across `landmarks.ts:14-20`, `protocol.ts:5-10`, `preprocessing/landmarks.py:12-21`, `model_contract.py:53-55`, `sequence.py`, `session_service.py`. Match CONFIRMED.
- Landmark ordering: frontend emits POSE+FACE+LH+RH; backend `normalize_pose_sequence` assumes first-33 = pose, hips 23/24, and `landmark_contract.json` is declared authoritative — but **neither the contract loader nor schemas compare order/range values**, only counts. A reordered bundle passes validation and silently mistranslates. See `END_TO_END_CONTRACT.md` for the exact pinned order.
- Normalization: frontend sends raw; backend `pose_normalize_v1` must match training export (notebook Sec 5a verbatim in code). Version string is pinned (`model_contract.py:55-58`); numeric equivalence relies on the export bundle's contract file, not re-verified at serve time.
- Sequence: frontend 10 fps, backend window 80 / stride 20 / max 160, `true_len` → padding mask. Consistent; `resize_sequence` downsamples by linspace index (temporal aliasing accepted, not a correctness break).
- Tokenizer: backend path uses 16k SentencePiece unigram (`PAD0/UNK1/BOS2/EOS3`); T5 POC uses google-t5 32k — intentionally incompatible, never cross-loaded (see `MODEL_READINESS.md`).

## 3. Model path (dataset → adapter)

| Component | Status | Note |
|-----------|--------|------|
| iSign dataset in repo | MISSING by design | `research/datasets/README.md`: streamed from HF, never committed; persistent copy lives on the old laptop |
| `research/training/*` SignTransformer V2 (canonical impl, SP-16k, export_bundle, READINESS plan, Kaggle notebook) | EXPERIMENTAL | Code defined, training not yet run |
| T5 POC `research/t5_pose_poc/*` (materialize/train/evaluate/model, self-contained Kaggle notebook) | EXPERIMENTAL | Research-only by README; real-gate memory + Phase-3 token bugs repaired; only synthetic-gate ever executed and it FAILED correctly (`t5_poc_gate/gate_report.json`: EM 0.0→0.09375 vs 0.5 threshold, `WIRING-FAIL`, correctly blocking 5k) |
| Checkpoints: 352868924 (WER 1.0046, diversity 0.99%, top-pred 44.8%) | BROKEN (as a translator) / WORKING (as a negative reference) | `promoted_to_backend:false`; reproducible collapse sample "the police said that" |
| T5 `best.pt` | MISSING | No 5k run has produced one; PoseT5 state dict could never load through `SignToTextAdapter` (different arch + 32k vocab) |
| Promotion (`backend/models/README`, `render.yaml` BUNDLE_URL/SHA) | WORKING gate, nothing promotable | Bar: beat-trivial + diversity_ok + `scripts/model_check` activity-gate |

## 4. Hallucination audit (can it invent English?)

- Previous unconstrained behavior is **gone from the serving path**: unpromoted model → `MODEL_UNAVAILABLE`, no text. The NO-GO weights cannot reach users even though they load.
- Residual UNSAFE property (latent, not currently reachable): **if** a future model is promoted with `min_confidence=0.0`, static/wrong-but-moving input yields fluent wrong sentences with `accepted=True` — there is no unknown-class, `unk` is banned from emission, and `compute_confidence` is uncalibrated. The gate structure is honest; its default threshold is not. Any promotion must set a calibrated `min_confidence` (see `PRODUCTION_BLOCKERS.md` B3).
- `decoding.py` has no fact filter (correct — filtering belongs in the gate/policies, which exist).
- Avatar-gloss LLMs are vocab-constrained and report unmatched words rather than guessing. No invented sign IDs possible.

## 5. Tests (useful vs executes-only)

Useful (assert behavior, all passing at audit time): frontend/backend contract parity (`contract/test_frontend_contract.py`); trust-boundary manifest rejects (`unit/ml/test_model_contract.py`); real-bundle load + static-reject + tamper/missing rejects (`integration/test_sign_to_text_adapter.py`); promotion refusal + protocol (`integration/test_websocket_translation.py`); domain invariants (`unit/domain/test_policies.py`); auth boundary (`unit/core/test_security.py`); vocab-filter/429-retry/fallback for gloss LLMs; `scripts/model_check.py` zero-motion-must-reject; `scripts/ws_diagnostic.py` transport-vs-accuracy separation (synthetic ramp proves plumbing, says nothing about recognition — documented in its own docstring).
Merely-executes: `__init__.py`, `conftest.py` fixtures, `__pycache__`; no assertion-free test was found — every named test above asserts a verdict.
Coverage gaps (no test): landmark **order** values, normalization numeric equivalence with training export, calibrated confidence threshold, T5↔backend incompatibility rejection (relies on shape errors, untested).
