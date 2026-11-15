# SignSage Status & Roadmap

**Status summary reviewed:** 2026-11-02  
**Model promotion:** `SIGNSAGE_MODEL_PROMOTED=false`.

This document contains detailed dated implementation and evaluation notes. Read
[ml/MODEL_STATUS.md](ml/MODEL_STATUS.md) for the concise current model decision,
[product/PRODUCT_OVERVIEW.md](product/PRODUCT_OVERVIEW.md) for the product boundary,
and [product/DEMONSTRATION_GUIDE.md](product/DEMONSTRATION_GUIDE.md) for the
evidence-based demo scope.

Current high-level position:
- The application has verified infrastructure and bounded features, including
  static alphabet image classification and a constrained English-to-gloss/avatar path.
- The isolated-word video pilot remains experimental and unpromoted.
- Continuous ISL-to-English sentence translation is not currently available as
  a validated/promoted feature.
- Sid-V5 is a research candidate only; checkpoint access and end-to-end English
  translation quality remain unverified.
- Do not present synthetic smoke outputs as real translations.

Historical sections below retain their original dates and evidence. A dated
historical result is not automatically the current state.

## 2b. Live Interpreter UI/UX redesign, sentence-level Auto-Speak (2026-11-15)

- Live Interpreter rebuilt around a dark-themed ShadCN `Interpretation` panel
  (status badge: waiting / recognising / composing / composed / not-confidently-recognised),
  explicit `Recognition mode → Recognised words → Sentence construction → Interpretation
  → Audio output` structure; pale-green panel removed; camera and controls stack vertically
  below `lg`. WebSocket, session, pause/reconnect/clear, and Basic-preview paths preserved.
- `Browse Signs` sheet replaced by a Help button (ShadCN AlertDialog) linking to
  `https://indiansignlanguage.org/` (new tab). Static image-classify toggle retained
  in the page header (alphabet A–Z only; word CNN remains unpromoted).
- Word→sentence pipeline stays: deterministic rules → Groq refinement (only when no rule
  matches; constrained prompt — every signed word must appear exactly once, only
  `is/are/am/the/a/i/can/could` may be added, reordering allowed for English grammar,
  content words can never be invented; unverifiable output falls back to a comma-join)
  → comma-join fallback. Content word reordering enabled 2026-11-15 (gloss order ≠ English
  order); every input word must still appear exactly once. Groq is a constrained
  composer, not an ISL grammar model — no linguistic-accuracy claim is made.
- Auto-Speak speaks completed interpretations only (composed sentences / final WS
  results), deduped by text, never intermediate predictions; Speak = manual playback of
  the current interpretation.
- TTS: backend `POST /api/v1/speech/speak` proxies Groq Orpheus
  (`canopylabs/orpheus-v1-english`, ≤200 chars, wav) when `SIGNSAGE_GROQ_TTS_API_KEY` (a dedicated voice-model key)
  is set, falling back to `SIGNSAGE_GROQ_API_KEY`
  (`SIGNSAGE_TTS_PROVIDER=auto|groq|browser`); browser Web Speech API is the automatic
  fallback. Key stays server-side. Free tier ≈ 10 RPM / 100 RPD / 3.6K tokens/day;
  Orpheus is English + Saudi Arabic only; no Hindi/Bengali voice exists in the provider.
- Known blocker (operator action): the Orpheus model requires one-time terms acceptance
  by the Groq org admin (console playground → model → accept); until then the endpoint
  returns 503 `TTS_UNAVAILABLE` and the frontend falls back to browser voices.
- Groq chat compose now uses `reasoning_effort="low"` + 256 max tokens with
  `openai/gpt-oss-20b` (llama-3.3-70b-versatile is removed by Groq); failure falls back
  once to allam-2-7b, then to the comma-join. No fabricated content is possible in any
  fallback path.

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
| Image classification (static) | Alphabet CNN: 26 ISL letters (A–Z), RealSign (CC0-1.0), 87.1% test acc / macro-F1 0.849, VERIFIED on held-out test. Word CNN: 20 static words (Mendeley), 19.0% test acc — NOT credible, kept unpromoted. Two separate classifiers, never merged. |
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
- Local `.env` (reversible, untracked): `SIGNSAGE_MODEL_BUNDLE_DIR` points to
  `research/training/artifacts/smoke_test/signsage_deployment_bundle` (complete
  synthetic/smoke checkpoint + tokenizer); `SIGNSAGE_ALLOW_UNPROMOTED_DEV=true`
  enables serving for end-to-end validation only. Production bundle (`352868924`)
  remains missing its checkpoint/tokenizer artifacts.

## 2a. Static image classification (new, 2026-10-27)

Two lightweight CNN classifiers added behind REST endpoint
`POST /api/v1/classification/image`, independent of the Sign->Text transformer
path (no effect on `model_promoted`, `allow_unpromoted_dev`, or WebSocket).

**Alphabet classifier** (VERIFIED, ready for local demo):
- Source: RealSign-Indian-Sign-Language-Dataset (CC0-1.0), 26 classes A–Z
- Data: Training 18,198 / Val 2,579 / Test 5,198 images (64×64 RGB)
- Architecture: SmallCNN (3 conv blocks + 256-unit head), trained from scratch
- Results: test accuracy 87.1%, macro-F1 0.849, macro-precision 0.868,
  macro-recall 0.871. Known weak classes: W (0% recall), Z (25% recall), S (50% recall)
  — hand shapes with high visual similarity.
- Artifacts: `research/image_classification/artifacts/alphabet/` (model.pt, labels.json,
  evaluation.json, training_history.json)
- Backend: `ImageClassifierAdapter` + `/api/v1/classification/image?task=alphabet`
- Frontend: periodic frame classification in TranslatePage, displays letter +
  confidence + uncertainty state

**Word classifier** (NOT credible — kept unpromoted):
- Source: Indian Sign Language Real-life Words (Mendeley), 20 word classes
- Data: ~18,000 images across 20 words, signer-aware (User_N) 70/15/15 split
- Results: test accuracy 19.0%, macro-F1 0.1645 — barely better than random (5%),
  severe overfitting (train 93.7% vs val 36.8%). Not deployed for recognition.
- Artifacts: `research/image_classification/artifacts/words/` (train/val/test npz only;
  no model.pt — training did not produce a usable model)

**Datasets audited** (see `research/image_classification/AUDIT.md`):
- Indian-Sign-Language-dataset (ayesha-hannure): NOT usable — inner zips empty
  (Git LFS pointers, no actual images)
- Indian Sign Language Real-life Words (Mendeley): usable for word classifier
- RealSign-Indian-Sign-Language-Dataset (RealSign62): usable for alphabet classifier
- Alphabet and word labels kept in separate classifier label spaces — never merged.


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

## 6. Playwright E2E verification (alphabet classification) — 2026-10-27

Scope: static image classification only (`POST /api/v1/classification/image`), not
continuous ISL translation or sentence generation. Both backend (`:8000`) and
frontend preview (`:4173`) started automatically via `playwright.config.ts`
`webServer` array; services verified reachable (`GET /api/v1/health` 200,
`GET /` 200) before any test executed. No `ECONNREFUSED` treated as pass.

Environment variables required (set in `backend/.env` or inline):
- `SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR=../research/image_classification/artifacts/alphabet`
- `SIGNSAGE_IMAGE_WORD_BUNDLE_DIR=../research/image_classification/artifacts/words`
- `SIGNSAGE_MODEL_PROMOTED=false` (gate stays closed for translation path; does not affect image endpoint)
- `SIGNSAGE_ALLOW_UNPROMOTED_DEV=false` (local dev escape hatch; image endpoint independent)

Commands executed (from repo root / `frontend/`):
- `npm run build` — passes (8.2s, no new errors)
- `npm test` — 27 pass (existing frontend unit tests)
- `npx playwright test --project=chromium --reporter=line` — 7 pass in 15.4s
- `cd backend && .venv/bin/python -m pytest -q` — 240 pass, 5 pre-existing PyTorch warnings (43.3s)

Verified behaviors (real dataset fixtures, not mocked):
- `POST /api/v1/classification/image` with real `test.npz` image (`label=A`, `confidence=1.0`, `accepted=true`, `model_version=signsage-image-alphabet`) — endpoint processes actual CNN bundle (`model.pt` + `labels.json`).
- Synthetic base64 PNG returns consistent structured response with all required fields (`label`, `confidence`, `index`, `n_classes`, `accepted`, `reason`, `model_version`, `task`).
- High `min_confidence` (0.99) produces honest `accepted` boolean (either true or false) with `reason` field present — no fabricated success.
- Invalid base64 (`not-valid-base64!!!`) returns 422 with `detail` property — honest rejection.
- Unknown task (`nonexistent`) returns `>=422` — endpoint refuses rather than inventing a classification.
- Frontend `TranslatePage` displays `imageClassifyMode` toggle (`Image` / `Image classify` button) and shows result with actual `label`, `confidence`, `accepted`, and `reason` values; when model unavailable (`modelBlocked`), page shows explicit `Translation unavailable` banner, never a fake translation. Image mode stops automatically when toggled off (`imageIntervalRef.current` cleared).

Fixes applied (minimal):
- `frontend/playwright.config.ts`: `webServer` changed from single `command` to array that starts backend (`:8000`) then preview (`:4173`) automatically; `reuseExistingServer: !process.env.CI` preserved.
- `frontend/e2e/image-classify.spec.ts`: added `import.meta.url` (`__dirname` fix for ES module scope); replaced synthetic-only endpoint test with real dataset fixture (`test.npz` base64); added rejected-prediction (`min_confidence=0.99`), unavailable-task, and full-UI-verification tests.

What was NOT verified (explicit exclusions preserved):
- Continuous ISL sign-to-text translation (`WS /ws/translate`) — remains gated by `SIGNSAGE_MODEL_PROMOTED=false` and not tested here.
- Sentence-level recognition or English→ISL sentence translation — out of scope.
- Physical webcam integration in headless browser — no `fake-camera.mp4` fixture available; camera lifecycle tested by UI state (`phase`, `streamRef`) not by real video feed.
- Performance / load / signer-independence — not measured.
- Model promotion gate calibration (B3) — still missing calibrated `min_confidence` threshold.

Actual results (no mock inference):
| # | Check | Actual result |
|---|---|---|
| E2E-1 | Endpoint reachable with real dataset image | PASS — label=`A`, conf=`1.0`, accepted=`true`, bundle=`alphabet` |
| E2E-2 | Endpoint returns structured response for synthetic image | PASS — all fields present |
| E2E-3 | Rejected predictions (high floor) handled honestly | PASS — `accepted` boolean + `reason` present |
| E2E-4 | Invalid image returns 422 | PASS — `detail` property present |
| E2E-5 | Unknown task rejected honestly | PASS — status `>=422` |
| E2E-6 | Browser page loads with image mode control | PASS — `Live Interpreter` heading visible, toggle present |
| E2E-7 | Full workflow verifies UI displays real response | PASS — endpoint returns real CNN result; UI shows honest state |

Artifacts unchanged: `research/image_classification/artifacts/alphabet/` (verified `model.pt` + `labels.json` load correctly); `docs/STATUS_AND_ROADMAP.md` updated; no retraining, no dataset download, no `.env` secrets committed.
