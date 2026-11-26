# SignSage — Testing

Prioritise **failure detection and research validity** over coverage numbers. A passing software test proves behaviour; it never proves translation quality.

## 1. Layers

| Layer | Proves | Does not prove |
|---|---|---|
| Unit | A function does what it says | Anything about models |
| Contract | Paths, message types, fields, error codes, landmark dimensions and vocab match between backend and frontend | Accuracy |
| Integration | Services, auth, history isolation, capability reporting and error handling work together | Real-camera behaviour |
| Browser E2E (Playwright) | The UI shows real backend responses and honest states | Continuous recognition |
| Model evaluation | Quality on held-out data against a baseline | Software correctness |
| Target-device | Latency and behaviour on the real machine and browser | — |

Passing one layer never substitutes for the next.

## 2. Commands

```bash
# Backend (from backend/, virtualenv active; pytest-asyncio is required)
pytest                                   # whole suite (~292 tests at last count)
pytest tests/unit tests/contract         # fast pre-commit set

# Frontend (from frontend/)
npm test                                 # unit tests (27 at last count)
npx tsc -b && npm run build              # type check + production build
npx playwright test --project=chromium   # E2E; starts backend :8000 and preview :4173

# Model and transport checks (from backend/)
python -m scripts.model_check            # bundle loads and all-zero input is REJECTED
python scripts/ws_diagnostic.py          # gated refusal and serving path on synthetic landmarks
```

Playwright may need system dependencies installed with elevated rights; it reuses running servers unless `CI` is set. An `ECONNREFUSED` is a failure, never a pass.

## 3. What must always be covered

- **Honest gate:** unpromoted → `MODEL_UNAVAILABLE` over WebSocket and 503 `MODEL_NOT_PROMOTED` over REST; `/capabilities` and `/health` agree.
- **Trust boundary:** NaN or ±Inf landmarks, wrong dimensions, oversized messages, over-long sequences are rejected with the documented code.
- **Auth and privacy:** another user's history id returns 404; history requires a token; the WebSocket authenticates in its first message; no raw-video path exists on the live route.
- **Wrong input must reject:** zeros, a static pose, random motion and a 5-frame fragment must not yield accepted text from a promoted model.
- **Vocabulary honesty:** unknown words are dropped, never mapped; glosses without clips show "No clip yet".
- **Composer safety:** every signed word appears exactly once; no content word is invented; unverifiable output falls back to the plain join.
- **Contract parity:** landmark order and dimensions, normalisation, tokenizer ids — pinned by tests that fail when only one side changes.

## 4. Research and model tests

- Sign → Text and isolated words: report accuracy, macro-F1 and per-class recall (isolated) or WER / BLEU / ROUGE-L (continuous), always with a **trivial baseline** and **output diversity**. Inspect failure cases by hand.
- Text → Pose: report normalised MSE and DTW **together with motion energy** — DTW/MSE alone hide near-static output.
- Data runs verify row ↔ pose correspondence, readability, splits (no signer or video leakage), and junk-label filtering.
- An overfit-gate pass, a lower loss, fluent text or a successful load are never success.
- Metrics from different tasks are never compared ([MODELS_AND_DATA](MODELS_AND_DATA.md)).

## 5. Pipelines and notebooks

The FDMSE-ISL extraction notebook is tested without the dataset: a small fake archive and a local range-capable server exercise streaming, archive and local-archive modes, resume, two-machine splitting, a rejected key, network failure, corrupt and truncated files. Use `DRY_RUN=True` to rehearse the full pipeline with a fake detector (output goes to a separate folder). `SignSage_03_LivePath_StressTest_v2.ipynb` replays real saved test-clip landmarks through the serving code under live-like conditions (frame count, dropped frames, landmark noise, missed hands, trimmed or padded windows, wrong aspect) and checks that the serving path reproduces the stored-feature accuracy. The stress test scores every condition with both the legacy and the robust hand handling so the effect of the missed-hand fix is measured on the real model. A dry run proves the machinery only; extraction speed and quality are measured by the notebook's benchmark cell on real clips.

## 6. Known gaps

- A full real-camera browser session on the target machine.
- Load and stress behaviour; any broad-signer claim.
- Live multi-account proof of row-level security (only mocked today).
- Numeric normalisation-equivalence (golden-vector) test; calibrated confidence floor ([ARCHITECTURE §10](ARCHITECTURE.md#10-known-gaps)).
- Latency on the target device.

## 7. Reporting what you ran

State the exact command, its result, and what you did **not** run. Do not claim a test passed unless it was actually run.
