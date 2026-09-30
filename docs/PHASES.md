# SignSage — Development & Research Phases

**Status:** Historical roadmap + current completion record

## Phase 0 — Research notebook

**Status: completed with major caveats.**

Two Kaggle runs were completed:

| Run | Train | Validation | Test |
|---|---:|---:|---:|
| Bounded | 2,000 | 300 | 300 |
| Scaled | 10,000 | 1,500 | 1,500 |

The pipeline successfully executed training/evaluation/provenance workflows.

However, the research results do not establish a useful Sign→Text model.

## Phase 0 findings

The documented experiments found:

- severe Sign→Text mode collapse,
- repetitive/generic output,
- weak general translation behavior,
- Text→Pose motion collapse,
- a need for motion-aware evaluation beyond DTW/MSE.

The historical `GO-CANDIDATE` label in an earlier notebook section was deliberately narrow and was later undermined by evaluation fixes. It must not be used as a synonym for model readiness.

## Phase 0.5 — Research-to-software extraction

**Status: implemented.**

Research modules were extracted into backend-serving infrastructure rather than leaving serving code embedded in the notebook.

The notebook remains a research artifact; the application owns serving.

## Phase 1 — Application integration

**Status: substantially implemented.**

Includes:

- FastAPI backend,
- WebSocket transport,
- browser MediaPipe input,
- frontend translation workflow,
- diagnostics/capability APIs,
- authentication/history integration.

## Phase 2 — English → ISL representation

**Status: implemented as application infrastructure; model quality remains experimental.**

Includes:

- text input,
- gloss generation,
- fallback chain,
- pose/skeleton rendering infrastructure.

Real sign animation assets remain a separate content/asset requirement if a richer avatar experience is desired.

## Phase 3 — Integrated application

**Status: application infrastructure exists; research model quality remains the primary unresolved issue.**

The full software path can be wired end-to-end, but a successful software round-trip must not be described as successful translation.

## Phase 4 — Validation

**Status: remaining work.**

Priority areas:

1. Improve Sign→Text decoding/model behavior.
2. Re-evaluate output diversity and WER.
3. Validate live landmark/training distribution compatibility.
4. Establish signer-independent evaluation if metadata permits.
5. Measure end-to-end latency on defined hardware.
6. Reassess Text→Pose with motion-aware criteria.
7. Only then consider checkpoint promotion.

## Deferred phases

- two-device relay,
- large-scale user data collection,
- production mobile deployment,
- high-stakes use,
- broad dialect coverage,
- photorealistic avatar.
