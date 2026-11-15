# SignSage — Master Project Guide

**Status:** Current project orientation  
**Last reviewed:** 2026-11-02  
**Production model gate:** `SIGNSAGE_MODEL_PROMOTED=false`

Start here for project scope and decisions. For quick navigation, use
[docs/README.md](README.md).

| Document | Purpose |
|---|---|
| `MASTER.md` (this file) | Product scope, architecture decisions, requirements, constraints |
| `STATUS_AND_ROADMAP.md` | Detailed status, dated verification, blockers, next work |
| `ml/MODEL_STATUS.md` | Current model decisions and promotion gates |
| `ml/DATASET_CATALOG.md` | Dataset summary and links to detailed audits |
| `architecture/SYSTEM_ARCHITECTURE.md` | High-level system structure |
| `architecture/BACKEND_ARCHITECTURE.md` | Backend layers and current blockers |
| `architecture/API_REFERENCE.md` | REST and WebSocket contract summary |
| `architecture/end-to-end-contract.md` | Pinned landmark, preprocessing, and transport contract |
| `development/DEVELOPMENT_SETUP.md` | Quick local setup |
| `development/DEVELOPMENT.md` | Development workflow, interfaces, security, testing |
| `development/RENDER_MODEL_DEPLOYMENT.md` | Render model-bundle deployment procedure |
| `product/PRODUCT_OVERVIEW.md` | Product scope and verified capabilities |
| `product/FEATURES_AND_REQUIREMENTS.md` | Feature-level evidence and requirements |
| `product/PRIVACY_AND_LIMITATIONS.md` | Privacy commitments and limitations |
| `product/DEMONSTRATION_GUIDE.md` | Evidence-based live-demo sequence |
| `AGENTS.md` | Instructions for AI coding agents |
| `reports/` | Historical audits and research evidence; check dates and supersession notes |

## 1. One-sentence definition

SignSage is an M.Tech research prototype for bidirectional ISL ↔ English
communication using browser-side landmark extraction, a FastAPI/WebSocket
backend, Sign→Text inference, and English→ISL gloss/pose generation.

## 2. Reality check

### Implemented / integrated

FastAPI backend, WebSocket translation contract, diagnostics REST, browser
MediaPipe Holistic integration, landmark transport, auth integration,
history API/isolation, English→gloss endpoint, text→pose/skeleton rendering
infrastructure, model capability reporting, automated verification.

### Experimental

Research Sign→Text checkpoint, Text→Pose generator, general continuous ISL
translation quality, generated-pose fidelity, broad signer generalization.

### Deferred / out of scope

Two-device relay, production mobile, photorealistic avatar, open-vocabulary and
dialect coverage, large-scale collection, high-stakes deployment, certified
human-interpreter replacement.

## 3. Most important research finding

Phase 0 establishes **no useful Sign→Text model**: trivial baseline WER ≈ 0.999,
GRU ≈ 0.991, Transformer ≈ 1.367 with ~21% output diversity in the early
snapshot; later filtering reversed the old GO-CANDIDATE reading — that label
must never be treated as readiness. Text→Pose motion energy ≈ 6.7% of ground
truth. The external T5 candidate scored 0/1887 exact match (see
`EXTERNAL_EVALUATION.md`). Live translation is therefore gated off by default.

## 4. How it works

```text
Browser: camera → MediaPipe Holistic → 543 landmarks/frame → sentence boundary
  → WebSocket → Backend: buffer → Sign→Text → acceptance gate → text or honest refusal
Browser: text → REST gloss → pre-authored avatar clips (unknown words dropped, never guessed)
```

The browser owns camera, extraction, boundary interaction, UI, and pose
rendering; the backend owns orchestration, inference, auth, history, status,
and optional LLM gloss. Raw video never leaves the browser. Training
(iSign → Kaggle notebook → preprocessing/training → evaluation/provenance →
checkpoint review → promotion decision → inference) is separate from serving;
the app never retrains. A checkpoint can be trained → saved → loadable →
runnable without being validated → useful → generalizable → deployable —
SignSage has the first group, not the second. Risks on record: landmark
distribution skew (studio training vs browser capture), sentence segmentation,
model collapse, free-hosting limits.

## 5. Product scope & requirements

Research objectives: spatial-temporal ISL→English modeling, latency
measurement, bidirectional workflow, English→ISL gloss/pose investigation.
In scope: responsive web UI, shared-device loop, continuous ISL in, English
out/in, gloss + pose/skeleton visualization, auth, history, capability
reporting, landmark-only transport, explicit uncertainty surfaces. Out of scope:
§2 list. Requirements: both directions (R1), landmark-only protocol (R2),
transparent availability/failure states (R3), account-scoped auth/history
(R4–R5), metrics separated from product claims (R6). Thesis success =
reproducible procedures + integrated app + measured results + failure analysis
+ defensible validation story; a mechanical demo proves nothing about quality.
Current caveat: Phase-0 models show severe collapse — the app must expose
status, never imply solved translation.

## 6. Product design (condensed)

Research-grade assistive instrument that looks serious without hiding
experimental state. Turn-taking obvious; communication result primary,
telemetry secondary (large readable captions, persistent output, quiet
status). Fail loudly on: camera denial, poor tracking, hand out of frame,
WS failure, unavailable model, uncertain result — never a fake translation.
Visual system: Space Grotesk headlines, Inter body, JetBrains Mono data;
spacing 4/8/16/24/32; 12-column desktop workbench; dark technical palette
(`#090E13/#101720/#253342/#151E28`), no diffuse glows. Mobile = single stream
(turn → camera → output → status → controls → telemetry). Pose output is a
research representation, never styled as a validated avatar. Landing page must
cover what it is, how it works, what is implemented, what research showed,
privacy, limitations, and app entry — never advertise unsupported accuracy.

## 7. Key decisions

| Decision | Rationale | Status |
|---|---|---|
| FastAPI | Native async/WebSocket + streaming | Adopted |
| React + TypeScript + Vite | Existing stack, responsive app | Adopted |
| Single-device MVP | No relay complexity before core loop works | Adopted |
| WebSocket translation loop | Streaming suits landmarks better than per-frame HTTP | Adopted |
| Browser-side MediaPipe | Raw video stays local; landmarks only on the wire | Adopted |
| GRU baseline, then plain Transformer | Simple baseline first; no graph complexity without evidence | Adopted |
| 543 image-space landmarks | Pose + face + hands, minus duplicated world block | Adopted |
| `video_id` split | Blocks same-video segment leakage | Adopted |
| Groq → Ollama → deterministic gloss fallback | Graceful degradation | Adopted |
| Optional backend TTS (Groq Orpheus) → browser Web Speech fallback | Natural voice without shipping keys to the browser; never hard-fails | Adopted (2026-11-15) |
| LLM-constrained sentence composer (rules → guarded Groq → comma-join) | Signed words can never be invented; degradation is deterministic | Adopted |
| Notebook separate from serving | Different lifecycles/reproducibility needs | Adopted |
| No raw-camera upload / relay / photoreal avatar / high-stakes use | Privacy-by-construction; deferred/out of scope | Adopted / Deferred / Out of scope |

Promotion rule: a checkpoint is promoted only on a documented evaluation with
baselines and failure checks — never for merely training, saving, loading,
running, improving one metric, or passing software tests. Text→Pose claims need
motion energy alongside DTW/MSE. Record architectural changes here.

## 8. History & phases

| Run | Train / Val / Test | Outcome |
|---|---|---|
| Bounded | 2,000 / 300 / 300 | Pipeline worked; no useful model |
| Scaled | 10,000 / 1,500 / 1,500 | Severe mode collapse; GO-CANDIDATE label retracted |

Phase 0 research (caveats above) → 0.5 extraction to backend serving → 1 app
integration (backend, WS, MediaPipe, diagnostics, auth/history) → 2 EN→ISL
representation (gloss chain + pose infra; quality experimental) → 3 integrated
app (round-trips work; ≠ translation) → 4 validation, remaining: fix
decoding/diversity, re-evaluate, check live-vs-training compatibility,
signer-independence if metadata permits, target-hardware latency, motion-aware
pose reassessment, and only then promotion. Deferred: §2 list.

## 9. Repository layout

```text
backend/    FastAPI + PyTorch inference (app/, scripts/, tests/, models/README)
frontend/   Vite + React UI (app/, features/, services/, components/)
research/   Training notebooks + Phase-0 record + eval artifacts (never serving code)
models/     External-candidate weight mirror (not SignSage-trained)
docs/       This map: 5 authorities + AGENTS + reference/ pins
tools/      Read-only inventory script
render.yaml Render blueprint (bundle via fetch_bundle.sh at start)
```

Ignored, never committed: `results.zip`, `staging_kaggle_results/`,
`research/artifacts/` (empty bundle skeleton — operator supplies a bundle),
`backend/.env`, `frontend/.env.local`. Entry points: `uvicorn app.main:app`,
`npm run dev`/`build`, `research/training/train_sign_to_text.py` via the
`signsage_kaggle.ipynb` runbook (see `research/training/READINESS.md`).

## 10. Documentation authority

When documents conflict: (1) code + tests describe implementation;
(2) `architecture/end-to-end-contract.md` pins values; (3) `development/DEVELOPMENT.md`
describes intended interfaces; (4) `STATUS_AND_ROADMAP.md` describes current
state; (5) recorded evaluations describe measured outcomes; (6) §7 explains
why; (7) §5 describes intended scope, not implementation. Historical phase
notes never override current implementation evidence.

## 11. Golden rule & claims policy

Do not document what the project should do as though it already does it.
`can execute ≠ produces output ≠ produces useful output ≠ generalizes ≠
is production-ready`. Use terms precisely: **Implemented** (code + check),
**Integrated** (talks through the contract), **Research result** (measured),
**Experimental** (available, unvalidated), **Deferred / Planned** (as named),
**Production-ready** (never without evidence). Never turn a metric into a
product-quality claim.

## 12. Thesis framing

An M.Tech research prototype contributing: continuous ISL↔English architecture;
privacy-preserving landmark-based browser input; reproducible iSign pipeline;
explicit evaluation and failure analysis; a bidirectional application layer.
Not a certified human-interpreter replacement or high-stakes tool.
