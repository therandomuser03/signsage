# SignSage

**Real-time, bidirectional assistive communication prototype for Indian Sign Language (ISL) ↔ English.**

> **Project status:** Research prototype / M.Tech thesis project. The software pipeline is substantially implemented, but the current research Sign→Text checkpoints are **not production-quality translation models** and must not be presented as such.

## 1. What SignSage is

SignSage explores a single-device, bidirectional communication workflow:

```text
ISL signer
  ↓
Browser camera
  ↓
MediaPipe Holistic
  ↓
Landmarks only
  ↓
WebSocket
  ↓
FastAPI
  ↓
Sign → Text inference
  ↓
English output
```

and:

```text
English text
  ↓
Text → ISL gloss
  ↓
Text → Pose / sign visualization
  ↓
Browser rendering
```

The project is designed as a research-grade communication instrument rather than a production accessibility service.

## 2. Current implementation

The current codebase/documentation describes:

- React + TypeScript + Vite frontend
- FastAPI backend
- Browser-side MediaPipe Holistic
- WebSocket translation transport
- REST diagnostics/capability endpoints
- Supabase authentication and translation-history storage
- English → gloss processing with Groq/Ollama/deterministic fallback
- Text → pose/skeleton rendering infrastructure
- Explicit model-capability reporting
- Tests and contract checks

### Important model status

The application infrastructure can execute the Sign→Text pipeline, but the verified research checkpoints show severe output collapse (see `research/evaluation/reports/352868924/evaluation_results.json` and `docs/MODEL_READINESS.md`; the historical path `research/outputs/phase0/RESULTS.md` cited by older documentation is TBD/missing in this checkout). Therefore:

- **Pipeline availability ≠ translation quality**
- **Passing tests ≠ model correctness**
- **A successful WebSocket round-trip ≠ useful ISL translation**
- A model must be independently evaluated before being promoted as a production/default checkpoint.

## 3. Research status

Two Kaggle research runs were completed.

The larger documented run used:

- 10,000 training examples
- 1,500 validation examples
- 1,500 test examples

The recorded results show severe Sign→Text output collapse and a Text→Pose motion-collapse problem. See:

- `research/evaluation/reports/352868924/evaluation_results.json` (verified NO-GO evidence; the historical `research/outputs/phase0/RESULTS.md` path is TBD/missing)
- `docs/PHASES.md`
- `docs/TESTING.md`

## 4. Privacy model

The intended application path is landmark-only:

```text
Camera frame → MediaPipe in browser → numeric landmarks → backend
```

Raw camera frames are not uploaded by the translation protocol. The backend receives numerical landmark data rather than a video stream.

See `docs/SECURITY.md` for the full privacy, licensing, secret-management, and safety boundary.

## 5. Documentation map

| Document | Purpose |
|---|---|
| `docs/MASTER.md` | Project orientation and source-of-truth map |
| `docs/STATUS.md` | Current implementation/research status |
| `docs/PRD.md` | Product/research scope and requirements |
| `docs/ARCHITECTURE.md` | Runtime and research architecture |
| `docs/API.md` | HTTP/WebSocket communication contract |
| `docs/DATA_MODEL.md` | Research/application data boundaries |
| `docs/DESIGN.md` | UI/UX and visual design specification |
| `docs/SECURITY.md` | Privacy, licensing, secrets, safety |
| `docs/TESTING.md` | Software verification and model evaluation |
| `docs/PHASES.md` | Historical and remaining development phases |
| `docs/DECISIONS.md` | Technical decision record |

## 6. Claims policy

Use these terms precisely:

- **Implemented** — code exists and has been verified by the stated check.
- **Integrated** — components communicate through the documented contract.
- **Research result** — measured in an experiment.
- **Experimental** — technically available but not sufficiently validated.
- **Deferred** — intentionally not implemented.
- **Planned** — intended future work.
- **Production-ready** — do not use unless there is evidence supporting that claim.

Never turn a research metric into a product-quality claim.

## 7. Thesis framing

SignSage should be presented as an M.Tech CSE / AI & Data Science research prototype whose contribution includes:

1. A continuous ISL↔English system architecture.
2. Landmark-based privacy-preserving browser inference input.
3. A reproducible research pipeline around iSign.
4. Explicit evaluation and failure analysis.
5. A bidirectional application layer connecting research outputs to a usable interface.

The project does **not** claim certified human-interpreter replacement or high-stakes deployment.
