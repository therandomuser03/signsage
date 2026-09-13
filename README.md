# SignSage

**Real-time, bidirectional assistive communication between Indian Sign Language (ISL) and English.**

An **M.Tech thesis research prototype** — continuous, sentence-level ISL ↔ English communication via spatial-temporal modeling, a non-manual-marker (face) ablation, and a Text-to-Pose pipeline. Single shared device, research/thesis evaluation scope.

> **Not a production accessibility product.** Not intended for medical, legal, emergency, or other high-stakes use — see [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Status (as of the Phase 0 scaled-up run, 2026-09-03)

**Phase 0 — bounded Kaggle validation — complete.** Full findings: [`research/outputs/phase0/RESULTS.md`](research/outputs/phase0/RESULTS.md). Read that before quoting any number from this project — the honest summary is:

- The pipeline (remote dataset streaming, training, evaluation, experiment provenance) runs correctly and reproducibly end-to-end on real iSign data, at two different data scales.
- Sign→Text: the GRU baseline has fully mode-collapsed to a single repeated output. The Transformer shows a small, real improvement in output diversity with more data, but its **WER is currently worse than a trivial baseline** due to a decoding-side repetition problem (greedy decoding, no repetition penalty) — not something more data alone fixes.
- Text→Pose: the generative model's favorable DTW/MSE numbers are misleading — a motion-energy diagnostic confirms it has collapsed to a **near-static average pose** (6.7% of ground-truth motion), a known failure mode of position+velocity MSE loss without an adversarial/perceptual term.
- The non-manual-marker (face) ablation replicated across both runs: removing face landmarks trains to a *lower* validation loss than keeping them, at this data scale — worth real discussion, not a bare number.
- A real, previously undocumented data-quality issue was found: roughly 1% of the dataset is junk/placeholder text (single digits, single letters, caption artifacts like `[Music]`), which contaminates any frequency-based baseline.

None of this blocks the thesis — it's exactly the kind of finding a bounded validation phase is supposed to produce. It does mean: don't present current checkpoints as demonstrating working translation or working pose generation without these caveats attached.

**Backend now built** (Phase 0.5 module extraction + the backend half of Phase 1 — see
`backend/README.md`); **frontend has a static Vite + React UI shell**, but no live
camera, MediaPipe, WebSocket, or pose rendering integration. See
[`docs/PHASES.md`](docs/PHASES.md) for remaining Phase 1/2 work.

---

## How it works

**Training (offline, Kaggle T4×2):** HuggingFace `Exploration-Lab/iSign` (gated, CC BY-NC-SA 4.0) → `iSign_v1.1.csv` (9.81 MB, downloaded) + the pose archive (measured ~158.5 GiB across 4 parts, streamed via a virtual ZIP64 over HTTP range requests, never downloaded or extracted) → GRU → Transformer → Text-to-Pose → `research/outputs/phase0/` checkpoints + `signsage_knowledge.db`.

**Inference (single device):** the WebSocket contract below is implemented and serving
the real Phase 0 checkpoints (`backend/`); the client-side half is not built yet.
```
Camera → MediaPipe Holistic (WASM, 543 landmarks, on-device) → motion-pause+Send → WebSocket → FastAPI → Sign→Text → captions
Text/speech → LLM (Groq→heuristic fallback) → ISL gloss → Text-to-Pose → pose → canvas skeleton
```
Raw video never leaves the device. Two-device relay is deliberately deferred — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```text
SignSage/
├── AGENTS.md                    # read by OpenCode / Zed / Google Antigravity automatically
├── CLAUDE.md                    # thin @AGENTS.md import, so Claude Code reads it too
│
├── research/                    # M.Tech research / training — the source of truth right now
│   ├── notebooks/
│   │   └── signsage.ipynb       # training/evaluation reference, run on Kaggle GPU
│   ├── outputs/
│   │   └── phase0/              # complete Kaggle experiment record
│   │       ├── checkpoints/     # 4 .pt files: GRU, Transformer (full/no-face), Text-to-Pose
│   │       ├── reports/
│   │       │   └── token_vocabulary.json
│   │       ├── signsage_knowledge.db          # queryable experiment/metrics record
│   │       ├── signsage_knowledge_manifest.json
│   │       └── RESULTS.md       # human-written findings — read this first
│   └── data/
│       └── README.md            # dataset access instructions only — raw data is gitignored
│
├── backend/                     # FastAPI + PyTorch — BUILT (Phase 0.5 + backend half of Phase 1)
│   ├── AGENTS.md                 # nested, backend-scoped rules (read in addition to root AGENTS.md)
│   ├── app/ · tests/ · scripts/ · weights/ · vocab/
│   ├── docs/API.md · MODEL_CARD.md · README.md
│   └── ...
├── frontend/                    # Vite + React static UI shell — no live translation yet
│
├── thesis/                      # human academic deliverables — not agent-facing, not referenced by AGENTS.md
│   ├── SignSage.pptx             # college/committee presentation
│   └── paper/                    # written report/research paper and its source material
│       ├── draft.md               # (or .docx/.tex, whatever you draft in)
│       ├── figures/               # curated, finalized charts pulled FROM research/outputs — not raw dumps
│       └── references.bib
│
├── docs/                        # specs and decisions (reference material, not auto-loaded)
│   ├── PRD.md · ARCHITECTURE.md · API.md · DATA_MODEL.md · DESIGN.md
│   └── SECURITY.md · PHASES.md · TESTING.md · DECISIONS.md
│
├── LICENSE
├── .gitignore
└── .env.example
```

**Why a separate `thesis/` instead of putting the pptx in `docs/` or `research/`:**
`docs/` is the agent-facing technical reference set (specs, contracts, decisions) —
a slide deck and a paper draft are a different audience (guide/committee) and a
different kind of content (narrative, not spec), and mixing them muddies both.
`research/outputs/phase0/` is explicitly the complete, unedited experiment record (see
below) — a paper's `figures/` are curated, finalized exports pulled *from* that record,
not part of it. `thesis/` keeps "outputs for a grade" separate from "outputs for the
pipeline" (`research/`) and "instructions for agents" (`AGENTS.md`, `docs/`).

**Why `AGENTS.md`/`CLAUDE.md` sit at the repo root instead of in `docs/` with everything
else:** OpenCode, Zed, and Google Antigravity all auto-discover `AGENTS.md` at the project
root — none of them look inside `docs/`. Claude Code reads `CLAUDE.md` specifically, not
`AGENTS.md`, so the root `CLAUDE.md` is a one-line `@AGENTS.md` import rather than a
second copy of the same content. `backend/AGENTS.md` is a nested, backend-scoped file
under the same convention (closest `AGENTS.md` to the file being edited applies, in
addition to the root one, not instead of it). Everything else in `docs/` is reference
material an agent is pointed to from `AGENTS.md`, not something any tool auto-loads.

**Why research/outputs vs. a future backend/weights:** `research/outputs/phase0/` is the complete, unedited experiment record. A future `backend/weights/` (once Phase 1 starts) will hold only the specific artifacts selected for serving — never the reverse; the research output is never overwritten by the deployment selection.

---

## Dataset

- **Primary:** [iSign](https://huggingface.co/datasets/Exploration-Lab/iSign) (Joshi et al., ACL Findings 2024) — 127,237 CSV rows / `.pose` files locally confirmed (the paper describes 118,228; this repo uses the local counts actually present, see `docs/DATA_MODEL.md`). License: **CC BY-NC-SA 4.0** (non-commercial, share-alike) — gated access, requires an accepted-terms HF account + `HF_TOKEN`.
- **Not stored in this repository** — neither the pose archive nor the CSV. Committing the CSV would violate the dataset's redistribution terms; see `research/data/README.md` for access instructions instead.
- **Secondary:** INCLUDE (263 isolated signs) — isolated-sign baseline reference only, not used for training.

---

## Evaluation

- **Sign→Text:** WER, BLEU, ROUGE-L vs. a (junk-text-filtered) most-common-sentence baseline; output-diversity check for mode collapse.
- **Text→Pose:** DTW + MSE vs. TF-IDF retrieval; **motion-energy check** vs. ground truth (catches static-pose collapse that DTW/MSE alone would miss).
- **System:** model-only latency (not end-to-end — that requires Phase 1's backend to exist).
- Every run writes full provenance to `signsage_knowledge.db` + a JSON manifest — see `docs/DATA_MODEL.md` for schema.

Full current numbers and their caveats: [`research/outputs/phase0/RESULTS.md`](research/outputs/phase0/RESULTS.md).

---

## Development

Research (this is what currently works):
```bash
# open research/notebooks/signsage.ipynb on Kaggle
# GPU T4 accelerator, Internet ON, HF_TOKEN secret attached
# run cells 0-8 first and confirm Device: cuda before running the rest
```

Backend: implemented — see `backend/README.md` for setup/run instructions. Frontend:
Vite + React static UI shell exists, but webcam capture, MediaPipe extraction, backend
WebSocket integration, and pose rendering are not yet implemented. See `docs/PHASES.md`
for remaining Phase 1/2 work.

---

## Privacy & security

Landmarks-only design: raw video never leaves the device — the backend has no camera/video upload endpoint by construction, not just by plan; no biometric storage; secrets via environment variables only; not a substitute for certified human interpreters in high-stakes contexts. Full detail: [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Documentation index

| Doc | Read when |
|---|---|
| [`AGENTS.md`](AGENTS.md) | rules for anyone (human or AI) working on this repo — read automatically by OpenCode, Zed, Google Antigravity |
| [`CLAUDE.md`](CLAUDE.md) | one-line `@AGENTS.md` import so Claude Code reads the same rules |
| [`backend/AGENTS.md`](backend/AGENTS.md) | additional rules scoped to `backend/` only |
| [`docs/PRD.md`](docs/PRD.md) | what this is/isn't, success criteria |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | training vs. MVP architecture, deferred features |
| [`docs/API.md`](docs/API.md) | the implemented WebSocket contract |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | dataset facts, knowledge-repo schema |
| [`docs/DESIGN.md`](docs/DESIGN.md) | interaction principles, turn-taking, fail-loudly |
| [`docs/SECURITY.md`](docs/SECURITY.md) | privacy, secrets, safety boundaries |
| [`docs/PHASES.md`](docs/PHASES.md) | roadmap, exit criteria, honest timeline math |
| [`docs/TESTING.md`](docs/TESTING.md) | evaluation strategy, preflight checks |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | why each technical choice was made |

---

## Boundaries (explicitly out of scope)

Unrestricted open-vocabulary ISL, regional dialect coverage, photorealistic avatars, production-grade mobile deployment, high-stakes translation use, remote/two-device relay, large-scale new volunteer data collection. See `docs/PRD.md`.

---

## Thesis

**SignSage** — M.Tech CSE (AI & Data Science), Anubhab Pal
Guide: Mr. Ponsuresh Manoharan · L&T EduTech · August 2026

Presentation deck and written report/paper live in [`thesis/`](thesis/) — kept separate
from `docs/` (agent-facing technical reference) and `research/outputs/` (the unedited
experiment record); see "Repository layout" above for why.

Ground-up redesign of an earlier prototype (single-frame MediaPipe Hands + Random Forest, 19-word vocabulary, Flask + MJPEG streaming).
