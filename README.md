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

**Next:** Phase 0.5 (extract reusable inference code from the notebook) → Phase 1 (backend) → Phase 2 (frontend). Per [`docs/PHASES.md`](docs/PHASES.md)'s own timeline math, Phases 1–3 are scoped and designed, not built — `backend/` and `frontend/` below are intentionally empty.

---

## How it works

**Training (offline, Kaggle T4×2):** HuggingFace `Exploration-Lab/iSign` (gated, CC BY-NC-SA 4.0) → `iSign_v1.1.csv` (9.81 MB, downloaded) + the pose archive (measured ~158.5 GiB across 4 parts, streamed via a virtual ZIP64 over HTTP range requests, never downloaded or extracted) → GRU → Transformer → Text-to-Pose → `research/outputs/phase0/` checkpoints + `signsage_knowledge.db`.

**Inference (single device, planned — not built yet):**
```
Camera → MediaPipe Holistic (WASM, 543 landmarks, on-device) → motion-pause+Send → WebSocket → FastAPI → Sign→Text → captions
Text/speech → LLM (Groq→heuristic fallback) → ISL gloss → Text-to-Pose → pose → canvas skeleton
```
Raw video never leaves the device. Two-device relay is deliberately deferred — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```text
SignSage/
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
├── backend/                     # FastAPI + PyTorch — NOT YET BUILT (Phase 1)
├── frontend/                    # Next.js — NOT YET BUILT (Phase 2)
│
├── docs/                        # specs and decisions
│   ├── PRD.md · ARCHITECTURE.md · API.md · DATA_MODEL.md · DESIGN.md
│   ├── SECURITY.md · PHASES.md · TESTING.md · DECISIONS.md · AGENTS.md
│
├── LICENSE
├── .gitignore
└── .env.example
```

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

Backend / frontend: not implemented yet. See `docs/PHASES.md` for the sequencing rule (`docs/AGENTS.md` calls it "the one rule that matters most") — they are built from modules extracted out of the notebook during Phase 0.5, not scaffolded ahead of a validated model.

---

## Privacy & security

Landmarks-only design: raw video never leaves the device (planned, once Phase 1 exists); no biometric storage; secrets via environment variables only; not a substitute for certified human interpreters in high-stakes contexts. Full detail: [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Documentation index

| Doc | Read when |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | what this is/isn't, success criteria |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | training vs. MVP architecture, deferred features |
| [`docs/API.md`](docs/API.md) | planned WebSocket contract |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | dataset facts, knowledge-repo schema |
| [`docs/DESIGN.md`](docs/DESIGN.md) | interaction principles, turn-taking, fail-loudly |
| [`docs/SECURITY.md`](docs/SECURITY.md) | privacy, secrets, safety boundaries |
| [`docs/PHASES.md`](docs/PHASES.md) | roadmap, exit criteria, honest timeline math |
| [`docs/TESTING.md`](docs/TESTING.md) | evaluation strategy, preflight checks |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | why each technical choice was made |
| [`docs/AGENTS.md`](docs/AGENTS.md) | rules for anyone (human or AI) working on this repo |

---

## Boundaries (explicitly out of scope)

Unrestricted open-vocabulary ISL, regional dialect coverage, photorealistic avatars, production-grade mobile deployment, high-stakes translation use, remote/two-device relay, large-scale new volunteer data collection. See `docs/PRD.md`.

---

## Thesis

**SignSage** — M.Tech CSE (AI & Data Science), Anubhab Pal
Guide: Mr. Ponsuresh Manoharan · L&T EduTech · August 2026

Ground-up redesign of an earlier prototype (single-frame MediaPipe Hands + Random Forest, 19-word vocabulary, Flask + MJPEG streaming).
