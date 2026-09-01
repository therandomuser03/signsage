# SignSage

**Real-time, bidirectional assistive communication for Indian Sign Language (ISL) ↔ English** — M.Tech thesis prototype (single shared device), not a production accessibility product.

> ~63 million people in India have significant auditory impairment (Sharma et al., IEEE ICCCIS 2022). Most systems do isolated sign recognition or one-way sign→text. SignSage targets **continuous, sentence-level, bidirectional** translation with non-manual markers — the gap literature hasn't closed.

## How it works

**Training (offline, Kaggle T4):** HuggingFace iSign (gated) → `iSign_v1.1.csv` (9.81 MB) + virtual ZIP64 over HTTP range requests (4 parts, ~231 GB, never extracted) → GRU → Transformer → Text-to-Pose → `research/outputs/phase0/` checkpoints + `signsage_knowledge.db`.

**Inference (single device):**
```
Camera → MediaPipe Holistic (WASM, 543 landmarks, on-device) → motion-pause+Send → WebSocket → FastAPI → Sign→Text → captions
Text/speech → LLM (Groq→heuristic fallback) → ISL gloss → Text-to-Pose → pose → canvas skeleton
```
Raw video **never leaves device**. Two-device relay deferred (`docs/ARCHITECTURE.md`).

## Repo layout (4 areas — compact)

```
SignSage/
│
├── frontend/                    # Next.js web application — currently empty, to be installed
│   ├── app/                     # App Router: pages, layouts, routes
│   ├── components/              # UI components
│   ├── lib/                     # API, WebSocket, MediaPipe utilities
│   ├── public/                  # Static assets / sign animations
│   ├── types/                   # TypeScript types
│   ├── package.json
│   └── ...
│
├── backend/                     # FastAPI + PyTorch
│   ├── main.py                  # FastAPI entry point
│   ├── inference.py             # Sign→Text / Text→Pose inference
│   ├── preprocessing.py         # Landmark preprocessing
│   ├── models.py                # Model architecture/loading
│   ├── weights/                 # DEPLOYMENT model artifacts only
│   │   ├── sign_to_text.pt
│   │   ├── text_to_pose.pt
│   │   └── token_vocabulary.json
│   ├── requirements.txt
│   └── .env.example
│
├── research/                    # M.Tech research / training
│   ├── notebooks/
│   │   └── signsage.ipynb
│   │
│   ├── outputs/                 # Complete notebook/Kaggle outputs
│   │   └── phase0/
│   │       ├── checkpoints/
│   │       ├── reports/
│   │       ├── signsage_knowledge.db
│   │       └── signsage_knowledge_manifest.json
│   │
│   └── data/                    # Dataset references only
│       └── README.md
│
├── docs/                        # Project documentation
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DATA_MODEL.md
│   ├── DESIGN.md
│   ├── SECURITY.md
│   ├── PHASES.md
│   ├── TESTING.md
│   ├── DECISIONS.md
│   └── AGENTS.md
│
├── .gitignore
├── .env.example
└── README.md
```

> `frontend/` is intentionally empty — install Next.js there (`npx create-next-app@latest frontend --typescript --app --eslint`). `research/outputs/phase0/` holds the 61M Kaggle research record; `backend/weights/` holds only the selected deployment artifacts.

Flow:
```
research/notebooks/signsage.ipynb → research/outputs/phase0/ (4 checkpoints, reports, knowledge.db)
                                    │ select validated
                                    ▼
                              backend/weights/ (sign_to_text.pt, text_to_pose.pt, vocab)
                                    │ FastAPI
                                    ▼
                                frontend/ (Next.js — to be installed)
```

## Quick start

```bash
# 1. Notebook (Kaggle GPU, Internet ON, HF_TOKEN secret) — run §0–8 first, confirm Device: cuda
pip install -r backend/requirements.txt
# open research/notebooks/signsage.ipynb

# 2. Backend (after Phase 0 GO)
uvicorn backend.main:app --reload

# 3. Frontend — currently empty, install framework first
# npx create-next-app@latest frontend --typescript --app --eslint  (or your setup)
# cd frontend && npm run dev
```

## Dataset

- **Primary:** [iSign](https://huggingface.co/datasets/Exploration-Lab/iSign) — 118k pairs, 127,237 `.pose` files `(frames, people, keypoints, xyz)` (Joshi et al., ACL 2024)
- **Secondary:** INCLUDE — 263 isolated signs
- Gated HF → needs `HF_TOKEN` + accepting terms. No volunteer collection this phase.

## Evaluation

- **Sign→Text:** WER · BLEU · ROUGE-L vs. "most common sentence" baseline
- **Text→Pose:** MSE + DTW vs. TF-IDF retrieval (capped 3000)
- **System:** end-to-end latency (SMART obj. 2)
- Every run writes `signsage_knowledge.db` + manifest — provenance, not claims.

## Status (honest)

**Phase 0:** bounded Kaggle run (2000/300/300, 5–6 eps) completed under `research/outputs/phase0/`; Section 17 `GO/NO-GO` pending audit. Phases 1–3 scoped, not built until GO (`docs/PHASES.md` — the one rule). Notebook statically validated (61 cells), not yet at thesis scale.

## Tech stack

Next.js + MediaPipe Holistic (WASM) · FastAPI + WebSocket · Transformer (GRU baseline first) · Groq `openai/gpt-oss-20b`/Ollama · SQLite · Kaggle T4 fp16. See `docs/DECISIONS.md` for why.

## Docs

| Doc | Read when |
|---|---|
| `docs/PRD.md` | what this is/isn't |
| `docs/ARCHITECTURE.md` | training vs. MVP, deferred features |
| `docs/API.md` | WebSocket contract |
| `docs/PHASES.md` | where we are, exit criteria |
| `docs/TESTING.md` | metrics + preflight |
| `docs/DECISIONS.md` | why each choice |
| `docs/DESIGN.md` | turn-taking, fail-loudly |

---
Thesis: Anubhab Pal (M.Tech CSE) · Guide: Mr. Ponsuresh Manoharan · L&T EduTech · Aug 2026. Prior: single-frame MediaPipe Hands + RF, 19 words, Flask+MJPEG — ground-up redesign.
