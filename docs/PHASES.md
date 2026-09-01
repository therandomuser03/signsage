# SignSage — Build Phases

Each phase has a go/no-go exit criterion. Do not start the next phase until the current
one's criterion is met — this is the single biggest risk-management tool available to a
solo dev on a thesis timeline (see PRD.md success criteria).

## Phase 0 — Core-concept notebook (current phase)

**Artifact:** `notebooks/signsage.ipynb` — one consolidated notebook
(see DECISIONS.md for why this replaced the earlier split `01_.../02_...` plan) covering,
as internally phase-gated sections within the single file:

- Sign->Text: GRU baseline -> spatial-temporal Transformer -> non-manual-marker
  (face) ablation, evaluated with WER/BLEU/ROUGE-L against a trivial baseline.
- Text->Sign: naive gloss heuristic (+ optional LLM upgrade) -> TF-IDF retrieval baseline
  -> autoregressive generative pose model, evaluated free-running with DTW and normalized
  MSE against ground truth.
- A genuine round-trip check (text -> gloss -> generated pose -> Sign->Text on that
  *generated* pose, not the dataset's).
- A model-only latency harness.
- A knowledge repository (SQLite + JSON) recording dataset version, splits, experiment
  configuration, metrics, artifacts, and hardware for every run — see DATA_MODEL.md.

Sections run 0 through 17, plus a Section 19 hand-off cell. The earlier Section 18
live-demo spike has been removed from the notebook (see DECISIONS.md); serving code
belongs to Phase 1, not to the research artifact.

**Goal:** prove every core piece of the project's concept produces a real, measurable
signal on iSign, before any of it is trusted at scale or wrapped in standalone
backend/frontend infrastructure.

**Current run state:** the bounded Kaggle validation run has now been **executed against real
iSign data and completed**, with the expected output structure and trained artifacts produced.
The run generated model checkpoints under `signsage_outputs/checkpoints/`, a vocabulary/report
artifact under `signsage_outputs/reports/`, and the SQLite/JSON project-knowledge artifacts.
The screenshots confirm that these artifacts exist; the exact experimental metrics and the
Section 17 `GO CANDIDATE`/`NO-GO` verdict still need to be read from the notebook/report before
being treated as validated results. The 2000/300/300 Kaggle run remains a bounded first GPU
validation pass, not a thesis-scale result.

`RUN_MODE` is now derived from the environment rather than hardcoded: `"kaggle"`
(2000/300/300 rows, 5-6 epochs) when `IS_KAGGLE`, otherwise `"smoke"` (300/60/60). The
Kaggle setting is a bounded first GPU validation pass, not a thesis-scale result.

**First-run sequence** (do not run all 61 cells blind — the cheapest failures are also
the earliest):

1. Kaggle prerequisites: Internet **enabled** (needs a phone-verified account), an
   `HF_TOKEN` Kaggle Secret whose account has accepted the gated iSign terms, and
   accelerator set to GPU.
2. Run cells 0-8 only. Confirm `Device: cuda`, the four part sizes print, the ZIP opens,
   and `Sample members:` shows paths matching what `RemotePoseStore` expects.
3. Continue to the one-batch smoke check (Section 6a) before touching training.
4. Only then run training top-to-bottom.
5. **Post-run verification:** preserve the completed `signsage_outputs/` directory and record
   the exact metrics, Section 17 verdict, checkpoint names, and run metadata before changing
   the notebook or starting another experiment.

**Exit criteria** — the notebook's Section 17 computes these automatically and prints
either `NO-GO` or `GO CANDIDATE`; it does not round a failing run up to a success:
- Sampled translations look coherent and pose confidence is reasonably high (data audit,
  read by a human — not automated).
- GRU and Transformer training loss actually decrease (not flat/NaN). Non-finite loss
  raises immediately rather than training on.
- Transformer beats the trivial "most common training sentence" baseline on WER **or**
  BLEU. Note this is deliberately an OR, i.e. a weak bar — one headline metric moving is
  enough to be called a `GO CANDIDATE`, which the report itself qualifies as "repeat at
  larger scale before treating it as a thesis result." Do not report a `GO CANDIDATE` as
  a validated result.
- The non-manual-marker ablation is actually measuring something — guaranteed here by the
  `FACE_LANDMARKS` assertion plus a no-face feature count strictly below the full count.
- The generative Text->Pose model's DTW distance is competitive with retrieval's (the
  report states plainly when generation is *worse*, and note retrieval is capped at 3000
  training rows — see TESTING.md).
- The round-trip check recovers *something* from a generated pose. Qualitative evidence
  only, explicitly not a benchmark.

Also worth reading before trusting any of the above: `RUN_MODE = "kaggle"` uses
2000/300/300 rows, and the go/no-go report itself qualifies a pass as "repeat at larger
scale before treating it as a thesis result." Smoke mode uses 300/60/60 rows, where these
criteria are pipeline checks, not results at all.

**Caveats now resolved** (from earlier local runs of Sections 0-4 against the real
archive, so these earlier open questions are answered): the CSV has **no** per-segment
timing columns, and none are needed because the archive is one `.pose` file per UID/row
(not per video); `pose.header.components` **does** expose `FACE_LANDMARKS` (468 points),
so the no-face ablation mask is verified rather than guessed, and the notebook `assert`s
it is present before running the ablation; the local CSV has **127,237 rows matching
127,237 `.pose` files 1:1** — the 118,228 figure is the published dataset description,
not this local archive. **Still true:** the optional LLM gloss step needs a
`GROQ_API_KEY` and silently falls back to a naive heuristic without one.

**If this fails:** the problem is almost certainly data quality or task difficulty at
this dataset size — do not respond by jumping to a bigger/fancier model. Debug the data
first.

**Compute caveat, revised:** training now runs on Kaggle GPU (T4) with fp16 mixed
precision, replacing the local CPU-only path (see ARCHITECTURE.md / DECISIONS.md). The
binding constraint has changed with it: GPU time is no longer the bottleneck, remote I/O
is. Pose members are fetched lazily from HuggingFace over HTTP range requests at
`NUM_WORKERS = 0` — roughly 8 sequential 256 KB requests per ~1.9 MB pose file — so the
cold-fetch phase, not training, dominates wall-clock on the first run. Budget for that
before assuming the effort estimates below still hold, and note Kaggle's limits: 12 h per
GPU session, ~30 h GPU per week. The pose cache lives in `/kaggle/temp` and does **not**
survive the session, so an interrupted run re-fetches from scratch.

## Phase 0.5 — Notebook to modules

**Goal:** once Phase 0 produces a go decision and exported artifacts, split the notebook's
logic into proper Python modules (data loading, pose normalization, models, inference).
Those modules — not the notebook — become the backend's internals.

**Why this is its own step:** the notebook is the reference implementation of the ML
system, deliberately not the backend. Copying cells into a FastAPI app would carry the
notebook's Kaggle-specific assumptions (remote ZIP streaming, `/kaggle/temp` cache,
`RUN_MODE` limits) into production code where none of them belong.

**Exit criteria:** a module-level function loads an exported checkpoint and runs inference
on one landmark sequence, producing the same output the notebook did for the same input.

## Phase 1 — Single-device Sign->Text demo

**Note on the removed Section 18 spike:** a FastAPI WebSocket server plus a MediaPipe
HTML client used to live inside the Phase-0 notebook behind `RUN_LIVE_DEMO = False`. It
has been removed (see DECISIONS.md), so there is no longer a flag to flip and no
shortcut into this phase. What that spike got wrong is still on record and is the
checklist to beat when the real backend is built: it sent un-normalized landmarks,
passed a wrong sequence length, auto-cut sentences at 160 frames instead of on
motion-pause, rendered only the first pose frame rather than animating, and had no error
surface at all. See ARCHITECTURE.md risk 1, API.md, and DESIGN.md for the full list.
Expect translation output to be meaningless until inference-time normalization matches
training.

**Goal:** webcam -> client-side MediaPipe Holistic -> WebSocket -> FastAPI -> Phase 0's
Transformer model -> captions on screen. One direction, one device, no relay.

**Exit criteria:** live demo works end-to-end on your own dev machine, with a
sanity-checked (not necessarily great) translation quality matching Phase 0's numbers
within a reasonable margin. If live quality is much worse than notebook quality,
investigate the train/inference landmark-extraction skew (see ARCHITECTURE.md) before
anything else.

## Phase 2 — Single-device Text->Pose demo

**Goal:** text/speech input -> gloss (LLM) -> pose sequence (Phase 0's generative
model) -> skeleton animation on canvas. Same device as Phase 1.

**Exit criteria:** live demo works end-to-end; animation is recognizable as attempting
ISL structure (not claiming fluency).

## Phase 3 — Combined shared-device MVP + evaluation

**Goal:** both directions live on one shared-device conversational UI (see DESIGN.md).
Run the full evaluation suite (WER/BLEU/ROUGE-L on a larger held-out iSign split, pose
metrics, end-to-end latency on target hardware — SMART objective 2) and write up
results, including honest failure modes.

**This is the MVP.** Everything below is optional, post-thesis-defense scope.

## Phase 4+ — Deferred (build only if Phase 3 demonstrates real need)

- Two-device WebSocket relay + room codes.
- GAT (graph attention) upgrade beyond Phase 0's Transformer, if time and results
  justify it (see DECISIONS.md).
- PWA packaging, persistence layer, polish.

## Rough timeline (solo dev, part-time alongside coursework)

| Phase | Effort estimate | Note |
|---|---|---|
| 0 | 4-10 weeks | Highest variance — covers both directions now, this is the actual risk in the project |
| 0.5 | 3-5 days | Extracting notebook logic into modules, once artifacts exist |
| 1 | 1-2 weeks | Mostly plumbing once Phase 0's Sign->Text model exists |
| 2 | 2-3 weeks | Mostly plumbing once Phase 0's Text->Pose model exists |
| 3 | 2-3 weeks | Integration + evaluation + writeup buffer |

Total: roughly 3-4 months full-time-equivalent, 5-7 months realistic calendar time
solo alongside coursework.

### Conflict with the guide's date — read this before planning against the table above

The project guide has set a completion date of **2026-09-05**. The estimates above total
months, not days, and Phase 0 has not yet produced a single result. These two facts cannot
both be satisfied, and the table is left unedited rather than quietly rewritten downward,
because inventing a schedule that fits the date would only move the failure later.

What is actually reachable by that date is Phase 0: a completed bounded Kaggle run
(2000/300/300) with real metrics, exported checkpoints, and a populated knowledge
repository — i.e. a defensible ML pipeline with evidence, plus an honest statement of what
remains. Phases 1-3 (live demo, combined MVP, full-scale evaluation) are not reachable and
should be presented as designed-and-scoped, not built. Prioritize accordingly: the Kaggle
run is the deliverable, and frontend/backend work started now would produce neither a demo
nor a result.
