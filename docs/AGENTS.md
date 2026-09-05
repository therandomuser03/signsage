# AGENTS.md — SignSage

Instructions for AI coding agents (in addition to any global/personal agent
instructions already in effect) working on this repository.

## Read this first

- `PRD.md` — what this project is and isn't.
- `PHASES.md` — what phase we're in right now, and what "done" looks like for it.
- `DECISIONS.md` — decisions already made and why; don't relitigate these without a new
  reason, and don't silently contradict them.
- `research/outputs/phase0/RESULTS.md` — the actual Phase 0 findings. Read this before
  assuming any WER/BLEU/DTW number from this project is good news; several are not what
  they look like at first glance (see below).

## The one rule that matters most here

**Nothing in `backend/`, `frontend/`, or any deployment/infra file should be built
before `research/notebooks/signsage.ipynb` produces a go decision.** Phase 0 has now produced one
(`GO-CANDIDATE`, per the notebook's own deliberately weak OR-based criterion), so
scaffolding Phase 0.5 module extraction is now appropriate — but read the qualifier
below before treating "GO-CANDIDATE" as "the model works."

The current notebook enforces the backend/frontend part of this rule by construction
rather than by a flag: the earlier Section 18 live-demo spike (FastAPI + WebSocket +
MediaPipe HTML client, gated behind `RUN_LIVE_DEMO = False`) has been **removed
entirely**. Sections now run 0–17 plus a Section 19 hand-off cell, and there is no
`RUN_LIVE_DEMO`, no FastAPI, and no INT8 quantization anywhere in the file. Do not
re-add serving code to the notebook — it belongs in Phase 1's backend, not in the
research artifact.

**The qualifier that matters:** `GO-CANDIDATE` is true because the Transformer beats a
trivial baseline on BLEU/ROUGE-L. It does not mean Sign→Text or Text→Pose currently work.
Two specific, evidenced problems exist and should inform how Phase 0.5 extraction is
prioritized, not be treated as separate future work to rediscover later:
- **Sign→Text has a decoding-side repetition problem**, not (only) a data-scale one — WER
  got *worse* (1.035 → 1.367) when training data was scaled 5x, because the model
  produces longer, more repetitive run-on output rather than better translations. Greedy
  decoding with no repetition penalty is the leading suspect.
- **Text→Pose has collapsed to a near-static average pose** — its favorable DTW/MSE
  numbers are explained by regression-to-the-mean (confirmed via a motion-energy check:
  generated motion is ~6.7% of ground-truth motion), not by working pose generation.

If asked to scaffold backend/frontend code as if these were solved, push back and point to
`RESULTS.md` — this is evidenced, not speculative caution.

## Project-specific context an agent should know

- This is a **solo M.Tech thesis project**, not a team/company codebase. Do not
  introduce multi-contributor process (CI gates, PR templates, contribution
  guidelines) unless explicitly asked — see DECISIONS.md for why several
  "standard project" files were deliberately deferred.
- The iSign dataset is **gated on HuggingFace** — it requires the human user's personal
  account to accept terms and generate a token. An agent cannot do this step; don't
  attempt to work around it, just flag it as a manual prerequisite. It is licensed
  **CC BY-NC-SA 4.0** (non-commercial, share-alike) — see SECURITY.md before ever
  considering committing any part of the raw dataset to this repository.
- **Training runs on Kaggle now, GPU (T4×2, one GPU used), not local CPU and not
  Colab.** The notebook detects its environment (`IS_KAGGLE`) and switches behavior:
  - *Kaggle mode* (the real, now-validated training path, two completed runs): the pose
    archive — documented in early planning as ~231 GB, but **measured directly by the
    notebook's own probe at 158.54 GiB (~170 GB)** across its four parts (see
    DATA_MODEL.md/DECISIONS.md for the discrepancy) — is **never uploaded and never
    extracted**. The notebook downloads only `iSign_v1.1.csv` (9.81 MB) from HuggingFace
    and streams individual `.pose` members out of the four
    `iSign-poses_v1.1_part_a{a,b,c,d}` files, treating them as one virtual concatenated
    ZIP64 and fetching 256 KB byte ranges on demand. This requires Kaggle **Internet
    enabled** and an `HF_TOKEN` Kaggle Secret (see SECURITY.md). Do **not** "fix" this
    by pointing `DATA_DIR` at `/kaggle/input` — no Kaggle Dataset upload exists, by
    design.
  - *Local mode* (fallback, still supported in code, but **no live machine currently
    backs it** — the storage laptop was wiped once the Kaggle path was validated twice
    at different scales): `SIGNSAGE_DATA_DIR` would point at an already-extracted
    archive on a separate storage machine, reached over Tailscale as an "Existing
    Jupyter Server." Treat this path as untested-since-migration if it's ever revived.
  - Mixed precision (`USE_AMP`) engages only on CUDA, so a CPU session silently trains
    far slower. Confirm `Device: cuda` in cell 3's output before timing anything.
- Column names in `iSign_v1.1.csv` and the `.pose` file layout are **confirmed** (see
  DATA_MODEL.md): the CSV has `uid` and `text` columns, and the archive is one `.pose`
  file per UID/row (127,237 rows, 127,237 files, 1:1) — not one file per `video_id`. The
  pose tensor is `(frames, people, keypoints, xyz)`, and the notebook uses the first 543
  image-space landmarks, excluding the trailing `POSE_WORLD_LANDMARKS` duplicate block.
- **A real, previously undocumented data-quality issue was found**: roughly 1% of the
  127,237 rows are junk/placeholder text (single digits, single letters, the literal
  word `blank`, auto-caption artifacts like `[Music]`), which is enough to dominate a
  naive "most common sentence" computation on a random sample. If you touch any
  frequency-based baseline or vocabulary-building logic, filter on the CSV's own
  `n_words` column (e.g. `>= 2`), not just non-empty text — see DECISIONS.md.
- The pose loader caches normalized per-UID arrays to disk under `POSE_CACHE_DIR` (see
  DATA_MODEL.md / DECISIONS.md), which resolves to
  `/kaggle/temp/signsage_cache/normalized_v1/` on Kaggle and
  `<SIGNSAGE_DATA_DIR>/cache/normalized_v1/` locally. It's derived data, not source data
  — safe to delete and let it rebuild if you ever suspect it's stale or corrupted; writes
  are already atomic and a failed load already falls back to a rebuild, so this shouldn't
  need manual cleanup in normal operation. On Kaggle it is deliberately *outside*
  `/kaggle/working` so it doesn't count against the 20 GB output limit, which also means
  it does not survive the session.
- **Every run writes a knowledge repository** (`signsage_knowledge.db` +
  `signsage_knowledge_manifest.json` under `signsage_outputs/`). If you add or change an
  experiment, evaluation, or artifact, record it there too — the traceability chain is a
  deliverable, not a nicety. See DATA_MODEL.md for the schema and the idempotency rules
  (`delete_keys` / `knowledge_metric` / `knowledge_artifact` replace on re-run;
  `knowledge_observation` is deliberately append-only). **Use one consistent name per
  experiment across `experiments`/`metrics`/`artifacts`** — an earlier version of this
  notebook used three different strings for the GRU model across those three tables,
  which silently broke joins. Fixed, but don't reintroduce the pattern when adding new
  experiments.
- Keep the model simple until proven insufficient: GRU baseline before Transformer,
  Transformer before GAT, the full 543 image-space landmarks (only the duplicated
  `POSE_WORLD_LANDMARKS` block dropped) before hand-picked blendshapes/landmark-subset
  optimization. Don't jump ahead in this ladder speculatively.
- **Groq's model lineup deprecates frequently** — `llama-3.1-8b-instant` (originally
  hardcoded in the Part 12 gloss step) was shut down on 2026-08-16, mid-project. The
  notebook now uses `openai/gpt-oss-20b` with a try/except that falls back to the naive
  gloss heuristic on any API failure, so a future deprecation degrades gracefully
  instead of crashing the notebook — but if that fallback keeps firing, check
  console.groq.com/docs/models before assuming it's a bug in this repo.

## When you make a new non-trivial technical decision

Add it to `DECISIONS.md` with a one-line rationale. Keep the format lightweight (this
project intentionally does not use a formal multi-file ADR process — see DECISIONS.md's
own header).

## Keep the docs honest

Several docs here (ARCHITECTURE.md, API.md, DESIGN.md) describe target/planned state,
not built state, because they were written before the corresponding code existed. When
you build the real thing and it diverges from the doc, **update the doc in the same
change** — stale docs that describe a system that doesn't match reality are worse than
no docs. (This file itself used to say the notebook "has not yet been run end-to-end" —
that was true when written and false by the time anyone read it next; don't let that
happen again. If you run something that changes a fact stated elsewhere in `docs/`,
fix that doc now, not "later.")
