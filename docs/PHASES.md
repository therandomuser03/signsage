# SignSage — Build Phases

Each phase has a go/no-go exit criterion. Do not start the next phase until the current
one's criterion is met — this is the single biggest risk-management tool available to a
solo dev on a thesis timeline (see PRD.md success criteria).

## Phase 0 — Core-concept notebook (complete, with caveats — read before trusting the label)

**Artifact:** `research/notebooks/signsage.ipynb` — one consolidated notebook
(see DECISIONS.md for why this replaced the earlier split `01_.../02_...` plan) covering,
as internally phase-gated sections within the single file:

- Sign->Text: GRU baseline -> spatial-temporal Transformer -> non-manual-marker
  (face) ablation, evaluated with WER/BLEU/ROUGE-L against a trivial baseline.
- Text->Sign: naive gloss heuristic (+ optional LLM upgrade) -> TF-IDF retrieval baseline
  -> autoregressive generative pose model, evaluated free-running with DTW, normalized
  MSE, and a motion-energy check against ground truth.
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

**Current run state — done, twice, with results recorded.** Two Kaggle runs have now
completed against real iSign data:

1. A bounded validation pass (`TRAIN_LIMIT/VAL_LIMIT/TEST_LIMIT = 2000/300/300`,
   5-6 epochs) — proved the pipeline (remote streaming, training, evaluation, provenance
   logging) works end-to-end.
2. A scaled-up pass (`10000/1500/1500`, 5/8/8 epochs, batch size 32) — run after two bugs
   found in the first pass's evaluation logic were fixed (a degenerate empty-string
   trivial baseline, and no check for the Text→Pose generator collapsing to a static
   pose). Full findings: **`research/outputs/phase0/RESULTS.md`.**

**The Section 17 verdict is `GO-CANDIDATE`, and that label is real but narrower than it
sounds.** It is true only because the Transformer beats the trivial baseline on
BLEU/ROUGE-L — the deliberately weak, OR-based bar described below. It does **not** mean
Sign→Text or Text→Pose currently produce useful output. Specifically, and this is the
part to actually internalize before presenting this phase as successful:

- Both Sign→Text models exhibit severe mode collapse (GRU: 0.4% unique outputs across
  1,500 test rows; Transformer: 21%, better but still mostly repetitive).
- The Transformer's WER got *worse* with 5x more training data (1.035 → 1.367), because
  more data made its repetition problem produce longer, more error-prone output rather
  than better translations — pointing at greedy decoding / lack of a repetition penalty
  as the actual bottleneck, not data volume.
- The Text→Pose generator's favorable DTW/MSE numbers are a static-pose artifact: a
  motion-energy check (added specifically because the pose visualization looked
  suspiciously frozen) confirmed generated motion is only ~6.7% of ground-truth motion.
- The trivial baseline itself required two rounds of fixing and, even fixed, revealed a
  genuine data-quality issue: ~1% of the full dataset is junk/placeholder text (see
  DATA_MODEL.md), which the notebook's baseline computation needs to filter more
  aggressively than it currently does.

None of this is a failure of Phase 0 — a bounded validation phase whose diagnostics
correctly caught two real, non-obvious problems (rather than reporting a falsely
reassuring "GO") is Phase 0 doing exactly its job. It does mean: **do not present
`GO-CANDIDATE` to a guide or committee without the caveats above attached**, and do not
assume Phase 0.5's module extraction can safely defer thinking about decoding strategy
or the pose loss function — both are now identified, specific problems, not vague future
polish.

**First-run sequence** (still correct guidance for any future re-run):

1. Kaggle prerequisites: Internet **enabled** (needs a phone-verified account), an
   `HF_TOKEN` Kaggle Secret whose account has accepted the gated iSign terms, and
   accelerator set to GPU.
2. Run cells 0-8 only. Confirm `Device: cuda`, the four part sizes print (measured at
   158.54 GiB total — see DATA_MODEL.md for why this differs from earlier ~231 GB
   estimates), the ZIP opens, and `Sample members:` shows paths matching what
   `RemotePoseStore` expects.
3. Continue to the one-batch smoke check (Section 6a) before touching training.
4. Only then run training top-to-bottom.
5. **Post-run verification: done for both completed runs.** `research/outputs/phase0/`
   holds the preserved output directory and `RESULTS.md` holds the exact metrics,
   Section 17 verdicts, and run metadata for both. Any future re-run should be recorded
   the same way, appended to `RESULTS.md`, not overwriting it.

**Exit criteria** — the notebook's Section 17 computes these automatically and prints
either `NO-GO` or `GO CANDIDATE`; it does not round a failing run up to a success. Now
annotated with what actually happened:

- Sampled translations look coherent and pose confidence is reasonably high (data audit,
  read by a human — not automated). **Text is not coherent** — see mode-collapse findings
  above; the human-read qualitative samples in both runs show repetitive, often
  ungrammatical output. Pose confidence itself (the MediaPipe-style per-point confidence
  score) was reasonably high in the audit; this criterion conflates two different things
  and should probably be split in any future revision of this doc.
- GRU and Transformer training loss actually decrease (not flat/NaN). **True in both
  runs** — no non-finite loss, no crash. This criterion is met and is a genuine positive
  signal, just a narrow one (it proves something is learnable, not that it's useful).
- Transformer beats the trivial "most common training sentence" baseline on WER **or**
  BLEU. **Technically true on BLEU/ROUGE-L in both runs, false on WER in the scaled-up
  run.** This is the criterion that most needs revisiting: an OR bar between WER and
  BLEU/ROUGE-L turned out to be gameable by a model that got *more* repetitive with more
  data (worse WER) while still nudging BLEU/ROUGE-L up slightly. A future version of this
  criterion should probably require WER improvement specifically, or check output
  diversity as a joint condition, not treat WER and BLEU as interchangeable evidence of
  the same thing.
- The non-manual-marker ablation is actually measuring something — guaranteed here by the
  `FACE_LANDMARKS` assertion plus a no-face feature count strictly below the full count.
  **Met, and now replicated**: no-face trained to a lower validation loss than full-face
  in *both* runs, at two different data scales — a real, repeated finding worth its own
  discussion in the thesis, not just a checkbox (see PRD.md success criteria, DATA_MODEL.md).
- The generative Text->Pose model's DTW distance is competitive with retrieval's (the
  report states plainly when generation is *worse*, and note retrieval is capped at 3000
  training rows — see TESTING.md). **Technically met on DTW/MSE in both runs, but now
  known to be a static-pose artifact** rather than genuine competitiveness — see the
  motion-energy finding above. This criterion, as originally written, does not detect
  this failure mode; TESTING.md has been updated to require the motion-energy check
  alongside DTW/MSE going forward.
- The round-trip check recovers *something* from a generated pose. Qualitative evidence
  only, explicitly not a benchmark. **Technically met** (recovers a short, garbled
  phrase) but unsurprising given the mode-collapse findings above — treat this as
  confirming, not independently validating, the other findings.

**If this fails:** the problem is almost certainly data quality or task difficulty at
this dataset size — do not respond by jumping to a bigger/fancier model. Debug the data
first. **Updated based on what actually happened:** this advice held for the trivial
baseline (a genuine data issue was found and partially fixed) but did not fully hold for
Sign→Text WER — scaling data 5x did not fix, and arguably worsened, the repetition
problem, because the actual bottleneck there is decoding strategy, not data volume. Keep
"debug the data first" as the default instinct, but don't assume it's the *only* possible
cause once a specific decoding-side symptom (long, repetitive, run-on generations) is
visible in the qualitative samples.

**Compute caveat, still accurate:** training runs on Kaggle GPU (T4) with fp16 mixed
precision. The binding constraint is remote I/O, not GPU time — confirmed directly: the
scaled-up run's ~13,000 unique pose fetches took the bulk of its 4h29m wall-clock time,
comfortably inside Kaggle's 12h/session cap and using roughly 15% of the ~30h/week quota.
Budget accordingly for any further re-run.

## Phase 0.5 — Notebook to modules

**Goal:** now that Phase 0 has produced a go decision and exported artifacts (with the
caveats above), split the notebook's logic into proper Python modules (data loading, pose
normalization, models, inference). Those modules — not the notebook — become the
backend's internals.

**Why this is its own step:** the notebook is the reference implementation of the ML
system, deliberately not the backend. Copying cells into a FastAPI app would carry the
notebook's Kaggle-specific assumptions (remote ZIP streaming, `/kaggle/temp` cache,
`RUN_MODE` limits) into production code where none of them belong.

**Before starting this phase, decide explicitly** whether to (a) proceed with
Phase 0.5/1 using the current checkpoints as-is, clearly labeled as a plumbing
proof-of-concept rather than a translation demo, or (b) spend additional Kaggle time
first on the two identified fixes (a repetition penalty/beam search for Sign→Text; a
loss-function change for Text→Pose) before extracting anything. Given the guide's
2026-09-05 date (see the timeline section below), (a) is the realistic choice — do not
silently default to (b) and run out of time, and do not silently default to (a) and then
present the resulting demo as more capable than RESULTS.md supports.

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
training — and, per Phase 0's results, expect it to still look repetitive/mode-collapsed
even once normalization is correct, since that's a model problem, not an integration one.

**Goal:** webcam -> client-side MediaPipe Holistic -> WebSocket -> FastAPI -> Phase 0's
Transformer model -> captions on screen. One direction, one device, no relay.

**Exit criteria:** live demo works end-to-end on your own dev machine, with a
sanity-checked (not necessarily great) translation quality matching Phase 0's numbers
within a reasonable margin. If live quality is much worse than notebook quality,
investigate the train/inference landmark-extraction skew (see ARCHITECTURE.md) before
anything else. If live quality *matches* Phase 0's numbers, remember what that quality
actually is (mode-collapsed, repetitive) before calling the demo successful.

## Phase 2 — Single-device Text->Pose demo

**Goal:** text/speech input -> gloss (LLM) -> pose sequence (Phase 0's generative
model) -> skeleton animation on canvas. Same device as Phase 1.

**Exit criteria:** live demo works end-to-end; animation is recognizable as attempting
ISL structure (not claiming fluency). Per Phase 0's results, expect the current
checkpoint's animation to look close to static — this is a known, documented model
limitation (see RESULTS.md), not something to debug as a rendering bug first.

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
| 0 | 4-10 weeks | **Complete** — see above. Actual elapsed effort landed inside this range |
| 0.5 | 3-5 days | Extracting notebook logic into modules, now that artifacts and known issues both exist |
| 1 | 1-2 weeks | Mostly plumbing once Phase 0's Sign->Text model exists |
| 2 | 2-3 weeks | Mostly plumbing once Phase 0's Text->Pose model exists |
| 3 | 2-3 weeks | Integration + evaluation + writeup buffer |

Total: roughly 3-4 months full-time-equivalent, 5-7 months realistic calendar time
solo alongside coursework.

### The guide's date, and what's actually being delivered against it

The project guide's completion date is **2026-09-05**. Phase 0 completed (both runs) on
2026-09-03, leaving on the order of a day or two before that date — nowhere near enough
calendar time for Phases 0.5 through 3 regardless of the effort estimates above, which
total weeks to months.

**What is actually being delivered by the guide's date is Phase 0**, and it is a
materially stronger deliverable than "a completed bounded Kaggle run" alone would have
been: two runs at different scales, a queryable experiment-provenance database (with a
naming bug found and fixed), a written results document with honest caveats, and two
specific, evidenced root causes identified for the problems found (decoding strategy for
Sign→Text; loss function for Text→Pose) rather than an unexplained "it doesn't work
great yet." Phases 1–3 (live demo, combined MVP, full-scale evaluation) remain
designed-and-scoped, not built, exactly as this document anticipated before Phase 0 ran.

Present it that way: the deliverable is a rigorously evaluated ML pipeline with a
defensible, self-critical account of its own limitations — not a demo, and not a claim
that the limitations are unknown or unaddressed. That is a legitimate, defensible thesis
Phase 0 outcome on this timeline.
