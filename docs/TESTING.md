# SignSage — Testing & Verification

This is a research prototype, not a production service — the test strategy below is
scoped accordingly. No heavy test pyramid, no CI/CD gate obsession. The goal is: every
non-trivial piece of logic has *one* cheap, runnable check that would catch it if it
silently broke.

## Model evaluation (the part that actually matters)

- **Metrics:** WER, BLEU, ROUGE-L for Sign->Text (via `jiwer`, `sacrebleu`,
  `rouge-score` — don't hand-roll these). Pose/keypoint MSE plus a sequence-similarity
  metric (DTW, or a back-translation-style check: run generated poses back through the
  Sign->Text model or a human reviewer and see if meaning survives) for Text->Pose. Raw
  pose MSE alone is a weak, gameable metric — a model predicting near-average poses can
  score deceptively well on MSE while being visually meaningless.
- **Always compare against a trivial baseline** (e.g. "predict the most common training
  sentence" for Sign->Text, retrieval for Text->Pose) — a good-looking absolute number
  means nothing without this comparison. Both are built into
  `notebooks/signsage.ipynb` already.
- **Report signer-independent splits if metadata allows it.** iSign likely has few
  distinct interpreters; a model can overfit to one person's signing style. Don't let a
  signer-dependent number stand in for generalization.
- **Data audit is a test.** The manual-read-15-random-samples step, the pose-
  confidence check, and the pose-file-count-vs-CSV-row-count sanity check (confirmed
  matching 1:1 once it was clear pose files are stored per-row/UID, not per-video_id —
  see DATA_MODEL.md) in the notebook are go/no-go gates, not optional exploration —
  see PHASES.md.

## Kaggle preflight checks (cheapest failures, run them first)

The notebook's early cells are tests, and they are ordered so the assumptions most likely
to be wrong fail first, before any expensive work:

- **Environment.** Cell 3 prints device, GPU count, data mode, run mode, and whether AMP
  is on. `Device: cpu` on Kaggle means the accelerator wasn't enabled — stop there, since
  every later timing number would be meaningless.
- **Credentials.** Cell 6 raises immediately if `HF_TOKEN` is absent rather than
  proceeding unauthenticated. A 401/403 from a gated repo surfaces later as a misleading
  "Internet enabled" range-request error, so verify the HF account has accepted the iSign
  terms first.
- **Archive layout.** Cell 8 probes each of the four part sizes, opens the virtual ZIP64,
  and prints five real member names. This is the notebook's single most important
  unverified assumption: that `concat(aa..ad)` is one valid ZIP and that member paths match
  `iSign-poses_v1.1/{uid}.pose` or `{uid}.pose`. A wrong prefix caught here costs seconds;
  caught later it is a `FileNotFoundError` on every UID.
- **fp16 cache fidelity.** Cell 19 clears the in-memory cache to force a real disk
  round-trip, then prints the max absolute difference between the fp16 cached array and a
  fresh fp32 computation for 3 UIDs. Small diffs relative to the ~unit normalized-pose
  scale are the check that float16 caching isn't quietly degrading inputs.
- **One-batch smoke check.** Section 6a runs a single batch through the loader before any
  training loop, so shape and collate errors don't surface mid-epoch.
- **Non-finite loss raises immediately** rather than training on (see PHASES.md).

Run cells 0-8, stop, read the output, and only then continue. Running all 61 cells blind
wastes the session's GPU quota on a failure that cell 8 would have shown in five minutes.

## Post-run verification (now required)

The first bounded Kaggle run completed and produced the expected checkpoint, report, and
knowledge-repository artifacts. Before treating the run as a Phase 0 pass, record the exact
run metadata and numerical results from the notebook/report: train/validation/test sizes,
experiment names, best validation losses, WER/BLEU/ROUGE-L, Text->Pose DTW/MSE, round-trip
observation, latency, and the Section 17 `GO CANDIDATE`/`NO-GO` verdict. Artifact existence is
only a pipeline-health signal; it is not evidence that the quality exit criteria were met.

## Reproducibility as a verification artifact

Every run writes its own record to `signsage_knowledge.db` (see DATA_MODEL.md): seed,
device, run mode, dataset row counts, split sizes and leakage flag, hyperparameters, all
metrics, and checkpoint paths. That is what makes a reported number checkable rather than
merely asserted — a metric with no recorded experiment, split, and dataset version behind
it cannot be defended. If you produce a number that isn't in the repository, that's a gap
in the test strategy, not just missing bookkeeping.

## System-level checks (once backend/frontend exist)

- One smoke test per direction: does a known landmark sequence produce *a* text output
  without crashing; does a known text input produce *a* pose sequence without crashing.
  Not correctness — just "the pipeline didn't fall over."
- Latency measurement harness for SMART objective 2 (end-to-end, on defined target
  hardware) — this should be a real, repeatable script, not a one-off manual stopwatch
  measurement, since it's a stated thesis deliverable.

## Device/environment testing (easy to skip, don't)

- Test MediaPipe Holistic performance on a budget Android phone, not just your dev
  laptop — WASM performance on weak hardware can drop well below usable frame rates,
  which undercuts any "works on any device" claim.
- Test in more than one lighting condition/room — landmark quality is sensitive to
  this, and the target users are not going to have your desk setup.

## What's explicitly not being done

- No unit-test-everything mandate for a solo research prototype — write tests for logic
  that's non-obvious or has bitten you before, not for the sake of coverage numbers.
- No load/stress testing — there's no concurrent-user scenario in the MVP scope.
