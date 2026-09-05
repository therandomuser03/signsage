# SignSage — Testing & Verification

This is a research prototype, not a production service — the test strategy below is
scoped accordingly. No heavy test pyramid, no CI/CD gate obsession. The goal is: every
non-trivial piece of logic has *one* cheap, runnable check that would catch it if it
silently broke.

## Model evaluation (the part that actually matters)

- **Metrics:** WER, BLEU, ROUGE-L for Sign->Text (via `jiwer`, `sacrebleu`,
  `rouge-score` — don't hand-roll these). Pose/keypoint MSE plus a sequence-similarity
  metric (DTW) for Text->Pose, **plus a motion-energy check** (mean absolute
  frame-to-frame change, compared against ground truth) — added after the first Kaggle
  run's pose visualization looked suspiciously static, and confirmed necessary: on the
  scaled-up run, the generative model beat retrieval on DTW/MSE while its motion energy
  was only ~6.7% of ground truth, meaning the DTW/MSE win was a near-static-pose
  artifact (regression-to-the-mean), not evidence of working generation. **Raw pose
  MSE/DTW alone is a weak, gameable metric for exactly this reason — a model predicting
  near-average poses can score deceptively well while producing no real motion. Do not
  evaluate Text->Pose on DTW/MSE without also checking motion energy.**
- **Always compare against a trivial baseline** (e.g. "predict the most common training
  sentence" for Sign->Text, retrieval for Text->Pose) — a good-looking absolute number
  means nothing without this comparison. Both are built into
  `research/notebooks/signsage.ipynb` already. **The trivial baseline needs its own sanity
  check, learned the hard way**: a naive "most frequent exact text value" computation
  degenerated to an empty string in the first run (391 blank-text rows dataset-wide were
  enough to be the mode of a 2,000-row sample), and after fixing that, degenerated to a
  single digit (`'1'`) in the second run — real sentences are almost always unique
  strings, so any systematic junk-text pattern in the dataset will win the "most common"
  computation. Exclude near-empty text using the CSV's own `n_words` column (e.g.
  `n_words >= 2`), not just non-empty text — see DATA_MODEL.md for the scope of this
  issue (~1% of the full dataset).
- **Report signer-independent splits if metadata allows it.** iSign likely has few
  distinct interpreters; a model can overfit to one person's signing style. Don't let a
  signer-dependent number stand in for generalization. **Not yet checked** — the current
  split is leak-safe by `video_id` (confirmed, both runs), but whether `video_id`
  diversity implies signer diversity has not been verified against any signer metadata
  iSign may expose. Flagged as a remaining gap, not resolved by either completed run.
- **Data audit is a test.** The manual-read-15-random-samples step, the pose-
  confidence check, and the pose-file-count-vs-CSV-row-count sanity check (confirmed
  matching 1:1 once it was clear pose files are stored per-row/UID, not per-video_id —
  see DATA_MODEL.md) in the notebook are go/no-go gates, not optional exploration —
  see PHASES.md. **This audit step is also what should have caught the junk-text
  baseline issue earlier than it was actually caught** — the 15-sample manual read is a
  small enough sample that single-digit/single-letter rows may simply not have appeared
  in it by chance. Consider enlarging this sample or adding an automated
  short-text-frequency check alongside the manual read in any future revision.
- **Knowledge-DB writes need consistent naming, and nothing currently enforces it.**
  The first run's DB had the GRU model recorded under three different strings across
  `experiments`, `metrics`, and `artifacts` (see DATA_MODEL.md/DECISIONS.md), silently
  breaking the most natural query (`JOIN` on name) without erroring. Fixed for the GRU
  case; treat this as a standing test-strategy gap for any new experiment added later —
  there is no automated check that would catch a repeat of this, only discipline.

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
  caught later it is a `FileNotFoundError` on every UID. **Confirmed correct across two
  runs**; this probe also gave the authoritative archive size (158.54 GiB total) that
  earlier docs' ~231 GB estimate turned out to be off from — see DATA_MODEL.md.
- **fp16 cache fidelity.** Cell 19 clears the in-memory cache to force a real disk
  round-trip, then prints the max absolute difference between the fp16 cached array and a
  fresh fp32 computation for 3 UIDs. Small diffs relative to the ~unit normalized-pose
  scale are the check that float16 caching isn't quietly degrading inputs.
- **One-batch smoke check.** Section 6a runs a single batch through the loader before any
  training loop, so shape and collate errors don't surface mid-epoch.
- **Non-finite loss raises immediately** rather than training on (see PHASES.md).

Run cells 0-8, stop, read the output, and only then continue. Running all 61 cells blind
wastes the session's GPU quota on a failure that cell 8 would have shown in five minutes.

## Post-run verification — done for both completed runs

The bounded run (2,000/300/300) and the scaled-up run (10,000/1,500/1,500) both
completed and produced the expected checkpoint, report, and knowledge-repository
artifacts. The exact run metadata and numerical results — train/validation/test sizes,
experiment names, best validation losses, WER/BLEU/ROUGE-L, Text->Pose DTW/MSE/motion
energy, round-trip observation, latency, and the Section 17 `GO CANDIDATE`/`NO-GO`
verdict for both runs — are recorded in `research/outputs/phase0/RESULTS.md`. This
requirement is satisfied; treat `RESULTS.md` as the artifact that discharges it, and
append to it (don't replace it) if a future run adds more evidence.

**What this verification found, in test-strategy terms:** artifact existence was never
in doubt for either run (checkpoints, DB, manifest all wrote correctly both times); what
the verification step actually caught was two evaluation-methodology problems (the
degenerate baseline, the missing motion-energy check) that would otherwise have let a
misleadingly positive `GO-CANDIDATE` stand unchallenged. That is exactly what this
section of the test strategy is for, and it worked as intended.

## Reproducibility as a verification artifact

Every run writes its own record to `signsage_knowledge.db` (see DATA_MODEL.md): seed,
device, run mode, dataset row counts, split sizes and leakage flag, hyperparameters, all
metrics, and checkpoint paths. That is what makes a reported number checkable rather than
merely asserted — a metric with no recorded experiment, split, and dataset version behind
it cannot be defended. If you produce a number that isn't in the repository, that's a gap
in the test strategy, not just missing bookkeeping. (One instrumentation gap currently
exists: `POSE_NORMALIZATION_STATS`, a counter for rare degenerate-pose edge cases added
alongside the motion-energy check, is tracked in-kernel but not yet printed or logged to
the DB — harmless to any result to date, but worth wiring up before treating this
reproducibility story as fully closed.)

## System-level checks (once backend/frontend exist)

- One smoke test per direction: does a known landmark sequence produce *a* text output
  without crashing; does a known text input produce *a* pose sequence without crashing.
  Not correctness — just "the pipeline didn't fall over." (Given Phase 0's results,
  expect these smoke tests to pass while producing low-quality output — don't confuse
  "didn't crash" with "works," the same distinction Phase 0 itself had to learn.)
- Latency measurement harness for SMART objective 2 (end-to-end, on defined target
  hardware) — this should be a real, repeatable script, not a one-off manual stopwatch
  measurement, since it's a stated thesis deliverable. Phase 0's model-only latency
  numbers (Sign→Text ~21-55ms, Text→Pose ~14-16ms depending on run) are a floor, not an
  estimate of end-to-end latency — they exclude client-side extraction, network, and any
  LLM gloss call entirely.

## Device/environment testing (easy to skip, don't)

- Test MediaPipe Holistic performance on a budget Android phone, not just your dev
  laptop — WASM performance on weak hardware can drop well below usable frame rates,
  which undercuts any "works on any device" claim.
- Test in more than one lighting condition/room — landmark quality is sensitive to
  this, and the target users are not going to have your desk setup.

## What's explicitly not being done

- No unit-test-everything mandate for a solo research prototype — write tests for logic
  that's non-obvious or has bitten you before (the trivial-baseline and motion-energy
  checks above are exactly this pattern: added because something non-obvious bit us,
  not proactively), not for the sake of coverage numbers.
- No load/stress testing — there's no concurrent-user scenario in the MVP scope.
