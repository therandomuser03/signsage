# SignSage Training Readiness Report — run v2 (post-352868924)

Date: 2026-10-02. Status: READY TO RUN on Kaggle GPU. No full training has been
started from this report; the expensive run is the operator's explicit next step.

## 1. What was wrong with 352868924?

**Not a code bug.** Masking, teacher-forcing shift, `ignore_index=PAD`, causal
mask, and the tiny-overfit gate (loss 0.0013 / 100% exact on 64 samples) were
all verified correct. Data was clean: continuous pose→English task (not
isolated signs), leak-safe split by `video_id` (99,699 / 11,927 / 12,153),
representation parity with serving (`pose_normalize_v1`, byte-identical port).

**Failure mechanism: severe under-training of a small post-norm Transformer on
a hard, diverse task.** 3 epochs (~4,674 steps) on 99.7k diverse pairs with a
d192×3-layer post-norm model; validation loss was still falling (5.20/5.26,
perplexity ~190) when the session budget stopped it. The decoder never left
the target-language prior regime, so greedy decoding samples the sharpest
prior mode: WER 1.005, BLEU 0.24 (worse than the trivial baseline's 0.40),
0.99% output diversity, top hypothesis covering 44.8% of test outputs.
Contributors: post-norm dynamics on a tiny step budget, zero-padded 160-frame
inputs with face-masked zeros weakening encoder signal, 16k-class output head.

## 2. What exactly changed (v2 config)?

| Aspect | 352868924 | v2 | Why |
|---|---|---|---|
| Norm placement | post-norm | **pre-norm** (`norm_first=True`) | stable gradients from step 0 on short budgets; same tensor shapes, backend loads via `model_config.norm_first` (default False = old bundles unchanged) |
| Capacity | d192 / 3L / ff512 (~4M) | **d256 / 4L / ff1024** (~15M) | 100k pairs × 16k vocab needs headroom; still fast on T4 |
| Label smoothing | 0 | **0.1** | dampens prior overconfidence (standard NMT practice) |
| Peak LR | 2e-4 | **3e-4** | pre-norm tolerates it; same 6% warmup + cosine, clip 1.0, AMP |
| Epochs | 3 (budget cutoff) | **up to 12, patience 3** on val loss | val loss was still falling at stop |
| Collapse guard | none (post-hoc) | **diversity gate every val epoch**: unique-frac on 512 val decodes; STOP if < 0.05 after epoch 2 | fail fast instead of burning GPU on a collapsed run |
| Overfit gate | notebook cell (pass) | **hard gate in script**: N=64 must reach ≥95% exact in 600 steps or abort | same proven check, now blocking |
| Implementation | notebook cells | **`research/training/train_sign_to_text.py` importing the backend's own `SignTransformer`, preprocessing, and masks** | single canonical implementation (§5 parity by construction); export layout identical so the serving stack loads v2 unchanged |

Unchanged deliberately: task (continuous iSign pose→English), video-level split
(seed 42), tokenizer procedure (SP unigram 16k, PAD0/UNK1/BOS2/EOS3), shift,
masks, `ignore_index`, batch 64, seed, decoding constraints, bundle layout.

## 3. Why should v2 avoid the collapse?

The collapse was prior-dominance from an under-trained decoder, not a structural
defect. v2 attacks each contributor: pre-norm + smoothing weaken the
sharp-prior attractor; 3–4× capacity and up to 4× steps give the decoder room
to actually condition on the encoder; the diversity gate converts a silent
8-hour collapse into a loud early stop. If v2 still collapses, the gate fires
by epoch 3 and the report says so — no sunk-cost full run.

## 4. Data / representation / targets

- iSign v1.1 `iSign_v1.1.csv` (HF `Exploration-Lab/iSign`, gated, HF_TOKEN) +
  4-part pose ZIP (158+ GiB, streamed, never fully downloaded).
- Cleaning: malformed UID (2,291) + junk captions (1,167) removed; duplicates
  refused; video-level 80/10/10 split. Audit: `audit_dataset.py`
  (`--self-test` passes locally; full run on Kaggle writes the JSON report).
- Representation: 543×3 raw → `pose_normalize_v1` → 160×1629 → model delta
  concat (3258). Identical code serving and training (backend import).
- Targets: SP-16k on train text only, BOS+EOS, ≤96 tokens (truncations ≈ 0).

## 5. Validation / promotion

Validation: val loss (checkpoint selection) + diversity gate (collapse stop).
Test (held-out, same split as 352868924 for direct comparison): WER, BLEU-1..4,
ROUGE-L, exact match, unique-prediction ratio, length stats, sample triples
incl. difficult cases, trivial-baseline comparison. Promotion stays manual:
`SIGNSAGE_MODEL_PROMOTED=true` only after the quality gate passes
(diversity ok, beats trivial on WER *and* BLEU, E2E + latency verified).

## 6. Expected GPU/time cost

Baseline throughput: 3 epochs in ~8.75h on 2×T4 (≈3h/epoch at batch 64, AMP).
v2 is ~3× params with identical sequence shapes: estimate ~3.5–4h/epoch.
Budget guard `--max-hours 8` stops gracefully after epoch checkpoints; resume
with `--resume` (recovery snapshot + best checkpoint reloaded from the output
dir; snapshots are per-epoch, so a preemption can lose up to ~one epoch).
Worst case (early collapse): ~2 epochs wasted,
not 8 hours. Expected full run: 2–3 Kaggle sessions to 6–12 epochs.

Data-pipeline note (2026-10-07 rework): the old workflow streamed poses over
HTTP range reads every epoch AND pre-tokenized ~99k captions every session,
which bottlenecked the run in preparation. The script now defaults to
`--data-mode stream` (FULL CACHE DISABLED: no materialization; every epoch
re-reads remotely, priced honestly by the feasibility gate) and supports
`--data-mode cache` (one-time resumable `pose_cache/` of verbatim `.pose`
bytes in sharded ZIPs + `tokenizer/tokens.npz`, both fingerprinted and reused
across sessions). Processed fp32 tensors (123,779 × 160 × 1629 ≈ 120 GiB,
≈60 GiB in fp16) are deliberately NEVER cached -- both exceed practical
Kaggle disk. Cache mode prints a projected total size after 200 samples and
REFUSES (keeps the partial cache) when it exceeds `--cache-budget-gib`
(default 10 GiB) or 90% of free disk. The overfit gate and the diversity gate
both reuse the token cache (no re-encoding) and run memoized against the local
cache when present.

Measured on Kaggle 2026-10-07 (123,779 usable; train=98,485 val=13,014
test=12,280): ~1,211 KiB/sample -> **~143 GiB full cache at 1.2 samples/s
sequential (~29h -- does NOT fit one session)**. Mitigations, in order:
(1) `--jobs 8` parallel fetch (default; expect ~4h, watch the printed ETA);
(2) let prepare span sessions (it resumes from the manifest); (3) explicit
`--subset N` pilot run (deterministic random N rows within each seeded split,
seed 42, loudly logged + recorded in dataset_audit.json -- representation
unchanged, just fewer videos) when the full cache cannot be held. The notebook
measure cell auto-selects the largest N fitting the measured quota with a
1.5h margin (certify cell re-scopes feasibility to the pilot before training). A ~143 GiB cache also strains Kaggle output-dataset persistence, so
confirm the saved output version actually contains `pose_cache/` before
relying on cross-session reuse.

### D-cache sizing: raw 143 GiB vs fp32 120.2 GiB vs fp16/int16 (rejected)

- Raw `.pose` bytes measured ~1,211 KiB/sample -> ~143 GiB for 123,779
  samples (Kaggle, 2026-10-07). The archive barely compresses, so "raw"
  means numerically safest, not smallest.
- Fixed 160x1629 fp32 tensors are exactly 123,779 x 1,042,560 B = 120.2 GiB
  -- smaller than raw with equal exactness, but still presumed too big.
- fp16 (60.1 GiB) and int16 are REJECTED without a validation experiment:
  both introduce train/serve skew (serving computes fp32; the velocity
  signal the model builds internally is exactly where quantization damage
  concentrates), and 60 GiB does not clear the persistence bar anyway.
- D-cache approval (fixed fp32 + lossless zstd) is CONDITIONAL on the
  empirical measurement below. No D-cache is built until the verdict is
  reviewed.

### Measurement + quota gate (run this before any prepare; minutes, no training)

```bash
python research/training/train_sign_to_text.py --csv $CSV --pose-source remote \
    --out /kaggle/working/s2t_v2 --stages measure --measure-samples 2000
# optional: --persistence-target-gib <your measured budget> for a FIT verdict
```

The `measure` stage (opt-in only, never in `all`) stride-samples ~2,000 items
across the authoritative train/val/test row order, runs each through the
exact V2 preprocessing (timed: `fetch_prep_per_sample_s`), zstd-compresses the
fixed fp32 tensor (level 3) with byte-exact round-trip verify, trains + times
the real 16k tokenizer (persisted, reused by later stages -- no retraining),
benchmarks real-config training steps/sec and decode hyps/sec, and writes TWO
reports: `measure_report.json` (true_len stats, ratios, padding-vs-data
attribution, 123,779 + per-split projections, FIT / BORDERLINE / DOES_NOT_FIT
or UNKNOWN-no-target verdict) and `feasibility.json` (projected epoch + total
hours vs the 13h34m quota / 11.5h target, PASS / FAIL banner). It creates NO
pose cache and NO checkpoints. The `train` stage REFUSES to start on a FAIL
verdict unless `--force` is passed (loudly logged; quota WILL burn). Cache-mode
measure leaves the total NaN (prepare unpriced) -- once the cache is COMPLETE
the verdict is recomputed with sunk prepare cost and overwrites
`feasibility.json`, so a FAIL-unknown-prepare can still turn PASS.

**Do not start training until the feasibility verdict is known.**
FIT / PASS -> proceed. BORDERLINE / DOES_NOT_FIT / FAIL -> subset pilot or
rethink persistence first. Notebook order is measure -> prepare -> train;
prepare needs `--data-mode cache` explicitly (stream is the default and
materializes nothing). The long run is `Save Version -> Save & Run All`
(batch execution is independent of the editor UI, so the ~40-minute
"Are you still there?" interactive warning is irrelevant -- never fake
activity to keep the editor awake).

## 7. Operator next step (exact)

On Kaggle (GPU T4×2), clone `https://github.com/therandomuser03/signsage.git`
(run the cells in `research/training/signsage_kaggle.ipynb` in order):

```bash
pip install pose-format jiwer sacrebleu rouge-score sentencepiece huggingface_hub requests tqdm pandas
# Provision iSign_v1.1.csv FIRST (notebook provisioning cell): an existing
# /kaggle/working copy, an attached Kaggle dataset (ISIGN_DATASET_SLUG), or a
# Hugging Face download (needs the HF_TOKEN secret). Training never starts
# without the CSV: the cell fails early otherwise.
python research/training/audit_dataset.py --csv $CSV --out $OUT/dataset_audit.json
# 1. MEASURE (minutes, fail-closed): writes measure_report.json + feasibility.json
python research/training/train_sign_to_text.py --csv $CSV --pose-source remote \
    --out $OUT --stages measure --measure-samples 2000
# 2. only on PASS/pilot decision: cache under a disk budget (refuses overruns)
python research/training/train_sign_to_text.py --csv $CSV --pose-source remote \
    --out $OUT --stages tokenizer,prepare --data-mode cache --cache-budget-gib 10 --jobs 8
# 3. train (refuses on FAIL unless --force); local cache auto-used when complete
python research/training/train_sign_to_text.py --csv $CSV --pose-source remote \
    --out $OUT --stages tokenizer,overfit,train --epochs 12 --max-hours 8 --resume
```

`--smoke` runs the CPU plumbing check (synthetic data, no dataset) — must pass
before any GPU spend. The overfit gate runs automatically and aborts the run
if it fails. Copy `$OUT` (bundle + manifest + report) back as the new
`research/artifacts/<RUN_ID>/` — never overwrite `352868924`.

### Resume across Kaggle sessions (read this: /kaggle/working is ephemeral)

`--resume` by itself does NOT survive a new Kaggle session. The exact workflow:

1. **End of session N:** use Kaggle's **Save Version** (save output) so the
    entire `$OUT` directory (`pose_cache/`, `checkpoints/`, `tokenizer/`,
    reports) is kept as a versioned output dataset. The notebook's persist
    cell lists exactly what will be kept; copying the `.ipynb` alone
    transfers nothing.
2. **Start of session N+1:** attach that saved output as an **input dataset**,
    then run the notebook's **restore cell** (it auto-detects
    `/kaggle/input/*/s2t_v2`; set `PRIOR_OUTPUT_SLUG` only if ambiguous).
    The cell copies the prior output to `/kaggle/working/s2t_v2` and reports
    exactly what is reused: pose-cache completeness, tokenizer + token cache,
    recovery epoch/best, best checkpoint.
3. **Session 1 runs measure first, then the prepare cell**
   (`--stages tokenizer,prepare --data-mode cache`):
    one-time resumable pose materialization under `--cache-budget-gib` +
    tokenizer/token cache (measure already built the tokenizer; it is reused).
    Interrupt any time; re-running skips present samples via the manifest.
4. **Training/eval cells use `--resume` and read the local cache**
    (`--stages tokenizer,overfit,train`, then `evaluate,export`).
    The recovery checkpoint now also stores bad-epoch count, history, and
    RNG states, so resume continues the schedule instead of restarting it.
