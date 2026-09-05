# SignSage — Phase 0 Results

*Covers two Kaggle runs on the same notebook lineage: the initial bounded validation run
(2,000/300/300 rows) and the scaled-up run (10,000/1,500/1,500 rows, 4h29m, GPU T4×2,
Run ID `20260903T134623Z`). Numbers below are read directly from `signsage_knowledge.db` /
`signsage_knowledge_manifest.json`, not just the printed cell output.*

## Status: GO-CANDIDATE (per the notebook's own weak, OR-based criterion) — with caveats that materially change what that label should be read as

The Transformer beats the trivial baseline on BLEU/ROUGE-L, which is technically enough to
satisfy PHASES.md's stated (deliberately weak) exit criterion. **This should not be read as
evidence of working translation or working pose generation.** Two specific, evidenced
problems explain why, detailed below. This is not a failure of the pipeline — the pipeline
correctly surfaced both problems instead of hiding them, which is the point of a bounded
validation run.

---

## 1. Sign → Text

| Model | WER | BLEU | ROUGE-L | Unique outputs / test set |
|---|---|---|---|---|
| Trivial baseline (`'1'`) | 0.999 | 0.00 | 0.005 | 1 |
| GRU baseline | 0.991 | 0.09 | 0.070 | 6 / 1,500 (0.4%) |
| Transformer (full) | **1.367** | 0.12 | 0.077 | 314 / 1,500 (21%) |
| Transformer (no face) | 1.097 | 0.11 | 0.074 | — |

**Finding 1 — the trivial baseline is still degenerate, and the reason why is a real dataset issue, not a sampling fluke.** The first run's baseline degenerated to an empty string; that bug is fixed (rows with blank text are now excluded before computing the mode). The corrected baseline still landed on `'1'` — a single digit. Checking the actual dataset (127,237 rows) confirms this isn't random sampling noise: after blank text (391 rows), the next most frequent exact "sentences" dataset-wide are `2` (99×), `1` (96×), `3` (87×), `5` (54×), `4` (49×), the literal word `blank` (46×), `B` (36×), `6` (32×), `[Music]` (30× — an auto-caption artifact), `A` (22×). Roughly **1% of the entire dataset is junk text** of this kind. **Fix:** filter the trivial-baseline computation (and arguably the training data itself) using the `n_words` column already present in the CSV — e.g. `n_words >= 2` or `>= 3` — not just non-empty text.

**Finding 2 — GRU mode collapse is unchanged by more data; Transformer diversity improved modestly.** GRU produces essentially one output ("the police have been arrested.") regardless of input, at both data scales (1.3% unique at 2k rows, 0.4% at 10k rows — if anything, relatively worse). This model is not learning input-conditioned translation. The Transformer improved from 10% to 21% unique outputs with 5x the data — a real, if small, positive signal that scale specifically helps the Transformer rather than the GRU.

**Finding 3 (new) — the Transformer's WER got worse at scale, and the qualitative outputs explain why.** WER rose from 1.035 to 1.367 despite BLEU/ROUGE-L improving slightly. Sample generated output at 10k rows: *"the police were upset to the police and the police and the government and the same level cricketer, politicians, hundreds"* — long, repetitive, run-on text. More training data made the underlying repetition problem more visible (longer generated sequences before hitting EOS/max length), not better. **This points specifically at the decoding strategy** (greedy decoding, no repetition penalty, no beam search) as the bottleneck, not data volume. Scaling data further without addressing decoding is unlikely to fix WER on its own.

**Finding 4 — the face ablation direction replicated.** At both 2k and 10k rows, removing the 468 face landmarks produced a *lower* validation loss than keeping them (9.66 vs 9.78 at 10k rows). This is now a repeated result across two independent runs at different scales, which makes it more credible than a single-run artifact. It cuts against the expected "non-manual markers help" narrative from SMART Objective 1 and deserves explicit discussion in any write-up — plausibly the 468-dimensional face block adds overfitting risk or needs different normalization, rather than being genuinely uninformative.

---

## 2. Text → Pose

| | Retrieval | Generative |
|---|---|---|
| DTW mean | 591.8 | **378.8** (lower/better) |
| MSE mean | 0.131 | **0.057** (lower/better) |
| Motion energy (mean frame-to-frame change) | 0.032 | **0.0025** |
| Ground truth motion energy | 0.038 | 0.038 |

**Finding 5 (confirmed, not just suspected) — the generative "win" is a static-pose artifact.** The first run's pose visualization looked visually frozen across all sampled frames; this run adds a quantitative check for exactly that, and it confirms it: generated motion energy is **6.7% of ground-truth motion energy**. The model has collapsed to predicting a near-static average pose. It still wins on DTW/MSE because outputting the "safe average" over time minimizes a position+velocity MSE loss without producing real motion — a well-known failure mode for this loss function without an adversarial or perceptual term. **Do not report the DTW/MSE numbers alone as evidence of working pose generation.**

**Round-trip check:** recovers `"the police were not be given to the police"` from the generated pose — consistent with, not independent evidence beyond, the Sign→Text mode-collapse finding above.

---

## 3. What this means for the thesis write-up

State plainly, in this order:
1. The end-to-end pipeline (data streaming, training, evaluation, provenance logging) runs correctly and reproducibly on real iSign data at two different scales — this is demonstrated and solid.
2. Sign→Text: GRU has fully collapsed to a single output; the Transformer shows a small, real improvement in output diversity with more data, but its WER is currently *worse* than a trivial baseline due to a decoding-side repetition problem, not (only) a data-scale problem.
3. Text→Pose: the generative model currently outputs a near-static pose. Its favorable DTW/MSE numbers are explained by regression-to-the-mean, confirmed via an explicit motion-energy check, and should not be presented as a working result.
4. The non-manual-marker (face) ablation produced a consistent, replicated finding — removing face landmarks trains better at this data scale — which is worth a real paragraph of discussion rather than being left as a bare number.
5. A real, previously undocumented data-quality issue was found and partially addressed: ~1% of the dataset consists of junk/placeholder text values (digits, single letters, caption artifacts), which contaminates any frequency-based baseline and likely also weakens training signal proportionally.

## 4. Recommended next actions, in order of cost

**Cheap, no retraining required:**
- Filter junk text (`n_words >= 2` or 3) from the trivial-baseline computation and from training data generally.
- Add a repetition penalty or simple n-gram blocking to greedy decoding — directly targets Finding 3.

**Moderate cost:**
- Beam search with length normalization for Sign→Text.
- Investigate face-landmark normalization/scaling specifically (Finding 4) rather than treating the ablation as settled.

**Deferred (explicitly, not silently dropped):**
- A loss function change for Text→Pose (e.g. an adversarial or perceptual term) to address the static-pose collapse — this is an architecture-level fix, appropriately scoped as future work rather than something to chase before a deadline.
- Larger-than-10k-row runs, once the decoding-side and data-cleaning fixes above are in, so the next scale-up isn't measuring the same known problems again.

## 5. Known gaps in this record

- `POSE_NORMALIZATION_STATS` (a counter added in this run for rare degenerate-pose edge cases) is tracked internally but was never printed or logged to the knowledge DB — an implementation oversight, not a data problem. Harmless to the results above; can be wired up in a future run if desired.
- This run's raw output bundle still includes `signsage_data/iSign_v1.1.csv` and HuggingFace cache files. These must not be committed to the repository (gated, CC BY-NC-SA 4.0 licensed dataset) — copy only `signsage_outputs/` into `research/outputs/phase0/`.
