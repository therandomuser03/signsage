# SignSage — Sign Recognition Expansion Plan

Date: 2026-10-12. Status: investigation + low-risk fix done; training NOT started.
Constraints honored: no training launched, `results.zip` and all checkpoints
untouched, deployment gate intact, no metrics fabricated, no auth/DB/UI changes.

## 1. Exact current recognition limitation

`backend/app/ml/inference/heuristic_signs.py` recognizes **5 static one-hand
shapes** by majority vote: open palm→Hello, thumbs-up→Yes, fist→No,
index-only→You, index+middle→Help. Everything else — including THANK_YOU,
SORRY, PLEASE, WATER, FOOD and all motion/contact signs — is **correctly
rejected** (`unrecognized_shape` / `no_hand_detected`). That is a scope
boundary, not a malfunction: THANK_YOU has no recognition method today, and
none was invented. The transformer path remains NO-GO behind its gate.

## 2. Bugs found, reproduced, fixed (with tests)

**BUG (fixed): right-hand shadowing — `classify_frame` rejected supported
left-hand gestures.** When the right-hand slice was present but
unclassifiable, the old code returned its `None` verdict without ever
examining the left slice — so a supported left-hand shape plus any
unclassifiable right-hand content (second hand, ghost detection) was
rejected. Single-hand frames were unaffected (intermittent, not systematic).

- Reproduction: frame with right=ring-only + left=fist returned `None`
  (expected `NO`). Test failed before the fix, passes after.
- Fix (`heuristic_signs.py::classify_frame`): right hand first; fall through
  to the left hand when the right yields `None`. Right-hand priority for
  two-supported-hands frames is preserved.
- Regression tests (`backend/tests/unit/ml/test_heuristic_signs.py`, 11 pass):
  shadowing case, priority-preservation case, single-hand/empty invariance.

**Investigated and cleared (evidence in code, no change):** MediaPipe finger
tuples all correct (thumb 2/3/4, fingers MCP/PIP/tip); z-scale cannot break
the scale-invariant 1.15 extension ratio; presence threshold (≥11/21) matches
the frontend's all-or-nothing 21-point hands; mirror/handedness cannot
systematically reject (either slice accepted single-handed); temporal vote
(10 frames / 0.6 agreement) accepts a clean 2 s hold; schemas/session pass
frames through unmutated; `compute_motion_energy` is reporting-only on this
path; static poses are immune to the motion gate by design. Known latency
quirk (not a bug): rolling interim starts at 80 buffered frames (~8 s), so
short demos only resolve on Stop/finalize.

## 3. Reusable datasets, checkpoints, cached landmarks

| Artifact | Path | Size / notes |
|---|---|---|
| iSign transcripts | `staging_kaggle_results/isign/iSign_v1.1.csv` | 9.4 MB, 127,237 rows, header `uid,text` only — **no signer IDs**; `uid=<video_id>--<seq>` (video split possible, signer split not) |
| Pilot pose cache | `staging_kaggle_results/s2t_v2/pose_cache/` | 13.6 GB, manifest n=7500 complete — random pilot uids, contains almost none of the §5 vocab (overlap: 1–2 rows each) |
| Pilot tokenizer | `staging_kaggle_results/s2t_v2/tokenizer/` | SP model 527 KB, pilot-scoped — retrain for any new vocab |
| Dataset audit | `staging_kaggle_results/s2t_v2/dataset_audit.json` | 123,779 usable rows / 3,797 videos, seed-42 video split 98,485/13,014/12,280 |
| 352868924 contracts | `research/evaluation/reports/352868924/` | landmark + deployment JSONs (representation reference, NOT promotable: NO-GO) |
| Trainer (canonical) | `research/training/train_sign_to_text.py` + `signsage_kaggle.ipynb` | flags: `--subset/--epochs/--batch/--lr/--decoder-input-dropout/--data-mode/--max-hours/--out/--resume/--smoke`; `export_bundle()` (l.1397) auto-runs on `export` stage |
| Servability check | `backend/scripts/model_check.py` | `python -m scripts.model_check`: loads bundle with `allow_unpromoted=True`, zeros-(10,543,3) must be REJECTED |
| Collapsed checkpoints | `results.zip:s2t_v2/checkpoints/*.pt` | 2× ~203 MB raw trainer snapshots, empty `model_config` — unloadable by the serving validator; reference only |
| Disk (this machine) | `/mnt/D` 92 GB free, `/tmp` 2.6 GB free | No GPU, 7 GB RAM — **no training here** ( Strom CPU-only laptop excluded by constraint) |

## 4. Root cause of the previous collapse (evidence-backed)

1. **Memorization regime:** s2t_v2 trained ~17 M params (d256/4L) on a
   `--subset 2500` pilot (~7k params/sample). The 64-sample overfit gate
   passed (95.3%) while held-out collapsed one epoch later — the textbook
   memorization signature, and it reproduced in an independent prior run
   (same epoch-2 shape, top-pred 63–70%).
2. **Same disease at smaller scale:** the 4 M-param 352868924 run never
   tripped the guard but degenerated anyway (0.99% output diversity,
   top-prediction 44.8%, BLEU below the trivial baseline) — collapse tendency
   tracks the capacity/data mismatch, not a seed.
3. **Untested exposure-bias path:** teacher forcing + greedy decode with
   `--decoder-input-dropout` defaulting to **0.0**; the mitigation was added
   post-collapse and has never run in a non-collapsed experiment.
4. **Supervision mismatch:** iSign labels are multi-word educational sentences
   (single-word rows are mostly fingerspelling/junk: "b", "a", "blank",
   "dash"); the avatar vocabulary has ~zero isolated examples in the data
   (`thank you`=2 rows, `sorry`/`yes`/`food`=1, `hello`/`please`/`help`=0).
   Sentence-level 16k-vocab seq2seq from poses on this budget is the wrong
   task shape — more epochs/data of the same kind will not fix it.

## 5. Proposed vocabulary and data split

**Core vocab (6 classes, 74 rows, every class multi-video) — genuine,
counted 2026-10-12 from the CSV:**

| Class | Rows | Videos | Static/temporal |
|---|---|---|---|
| YOU | 21 | 10 | temporal (needs sequence) |
| MUSIC | 20 | 18 | temporal |
| THREE | 12 | 12 | static-capable (number handshape) |
| ONE | 9 | 6 | static-capable |
| WATER | 8 | 8 | temporal |
| TWO | 4 | 4 | static-capable (thin — floor class) |

- **Static vs temporal:** number handshapes can be classified from static
  poses; YOU/WATER/MUSIC require multi-frame sequences (keep the seq2seq
  architecture so the serving path is unchanged).
- **Explicitly excluded:** THANK_YOU (2 rows, temporal, unlearnable here —
  support is NOT claimed); single letters A–E (17–36 rows each) pending
  video spot-checks that they are fingerspelling, not lesson labels; every
  other avatar-vocab word (≤1 row).
- **Split (no leakage):** filter CSV to the 6 classes (builder below) and pass
  it as `--csv` — the trainer's existing `video_split(seed=1)` groups by
  `video_id` with no video across splits (signer IDs don't exist; video split
  is the only available control). The split is recorded in the run's
  `subset_manifest`-style audit. Expected: ~52/11/11 rows.
- **Poses:** extract only the subset's clips via the trainer's `stream`
  data-mode (range reads against the HF pose archive; preflight-proven
  reachable). Estimate <100 MB cache into the persistent `--out` dir —
  no 13.6 GB reuse possible (no overlap), no large-archive extraction.

## 6. Resource estimate and explicit stop conditions

- **Where:** Kaggle GPU (T4), never the local laptop. **Budget: ≤4 GPU-hours**
  (`--max-hours 4`); small-arch model d128/2L pre-norm (~1–2 M params, ~10×
  smaller than the collapsed run — applied as the explicit patch below, since
  `D_MODEL/NHEAD/NLAYERS/FF` are module constants, not flags), batch 32,
  epochs ≤8, new SP tokenizer trained on subset texts only (~64 pieces),
  `--decoder-input-dropout 0.2`, label smoothing 0.1 (existing constant),
  LR 1e-4 + warmup/cosine (existing scheduler).
- **Stop (first hit ends the run):** 4 h budget · existing `DIVERSITY_FLOOR`
  collapse gate fires · val exact-match plateaus 2 epochs (patience 2) ·
  overfit-gate (<95% exact on 64 train samples) fails before epoch 1 ends.
- **Persistence:** `--out` on Kaggle working disk + copy checkpoints, logs,
  metrics and `subset_manifest.json` to a Kaggle Output dataset every epoch;
  `df` check before any download; `results.zip`, existing checkpoints and
  staging artifacts are read-only inputs, never overwritten.

## 7. Evaluation and deployment-bundle acceptance criteria (all required)

On the video-held-out test split, with the majority-class baseline
(mode-copy, ≈28% EM on this vocab) computed by the trainer's existing
trivial-baseline code:

1. Exact-match **≥70%** AND **≥20 pts above** the baseline.
2. Per-class recall **≥50% for every one of the 6 classes** (exposes a
   memorizing or single-class-collapsed model even if EM looks fine).
3. Output diversity: unique predictions covering **≥5 of 6 classes**
   (`unique_fraction ≥ 0.5` on test decodes — scaled to the 6-class task).
4. Overfit gate passes; `export_bundle()` succeeds; **`model_check.py`
   exits 0** against the exported bundle (load OK + zeros rejected).
5. Promotion conjuncts are computed and reported, but
   **SIGNSAGE_MODEL_PROMOTED is not flipped by the experiment** — promotion
   stays a separate human decision under `backend/models/README.md`.

If any criterion fails, the result is a failed experiment with logs kept —
not a bundle, not a fallback-vocab addition. No word enters the served
vocabulary without a recognition method meeting §7 (satisfies §8: the app
interface — `TranslationService.transcribe` / capabilities fields / WS
protocol — is unchanged; only `FALLBACK_VOCABULARY` membership or the bundle
behind the existing gate may grow).

## 8. Exact commands for the bounded experiment

Build the filtered subset (run locally — CSV only, seconds):

```bash
python3 - <<'EOF'
import csv
VOCAB = ['you','music','three','one','water','two']
rows = list(csv.DictReader(open('staging_kaggle_results/isign/iSign_v1.1.csv')))
sub = [r for r in rows if r['text'].strip().lower() in VOCAB]
vids = {r['uid'].rsplit('--',1)[0] for r in sub}  # rsplit: uid may contain '--'
with open('subset6.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['uid','text'])
    w.writeheader(); w.writerows(sub)
print(len(sub), 'rows,', len(vids), 'videos')
EOF
```

Pre-step (experiment branch only — arch size is a module constant at
`train_sign_to_text.py:88`, not a flag; everything else below is a real flag):

```diff
-D_MODEL, NHEAD, NLAYERS, FF = 256, 4, 4, 1024
+D_MODEL, NHEAD, NLAYERS, FF = 128, 4, 2, 512  # small6 experiment only
```

Train on Kaggle GPU (small model, dropout mitigation on, hard time budget;
poses stream-extracted for the subset only, everything persisted to `--out`):

Train on Kaggle GPU (small model, dropout mitigation on, hard time budget;
poses stream-extracted for the subset only, everything persisted to `--out`):

```bash
python research/training/train_sign_to_text.py \
  --csv subset6.csv \
  --data-mode stream --cache-budget-gib 2.0 \
  --epochs 8 --patience 2 --batch 32 --lr 1e-4 \
  --decoder-input-dropout 0.2 \
  --max-hours 4.0 --seed 1 \
  --out /kaggle/working/s2t_small6 --stages all
cp -r /kaggle/working/s2t_small6 /kaggle/output/   # every epoch / at end
```

Serve-check the output (any machine with the repo env; bundle path from `--out`):

```bash
SIGNSAGE_MODEL_BUNDLE_DIR=/kaggle/output/s2t_small6/signsage_deployment_bundle \
  python -m scripts.model_check   # expect: Loaded bundle OK + activity gate passed, exit 0
```

Then evaluate §7 against `evaluation_report.json` in the output dir. Notebook
alternative: `research/notebooks/signsage_kaggle.ipynb` cells measure →
certify → prepare (with the subset CSV) → train with the flags above.
