# SignSage — Models and Data

Every dataset and model the project uses or has evaluated, what each is allowed to claim, the gates a model must pass, and how the FDMSE-ISL training pipeline works.
Metrics belong to a **task and dataset**. Isolated-sign accuracy, continuous gloss WER and sentence BLEU measure different things and must never be compared as if they were the same.

## 1. Task levels

| Level | Example | Where it is used |
|---|---|---|
| Static image | A hand-shape photo → letter | Alphabet classifier |
| Isolated sign (video) | 2–3 s clip → one word | FDMSE-ISL and `word_v3` recognisers |
| Continuous signing | Sentence-length video → gloss or English | Gated transformer path; research candidates |
| English → ISL | Text → gloss IDs → avatar motion | Text → Sign |

Never merge these into one task without an explicit definition. The 14 avatar gloss IDs are **generation** labels, not recognition labels.

## 2. Datasets

| Dataset | Used for | Size and facts | License / terms | Status |
|---|---|---|---|---|
| **FDMSE-ISL** | Isolated-word recognition (primary) | 2,002 words; 40,033 videos; 20 signers; ≈30 GB archive; official split train 20,016 / val 4,003 / test 16,014 videos, signers 10 / 2 / 8, disjoint | Access by key from the authors' portal; used with permission for this project; research use; do not redistribute | **In progress** — §3 |
| **iSign v1.1** | Continuous sign → text research | CSV `uid`, `text`; 127,237 rows and `.pose` files; no signer column | CC BY-NC-SA 4.0 (gated). Derived weights do not inherit MIT | Previous transformer trained on it and collapsed |
| `isl-isolated-40words` (aggregate) | Early 6-word pilot | 642 clips, 40 glosses, mixed sources; only 122 INCLUDE clips used | Mixed; INCLUDE subset CC-BY-4.0. Do not assume every source inherits it | Superseded by FDMSE-ISL |
| ISLRTC dictionary videos | Avatar clip source (HELLO, THANK_YOU) | 21 clips each, multiple signers | Reference material; check terms before redistribution | In use for clips |
| RealSign alphabet images | Alphabet classifier | 26 classes; 18,198 / 2,579 / 5,198 train/val/test images, 64×64 | CC0-1.0 | Verified |
| Mendeley "ISL real-life words" images | Static word classifier | 20 classes, ≈18,000 images | Check source | Unreliable (19.0% accuracy) |
| Local signer recordings (`data/videos/`) | Avatar clips (WHERE) and tests | One signer, low resolution; gitignored | Own recordings | Single-sample quality |

**Candidates for expanding training later** (none downloaded or audited yet; audit license, signers, splits and label overlap before use):

| Dataset | Why it is interesting | Caveat |
|---|---|---|
| INCLUDE (≈ 4,284 videos, 250 categories) | Real isolated-sign video | License is stated inconsistently (CC-BY-4.0 for the pilot subset, CC BY-NC-SA 4.0 elsewhere) — verify at the source |
| CISLR | Isolated-word clips (81 seen in the aggregate) | Research license; no local archive |
| ISLRTC dictionary | Reference signs for avatar and vocabulary checks | Reference material, not training labels |
| Other ISL isolated-word sets | More signers and words | Map labels to the FDMSE-ISL vocabulary; keep sources separable so each can be credited and reported |

Using several datasets must keep each one's provenance, split and credit; report how many videos came from each.

Rules for any dataset: keep provenance, original labels, license and preprocessing metadata together; keep splits reproducible and leak-free (split by video, and by signer where metadata allows); never commit data, archives or caches; never claim a dataset is available locally because a link or manifest exists.

## 3. FDMSE-ISL

**Citation** — please keep it in every report and in the product credits:
Patra et al., *Hierarchical Windowed Graph Attention Transformer Encoder and a Large Scale Dataset for Indian Sign Language Recognition*, Pattern Analysis and Applications 28(3):148, 2025 (BibTeX in [references.bib](references.bib)). The paper's 93.86% top-1 is the authors' graph-attention model on their data; it is not a claim for SignSage's models.

**Access.** The portal sends a time-limited key by email after approval. The key lives only in a Colab secret named `FDMSE_ISL_KEY` or an environment variable of the same name; it is never printed, committed or pasted into a cell. Written permission from the authors should be kept with the project. Current permission clearly covers the college project; confirm in writing that a **public, free, non-commercial thesis app and trained weights** are covered before opening to the public.

**Usage rule (project policy).** Use the whole dataset — all 2,002 words and all signers — and record how many videos were used per split, how many failed, and why. Do not silently train on a convenient subset.

**Splits.** Report metrics on the official signer-independent split: select a model on validation, evaluate on test **once**. The deployment model is then retrained on all signers with the same recipe; the reported test number stays from the first run.

**Known issue in the earlier 200-word notebook.** Its vocabulary "ranking" used a coverage score on which 2,001 of 2,002 classes tie, so it kept the first 200 words alphabetically (digits, then a–"bhutan"). Everyday words such as "hello" were never included. The v2 pipeline uses all classes.

## 4. Models

| Model | Task | Status | Evidence / decision |
|---|---|---|---|
| Alphabet CNN (`SmallCNN`, A–Z) | Static image | **Verified** | 87.1% test accuracy, macro-F1 0.849; weak: W (0% recall), Z (25%), S (50%) |
| Static word CNN | Static image | Unpromoted | 19.0% accuracy, heavy overfit; do not present as reliable |
| `word_v3` | Isolated word, 9 words, one signer | **Experimental** | Not promoted; not signer-independent |
| Earlier 6-word pilot (`isl_bundle`) | Isolated word | Superseded | 61.1% accuracy on 18 test samples; per-class recall: friend 0%, hello 100%, hospital 66.7%, market 100%, school 33.3%, thank you 66.7% |
| FDMSE-ISL recogniser (BiGRU baseline) | Isolated word, up to 2,002 words | **In progress** | See §7 |
| `352868924` SignTransformer | Continuous sign → text | **NO-GO** | WER 1.0046, BLEU 0.24 (below the trivial baseline 0.40), exact match 0, diversity 0.99%; trained on synthetic text→pose data, not real ISL video. Kept as a negative reference |
| Checkpoint-free heuristic | 5 static one-hand shapes | Fallback (to verify in code) | Open palm→Hello, thumbs-up→Yes, fist→No, index→You, index+middle→Help |
| External `t5-small-motion` | Continuous | **NO-GO** | 0 / 1,887 exact match; 21% of outputs "isn't that amazing!" |
| ISLiM | Continuous (94-landmark input) | Not integrated | Loads; smoke test only; input contract differs from the 543-point pipeline; no paired evaluation data |
| Sanket | Isolated/category, video + pose | Rejected | Video input required; task is not continuous translation; weights deleted locally |
| Sid-V5 (ST-GCN + Transformer + CTC) | Continuous gloss | Research candidate | Reports 30.77% WER on gloss recognition; checkpoint not downloadable when tested; needs a separate gloss → English stage |
| ISH-NEWS, multilingual pose-based transformer | Sentence level | Research leads | No verified public weights |

Weights are never committed; the full inventory of what is kept locally is in the repository's model README.

## 5. Evaluation history (what was learned)

| Run | Outcome |
|---|---|
| Bounded (2,000 / 300 / 300) | Pipeline worked; no useful model |
| Scaled (10,000 / 1,500 / 1,500) | Severe mode collapse; an early "GO-CANDIDATE" label was retracted and must never be cited as readiness |
| Text → Pose experiment | Motion energy ≈ 6.7% of ground truth — low DTW/MSE hid near-static output |
| Alphabet classifier | Held-out verified; Playwright E2E passes on real fixtures |
| External T5 | NO-GO (above) |

Lessons: a model that **overfits 64 samples is not a model that generalises**; loss decrease, fluent text and a successful load prove nothing; always compare against a trivial baseline and check output diversity.

## 6. Promotion gates

A recognition model is promoted only when **all** hold, in this order:

1. Real weights beat a trivial baseline and show output diversity on held-out data (collapse gate: unique fraction < 0.05 after epoch 2 stops training).
2. Documented evaluation with baselines, per-class results and qualitative failure review. Isolated-word bar so far: **80% accuracy, 80% macro recall, 60% per-class recall** (50% for expanded vocabularies).
3. Signer-independent evaluation, or an explicitly accepted written limitation.
4. Calibrated confidence floor and a test that a wrong or empty input is **rejected** (today the floor ships at 0.0).
5. Preprocessing is numerically identical between training export and serving (golden-vector test) and landmark order is verified at load.
6. Browser landmarks are shown to match training landmarks (extractor parity).
7. Bundle export, `model_check.py` passes, adapter and API tests pass.
8. Licence review (§7) and an explicit human release decision.

Then, and only then: set the bundle directory and `SIGNSAGE_MODEL_PROMOTED=true`. The honest default state is `MODEL_UNAVAILABLE`.

## 7. FDMSE-ISL training pipeline (v2)

Notebook: `SignSage_01_FDMSE_Extraction_v2.ipynb` (suggested home: `research/FDMSE/`). A training notebook that reads only the saved features is next.

**Why it was rebuilt.** Landmark extraction (MediaPipe on CPU) is the slow step — hours for thousands of videos — and the earlier notebook saved nothing until the end, so a Colab limit wasted a night.

**How it is built**
- **Streaming or local archive.** Videos are read one at a time from the portal with HTTP range requests (no 30 GB download), or from an archive you already downloaded (`LOCAL_ARCHIVE`, no key needed), or from a one-time resumable download if the portal refuses ranges. A pre-flight check says which, and explains a rejected key in plain words.
- **Chunks.** Work is cut into deterministic chunks of 50 videos; each finished chunk is written atomically (`chunks/chunk_#####.npz`). A disconnect costs at most the chunks in progress. The class order is interleaved so a partial run still covers every class.
- **Raw landmarks are kept** (pose, hands, face on every K-th frame), so features can be rebuilt without re-extracting.
- **Locked settings.** `N_FRAMES`, `FACE_EVERY_K`, `POSE_MODEL`, `MAX_SIDE` and `CHUNK_SIZE` are stored in `meta/extraction_config.json`; a later run with different values refuses to continue. Defaults: 48 frames, face every 4th frame, full pose model, 720 px. The notebook's benchmark cell measures the real cost and the feature change versus face-every-frame — decide before the first real chunk is saved.
- **Two machines.** `MACHINE_INDEX` / `MACHINE_COUNT` split chunks by modulo; chunk files from either machine merge by copying. Use a **CPU** runtime — a GPU does not speed this up.
- **Features.** `sign_features_v3` (60 × 122) built per chunk into `features/feat_chunk_#####.npz`, with per-clip usability flags (pose and hand visibility) instead of silent drops.
- **Hands** are stored in detection order, not by MediaPipe's Left/Right label, matching how `sign_features_v3` ignores labels.
- **Dry run.** `DRY_RUN=True` exercises the whole pipeline with a fake detector in a separate output folder.

**Planned training and evaluation.** BiGRU baseline over `(60, 122)` features; model selection on validation macro-F1; test once; per-class report and confusion matrix; export `model.pt`, label map, preprocessing and metrics; then retrain on all signers for deployment. Letter and digit classes in the dataset are to be checked for the fingerspelling fallback.

## 8. Licence gate

Code is MIT. Weights trained on iSign inherit CC BY-NC-SA 4.0 obligations; FDMSE-ISL weights follow the authors' terms. Before any public deployment record: (1) whether non-commercial use covers a hosted app, (2) whether serving predictions counts as commercial use, (3) attribution and share-alike duties for the bundle, (4) whether tokenizers derived from the data need the same treatment. Local research is not blocked; presenting a checkpoint as commercially cleared without this review is.
