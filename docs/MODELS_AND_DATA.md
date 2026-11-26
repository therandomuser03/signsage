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
| **FDMSE-ISL** | Isolated-word recognition (primary) | 2,002 words; 40,033 videos; 20 signers; ≈30 GB archive; official split train 20,016 / val 4,003 / test 16,014 videos, signers 10 / 2 / 8, disjoint | Access by key from the authors' portal; used with permission for this project; research use; do not redistribute | **Extraction complete (Nov 2026); training next** — §3, §7 |
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
| FDMSE-ISL recogniser `baseline_v1` (BiGRU, 1.54 M parameters) | Isolated word, 2,002 words | **Trained offline; not promoted** | Official signer-independent test split (8 unseen signers, 16,014 clips): **top-1 82.1%, top-5 95.1%**, macro-recall 82.1%; validation top-1 85.8%. Studio video only; webcam accuracy unmeasured. See §7 |
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
6. Browser landmarks are shown to match training landmarks (extractor parity: pose model, tracking vs per-image detection, camera aspect via `video_width`/`video_height`, 3 s capture resampled to 48 frames, webcam vs studio video). The browser now loads the **full** pose model, as in training; notebook 03 measures the effect of live-like frame counts, noise, missed hands, timing and aspect on the real test landmarks; the camera itself (lighting, background, angle) can only be measured with webcam footage.
7. Bundle export, `model_check.py` passes, adapter and API tests pass.
8. Licence review (§7) and an explicit human release decision.

Then, and only then: set the bundle directory and `SIGNSAGE_MODEL_PROMOTED=true`. The honest default state is `MODEL_UNAVAILABLE`.

## 7. FDMSE-ISL training pipeline (v2)

Notebooks (suggested home: `research/FDMSE/`): `SignSage_01_FDMSE_Extraction_v2.ipynb` (extraction and features) and `SignSage_02_FDMSE_Training_v2.ipynb` (training, which reads only the saved features). A third notebook, `SignSage_03_LivePath_StressTest_v2.ipynb`, replays the saved test-clip landmarks through the serving code under live-like conditions. Both are designed for Colab: extraction on a **CPU** runtime (the T4 does not speed it up), training on a CPU runtime or a GPU. Each notebook's first cell explains the settings.

**Why it was rebuilt.** Landmark extraction (MediaPipe on CPU) is the slow step — hours for thousands of videos — and the earlier notebook saved nothing until the end, so a Colab limit wasted a night.

**How it is built**
- **Streaming or local archive.** Videos are read one at a time from the portal with HTTP range requests (no 30 GB download), or from an archive you already downloaded (`LOCAL_ARCHIVE`, no key needed), or from a one-time resumable download if the portal refuses ranges. A pre-flight check says which, and explains a rejected key in plain words.
- **Chunks.** Work is cut into deterministic chunks of 50 videos; each finished chunk is written atomically (`chunks/chunk_#####.npz`). A disconnect costs at most the chunks in progress. The class order is interleaved so a partial run still covers every class.
- **Raw landmarks are kept** (pose, hands, face on every K-th frame), so features can be rebuilt without re-extracting.
- **Locked settings.** `N_FRAMES`, `FACE_EVERY_K`, `POSE_MODEL`, `MAX_SIDE` and `CHUNK_SIZE` are stored in `meta/extraction_config.json`; a later run with different values refuses to continue. Defaults: 48 frames, face every 4th frame, full pose model, 720 px. The notebook's benchmark cell measures the real cost and the feature change versus face-every-frame — decide before the first real chunk is saved.
- **Two machines.** `MACHINE_INDEX` / `MACHINE_COUNT` split chunks by modulo; chunk files from either machine merge by copying. Use a **CPU** runtime — a GPU does not speed this up.
- **Features.** `sign_features_v4` = `sign_features_v3` (60 × 122) on isotropic coordinates with pose-derived face points, built per chunk into `features_v4_pose_face/feat_chunk_#####.npz` with a `feature_spec.json` that records the recipe (a feature folder can never mix recipes), and per-clip usability flags (pose and hand visibility) instead of silent drops. The dataset's video size and fps are measured once into `meta/video_format.json`: **1200 × 950 (aspect 1.2632), 60 fps, median 180 frames (3 s)** in the 12 sampled videos, all the same size. A webcam is typically 4:3 or 16:9, so live serving must pass its own width/height for the aspect correction.
- **Hands** are stored in detection order, not by MediaPipe's Left/Right label, matching how `sign_features_v3` ignores labels.
- **Dry run.** `DRY_RUN=True` exercises the whole pipeline with a fake detector in a separate output folder.

**Run record (Nov 2026).** Extraction finished on a Colab CPU runtime (MediaPipe 1.2.3, 2 workers, archive on Drive, about four days): **801 of 801 chunks, 40,033 of 40,033 videos** (train 20,016 / validation 4,003 / test 16,014), 2,002 of 2,002 classes (13 to 20 videos per class), no corrupt chunk, features built for all clips with 0 unusable. Settings fingerprint `7808ddbe5a1a`, manifest sha1 `d94f31c17ca70190…`, `N_FRAMES=48`, `FACE_EVERY_K=4`, full pose model, 720 px. Benchmark on real clips: decoding and resizing 1.6 s per clip, pose 53 ms, hands 58 ms, face 6 ms per frame (about 7 s per clip in total). **Finding (Nov 2026): the face model detected a face in about 0% of frames** (mean 0.001) while pose was detected in 100% and hands in 99.6%. With no face, the hand-to-face features of `sign_features_v3` collapse to "hand position in the image ÷ shoulder width", which depends on where the signer stands and would not match the live app. **Fix (`sign_features_v4`):** the four face reference points (nose, chin, mouth, forehead) are derived from the pose model's head points, and x coordinates are multiplied by the video's width/height so x and y share one unit; the `sign_features_v3` code itself is unchanged. Features are rebuilt from the saved raw landmarks (no re-extraction) into `features_v4_pose_face/`; the earlier `features/` folder must not be used for training. Tests confirm the new features no longer change when the signer moves in the frame (change 0.0, versus 0.5 with the empty face) or when the same scene is filmed at 4:3 instead of 16:9.

**Training result, `baseline_v1` (Nov 2026).** 40 epochs on a Colab CPU (about 2 minutes per epoch); validation top-1 rose to 85.8% (best epoch 40, still creeping up). On the official test split, used once: **top-1 82.1%, top-5 95.1%, macro-recall 82.1%** over all 2,002 words and 8 signers never seen in training. The confidence threshold for 90% precision on validation is 0.24, but the frozen-pose probe (one frame repeated) scored 0.525, so the final threshold is **0.53**: on the test split that accepts 66.8% of clips at 94.4% precision. Weakest words (recall 12%, one clip of eight) are mostly near-synonyms or look-alikes (Why/What, Minute/Hour, Address/Clap). The deployment model was retrained on all 40,033 clips for 40 epochs (about 3.3 minutes per epoch) and exported to `training/baseline_v1/export/` (weights, preprocessing, labels, metrics, model card); it cannot be tested again because the test signers are in its training data. **Per-word spread (test, 8 clips per word):** 688 words at 90–100% recall, 885 at 75–90%, 324 at 50–75%, 89 at 25–50%, 16 below 25%, so 78.6% of words reach at least 75%. **Letters:** all 26 single letters are classes but average only 64.4% recall among all 2,002 words (worst T 25%, D 25%, then Q, O, N, X, M, J at 50%), confused with words such as "Add", "Delhi", "November", "Multiply" and "0/Zero". A fingerspelling mode should restrict the output to the 26 letters (not yet measured) or use a dedicated letter model. **Motion floor (`metrics.json`):** the wrist-velocity feature over the 40,033 real clips has median 0.011, 1st percentile 0.0036 and 0.1st percentile 0.0027; the stored floor is half the 1st percentile, **0.00181**. It stops only (nearly) perfectly still input: a hand held still in a real webcam will jitter above it, so the 0.53 confidence threshold and, later, a "no sign" class remain necessary. **Live-path stress test (notebook 03, 4,004 real test clips replayed through the serving code, `baseline_v1`, Nov 2026).** Top-1 on the 48-frame layout 81.0% (equal to the stored-feature accuracy, so the serving code reproduces training). Changes: 30 frames −6.1 points, 24 frames −6.9, 30 frames with 20% dropped −7.9; landmark jitter 0.002 −0.2, 0.005 −1.4; **a hand missed in 10% of frames −15.8, in 30% −29.5**; first 15% cut off −6.0, last 15% −2.7, 20% idle frames before and after −9.7; **wrong aspect (16:9 assumed) −15.9** (accepted clips fall from 66% to 28%), no aspect correction −3.7; webcam-like combination 56.7% top-1, accepting 27% of clips at 90.7% precision. This is studio landmarks with simulated live conditions, a best case that excludes the camera itself. **Two flaws of `sign_features_v3` explain the missed-hand loss:** it chooses the active hand by wrist travel measured on zero-filled arrays (each missed frame counts as two jumps to the image corner, so a resting hand with a few dropouts looks like the signing hand), and when it mirrors a sign it mirrors the zeros too, turning every missed frame into a phantom hand. The robust variant (`sign_features_v4` revision 2, `robust=True`) fixes both and is identical to v3 when no hand is missed; on a synthetic clip with a 10% miss rate the feature change drops from 0.344 to 0.007. The remaining weaknesses (frame rate, window alignment, aspect) need timestamped capture and landmark-level augmentation in training. **Limits:** studio recordings only; the model confidently labels a frozen pose (so the live motion gate must stay); per-word test recall has 1/8 granularity; live webcam accuracy, segmentation and extractor parity are unmeasured. It is not promoted: gates 3–8 in §6 remain open.

**Training and evaluation (notebook 02, CPU-friendly).** A BiGRU with attention and max pooling (about 1.5 M parameters) over the `(60, 122)` features, with speed-jitter, noise and frame-dropout augmentation and per-feature standardisation. A checkpoint is written every epoch and resuming a changed run is refused. The model is chosen on validation top-1; the **test split is evaluated once** (a saved report blocks re-testing) and reports top-1, top-5, macro recall and per-class recall. The confidence threshold is the smallest value that reaches the target precision on validation **and** sits above the confidence of non-sign probes (zeros, noise, frozen pose, 5 frames). The deployment model is retrained on all signers for the best epoch count and exported with weights, preprocessing (including the exact extraction settings), labels, metrics and a model card. Letter and digit classes in the dataset are to be checked for the fingerspelling fallback.

## 8. Licence gate

Code is MIT. Weights trained on iSign inherit CC BY-NC-SA 4.0 obligations; FDMSE-ISL weights follow the authors' terms. Before any public deployment record: (1) whether non-commercial use covers a hosted app, (2) whether serving predictions counts as commercial use, (3) attribution and share-alike duties for the bundle, (4) whether tokenizers derived from the data need the same treatment. Local research is not blocked; presenting a checkpoint as commercially cleared without this review is.
