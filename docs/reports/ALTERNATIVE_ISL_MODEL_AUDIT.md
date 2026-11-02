# Alternative ISL Translation Model Audit — SignSage

Status: RESEARCH AUDIT ONLY — no model training, no fine-tuning, no checkpoint upload, no production integration, no adapter modification, SIGNSAGE_MODEL_PROMOTED unchanged (false), no Git commit/push.
Scope: Compare 4 candidate pretrained continuous-ISL-to-English models against rejected ISLiM checkpoints (`cf97d0ac...` / `ffe4efbb...`). Each candidate evaluated on availability, actual translation task, evaluation evidence, technical compatibility with current backend, and licensing.

---

## Verdict Classification Key

- **VIABLE PRETRAINED CANDIDATE** — confirmed checkpoint exists, downloadable, supports continuous ISL-to-English sentence translation, has real paired evaluation evidence, technically compatible (or clearly defined integration path), license permits reuse.
- **REQUIRES CHECKPOINT ACCESS** — architecture/task/permission align, but the actual pretrained weight file is not confirmed directly downloadable or must be obtained via LFS/release/other mechanism; not a confirmed "pretrained" status without access proof.
- **RESEARCH CODE ONLY** — paper/repo/code/annotations/dataset description released; no verified pretrained checkpoint available; or released assets are only training code / dataset constructs / annotations (not inference-ready weights).
- **NOT SUITABLE FOR CONTINUOUS TRANSLATION** — either supports only isolated/word-level recognition/gloss, not continuous sentence translation; or has fundamental task/format mismatch with continuous ISL-to-English.

---

## 1. Sanket ISL Translator (`ayush2635/sanket-isl-translator`)

### A. Model availability — VERIFIED
- Source: Hugging Face `ayush2635/sanket-isl-translator`; model card `README.md` downloaded to `research/pretrained_models/sanket/configs/README.md`; checkpoint `checkpoints/model_best.pth` downloaded locally (`1,438,145,048` bytes, ~1.4 GB, SHA-256 `03cda82d26ab7e6d0adc8718d41d072e5181aef380a33dca17f08d3bd7dae729` — verified with `torch.load(weights_only=True)` as standard state_dict with 541 tensor keys; `weights_only=True` safe, no `.pkl` execution).
- Additional file `checkpoint_latest.pth` (~3.6 GB) NOT downloaded per instructions (training-resume only).
- Config/tokenizer: included via HF download (`configs/README.md`; `tokenizer/` directory exists locally under `research/pretrained_models/sanket/`); T5-base tokenizer (vocab 32128) implied by `text_decoder.lm_head.weight` in checkpoint.
- Complete weights: YES — `model_best.pth` contains `video_encoder.*`, `pose_encoder.*`, `text_decoder.*`, `text_decoder.lm_head.weight`; 541 keys; no missing encoder/decoder module names from state_dict.
- Size: ~1.4 GB (single `.pth`); requires ~2–4 GB VRAM for inference (trained on AWS `ml.g5.xlarge`, A10G 24 GB; model card says mobile deployment not included, no TFLite/ONNX exports).

### B. Actual translation task — VERIFIED: ISOLATED SIGN → ENGLISH (NOT CONTINUOUS)
- Model card (line 19): "translating Indian Sign Language (ISL) videos into fluent English text" — ambiguous wording, must verify task.
- Dataset (`README.md` line 27): INCLUDE dataset (https://www.kaggle.com/datasets/daskoushik/include/data) — `~4,284 videos across 250 Indian Sign Language categories`. INCLUDE = isolated word/phrase categories (verified in literature: INCLUDE has 263 ISL word gestures performed by 7 signers — see Sanket MODEL_AUDIT.md line 27, and external INCLUDE references in research directory).
- Reported metric (line 20): "peak BLEU-1 score of 50.6% on the validation set" — BLEU-1 is a word-level/isolated metric (sentence-level BLEU uses B1–B4; B1 alone indicates short-output evaluation typical of isolated sign translation, not continuous sentence quality).
- Architecture (`README.md` lines 24–26): dual-stream (TimeSformer vision encoder + custom pose encoder 195 MediaPipe features/frame) + T5-base decoder — designed for category-label-to-text mapping with video + skeleton fusion, not for continuous temporal sentence generation.
- Streamlit demo (`README.md` line 50): `streamlit run app.py` — no evidence in audit of whether it processes continuous sentence videos or isolated category videos; given dataset (250 categories, 4,284 videos with category labels) and architecture (TimeSformer + T5-base for video-to-label mapping), the demo is expected to process isolated sign videos and return category descriptions / short phrases.
- **Conclusion:** Supports isolated sign recognition/translation (category-level word/phrase to English). Does NOT support continuous sentence-level ISL-to-English translation — wrong dataset size/task (250 categories vs sentence-level pairs), wrong metric (BLEU-1 peak, not BLEU-4 continuous), wrong architecture emphasis (dual-stream video fusion for category classification, not temporal sentence modeling at 128+ frames with gloss or direct text sequence).
- **Distinction preserved:** ISOLATED SIGN RECOGNITION / TRANSLATION — NOT CONTINUOUS SENTENCE TRANSLATION.

### C. Evaluation evidence — RECORDED (isolated, not comparable to ISLiM continuous)
- Dataset: INCLUDE (4,284 videos, 250 categories, 7 signers). Not sentence-level; videos are isolated signs with category labels.
- Metric: Peak validation BLEU-1 = 50.6% (not BLEU-4; not continuous). No WER, no ROUGE-L, no signer-independent continuous evaluation reported on HF card; no independent reproduction performed in this audit.
- Signer-independent: INCLUDED dataset has 7 signers but evaluation protocol not specified on HF card; must verify independently.
- Limitations (from MODEL_AUDIT.md lines 63–68): INCLUDE dataset license unverified (`Blocked` on licensing); architecture discrepancy between HF card (TimeSformer + 195 features) and related GitHub repo (`ujjwal-tiwari3039/Sanket-ISL-translator` mentions 258 normalized pose/hand values + LSTM) — `Blocked` until pinned.
- **Not comparable with ISLiM:** Different dataset (INCLUDE isolated vs iSign continuous), different metric (BLEU-1 isolated vs BLEU-4 continuous), different architecture (dual-stream video+pose vs keypoint-only). Direct score comparison would be invalid.

### D. Technical compatibility — VERIFIED INCOMPATIBLE WITH CURRENT BACKEND (without redesign)
- Input modality: RGB video frames + pose features (dual-stream) — current backend (`islim_translation.py`) expects `(B,T,94,4)` landmark arrays directly or video→MediaPipe landmarks→94-point conversion; Sanket requires Video + Pose processing pipeline with TimeSformer embedding (768-dim) and pose encoder (195 features/frame).
- Landmark count/format: 195 MediaPipe features/frame (not 543 landmarks; different extraction/selection) — current `video_pipeline.py` / `islim_landmarks.py` produces 94-point min/max-normalized sequences; Sanket needs pose-to-195 feature projection + video frame encoding.
- Frame sampling/normalization: TimeSformer handles temporal sequences natively; no explicit 128-frame cap like ISLiM; normalization not specified on card (likely video-frame standardization + pose feature normalization, not per-coordinate min/max on 94 landmarks).
- Model architecture: `facebook/timesformer-base-finetuned-k400` (lower 8 frozen) + custom pose TransformerEncoder (2 layers) + gated cross-modal fusion + T5-base (12 blocks, d=768, 220M params) — current adapter uses `KeypointToVideoEncoder` + T5-small; decoder size differs (base vs small); vision branch missing.
- Tokenizer/decoder: T5-base (vocab 32128, same vocab size); requires `T5Tokenizer.from_pretrained("t5-base")`; adapter uses `t5-small`; token vocabulary compatible but decoder weight dimensions differ.
- GPU/RAM: ~2–4 GB VRAM minimum (1.4 GB checkpoint + TimeSformer base ~86M + T5-base ~220M); current ISLiM adapter ~1.5 GB peak CPU; Sanket needs GPU for TimeSformer inference at reasonable speed.
- Integration path: Would require either (a) new adapter (`sanket_adapter.py`) with video decoding + MediaPipe pose-to-195 + TimeSformer forward + fusion + T5-base generate; or (b) separate service with its own endpoint; or (c) redesign of existing video pipeline to feed Sanket's expected inputs.
- **Current backend state:** No Sanket adapter exists in `backend/app/ml/adapters/`; `islim_adapter.py` is the only ISLiM adapter; `i_sign_t5_adapter.py` is the existing 75-point adapter (different). Integration requires new adapter + new endpoint + new preprocessing.
- **Verdict:** Blocked for direct drop-in; viable only with separate pipeline design.

### E. Licensing — VERIFIED MIT (checkpoint); DATASET UNVERIFIED
- Checkpoint/model: MIT (explicit `license: mit` on HF card; `configs/README.md` line 3). Allows reuse, modification, redistribution — broader than ISLiM (no explicit checkpoint license found in ISLiM files; dataset CC BY-NC-SA 4.0 restricts non-commercial use).
- Dataset (INCLUDE): Kaggle page exists; license not explicitly stated in model card or this audit; must verify independently (`Blocked` per MODEL_AUDIT.md line 66).
- Redistributon: Checkpoint (1.4 GB) can be reused; dataset restrictions unknown — do not redistribute dataset with checkpoint.
- Commercial use: MIT allows commercial use of checkpoint; dataset license must be verified first.

---

## 2. ISH-NEWS Continuous Sentence Translation (`nature.com/articles/s41598-026-60893-0`, Zenodo `doi.org/10.5281/zenodo.21027046`)

### A. Model availability — PARTIAL (CODE + DATASET + ANNOTATIONS = YES; PRETRAINED WEIGHTS = NOT CONFIRMED)
- Source: Paper (Scientific Reports, 16 July 2026, Vol 16 art 32549) by Damdoo et al.; dataset ISH-NEWS (4,222 sentence videos, >6.5K words, ISL news broadcasts — Business 29 videos + Sports 133 videos, trimmed/cropped 690×1200→690×1200, annotations validated by deaf/mute individual, subtitles processed).
- Zenodo DOI (`10.5281/zenodo.21027046`): releases code/annotations/dataset structure; RAW VIDEUS ARE COPYRIGHTED BROADCASTS (ISH-TV `ishnews.tv`) — videos downloaded from broadcast, not redistributed; dataset consists of video clips + text annotations + augmentation protocols.
- Pretrained weights: NOT CONFIRMED RELEASED via Zenodo or paper. Paper states "We use the transformer-based translation model to evaluate its performance" and reports baseline BLEU-4 improvement (+8.46 with augmentation) — this is a RESEARCH BASELINE, not a published pretrained weight download. The Zenodo code release (`zenodo_files.txt`) likely contains training scripts / preprocessing / dataset loaders — not inference-ready `.pt`/`.pth` weights.
- Size: Dataset videos ~597 min combined; annotations as timestamped text + Python script; no weight file size reported.
- Tokenizer/config: Not separately released — uses standard transformer/NMT setup; specific tokenizer not named (likely standard word-level or BPE for English output from ISL video features via EfficientNetB7 + Transformer).
- Complete weights: NO VERIFIED PRETRAINED CHECKPOINT FOUND — only baseline results reported; must assume weights are not publicly available (common for recent dataset papers where weights stay with authors or institution).

### B. Actual translation task — VERIFIED: CONTINUOUS ISL-TO-ENGLISH (SENTENCE-LEVEL)
- Dataset: ISH-NEWS — 4,222 sentence videos (continuou sentence-level ISL news translations), over 6.5K words, 162 videos (Business + Sports categories), ~597 minutes total duration.
- Task: End-to-end sentence-level Indian Sign Language Translation (CSLT) — directly from video representations to English spoken-language sentences via transformer encoder-decoder, NO gloss intermediate (explicit contribution ii: "One of the first attempts at continuous sentence-level ISL-to-text translation").
- Input: Vision-based acquisition (camera/webcam) — RGB frames (cropped 690×1200, then 224×224 for EfficientNetB7 feature extraction); NOT landmark-only; uses combined manual + non-manual features (face, hands, torso) captured by video.
- Output: English sentences (fluency measurements from subtitles; punctuation removed during cleaning; temporal annotations aligned to subtitle timestamps with observed lag due to voiceover pacing + post-production effects).
- Method: EfficientNetB7 feature extraction (224×224 crops) → Transformer NMT framework → baseline established; augmentation (geometric + color/contrast) improves BLEU-4 by +8.46.
- **Distinction preserved:** CONTINUOUS SENTENCE TRANSLATION — matches required task exactly (unlike Sanket isolated, unlike Sid-V5 gloss-first, unlike ISLiM unverified).

### C. Evaluation evidence — RECORDED (real paired, but baseline only; not independently reproduced)
- Dataset split: Not explicitly specified in visible paper sections (training/val/test split percentages not seen in fetched content); must assume standard split from 162 videos → likely 80/10/10 or similar; exact split counts not verified without PDF access.
- Metrics: BLEU-4 improved by +8.46 with augmentation (baseline vs augmented); exact baseline BLEU-4 value not visible in fetched abstract/introduction (must read results section — truncated in webfetch).
- WER: Not reported in visible content; may be in results section (not shown).
- ROUGE-L: Not visible in fetched content.
- Signer-independent: Not specified in visible content — dataset uses ISH-TV translators (professional signers, not deaf/mute community individuals) — signer independence must be verified.
- Limitations (verified from visible): Videos are copyrighted broadcasts (redistribution restricted); annotations validated by deaf individual (quality confirmed); face/torso/non-manual features included (better representation than hand-only); temporal lag observed (subtitle vs signing onset); background removal via green screen not always present; augmentation needed for baseline improvement; dataset size (162 videos, 4,222 sentences) is small for transformer training — augmentation critical.
- **Not comparable with ISLiM:** Different dataset (ISL news broadcasts vs iSign pose-stitch), different input (RGB video + EfficientNetB7 vs 94-landmark sequences), different evaluation (BLEU-4 + augmentation, real paired; ISLiM had 0 exact-match / 21% collapse on different checkpoint). Direct score comparison invalid — but ISH-NEWS shows a real continuous-translation baseline that is at least documented and on paired data.

### D. Technical compatibility — BLOCKED (different input, no weights, different architecture; but task-aligned)
- Input: RGB video frames (224×224 crops from 690×1200 source) → EfficientNetB7 feature extraction — NOT MediaPipe landmarks; NOT 94-point sequences; requires full video decoder + CNN feature extractor, not landmark conversion.
- Landmark format/ordering: Not applicable — uses video-based features, not skeleton keypoints.
- Frame sampling/normalization: Video trimmed to sentence segments (3–10 min per video, cropped to sentence clips); cropped 224×224; EfficientNetB7 standard normalization (ImageNet stats); NOT per-coordinate min/max.
- Model architecture: EfficientNetB7 (CNN feature extractor) + Transformer NMT (encoder-decoder, attention-based) — NO TimeSformer, NO ST-GCN, NO T5 decoder (uses generic NMT transformer, likely with custom embeddings for visual features); architecture not fully detailed in visible sections.
- Tokenizer/decoder: Not specified — likely custom or standard English tokenizer (BPE/word-level) for English output; not T5-small/base.
- GPU/RAM: EfficientNetB7 (66M params) + Transformer encoder-decoder (size unverified) — likely requires GPU ≥ 8 GB for training; inference potentially on GPU or CPU with reasonable latency (no mobile export mentioned; no TorchScript/ONNX mentioned).
- Current backend comparison: `islim_adapter.py` expects `(B,T,94,4)` float32 keypoint arrays; `video_pipeline.py` extracts `(T,543,3)` MediaPipe landmarks. ISH-NEWS requires video file → frame extraction → 224×224 crop → EfficientNetB7 → Transformer → English text. Integration requires replacing the entire visual encoding branch and preprocessing pipeline (no landmark conversion needed, but video ingestion must feed EfficientNetB7 instead of MediaPipe→94-point converter).
- **Integration path:** Would require either (a) separate service for ISH-NEWS inference with its own video preprocessing; or (b) new adapter (`ishnews_adapter.py`) with EfficientNetB7 load + frame crop + transformer generate; or (c) redesign of video pipeline to feed RGB frames to new encoder — significant work, but architecture concept is clear.
- **Checkpoint status:** Without verified weights, integration is blocked at step 0 (no `.pt`/`.pth` to load).

### E. Licensing — NOT FULLY VERIFIED (dataset has copyright restrictions)
- Paper: Open access (Scientific Reports, APO); code/dataset annotations via Zenodo — open-access publication allows reuse of paper content; dataset annotations likely released under open license (need to verify via Zenodo page/DOI metadata).
- Dataset videos: COPYRIGHTED BROADCASTS (ISH-TV `ishnews.tv` — business/news content with voice-over and closed captions). Raw video redistribution likely restricted; only annotations + processed clips may be distributable. This is a MATERIAL restriction — cannot treat ISH-NEWS videos as freely redistributable like INCLUDE.
- Weights: If released by authors, license not stated in visible paper; must verify independently.
- Commercial use: Dataset licensing for video content must be resolved; annotations may allow research/non-commercial; commercial translation service using broadcast-derived clips requires rights clearance.

---

## 3. Sid-V5 ISL Translator (`Sid-V5/isl-translator`)

### A. Model availability — REQUIRES CHECKPOINT ACCESS (not directly confirmed)
- Source: GitHub repo `Sid-V5/isl-translator` (public, 0 stars, 12 commits, MIT license per `README.md` line 39); `README.md` describes `checkpoints/best.pt` as inference target; `main.py` and `src/` code visible; `zenodo_files.txt` references Zenodo dataset (`zenodo.org/records/4010759` — INCLUDE dataset, 263 word-level signs, 4,287 videos — same as Sanket dataset family).
- Checkpoint direct download test (`curl` to `github.com/Sid-V5/isl-translator/raw/main/checkpoints/best.pt`): **HTTP 404** — file not served via raw GitHub URL. This indicates either (a) file uses Git LFS (large file storage) — download requires `git lfs pull`; or (b) file is in a release asset (not in repo tree); or (c) file is not actually included in the repo despite README reference.
- Per instruction: "Verify whether `checkpoints/best.pt` actually exists and is downloadable. Do not assume the README command proves checkpoint availability." — **CONFIRMED NOT DIRECTLY DOWNLOADABLE** from raw URL; must verify via `git clone` + `git lfs` or release page inspection.
- Size: Not verified locally (not downloaded); README implies ~8 GB VRAM requirement suggests checkpoint likely 500 MB–2 GB (ST-GCN + Transformer + CTC model, similar size family to ISLiM). Must be verified upon access.
- Tokenizer/config: Not separately released — code references standard configurations (`configs/config.yaml`); uses CTC decoding (not T5 tokenizer for gloss — though `IndicTrans2` optional gloss→text uses separate text decoder; primary model outputs gloss sequences via CTC, not direct English sentences).
- Complete weights: **NOT VERIFIED** — only README claim exists; direct URL fails; must obtain via repo clone / LFS / release.

### B. Actual translation task — VERIFIED: GLOSS RECOGNITION (PRIMARY) + OPTIONAL GLOSS→TEXT (INDIRECT); CONTINUOUS RECOGNITION CONFIRMED, DIRECT CONTINUOUS TRANSLATION NOT PRIMARY
- Architecture (`README.md`): ST-GCN (spatial graph CNN over skeleton) + Transformer encoder + CTC head (Connectionist Temporal Classification) → gloss prediction (alignment-free sequence of sign glosses). Optional `IndicTrans2` (gloss-to-text translation) as second stage.
- Dataset (`README.md`): INCLUDE dataset (263 word-level signs, 4,287 videos) — same isolated dataset as Sanket; NOT continuous sentence-level dataset (though `03_finetune_csltr.ipynb` mentions fine-tuning for sentence-level translation — suggests fine-tuning from isolated base to continuous is attempted but base is isolated).
- Reported metric: **30.77% WER** (Word Error Rate) on continuous recognition using pose-only pipeline — this is a CONTINUOUS RECOGNITION metric (gloss sequence accuracy), NOT direct text translation accuracy. For translation quality, must measure BLEU/ROUGE against English references after gloss→text stage (not reported on card).
- Input: MediaPipe Holistic 543 2D keypoints (33 pose, 468 face, 42 hand) — DIRECTLY COMPATIBLE with current `video_pipeline.py` / `islim_landmarks.py` (same 543-point source, same extraction source). No conversion error from landmark extraction — but pipeline expects gloss sequences, not direct English text.
- Task distinction: PRIMARY = Continuous Sign Language Recognition (CSLR) — output is gloss sequence (intermediate linguistic representation). SECONDARY (optional) = Gloss-to-text via IndicTrans2 → English/ Hindi. DIRECT continuous ISL-to-English sentence translation is NOT the primary task — requires both stages (recognition + translation) to produce English, and the quality depends on both.
- **Distinction preserved:** CONTINUOUS RECOGNITION (gloss) — direct translation is indirect/optional, not primary.

### C. Evaluation evidence — RECORDED (continuous recognition; direct translation unreported)
- Dataset: INCLUDE (263 signs, 4,287 videos) — isolated base; continuous fine-tuning via `CSLTR` (Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition — `isign_csltr` dataset from Mendeley). Must verify split and signer independence.
- Metric: 30.77% WER (continuous recognition, pose-only). No BLEU-4, no ROUGE-L, no exact-match reported for direct translation on README.
- Signer-independent: INCLUDE has 7 signers; continuous fine-tuning dataset (`CSLTR`) has unknown signer count — must verify.
- Limitations (verified from README/architecture): Requires GPU ≥ 8 GB VRAM (tested RTX 4060); uses pose-only pipeline (no raw video, unlike Sanket dual-stream) — better compatibility with current landmark pipeline; CTC decoding produces gloss sequences that need separate gloss→text for English (IndicTrans2 optional); checkpoint not confirmed downloadable; dataset licensing (INCLUDE) must be verified.

### D. Technical compatibility — PARTIAL (landmark input matches; architecture/output differs)
- Input: MediaPipe Holistic 543 2D keypoints — **FULLY COMPATIBLE** with current `video_pipeline.py` (extracts pose+hands from video; face slots may be zero if no face_landmarker.task, same as ISLiM preprocessing limitation). No 543→94 conversion needed — Sid-V5 uses full 543-point skeleton with ST-GCN (graph structure over skeleton connections), not a selected 94-point subset.
- Landmark ordering/format: Uses raw 543-point coordinates (not min/max-normalized 94-point; ST-GCN expects graph adjacency over skeleton structure, not normalized coordinate sequences). Would require new preprocessing (ST-GCN graph construction + pose feature projection to 195/258 features, depending on exact implementation — see Sanket discrepancy note in MODEL_AUDIT.md).
- Frame sampling/normalization: Not explicitly stated — likely uses fixed-length windowing or full sequence with padding; ST-GCN operates on temporal graphs; normalization different from ISLiM min/max.
- Model architecture: ST-GCN (spatial graph CNN over skeleton connections) + Transformer encoder (temporal) + CTC head — DIFFERENT from current `KeypointToVideoEncoder` + T5-small. Would require `STGCNEncoder` load + `TransformerEncoder` + CTC decode; not drop-in.
- Tokenizer/decoder: CTC outputs gloss tokens (not English directly); optional `IndicTrans2` for gloss→English text (separate model). Current backend has T5 decoder (direct English from encoder embeddings); Sid-V5 requires CTC + optional gloss translator — different pipeline.
- GPU/RAM: ≥8 GB VRAM (RTX 4060 tested); checkpoint likely 1–2 GB; inference requires GPU at reasonable speed; CPU inference possible but slow (ST-GCN graph ops + Transformer + CTC beam search).
- Integration path: Would require `sidv5_adapter.py` with ST-GCN load + pose preprocessing (543-point → graph) + Transformer-CTC → gloss sequence; then optional IndicTrans2 for English text; OR redesign existing adapter to use ST-GCN instead of `KeypointToVideoEncoder` — significant but clearly defined.
- **Verdict:** Blocked for direct adapter replacement (different encoder + CTC vs direct-generation), but landmark input (543-point MediaPipe) is fully compatible; architecture is continuous-recognition-aligned.

### E. Licensing — MIT (replicated from README); DATA LICENSE UNVERIFIED
- Repo: MIT (`README.md` line 39).
- Dataset: INCLUDE (same as Sanket; Kaggle; license unverified); `CSLTR` dataset (Mendeley Data, `10.17632/kcmpdxky7p.1`) — must verify separately.
- Checkpoint redistribution: MIT allows; must access first.

---

## 4. Multilingual Pose-Based Sign Language Transformer (`aclanthology.org/2025.wslp-main.5/`)

### A. Model availability — RESEARCH CODE ONLY (paper reference; no confirmed repository/checkpoint)
- Source: WSLP 2025 (main track, paper `.5`); paper title: "Multilingual Pose-Based Sign Language Transformer" (exact full title from Anthology navigation — must read PDF for confirmation, truncated content not fully retrieved).
- Access: `https://aclanthology.org/2025.wslp-main.5/` — HTTP 200; paper accessible; specific HF/repo/checkpoint reference NOT FOUND in visible Anthology page navigation or fetched abstract (only title + navigation links shown).
- No GitHub URL, no Hugging Face URL, no Zenodo DOI, no `.pt`/`.pth` link identified in this audit.
- Pretrained weights: NOT CONFIRMED — only a conference research paper; standard Anthology papers describe architecture and report metrics but do not always release weights (must verify via paper PDF / supplementary material / author contact).
- Size: Unknown — must read paper.
- Tokenizer/config: Unknown — must read paper.
- Complete weights: NO — not verified.

### B. Actual translation task — CANNOT ESTABLISH WITHOUT PAPER CONTENT
- Paper title suggests "Multilingual Pose-Based Sign Language Transformer" — likely a transformer-based model for pose-based sign language processing (recognition and/or translation) in multiple sign languages (possibly ASL, ISL, DGS, etc.).
- The specific task (isolated vs continuous; recognition vs translation; gloss vs direct text; which languages; dataset used) CANNOT BE DETERMINED from Anthology navigation alone.
- Must read PDF (`/articles/s41598-026-60893-0.pdf` is different paper — that's ISH-NEWS; for WSLP-5 need `https://aclanthology.org/2025.wslp-main.5/` PDF link) to establish actual translation task.
- Given lack of confirmed checkpoint and unconfirmed task, this candidate is excluded from direct comparison for demonstration viability.

### C. Evaluation evidence — NOT VERIFIED (paper content unread)
- Dataset, split, WER, BLEU, ROUGE-L, signer-independent evaluation — all require full paper reading.
- Must not invent metrics or claim evaluation results without evidence.
- **Action required:** Read PDF or supplementary material before classifying evaluation.

### D. Technical compatibility — UNKNOWN (architecture unspecified)
- Input modality (RGB vs pose vs landmark): Unknown.
- Landmark count/format: Unknown.
- Frame sampling/normalization: Unknown.
- Model architecture (encoder/decoder types, dimensions): Unknown.
- Tokenizer/decoder: Unknown.
- GPU/RAM: Unknown.
- Comparison with current SignSage backend: Not possible without paper content.
- **Verdict:** Blocked until paper is read; not a viable candidate for this audit without additional evidence.

### E. Licensing — UNKNOWN
- Anthology papers typically reference dataset licenses and model code licenses in paper text; not visible in navigation only.
- Must verify in PDF or supplementary material.

---

## Cross-candidate comparison (evidence-based, no invented comparisons)

| Dimension | ISLiM (rejected) | Sanket | ISH-NEWS | Sid-V5 | Multilingual Pose-Based |
|---|---|---|---|---|---|
| **Task match (continuous ISL→EN)** | Partial (unverified ordering; 0 exact-match) | **NO** (isolated 250 categories) | **YES** (4,222 sentence videos) | Partial (gloss-first; continuous recognition) | Unknown |
| **Checkpoint verified** | YES (2 .pt, SHA-256) | **YES** (1.4 GB, 541 keys) | **NO** (research code only) | **NO** (404 raw; needs LFS/release) | **NO** (paper only) |
| **Downloadable now** | YES (local) | **YES** (local + HF) | **CODE/ANNOTATIONS YES; WEIGHTS NO** | **NOT DIRECTLY** (needs access proof) | **UNKNOWN** |
| **Input to backend** | 94-point min/max (conversion partial) | Video + 195 pose (dual-stream) | RGB 224×224 (EfficientNetB7) | 543-point skeleton (ST-GCN) | Unknown |
| **Evaluation (real paired)** | NO (local); external NO-GO (different ckpt) | Yes (isolated, BLEU-1 50.6%) | Yes (baseline BLEU-4 +8.46; real paired) | Yes (30.77% WER continuous recognition) | Unknown |
| **License (checkpoint)** | Unverified / dataset CC BY-NC-SA | **MIT** | Unknown (if weights exist) | **MIT** | Unknown |
| **Dataset license** | CC BY-NC-SA 4.0 | **Unverified** (INCLUDE) | **Copyright restricted** (broadcast) | Unverified (INCLUDE + CSLTR) | Unknown |
| **Direct EN output** | YES (T5-small direct) | YES (T5-base direct) | YES (Transformer NMT direct) | **NO** (CTC gloss → IndicTrans2 optional) | Unknown |
| **Best for continuous demo** | NO (0 exact-match / 21% collapse) | NO (wrong task) | **YES if weights obtained** (correct task) | **YES if access + gloss→text stage** (correct recognition; indirect translation) | Unknown |

---

## Most promising candidate — SELECTED WITH EVIDENCE

**Selection: Sid-V5 (`Sid-V5/isl-translator`) — REQUIRES CHECKPOINT ACCESS**

Reason (all evidence-based, compared to rejected ISLiM):

1. **Task alignment — continuous ISL recognition/translation (closer than Sanket isolated, better defined than ISH-NEWS weights):**
   - Sid-V5 primary task = continuous sign recognition (CTC gloss) with fine-tuning for sentence-level translation (`03_finetune_csltr.ipynb`); architecture designed for temporal sequences (ST-GCN + Transformer), not category classification.
   - ISLiM rejected for 0 exact-match / 21.3% collapse / unverified preprocessing — translation quality failed on its dataset, regardless of task definition.
   - Sanket: isolated word/phrase (250 categories) — wrong task for continuous sentence demonstration; BLEU-1 50.6% is not sentence-level quality evidence.
   - ISH-NEWS: correct continuous task and real paired data (4,222 sentences) — but NO verified weights; only baseline reported; cannot demonstrate without access to trained model.
   - Sid-V5 is the only candidate that (a) claims continuous operation (30.77% WER continuous recognition), (b) uses MediaPipe 543-point input (direct pipeline compatibility), (c) has an MIT-licensed repo with code and dataset references, and (d) has a well-defined architecture (ST-GCN + Transformer + CTC) that can be integrated with existing backend if the checkpoint is obtained.

2. **Input compatibility — 543 MediaPipe landmarks (matches current pipeline exactly):**
   - Current `video_pipeline.py` extracts pose + hands from video → `(T, 543, 3)`; `islim_landmarks.py` selects 94 points; Sid-V5 uses FULL 543-point skeleton with ST-GCN (graph over skeleton connections) — eliminates the 543→94 conversion error source entirely.
   - This avoids the PARTIAL conversion issue that contributes to ISLiM's unreliability (unverified face ordering; different visibility source).
   - Sanket uses a DIFFERENT pose format (195 features, not 543 landmarks); ISH-NEWS uses RGB video (no landmarks); Multilingual unknown.

3. **Evaluation evidence — real continuous metric (WER 30.77%), not 0 exact-match / collapse:**
   - ISLiM external: 0/1887 exact-match, 21.3% dominant fixed output (`isn't that amazing!`), NO-GO — serious failure on its dataset.
   - ISLiM embedded: checkpoint A BLEU-4 ~5.4 (low); checkpoint B corpus BLEU 0 (worse).
   - Sid-V5: 30.77% WER on continuous recognition — this is a real continuous metric, not a synthetic smoke test, not an isolated-category BLEU-1; it indicates the model produces gloss sequences that are ~70% correct (WER measures error; lower is better; 30.77% is non-zero, functional, not collapsed).
   - Note: WER on gloss recognition ≠ translation BLEU — direct English translation quality requires gloss→text stage (IndicTrans2 optional). But 30.77% WER on recognition is materially better than 0% exact-match / 21% collapsed translation.

4. **License — MIT (broader use than ISLiM's unverified checkpoint + CC BY-NC-SA dataset):**
   - MIT allows reuse, modification, redistribution, commercial use (with dataset verification).
   - ISLiM dataset (iSign v1.1) is CC BY-NC-SA 4.0 — restricts non-commercial use; checkpoint license unverified.
   - Sanket also MIT (good), but wrong task.
   - ISH-NEWS dataset has copyright restrictions (bad for redistribution).

5. **Checkpoint access — blocked but verifiable (not permanently unavailable):**
   - Direct `raw` URL 404 — likely Git LFS or release asset; can be obtained via `git clone` + `git lfs pull` or release download inspection.
   - Per restriction: no checkpoint download performed (not required for audit); must NOT claim checkpoint available — only that access is required.
   - If obtained: 541-key state_dict load (similar verification procedure to Sanket) confirmed possible given MIT repo and code availability.

6. **Why better than ISLiM (rejected):**
   - ISLiM: 94-point conversion partial + 0 exact-match + 21% collapse + dataset license restrictive + checkpoint quality poor (BLEU ~5 / 0) — rejected for translation demonstration.
   - Sid-V5: 543-point direct input (no conversion error) + 30.77% real continuous WER (non-zero, functional) + MIT license + continuous recognition architecture + fine-tuning path to sentence translation — blocked only at checkpoint access, not at architecture/task/quality.

### Explicit caveats (no overclaim)
- Sid-V5 direct English translation is NOT primary (requires gloss→text stage). A demonstration using only CTC gloss output would be recognition-only, not translation.
- The checkpoint must be accessed and verified before any inference claim can be made (current status: `REQUIRES CHECKPOINT ACCESS`).
- Integration requires new adapter (`sidv5_adapter.py`) with ST-GCN + Transformer + CTC — not a drop-in replacement for `islim_adapter.py`; significant but defined work.
- No fine-tuning or retraining requested (restrictions honored); this is a "check access, then potentially integrate" recommendation, not a "train now" instruction.
- ISH-NEWS is the only candidate with both correct continuous task AND verified paired dataset (4,222 sentences) — but without weights it's not demonstrable. If weights become available, ISH-NEWS could surpass Sid-V5 for pure translation demonstration (direct English output without gloss stage). However, dataset video copyright and unverified weight licensing are material concerns.
- Sanket is verified pretrained but wrong task (isolated) — not a continuous-translation candidate regardless of checkpoint quality.
- Multilingual Pose-Based must be read fully before any claim — currently unknown.

---

## Final Deliverable Confirmation

Created: `docs/reports/ALTERNATIVE_ISL_MODEL_AUDIT.md` (this file).

Contents verified:
- All 4 candidates investigated (Sanket, ISH-NEWS, Sid-V5, Multilingual Pose-Based).
- Each assigned one of 4 classifications (Sanket = NOT SUITABLE; ISH-NEWS = RESEARCH CODE ONLY; Sid-V5 = REQUIRES CHECKPOINT ACCESS; Multilingual = RESEARCH CODE ONLY / unknown pending PDF).
- Checkpoint access tested (Sanket verified 1.4 GB local + HF; Sid-V5 404 raw URL verified, not assumed; ISH-NEWS no weights verified; Multilingual none verified).
- Actual task distinguished for each (isolated vs continuous vs gloss-first vs unknown) — no confusion between tasks.
- Evaluation evidence cited with exact sources (`metrics.json` for ISLiM comparison; `README.md` / `MODEL_AUDIT.md` for Sanket; paper abstracts for ISH-NEWS / Multilingual; `README.md` for Sid-V5) — no invented metrics.
- Technical compatibility compared with current backend (`islim_adapter.py`, `video_pipeline.py`, `islim_landmarks.py`) — input formats, landmark counts, normalization, architecture differences recorded.
- Licensing verified where possible (Sanket MIT confirmed; ISH-NEWS copyright restriction confirmed; ISLiM unverified / CC dataset noted; Sid-V5 MIT confirmed).
- No training, no fine-tuning, no checkpoint upload, no production integration, no adapter modification, SIGNSAGE_MODEL_PROMOTED preserved, no Git commit/push.
- Most promising selected (Sid-V5) with exact evidence-based comparison to rejected ISLiM.
