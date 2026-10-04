# SignSage Audit + Model Strategy Decision

All findings below are from reading the actual repository (file paths cited). Web evidence is confined to §7 and explicitly labeled.

---

# 1. Current Architecture

End-to-end path as **actually implemented**:

```
Browser camera
  → frontend/src/features/translate/landmarks.ts (HolisticExtractor: MediaPipe Tasks-Vision
    PoseLandmarker 33 + FaceLandmarker 468 + HandLandmarker 21+21 = 543×3 raw, zero-fill missing)
  → frontend/src/services/websocket/client.ts (validates exactly 543×3, sends landmarks.frame @ ~10fps)
  → backend/app/api/websocket/translation.py (/ws/translate: auth → TranslationSession buffer →
    rolling interim every 80-frame window / 20-frame stride → final on sentence.boundary)
  → backend/app/services/translation_service.py → SignToTextAdapter.transcribe()
  → prepare_live_landmarks → normalize_pose_sequence (pose_normalize_v1) → SignTransformer →
    greedy_decode_ids (constrained greedy) → decode_token_ids → acceptance gate → English text
```

What is implemented vs intended — per-component verdict:

| Component | Intended | Actually implemented | Verdict |
|---|---|---|---|
| Frontend MediaPipe extraction | real webcam landmarks | `landmarks.ts` — real Tasks-Vision landmarkers, correct 33+468+21+21 order, raw coords, documented zero-fill | Correct, keep |
| WebSocket protocol | streaming frames + boundary | `translation.py` + `protocol.ts` — rolling interim (80/20) with repeat-agreement stabilization (`session_service.py`), `final:false/true` split, transport-tested | Correct, keep |
| FastAPI backend/services | translation use-case | `translation_service.py`, `session_service.py`, history/auth/capabilities — tested | Correct, keep |
| Adapter | raw→text pipeline | `sign_to_text_adapter.py` — full pipeline (preprocessing + motion gate + inference + decode + acceptance), strict bundle cross-validation | Correct, keep |
| Model | trained Transformer | `sign_to_text.py` d192×3L post-norm (v1 weights) — loads, runs, produces English-looking text | **Runs correctly; weights are NO-GO quality** |
| Gating | no silent bad output | `registry.py` refuses to serve unless `SIGNSAGE_MODEL_PROMOTED=true`; `render.yaml` pins `false`; `models/README.md` documents NO-GO | Correct, keep |
| Deployment | Render + bundle fetch | `render.yaml` + `fetch_bundle.sh` — download-at-start, checksum-verified, starts honestly without bundle | Correct, keep |

**The software pipeline is genuinely complete and honest. The only missing piece is a promotable checkpoint.** Nothing in the serving path needs rewriting.

## Exact live-inference contract answers (§3, items 1–11)

1. **Input shape**: raw `(T, 543, 3)` float32, component order POSE(33)+FACE(468)+LHAND(21)+RHAND(21). Asserted at `landmarks.py:assert_landmark_shape`, re-validated at WS schema boundary and in `prepare_live_landmarks`.
2. **Preprocessing**: reshape→`normalize_pose_sequence`→`resize_sequence(norm, max_frames=160)`. Returns `(processed (160,1629), true_len=min(T,160))`.
3. **Normalization** (`pose_normalize_v1`): per-sequence root-center on hip midpoint (global indices 23/24, fallback = mean of first 33 body landmarks if non-finite); scale = median-over-frames std of first-33 landmarks (fallback 1.0); `nan_to_num→0`. Backend port claims byte-parity with training.
4. **Masking/padding**: long sequences uniformly downsampled via `linspace` rounding; short ones zero-padded; `true_len` drives `padding_mask` for encoder self-attention and decoder cross-attention (`memory_key_padding_mask`). Correct.
5. **Temporal buffering**: `TranslationSession` buffers up to `max_frames*4`=640 frames (DoS cap); interim inference on trailing 80-frame window every 20 frames (~2 s cadence, measured ~50–70 ms CPU per interim, <5% overhead); first accepted interim emits immediately, changed text requires repeat-agreement on two consecutive windows; `sentence.boundary` consumes the full buffer for the final result.
6. **Architecture** (loaded bundle): `FrameProjection` (Linear 3258→384 → LayerNorm → GELU → Dropout → Linear →d → LayerNorm + depthwise Conv1d-k5) + sinusoidal PE + `nn.TransformerEncoder/Decoder`, **d_model=192, nhead=4, layers=3, ff=512, dropout 0.15-train/0.0-infer, post-norm** (`norm_first=False`). Adapter reconstructs from checkpoint `model_config`, defaulting missing `norm_first`→False so old bundles load byte-identically.
7. **Tokenizer**: SentencePiece unigram 16k, pad0/unk1/bos2/eos3, trained on train targets only; adapter asserts loaded SP IDs match manifest.
8. **Decoding**: autoregressive constrained greedy from BOS; bans PAD/BOS/UNK always; bans EOS for first `min_output_tokens=1` step; no-repeat-ngram n=3; repetition penalty 1.15; max 94 tokens; stop at EOS.
9. **Confidence/acceptance gates**: confidence = `exp(mean per-step logprob)` — explicitly documented as uncalibrated diagnostic. Gate: `motion ≥ 0.002`, `confidence ≥ 0.0` (disabled by default — deliberate, the notebook never validated a floor), non-empty text. Rejections return `accepted=false` with reason, never fake text.
10. **Frontend/backend consistency**: yes. Frontend sends raw coords, backend normalizes — exactly as the contract requires. Counts match (33/468/21/21). Three documented caveats, none blocking: (a) selfie-preview mirroring can swap L/R hands — noted in code as accepted calibration limit, consistent every frame; (b) training poses came from offline MediaPipe Holistic, live poses from Tasks-Vision landmarkers — same counts, small systematic differences possible; (c) training zeroes NaN/Inf inside normalization, serving rejects non-finite frames at the WS boundary — both guarantee the model never sees non-finite input, but an all-missing training frame becomes zeros while the equivalent live frame is rejected outright. Minor distributional edge, not a parity break.
11. **Train/inference preprocessing consistency**: yes by construction — `train_sign_to_text.py` imports the backend's own `normalize_pose_sequence`, `resize_sequence`, `SignTransformer`, and mask functions. **One real exception found**: the script's `batch_greedy` (used for the per-epoch diversity gate AND final test evaluation) does **not** apply the no-repeat-ngram ban or the repetition penalty that serving `greedy_decode_ids` applies (`grep` confirms `NO_REPEAT_NGRAM_SIZE`/`REPETITION_PENALTY` appear only as constants and in the export manifest, never in `batch_greedy`). So eval-measured diversity/quality is computed under slightly different decoding than what users would get. Small fix, must be applied before trusting gate numbers.

---

# 2. Current Training Architecture

- **Dataset**: iSign v1.1 (`iSign_v1.1.csv` + 4-part pose ZIP, ~158 GiB poses / ~228 GB total). Cleaning (`audit_dataset.py`, mirrors notebook): drop 2,291 malformed UIDs + 1,167 junk captions → ~123,779 usable; **video-level 80/10/10 split, seed 42** (train 99,699 / val 11,927 / test 12,153 — matches manifest; prompt's 98,485/13,014/12,280 are the pre-cleaning estimate, use the manifest numbers). Task is genuinely continuous pose→English. No leakage controls broken.
- **Pose pipeline**: `RemoteConcatFile` (`train_sign_to_text.py:127–272`) presents the 4 HF parts as one seekable file (probe via `Range: 0-0` + `Content-Range` parse; 32 MB chunks; 8-entry LRU ≈ 256 MB; 5 retries with exponential backoff; CDN re-resolution on expiry; strict range-coverage validation) → `zipfile` reads central directory → per-sample member fetch → `pose-format` decode → `normalize_pose_sequence` → mask → resize to 160×1629. Each DataLoader worker opens its own store (fork-safe). `num_workers=2` on CUDA, else 0. **No disk cache, no `persistent_workers`, no prefetch tuning in the v2 script** (the old notebook had an optional disk cache; v2 dropped it entirely).
- **Model (v2, verified in code)**: d256 / 4 heads / 4 layers / ff1024 / dropout 0.15 / **pre-norm** / label smoothing 0.1 / peak LR 3e-4 (6% warmup + cosine to 0.1×) / AdamW wd 1e-4 / grad clip 1.0 / seed 42 / batch 64 / AMP / ≤12 epochs / patience 3 on val loss / `--max-hours` graceful stop / `--resume` from recovery snapshot. **All V2 values confirmed present** (`train_sign_to_text.py:61–88, 612–632, 717–721, 753–756`). ~15M params vs v1 ~4M.
- **Tokenizer**: same procedure as v1 (SP unigram 16k on train text only, ids 0/1/2/3), pre-tokenized once in `Dataset.__init__`.
- **Gates**: (a) mandatory overfit gate — 64 samples, ≥95% exact within 600 steps at lr 1e-3, abort otherwise; (b) per-epoch diversity collapse gate — unique-fraction on 512 val decodes, STOP if <0.05 after epoch 2 (`COLLAPSED.txt`); (c) final promotion gate — beats trivial baseline on WER **and** BLEU, diversity OK, exact-match nonzero.
- **Eval/export**: WER (jiwer), BLEU (sacrebleu), ROUGE-L, exact match, nonempty rate, diversity, length stats, 30 samples, trivial-baseline comparison → `evaluation_report.json` → deployment bundle in the identical layout the backend loads + run manifest. Never overwrites `352868924`.

**Authoritative vs duplicated**: `research/training/train_sign_to_text.py` + `research/training/signsage_kaggle.ipynb` are the authoritative v2 entrypoint. `research/notebooks/signsage/signsage.ipynb` (81 cells) is the **v1 run record** — authoritative for v1 provenance (throughput benchmark, dynamic epoch budget, tiny-overfit loss 0.0013/100%), outdated as a training entrypoint. The v2 script is a faithful port **minus** the old notebook's throughput benchmark and dynamic epoch resolution (replaced by fixed `--epochs 12` + `--max-hours`), which is a minor regression worth noting.

---

# 3. What Is Already Correct (do NOT rewrite)

1. Frontend landmark extraction, WS protocol/client, reconnect/teardown.
2. WS translation route, session buffering, rolling-interim stabilization.
3. `SignToTextAdapter` + all four contract modules + preprocessing/postprocessing ports.
4. `SignTransformer` definition (loads both pre-norm and post-norm via `model_config`).
5. Promotion gating (`model_promoted=false` default, Render pins it, honest `MODEL_UNAVAILABLE`).
6. Cleaning policy, video-level split, tokenizer procedure, shift/masks/`ignore_index`, eval metric set + trivial baseline + diversity reporting.
7. `RemoteConcatFile` engineering (correct, well-validated HTTP logic) — its *existence* is fine; its *use as the per-epoch training feed* is the problem (§5).
8. Bundle/export layout and adapter-load self-check in the Kaggle notebook.

---

# 4. What Is Actually Broken / Weak

## Critical blockers

1. **v1 weights are collapsed (evidence: `evaluation_results.json`)** — WER 1.005, BLEU 0.24 (worse than trivial 0.40), ROUGE-L 0.089, 0.99% unique outputs, top hypothesis on 44.8% of test, generated mean length 8.2 vs reference 10.9. Decoder is in the target-language-prior regime. Correctly marked NO-GO everywhere.
2. **Training feed is remote-random-access with zero amortization** — every epoch re-fetches and re-decodes ~100k pose members over HTTP Range requests. At ~3 h/epoch for a 4M-param model on 2×T4 (READINESS §6), the pipeline is I/O-bound by roughly two orders of magnitude over GPU time. Any model choice fails through this feed.
3. **Even the 64-sample overfit gate traverses the remote feed** — the reported KeyboardInterrupt-during-gate is consistent with workers each opening a remote store (probes + EOCD fetch) then stalling on random member reads. The current Kaggle notebook works around this by *skipping* the gate (`--stages tokenizer,train`, "already passed 95.3%") — understandable triage, but it removes the safety gate from the exact runs that need it most.

## Major weaknesses

4. **Eval decode ≠ serving decode** (§1.11): `batch_greedy` omits the n-gram ban + repetition penalty. Diversity/gate numbers are measured under different decoding than production.
5. **V2 budget vs throughput**: ~3.5–4 h/epoch projected → 12 epochs ≈ 42–48 h ≈ 6–7 Kaggle sessions with manual save/restore/resume (`/kaggle/working` is ephemeral). One interrupted session without a saved output loses up to an epoch. The schedule, not the model, is the likely cause of the next failure.
6. **No temporal augmentation** (verified absent; `GESTURETALK_AUDIT.md` item 7 recommends adding time-masking/jitter before the run — still not implemented).
7. **Signer-independence unverified**: split is leak-safe by `video_id`, but signer overlap across videos was never checked (`audit_dataset.py` now reports it *if* a signer column exists; unconfirmed on the real CSV).

## Minor issues

8. `.env.example` default bundle path points at the NO-GO `352868924` bundle — safe today (promotion flag defaults false), but invites a dev-mode accidental load; should point at a placeholder.
9. Absolute quality ceiling is low regardless (see §6/§7: published iSign pose-to-text baselines peak at BLEU-4 ~1.5).
10. Scale ambiguity in BLEU reporting (sacrebleu 0–100 vs paper BLEU-4 0–100 points) — report both scales explicitly going forward to avoid comparing 0.24 against 1.47 incorrectly (both are ~zero-quality regime; the comparison direction still holds).

---

# 5. Current Bottleneck

**Data / I/O-bound. Primarily: remote ZIP random access + per-sample CPU pose decoding, repeated every epoch with no cache.** Evidence:

- `RemoteConcatFile` serves at most 8×32 MB cached chunks; a random 100k-sample epoch over 158 GiB guarantees near-zero chunk reuse across batches → effectively one or more HTTP Range round-trips per sample, each followed by zip decompression + `pose-format` parse + normalization in Python, under `num_workers=2` without persistent workers.
- Observed: 3 epochs in ~8.75 h on 2×T4 for a ~4M-param model at batch 64/AMP — GPU compute for this shape is milliseconds per batch; ~3 h/epoch can only be the feed. The reported "seconds per batch data loading vs dramatically shorter GPU step" matches exactly.
- The overfit gate stalling on 64 samples through the same feed confirms the bottleneck sits *before* model computation, not in it.
- Not GPU-bound (a faster GPU changes nothing while the feed starves it), not model-compute-bound, not memory-bound (8×32 MB cache + batch 64 × 160×1629 is modest).

Consequence per RULE 8: **do not spend anything on GPU upgrades; spend it on materializing the feed once** (local normalized pose cache reused across all epochs and all future model variants).

---

# 6. Model Alternatives

Common context: all options train on the same iSign continuous pose→English pairs; all can sit behind the current adapter with at most a projection-layer change; none can reuse v1 weights except A/F.

**A. Current V2 Transformer from scratch (d256×4L pre-norm, ~15M)**
Architecture: as §2. Data: full iSign via materialized cache. Compute: ~3.5–4 h/epoch local-feed estimate on T4-class; 6–12 epochs. Difficulty: zero (built). Expected quality: poor-to-marginal — v1 BLEU 0.24 after 3 stopped-early epochs; v2 with full schedule might reach ~1–2 BLEU, i.e. the published from-scratch regime, still not useful. Contract compat: native (1629→3258 internal). Risk: **high** (likely burns 20–40 GPU-hours to reproduce a known-weak regime). Realistic under deadline: runnable, but fails RULE 6 (low probability of useful output).

**B. Pretrained pose encoder + translation decoder (RECOMMENDED core, see §8)**
Architecture: frozen/pretrained text seq2seq (T5-small/base) as the language prior + a learned pose projection (160×1629 → T5 `d_model` via the existing `FrameProjection` + linear map) feeding the T5 encoder; fine-tune on iSign. This mirrors the iSign paper's own best pose-to-text family (T5+MediaPipe features, BLEU-4 1.47 — the highest published on this task). Data: same pairs. Compute: T5-small (~60M) fine-tune is heavier per step but needs far fewer steps because English generation is already learned — the exact deficit that collapsed v1. Difficulty: medium (one new module + HF transformers wiring + export change to bundle the projection with T5 weights). Quality: best expected of any option — beats from-scratch, plausible BLEU-4 1.5–3.5 range per published numbers. Contract compat: adapter loads a different checkpoint class; landmark contract unchanged (still 543×3 in, projection inside model). Risk: medium (engineering + export-schema change). Realistic: yes, if scoped to T5-small first.

**C. Fine-tune an existing iSign pose-to-text checkpoint**
Would be ideal; **no such downloadable checkpoint was found** (repo contains none; HF `Exploration-Lab/iSign` hosts data, not models; iSign paper links code/data; PoseStitch repo releases code+data, no weights located). Verdict: not actionable — do not plan around it. Revisit only if a checkpoint URL surfaces.

**D. Isolated-sign pretraining → continuous translation**
Use CISLR word-level ISL recognition (the iSign Task-3 dataset, I3D/pose classifiers ~16–20% acc. per paper) to initialize the pose encoder, then fine-tune the full encoder-decoder on continuous pairs. Legitimate transfer (encoder init + auxiliary loss), keeps the continuous task intact. Data: CISLR + iSign. Compute: two-stage, moderate. Difficulty: medium-high (second data pipeline + staged training). Quality: modest gain expected; weaker evidence than B on this dataset. Compat: encoder-swap only, contract unchanged. Risk: medium-high, slower payoff. Realistic: only as a follow-up if B underperforms; not the first bet.

**E. CTC gloss model → English generation**
Requires gloss (frame-aligned sign-word) labels; **iSign has no gloss annotations** (paper §5: gloss annotation at 118k scale infeasible; gloss-free is the explicit setting). Would need synthetic glosses (noisy) or a different dataset. Verdict: rejected — violates the dataset's actual annotation reality; high work for negative expected value.

**F. Temporal CNN/TCN + Transformer**
Replace/augment the depthwise Conv1d-k5 front-end with a deeper TCN before the same encoder-decoder. Data/compute similar to A. Difficulty: low-medium. Quality: incremental at best — v1's failure was decoder prior-dominance from under-training, not local temporal modeling (GESTURETALK audit item 8/12 confirm temporal modeling already present). Compat: native. Risk: medium (burns schedule on the wrong cause). Realistic: no.

**G. PoseStitch-style synthetic pretraining (BERT-GPT2 + template-stitched poses)**
The strongest published result on iSign pose-only gloss-free translation: BLEU-4 0.55→3.43 via pretraining on millions of synthetic sentences stitched from CISLR/WLASL word poses + linguistic templates, then fine-tuning (standard Transformer, BERT-4L/GPT2-4L, BPE 15k, 76-keypoint×xy 152-dim input). Architecture proven; code+data released. But: requires building the CISLR-word-pose pipeline, template generation at BPCC scale, 60k-step pretraining, AND a representation change (152-dim vs our 1629-dim contract + adapter/export rework). Difficulty: high; training time: longest of all options. Verdict: best long-term ceiling, **wrong choice under current time/compute** — adopt its insight (pretrained language prior + synthetic data help enormously) without adopting its full pipeline. Revisit post-deadline.

---

# 7. Pretrained Model Investigation

**Repository evidence**: no pretrained third-party weights exist in the repo. Only v1 run artifacts (git-ignored `research/artifacts/352868924/`), manifests, and reports. Nothing to fine-tune from locally.

**Web evidence** (separated as required):

- iSign paper (ACL 2024 Findings): pose-to-text baselines on iSign are uniformly weak — best BLEU-4 **1.47** (T5-base+MediaPipe(75)), SLT 0.36, T5-small 0.8, T5-large 0.09; ROUGE-L 7.6–19.1. Authors attribute this to gloss-free training + iSign's scale/diversity (118k vs Phoenix 8k). No downloadable model weights found.
- PoseStitch-SLT (2025, same lab): BERT-encoder/GPT2-decoder + synthetic pose-stitched pretraining → BLEU-4 **3.43 test** on iSign, SOTA for pose-only gloss-free. Uses 76 keypoints×(x,y)=152-dim input — **incompatible with the 543×3 contract without rework**. Code + synthetic data released; weights not located.
- CISLR (word-level ISL, ~4.7k prototypes) exists as a recognition resource — useful for option D, not a translation model.
- No verified ISL landmark-in/English-out pretrained checkpoint with compatible preprocessing, license, and continuous-translation support was found. Anything claiming otherwise without a URL + input-spec + license check is to be treated as unverified.

**Inference**: transfer must come from *general* pretrained models (T5 family — already proven on this exact task by the dataset authors), not from a mythical ISL-specific checkpoint.

---

# 8. Best Practical Strategy

**Primary: materialize the feed once, then fine-tune T5-small with a pose-projection front-end on iSign — keeping the entire serving stack and 543×3 contract.**

1. **One-time pose materialization job** (Kaggle, CPU-adequate): stream every needed `.pose` member through the *existing* `read_normalized_pose` once, write local `(160×1629 float16 + true_len + token ids)` shards (memmap/NPZ, ~100k×160×1629×2 B ≈ 50 GB in fp16 — fits Kaggle output), publish as a versioned Kaggle dataset. All subsequent training/gates/eval read locally. This single step removes the critical blocker for every option, restores the overfit gate to a seconds-long check, and makes epoch time GPU-determined.
2. **Model**: T5-small encoder-decoder (off-the-shelf HF weights) + `FrameProjection`-style pose front-end mapping 1629-dim frames (+internal deltas, reuse `FrameProjection` verbatim) to T5 `d_model=512`. Train front-end + fine-tune T5 (optionally frozen-encoder-first-epoch) with the existing loop mechanics (label smoothing, cosine, clip, patience 3, diversity gate with the §1.11 decode fix, same eval+trivial-baseline+export discipline; extend bundle schema to hold T5 weights + projection).
3. **Why this one**: it attacks the verified failure cause (decoder never left the language-prior regime — so *start* from a decoder that already models English) using the exact model family the dataset authors found best; it preserves 100% of the correct serving architecture; and the materialization step is reusable sunk value even if the model underperforms.

Honest expectation: success looks like BLEU-4 in the 1.5–3.5 band with diversity >5% and consistent trivial-beating — genuine, measurable, demonstrable ML translation of simple sentences, **not** general-purpose reliable translation (nobody has that on iSign; the SOTA is 3.43).

---

# 9. Fallback Strategy

**If transfer fails or time runs out: run the frozen V2 script unchanged against the materialized cache** (fix only the `batch_greedy` decode parity, §1.11). Cost: near-zero engineering; the cache makes even the full 12-epoch run plannable across sessions. It will likely reproduce the published from-scratch regime (BLEU ~1–2, possibly still gated NO-GO) — but it yields an honest, like-for-like baseline with a defensible evaluation record rather than a burned remote-streaming run. If it passes the gate, promote it; if not, the report still demonstrates genuine end-to-end ML inference with measured limits.

---

# 10. What NOT To Do

1. **Do not deploy or demo 352868924** as anything but a documented NO-GO (its random-input output, "the police said that…", reproduces the collapse — good for pipeline tests, poison for users).
2. **Do not run another 9-hour remote-streaming training** (v1 or v2) before materializing the feed — it re-pays the I/O tax every epoch and will likely stall at the same gate.
3. **Do not download 228 GB to the laptop as a training plan** (§9 below / §12 laptop answers) — and do not train on the laptop at all.
4. **No DTW/template matching, phrase dictionaries, hardcoded mappings, mock inference, or LLM-on-landmarks as the translator.** DTW stays an evaluation-side motion metric; LLMs stay on the English→gloss side where they already legitimately work (vocabulary-filtered). Any of these as the sign→text mechanism violates the project's core rule and the promotion policy.
5. **No isolated-sign classifier relabeled as translation.** Word-level work is legitimate only as encoder pretraining (option D) with honest labeling.
6. **No GPU upgrade, no TCN/param-tweaking, no representation changes** (wrist-centering, smoothing, interpolation, 32-frame windows — all correctly quarantined as v3 experiments in `GESTURETALK_AUDIT.md`) until the feed is local; none address the verified bottleneck or failure cause.
7. **Do not judge any run by loss curves or English-looking output** — promotion stays gated on WER+BLEU vs trivial, diversity, length stats, and qualitative samples.

---

# 11. Concrete Implementation Plan (no code changed yet — approve first)

1. **Isolate v1**: confirm `research/artifacts/352868924/` is read-only baseline; repoint `.env.example` bundle path to a placeholder (keep manifest/report references — they are the audit trail, not the hazard).
2. **Fix eval parity** (`train_sign_to_text.py:batch_greedy`): add the no-repeat-ngram ban + repetition penalty identical to serving; one-function change.
3. **Add temporal augmentation** (per GESTURETALK item 7): time-masking + temporal jitter in `SignToTextDataset.__getitem__` (train split only); representation-preserving.
4. **Write materialization script** (new, small): reuse `RemoteConcatFile` + `read_normalized_pose` + tokenizer to emit fp16 shards + manifest; run once on Kaggle; publish output as versioned dataset; verify with `--preflight-only` + shard checksums.
5. **Retarget loaders** to local shards (keep remote path as fallback for the gate/audit); restore the overfit gate into the notebook stages; re-add a lightweight throughput benchmark (v1 notebook had it; v2 dropped it).
6. **Implement T5-small pose-to-text module** (new file under `backend/app/ml/models/`, reusing `FrameProjection` + contracts); extend `export_bundle` + adapter `load()` for the new checkpoint class; keep v1-class loading intact.
7. **Train**: smoke → overfit gate (seconds, now local) → short fine-tune (patience 3, diversity gate armed) → evaluate/export → adapter-load check through the real serving path.
8. **Promote only on a passing gate**, per the existing rule. Otherwise record honestly and fall back (§9).

Files touched: `train_sign_to_text.py` (2 small fixes + local loader), one new materialize script, one new model file + adapter/export extensions, `.env.example` (1 line), `signsage_kaggle.ipynb` (stages). Serving path, frontend, contracts, preprocessing: untouched.

---

# 12. Final GO / NO-GO Decision

- **CURRENT MODEL (352868924): NO-GO** — collapsed (WER 1.005, diversity 0.99%, top prediction 44.8%), correctly gated, must never serve users. Safe to keep only as a pipeline-test baseline behind the dev flag.
- **BLIND V2 RETRAIN OVER REMOTE STREAMING: NO-GO** — correct config, wrong feed; high probability of repeating the stall at 6–7 sessions of cost.
- **RECOMMENDED APPROACH: materialize the pose feed once → fine-tune T5-small with pose-projection front-end on iSign, existing serving stack unchanged.**
- **WHY**: the failure was an under-trained decoder stuck in the language prior + a starving I/O feed; transfer supplies the missing English prior (the dataset authors' own best result uses this family) and materialization removes the verified bottleneck — the only combination with a reasonable probability of genuine, demonstrable translation under available compute.
- **ESTIMATED IMPLEMENTATION TIME**: materialization script + loader retarget + decode-parity fix + augmentation ≈ 1–2 focused days; T5 module + export/adapter extension ≈ 2–3 days.
- **ESTIMATED TRAINING TIME**: materialization ≈ 1 Kaggle session (one streaming pass, CPU-adequate); overfit gate ≈ minutes; fine-tune ≈ 1–3 sessions (fewer, shorter epochs than from-scratch; local feed makes time plannable).
- **EXPECTED RISK: MEDIUM** — engineering is bounded and serving is untouched; residual risk is the dataset's intrinsic difficulty (SOTA BLEU-4 3.43), so "useful" must be scoped to simple-sentence demonstration, not general translation.

## Laptop questions (§9), answered directly

1. **Can 228 GB fit?** Yes — 512 GB SSD + 1 TB HDD ≈ 1.5 TB raw; the 158 GiB pose archive fits on the HDD with room to spare.
2. **Technically feasible?** Yes, as dumb storage (plus `pose-format` for reads).
3. **Practical for training?** No — i3/8 GB/no-CUDA means CPU-only training measured in weeks per run; 8 GB RAM also constrains batching/caching.
4. **Faster than remote ZIP?** Yes — even HDD random seeks (~5–15 ms) beat per-sample HTTP round-trips plus 32 MB chunk amplification; local sequential reads are orders of magnitude faster.
5. **Does no-GPU kill it?** For training, yes, definitively.
6. **As dataset cache/source?** Plausible but low-value: the one-time download still has to happen somewhere, and getting 158 GiB from the laptop into Kaggle means re-uploading it (slow, version-fragile). Download directly onto Kaggle/output-dataset instead.
7. **Transfer to GPU env efficiently?** Same answer — the efficient move is downloading straight into the GPU environment (or a cloud bucket both can see), not laptop-as-middleman.
8. **Would it solve the bottleneck?** Only if training also moved to where the data *and* a GPU are. Local data + no GPU = a different bottleneck, not a solved one. **Do not do the laptop download.**
