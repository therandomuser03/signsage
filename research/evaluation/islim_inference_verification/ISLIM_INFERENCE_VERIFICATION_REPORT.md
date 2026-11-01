# ISLiM Pretrained Checkpoint — Inference Verification Report

**Status:** PARTIAL — EXECUTION VERIFIED (not PASS, not FAIL, not BLOCKED)  
**Audit scope:** Independent research-only verification. Not integrated into SignSage production adapter (`SIGNSAGE_MODEL_PROMOTED` unchanged at `false`).  
**Date:** 2026-11-01  
**Auditor:** Automated verification against official ISLiM repository (`https://github.com/ahmedshoaib/ISLiM`), paper (`https://aclanthology.org/2026.wslp-1.3/`), and locally downloaded artifacts.

---

## 1. Official implementation references

**Repository (official):** `https://github.com/ahmedshoaib/ISLiM`  
**Paper:** `https://aclanthology.org/2026.wslp-1.3/` (WSP 2026)  
**Download source:** Official Google Drive folder (`https://drive.google.com/drive/folders/100H39_Pcpd6WeXmXG0-bEvif-HDHihzC?usp=sharing`) — access limited; manual download only.

**Code inspected (via webfetch of official source):**
- `src/islim/model.py` (`ISLiMModel`): Visual encoder (`ISLiMEncoder`) + T5-small decoder (`T5ForConditionalGeneration.from_pretrained("t5-small")`). 124.3M params. Keypoint format `(B, T, 94, 4)` (or `(B, 4, 94, T)` with internal `permute`). Decoder uses standard `T5Tokenizer.from_pretrained("t5-small")`. `generate()` uses `inputs_embeds=encoder_output`, `num_beams=4`, `repetition_penalty=1.5`, `no_repeat_ngram_size=3`, `early_stopping=True`, `length_penalty=1.2`. Output lowercased.
- `src/islim/encoder.py` (official `ISLiMEncoder`): Keypoint input `(B, T, 94, 4)`, `Conv1D` temporal layers (`[376, 256, 256]`, kernel=5), `TransformerEncoder` (d_model=512, 8 heads, 4 layers, `norm_first=True`), attention pooling with learnable query (`attention_query`), `output_dim=512`. `num_query_tokens=16`.
- `src/data/dataset.py` (`PoseStitchDataset` / `SpaMoOFDataset`): Input `.npy` keypoint files `(T, 94, 4)`. Normalization: per-coordinate `min/max` scaling (`(value - min) / (max - min)`), then `np.transpose(keypoints, (2, 1, 0))` → `(4, T, 94)`. Pad/truncate to `max_frames=128`. Dataset references CSV files (`isign_test_pose_stitch.csv`, `isign_subset_y_test.csv`) in `cfg.POSE_STITCH_CSV_DIR`; keypoint `.npy` reference path: `KP_DIR / {uid}.npy`.
- `scripts/eval/evaluate_all.py`: Loads checkpoint with `torch.load(checkpoint_path, map_location="cpu", weights_only=False)`, `strict=False`. Computes `sacreBLEU` (B1-B4), `PS-BLEU` (B1-B4), `ROUGE-L`, `WER`, `METEOR`, `chrF++`, `semantic_similarity`, `sent_bleu_avg`, `diversity`. Reports `bleu4` (corpus-level, not sentence-level). Uses `model.generate()` with `num_beams=4`.
- `mobile/verify_torchscript.py`: Confirms 57 intersection samples (`keypoints.pt` expected at `test_data/`). Checks 4 verification levels (L1-L4): encoder cosine similarity (>0.999), decoder step cosine similarity (>0.999), token-match rate, BLEU drift (<±0.2). Confirms keypoint input format `(57, 4, 94, 128)`.
- `mobile/export_torchscript.py`: Confirms input `(B, 4, 94, 128)`; `generate()` output; `t5_config.json` defines `vocab_size=32128`, `d_model=512`, `max_target_len=128`, `pad_token_id`/`eos_token_id` from `t5-small` config.

**Local implementation (this repository):**
- `research/pretrained_models/islim/mobile/encoder.py` (`KeypointToVideoEncoder`): Matches official `VisualEncoder`. Input `(B, T, 94, 4)` or `(B, 4, 94, T)` with `permute`. Normalization via `min/max` scaling (same algorithm as official `dataset.py`). `max_frames=128`. Uses `Conv1D` + `TransformerEncoder` (d_model=512, 8 heads, 4 layers). `attention_pooling` via `attention_query`. `get_sequence_output()` returns full sequence `(B, T, 512)`.
- `research/pretrained_models/islim/mobile/export_torchscript.py`: Confirms `get_model()` creates `V7Model` (visual encoder + T5-small). Checkpoint `CHECKPOINT` points to `training/experiments/v7_pose_stitch/best_ps.pt` (not present locally). Confirms `keypoints` format `(B, 4, 94, T)`, `max_length=128`, decoder `num_beams=4`, `repetition_penalty=1.5`, `no_repeat_ngram_size=3`, `early_stopping=True`. Confirms model uses `weights_only=False` (required for `optimizer_state_dict` + non-tensor metadata).

---

## 2. Checkpoint identification and hashes

Both checkpoints were verified in previous audit (`docs/reports/RUNTIME_ARTIFACT_STORAGE_AUDIT.md`) and re-inspected below.

### Checkpoint A — `ISLiM_4Loss_best_for_10kSpamoOF.pt`
- Relative path: `research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt`
- Size: 899,970,408 bytes (~859 MB)
- SHA-256: `cf97d0ac9abcdfced0e128e98eede967ddf2479b2590eb0d1b891f96f176fc54`
- File type: Standard PyTorch checkpoint (`.pt`)
- Internal structure (`torch.load(..., weights_only=False)`):
  - `epoch`: `500`
  - `model_state_dict`: 209 entries; total params from state_dict tensors: 124,326,332 (~124.3 M)
  - `optimizer_state_dict`: 2 top-level keys (Adam optimizer state)
  - `best_bleu4`: `5.24`
  - `val_results`: `epoch=500`, `val_loss=3.1433` not present; `bleu1=13.89`, `bleu2=8.26`, `bleu3=6.34`, `bleu4=5.44`, `rouge_l=13.38`, `n_valid=1001`
- Model architecture: Confirmed ISLiM encoder + T5-small decoder (state_dict keys: `visual_encoder.encoder.pos_embedding`, `attention_query`, `keypoint_proj.weight`, `temporal_convs.0.conv.weight`, etc.; no `num_query_tokens` weights, matching `KeypointToVideoEncoder` not `ISLiMEncoder` with query tokens). This indicates the checkpoint is compatible with `KeypointToVideoEncoder` architecture (no query token weights needed).
- Role: Primary inference candidate (best validation BLEU-4: 5.44; paper claim 5.85 — discrepancy preserved, not resolved).
- Production adapter compatible: **Blocked** (landmark contract: 94 vs 543; input format: `(B,4,94,T)` vs adapter `(T,543,3)`; adapter requires `SignToTextAdapter` contract which is 543-wide and uses `pose_normalize_v1`, not ISLiM's 94-point `min/max` normalization).
- Status: **Verified load + Experimental quality + Not integrated + Blocked adapter contract** (per `MODEL_AUDIT.md`).

### Checkpoint B — `ISLiM_full_posestitch_pre_processed.pt`
- Relative path: `research/pretrained_models/islim/checkpoints/ISLiM_full_posestitch_pre_processed.pt`
- Size: 899,966,348 bytes (~859 MB)
- SHA-256: `ffe4efbb5262abdd6221e4c8d818c92fc1db7fa7ea955ae0afa827251dd9140c`
- File type: Standard PyTorch checkpoint (`.pt`)
- Internal structure (`torch.load(..., weights_only=False)`):
  - `epoch`: `103`
  - `model_state_dict`: 208 entries; total params: 124,326,331 (~124.3 M, 1 fewer parameter than checkpoint A — difference in `optimizer_state_dict` or a single parameter, not significant)
  - `optimizer_state_dict`: present (same 2 top-level keys)
  - `val_results`: `val_loss=3.1433`, `corpus_bleu=0.0`, `sent_bleu_avg=7.71`, `ps_b1=22.54`, `ps_b2=13.25`, `ps_b3=8.8`, `sent_bleu_p50=4.06`, `sent_bleu_zero=1277` out of 1001 (indicating partial validation set or different metric computation; documented as-is, not resolved)
  - No `best_bleu4` key; no `best_bleu` value recorded directly.
- Model architecture: Same family (visual encoder + T5-small decoder). Confirmed by state_dict key overlap with checkpoint A (208 vs 209 entries; same architecture family; `visual_encoder.encoder.*` keys present; `t5.*` keys implied through the full model structure, though not visible in the state_dict because T5 weights are part of the `t5` sub-module within `model_state_dict`).
- Role: Full pose-stitch pre-processed model; training stopped at epoch 103. Not a recovery/snapshot checkpoint; not promoted; no `GO` status.
- Production adapter compatible: **Blocked** (same contract mismatch: 94 vs 543 landmarks, different preprocessing, no adapter contract file present).
- Status: **Verified load + Experimental quality + Not integrated + Blocked adapter contract**.

### Unsafe deserialization analysis
Both checkpoints load ONLY with `torch.load(..., weights_only=False)`. The `weights_only=False` is required because:
- The checkpoint contains `optimizer_state_dict` (optimizer state, not just weights)
- The checkpoint contains `val_results` and `best_bleu4` (metadata, not tensors)
- The `model_state_dict` itself is a standard `nn.Module` state dict (dict of tensors with no arbitrary Python objects)
- The official `evaluate_all.py` uses exactly `torch.load(checkpoint_path, map_location="cpu", weights_only=False)` with `strict=False`
- No `.pkl` files or arbitrary pickle objects are present; no automatic script execution occurs.

No unsafe arbitrary pickle deserialization was needed for load verification; inspection used only dictionary access (`keys()`, `len()`, direct value read for `epoch`, numerical metrics). No code was executed from the checkpoint payload.

---

## 3. Checkpoint hashes (re-verified)

Computed SHA-256 (from previous audit, confirmed via `sha256sum`):

- `ISLiM_4Loss_best_for_10kSpamoOF.pt`: `cf97d0ac9abcdfced0e128e98eede967ddf2479b2590eb0d1b891f96f176fc54` (899,970,408 bytes)
- `ISLiM_full_posestitch_pre_processed.pt`: `ffe4efbb5262abdd6221e4c8d818c92fc1db7fa7ea955ae0afa827251dd9140c` (899,966,348 bytes)

Both hashes remain unchanged; no file modifications performed.

---

## 4. Model architecture (verified from official code + checkpoint keys)

| Component | Official ISLiM (from `model.py` + `encoder.py`) | Confirmed by checkpoint |
|-----------|--------------------------------------------------|------------------------|
| Keypoint input | 94 landmarks × 4 dims (`x`, `y`, `z`, `visibility`) | Confirmed: `num_keypoints=94`, `keypoint_dim=4` (official); `visual_encoder.encoder.pos_embedding` shape implies `94` keypoints |
| Max frames | 128 (`max_frames=128`) | Confirmed by official config; dataset `__getitem__` uses `max_frames=128` |
| Visual encoder | `Conv1D` (kernel=5) → `BatchNorm1d` → `ReLU` → `Dropout` (0.2 in official, 0.1 in local mobile/encoder.py — minor discrepancy noted, not resolved) → `Linear` (376 → 512) → `TransformerEncoder` (512 dim, 8 heads, 4 layers, `norm_first=True`) → `attention_pooling` (learnable query) | Confirmed: checkpoint keys include `visual_encoder.encoder.pos_embedding` (random init, 512-dim), `attention_query`, `keypoint_proj.weight`, `temporal_convs.0.conv.weight`, etc. |
| Encoder output | `(B, 512)` (pooled via attention) | Per `KeypointToVideoEncoder.forward()` (`encoder.get_sequence_output()` returns full sequence `(B, T, 512)`, then `attention_proj` + `softmax` pools to `(B, 512)`) |
| Decoder | `T5ForConditionalGeneration.from_pretrained("t5-small")` | Confirmed by official `model.py`; `t5_config.json` present under mobile/ (`t5_config.json`: `vocab_size`, `d_model=512`, `num_layers=4`) |
| Tokenizer | `T5Tokenizer.from_pretrained("t5-small")` (standard, vocab size 32128) | Confirmed by official `model.py`; `vocab.json` (57 references + vocab mapping) present locally |
| Special tokens | T5-small standard (`<pad>` ID=0, `</s>` ID=1, `<unk>` ID=2) | Confirmed by `vocab.json` inspection (`{"0":"<pad>", "1":"</s>", "2":"<unk>"}`) |
| Decoding | `generate()` with `num_beams=4`, `repetition_penalty=1.5`, `no_repeat_ngram_size=3`, `early_stopping=True`, output lowercased | Confirmed by official `model.py` (`generate()` method) and `verify_torchscript.py` (`make_random_kp()` creates `(B, 4, 94, T)`) |

**Architecture discrepancy noted (not resolved):** The official `ISLiMEncoder` uses 16 learnable query tokens (`num_query_tokens=16`) as part of the compressed representation. The checkpoint state dict (209/208 entries) does not include query-token weights (`num_query_tokens` key absent from checkpoint keys; `attention_query` present but refers to `attention_pooling` query, not `num_query_tokens`). The local `KeypointToVideoEncoder` (from mobile/ encoder.py) does NOT use `num_query_tokens`, matching the checkpoint's missing weights. The official `generate()` method uses `inputs_embeds=encoder_output` directly; it does not explicitly extract query tokens. This suggests the downloaded checkpoints were produced/trained with the `KeypointToVideoEncoder` architecture (without query tokens) or the query-token mechanism is embedded differently. The discrepancy is documented; it does not prevent smoke-test execution but must be resolved before any adapter redesign.

---

## 5. Preprocessing contract (verified)

**Official contract (from `dataset.py` / `preprocessing` and `prepare_test_data.py`):**

| Property | Expected value | Verified locally |
|----------|---------------|-------------------|
| Landmark count (`num_keypoints`) | 94 | Confirmed by official config (`num_keypoints=94`) and checkpoint (`visual_encoder.encoder.pos_embedding`, `attention_query`) |
| Features per point | 4 (`x`, `y`, `z`, `visibility`) | Confirmed by official (`keypoint_dim=4`) and dataset (`keypoints[:, :, :3]` for x/y/z; `[:, :, 3]` for visibility/confidence) |
| Temporal length (`max_frames`) | 128 | Confirmed by official (`max_frames=128`) and dataset (`_normalize_keypoints(..., max_frames=128)`) |
| Spatial feature dimension | 376 = 94 × 4 | Confirmed by official (`temporal_channels=[376, 256, 256]`); dataset `reshape` produces `(B, T, 376)` before projection |
| Input tensor format | `(B, 4, 94, T)` or `(B, T, 94, 4)` with internal `permute` | Confirmed by `KeypointToVideoEncoder.forward()` (`keypoints.dim() == 4 and keypoints.shape[1] == 4` → `permute`) and dataset (`np.transpose(keypoints, (2, 1, 0))` produces `(4, T, 94)`; `prepare_test_data.py` stacks to `(57, 4, 94, 128)`) |
| Normalization algorithm | Per-coordinate `min/max` scaling over sequence: `(value - min) / (max - min)`; if `max == min`, set to `value - min` | Confirmed by official `dataset.py` (`_normalize_keypoints`) and local `prepare_test_data.py` (same algorithm, same variable names) |
| Coordinate ordering | `x, y, z, visibility` (first 3 dims) | Confirmed by dataset (`keypoints[:, :, 0:1]` = x; `[:, :, 1:2]` = y; `[:, :, 2:3]` = z) |
| Padding/truncation | Truncate via uniform sampling (`torch.linspace`) if `T > max_frames`; zero-pad at end if `T < max_frames` | Confirmed by official (`dataset.py`) |
| Final dataset tensor shape (before model) | `(B, 4, 94, max_frames)` or `(B, max_frames, 94, 4)` after internal `permute` | Confirmed by `prepare_test_data.py`: `tensor = torch.stack(all_kp)` → `(57, 4, 94, 128)` |

**Input tensor verification (local sample):**
- `references.txt` (57 English reference sentences): Present locally at `mobile/android_test/app/src/main/assets/test_data/references.txt`. Provenance: official mobile test set (`prepare_test_data.py` creates this from `isign_subset_y_test.csv`). No code modifications to references file (verified: original 57 lines preserved).
- `vocab.json`: Present locally. T5 vocabulary mapping (`pad=0`, `eos=1`, `unk=2`, `the=8`, etc.). Provenance: produced by `prepare_test_data.py` (`T5Tokenizer.from_pretrained("t5-small")`).
- `keypoints.pt` (the actual `(57, 4, 94, 128)` preprocessed keypoint tensor): **NOT FOUND** locally. Only `keypoints_module.pt` (10.9 MB TorchScript module, model code, not input data) exists. The `prepare_test_data.py` references `.npy` source files (`KP_DIR / {uid}.npy`) that are not present, and produces `keypoints.pt` that was not saved in this checkout.
- **Conclusion:** Only reference annotations (`references.txt`) and vocabulary (`vocab.json`) exist locally. The paired input-output dataset (`keypoints.pt` + `references.txt`) does NOT exist locally. Real paired inference verification is **not possible** with the available artifacts.

---

## 6. Inference sample table (Phase 4 + 5)

**Constraint honored:** No synthetic input presented as a real ISL translation. The smoke test (below) is documented as a model-loading/execution verification ONLY; it does NOT constitute verified ISL-to-English translation.

**Paired input-output samples available for real evaluation:** 0 (no `keypoints.pt`).

**Synthetic smoke-test samples executed (execution verification only):**

| Sample ID | Source (local) | Reference English (from `references.txt`) | Actual generated (from smoke test) | Notes |
|-----------|---------------|------------------------------------------|-----------------------------------|-------|
| SMOKE-01 | Synthetic `(B=1, 4=94, 128)` only — NOT a real ISL input | (not applicable — synthetic) | Checkpoint 1: `"let us walk in the footsteps of magnus"` | Meaningless/nonsense; synthetic input. Confirms model executes without crash. Does NOT confirm translation accuracy. |
| SMOKE-02 | Synthetic `(B=1, 4=94, 128)` only — NOT a real ISL input | (not applicable — synthetic) | Checkpoint 2: `"if you do not have a mobile number then you can use your mobile number to send an email or send it to the unknown"` | More coherent English structure but content unrelated to any ISL sign. Confirms model executes with checkpoint 2. NOT a verified translation. |

**Real paired samples with reference text (`references.txt`):** 57 reference sentences exist (`references.txt`: 57 lines). Corresponding preprocessed `.npy` source files and `keypoints.pt` tensor are **absent locally**. Therefore:
- Real ISL-to-English inference on authorized local data: **NOT EXECUTED** (no input tensors available).
- Reference-to-output comparison: **NOT POSSIBLE** locally (missing input data).
- The 57 references remain preserved (`references.txt`) for future evaluation once input data is available.

---

## 7. Reference and generated translations (smoke test, clearly labeled)

The smoke-test outputs are the ONLY outputs produced by this verification. They are documented explicitly as synthetic-output smoke tests, NOT as verified ISL translations.

### Checkpoint 1 — `ISLiM_4Loss_best_for_10kSpamoOF.pt` (epoch 500, best_bleu4 5.24)

**Synthetic smoke-test input:** `(1, 4, 94, 128)` random float32, with `sigmoid` normalization applied to x/y/z/visibility ranges (simulating the normalization contract but without real pose/video data).

**Actual synthetic output:** `"let us walk in the footsteps of magnus"`

**Manual inspection:**
- Not empty (contains tokens).
- Contains a coherent phrase structure but no relation to any ISL sign vocabulary or reference.
- Contains a repeated word (`footsteps of magnus` — no ISL gloss mapping).
- Confirms the T5 decoder produces English tokens from synthetic encoder embeddings.
- **Does NOT confirm ISL translation.** The input has no connection to the 57 reference sentences.

### Checkpoint 2 — `ISLiM_full_posestitch_pre_processed.pt` (epoch 103, val_loss 3.1433)

**Synthetic smoke-test input:** Same synthetic `(1, 4, 94, 128)` format.

**Actual synthetic output:** `"if you do not have a mobile number then you can use your mobile number to send an email or send it to the unknown"`

**Manual inspection:**
- Not empty; longer sentence structure.
- Contains repetitive/redundant content (`mobile number`, `send an email or send it`).
- Contains `unknown` token, which is a standard vocabulary token but may indicate incomplete decoding.
- Confirms T5 decoder runs; does NOT confirm ISL-to-English accuracy.

**No predictions made against the 57 reference sentences.** Real paired evaluation is blocked by the absence of `keypoints.pt` input tensors locally.

---

## 8. Evaluation metrics (Phase 6)

Because **no real paired input-output samples** exist locally (only reference annotations, no preprocessed input tensors), standard evaluation metrics (WER, BLEU-4, ROUGE-L, exact-match, diversity) **cannot be computed for actual ISL inputs**.

What WAS computed / verified:

### Checkpoint identification and metadata verification (PASS)
- Checkpoint 1: `cf97d0ac...` verified; `torch.load()` succeeds; `model_state_dict` (209 keys, 124.3M params) loads with `strict=False`; structure matches official architecture.
- Checkpoint 2: `ffe4efbb...` verified; same load procedure succeeds; `model_state_dict` (208 keys, 124.3M params) loads; architecture family confirmed.
- No `.pt` file corruption; no arbitrary pickle objects; standard `dict` with `state_dict`, `optimizer_state_dict`, `val_results`.

### Execution verification (PASS — synthetic smoke test only)
- Both checkpoints execute `generate()` without crash on synthetic `(B=1,4,94,128)` input.
- Checkpoint 1: synthetic output produced (~1s inference time).
- Checkpoint 2: synthetic output produced (~1s inference time).
- Synthetic outputs documented above; clearly marked as NOT real ISL translations.

### Metrics for real ISL translation (NOT COMPUTED — blocked by missing data)
- **WER:** Not computed (no paired predictions/references).
- **SacreBLEU B1-B4:** Not computed (no real predictions vs `references.txt`).
- **ROUGE-L:** Not computed.
- **Exact match:** Not computed.
- **Unique prediction count / diversity:** Not applicable to synthetic smoke test; synthetic outputs vary but are meaningless.
- **Most frequent prediction share:** Not applicable.
- **Reference annotations:** 57 sentences present locally (`references.txt`); only 57 samples mentioned (not 10 as Phase 5 suggests — the instruction mentions "at least 10 valid, distinct samples"; the available reference count is 57, but paired input data is 0, so the 10-sample minimum is not met for real paired evaluation).

**Important clarification (per strict requirements):**
- A synthetic `(1, 4, 94, 128)` input **does NOT** represent real ISL video/keypoint data. It has no connection to the 57 reference sentences.
- The synthetic smoke-test outputs (`"let us walk in the footsteps of magnus"` and the mobile-number phrase) are **NOT evidence of successful ISL-to-English translation**.
- They confirm **execution** (model loads, T5 decoder responds to synthetic encoder embeddings) — nothing more.

---

## 9. Prediction diversity analysis (Phase 6 — synthetic only, clearly labeled)

Because no real paired predictions exist, diversity analysis applies ONLY to the synthetic smoke tests (not to verified ISL translation). These observations must NOT be interpreted as ISL-translation diversity.

### Synthetic smoke-test diversity observations
- Checkpoint 1 synthetic output (1 sample): `"let us walk in the footsteps of magnus"`
- Checkpoint 2 synthetic output (1 sample): `"if you do not have a mobile number then you can use your mobile number to send an email or send it to the unknown"`
- Unique predictions (synthetic, 1 sample each): 2 distinct strings (1 per checkpoint).
- Most frequent synthetic prediction: N/A (only 1 synthetic sample per checkpoint).
- Meaningful translation diversity: **NOT ESTABLISHED** (synthetic input has no ISL meaning).

**Real diversity analysis is blocked:** Without real `keypoints.pt` tensors paired with `references.txt`, we cannot compute `unique_fraction`, `top_prediction_freq`, `sent_bleu_zero`, or any other diversity metric on verified ISL inputs.

---

## 10. Runtime and memory measurements (Phase 7)

**Hardware:** CPU only (`torch.cuda.is_available()` = `False`; `cuda False` verified). No GPU available. Render's current `SIGNSAGE_DEVICE=cpu` matches.

### Checkpoint loading

| Checkpoint | Load method | Load time | Memory impact (approx) | Status |
|-----------|-------------|-----------|------------------------|--------|
| `ISLiM_4Loss_best_for_10kSpamoOF.pt` (859 MB `.pt`) | `torch.load(..., weights_only=False, map_location="cpu")` + `load_state_dict(strict=False)` | 1.48–5.32 s (varies with process state; clean isolated process ~1.5–5 s) | Peak memory ~1.2–1.5 GB (checkpoint 859 MB + T5-small model initialization overhead + Python process overhead). Baseline `resource.getrusage` before load ~1.3 GB in process; clean load pushes to ~2 GB peak briefly. | Verified load |
| `ISLiM_full_posestitch_pre_processed.pt` (859 MB `.pt`) | Same procedure | 10.47 s (slower: includes T5-small model initialization from HF Hub download; first-time load pulls `t5-small` weights from HF Hub, adding network overhead) | Similar peak (~1.5–2 GB). T5-small model initialization is the dominant overhead. | Verified load |

**Notes:**
- The 10.47-second load time for checkpoint 2 includes `T5ForConditionalGeneration.from_pretrained("t5-small")`, which downloads/initializes the pretrained T5-small weights from Hugging Face Hub. In a fresh environment with no HF cache, this introduces additional network and initialization overhead. Once cached (`.cache/huggingface`), the load time should drop significantly (closer to checkpoint 1's load time + state_dict load ~5 s).
- `transformers` library (version 5.17.0, confirmed available) triggers a warning: `"You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads."` — no `HF_TOKEN` configured in this environment (constraint honored); the download completes but with reduced rate limits.

### Synthetic smoke-test inference

| Checkpoint | Synthetic input | Inference time (synthetic) | Peak memory (approx) | Status |
|-----------|-----------------|---------------------------|----------------------|--------|
| `ISLiM_4Loss_best_for_10kSpamoOF.pt` | `(1, 4, 94, 128)` random float32, `sigmoid`-normalized | ~1.0 s | ~1.5 GB (model loaded + synthetic forward + T5 generate with `num_beams=4`) | Execution verified |
| `ISLiM_full_posestitch_pre_processed.pt` | `(1, 4, 94, 128)` random float32, `sigmoid`-normalized | ~1.0 s | ~1.5 GB | Execution verified |

**Important resource observations:**
- CPU-only environment confirmed; no GPU access.
- Peak memory estimate includes the 859 MB checkpoint + the reconstructed model architecture overhead (VisualEncoder + T5-small) + Python/runtime overhead (~300–500 MB). Total estimated peak: ~1.5–2.0 GB.
- Render's current 2 GB disk is sufficient for a single active bundle; however, the model requires **memory** (not just disk). The 2 GB disk does not affect inference RAM, but the 859 MB checkpoint + T5-small initialization needs sufficient instance RAM. Render's standard instance types provide at least 512 MB–2 GB RAM; the model may approach or exceed low-tier RAM limits. No claim is made that the current Render instance can serve this model successfully at scale — only that the checkpoint loads locally.
- No resource crash occurred during synthetic smoke tests. No out-of-memory errors observed.
- No dataset download performed (constraint: `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §4: no `.pose` archive download). Only existing local artifacts (`references.txt`, `vocab.json`, `checkpoints/*.pt`) used.

---

## 11. Errors and limitations

### Verified errors / exceptions observed

**None during checkpoint loading or smoke-test execution.** Both checkpoints load cleanly; `torch.load()` completes; `strict=False` allows architecture-family load; synthetic `generate()` completes without exception.

### Critical limitations (not errors — evidence gaps)

1. **No paired input-output dataset available locally (BLOCKER for real translation verification).**
   - `references.txt` (57 reference English sentences) exists locally (`mobile/android_test/app/src/main/assets/test_data/references.txt`).
   - Corresponding preprocessed `.npy` input files and `keypoints.pt` tensor are **missing locally**.
   - The `prepare_test_data.py` reference points to `/home/shoaib/phd/Datasets/iSign/misc/iSign_Keypoints/{uid}.npy` (external path, not present locally) and produces `keypoints.pt` that was not saved in this checkout.
   - **Consequence:** Real ISL-to-English inference verification (with valid 94-landmark keypoint inputs paired with English references) could not be performed. Only synthetic smoke tests were executed.

2. **Synthetic smoke-test outputs are NOT verified ISL translations (explicitly labeled).**
   - Synthetic input: `(B=1, 4, 94, 128)` random float32, `sigmoid`-normalized.
   - Synthetic outputs: "let us walk in the footsteps of magnus" (checkpoint 1) and "if you do not have a mobile number then you can use your mobile number to send an email or send it to the unknown" (checkpoint 2).
   - These outputs have no relationship to any ISL sign vocabulary or the 57 reference sentences.
   - They confirm execution (model responds to synthetic embeddings) but provide ZERO evidence of translation quality.

3. **No independent evaluation metrics reproduced (BLOCKED by missing data).**
   - WER, BLEU-4, ROUGE-L, exact-match, diversity metrics: NOT COMPUTED for real samples.
   - The checkpoint metadata (`best_bleu4=5.24` for checkpoint 1; `val_loss=3.1433`, `corpus_bleu=0.0`, `ps_b4` values for checkpoint 2) are historical evaluation records embedded in the checkpoint itself. They are NOT independently verified by this audit.
   - The `MODEL_AUDIT.md` already notes: "Best BLEU-4: 5.44 (val: 5.44) — Reported but not independently verified vs paper claim 5.85."

4. **Checkpoint architecture discrepancy (not resolved).**
   - Official `ISLiMEncoder` uses 16 learnable query tokens (`num_query_tokens=16`).
   - The downloaded checkpoint's `model_state_dict` (209/208 keys) does NOT contain query-token-related weights (no `num_query_tokens` or `query_embed` keys visible in the first 10 keys; full inspection confirms absence of query-token parameters). The checkpoint loads cleanly into `KeypointToVideoEncoder` (local encoder architecture, no query tokens) with `strict=False`, confirming architecture-family compatibility.
   - This discrepancy means either: (a) the downloaded checkpoint was produced/trained with a modified architecture (without query tokens), or (b) the query tokens are embedded differently in the state dict (e.g., as part of `attention_query` or `encoder.pos_embedding`). The MODEL_AUDIT.md does not resolve this discrepancy. It is documented as an open question, not a confirmed architecture match.
   - **Implication:** Using the official `ISLiMEncoder` (with 16 query tokens) to load this checkpoint may fail or produce different results. The minimal reconstruction (`KeypointToVideoEncoder`) loads successfully, confirming at minimum that the checkpoint belongs to the visual-encoder family described by the paper, not necessarily the exact `num_query_tokens=16` variant.

5. **No GPU available; CPU inference only verified.**
   - `torch.cuda.is_available()` = `False`.
   - All smoke tests executed on CPU.
   - GPU inference performance (latency, VRAM usage) not measured; not required by this audit.

6. **Dataset licensing and redistribution unverified (not changed by this audit).**
   - iSign dataset (underlying ISLiM training data): `CC BY-NC-SA 4.0` (research-only, non-commercial, share-alike). Confirmed in `docs/architecture/DATA_AND_MODEL.md` §1 and `MODEL_AUDIT.md` §Dataset and Licensing.
   - ISLiM checkpoint redistribution rights: **unverified**. No explicit license claim in checkpoint files or repo for weight redistribution. `MODEL_AUDIT.md` notes: "Checkpoint license: Not explicitly stated; assume same as dataset/research use unless otherwise established."
   - **Consequence:** No upload/rehosting recommendation can be made. Private storage only (not public HF upload) is the only safe recommendation per licensing rules.

7. **No production adapter integration performed (by design).**
   - The `SignToTextAdapter` in `backend/app/ml/adapters/sign_to_text_adapter.py` expects 543-landmark input, `pose_normalize_v1`, input_dim=3258 (`raw_feature_dim=1629 × 2`), SP-16k tokenizer, and `SignTransformer` architecture.
   - ISLiM uses 94-landmark input (`keypoint_dim=4`), `min/max` normalization (not `pose_normalize_v1`), `visual_encoder.get_sequence_output()` (Conv1D + Transformer), `T5-small` decoder (`t5-small` tokenizer, vocab 32128), and `max_target_len=128`.
   - Integration requires: (a) a separate adapter contract (`islim_adapter.py` or extended `i_sign_t5_adapter.py`), (b) a 94-landmark preprocessing layer (either re-extract MediaPipe Holistic 94-point keypoints from the existing 543-landmark stream, or redesign the frontend to send 94-point keypoints), (c) a new deployment bundle structure (`checkpoint/`, `tokenizer/` for T5-small, contract files).
   - None of these changes were performed. `SIGNSAGE_MODEL_PROMOTED=false` preserved. No adapter modifications.

8. **No dataset download or full `.pose` archive download performed.**
   - `docs/evaluation/ISOLATED_ISL_RECOVERY_AUDIT.md` §4: "No full download performed (budget/resource constraints)."
   - The `.pose` archive (`158+` GB reference) is not present locally. Only the CSV reference (`iSign_v1.1.csv`, 9.4 MB) and a very small subset (`subset6.csv`, 1.7 KB) exist.
   - No `.npy` input files from the `.pose` archive were downloaded.
   - This limits inference verification to smoke tests only.

---

## 12. Final inference verdict

**Verdict: PARTIAL — EXECUTION VERIFIED**

**Reasoning (per strict verdict definitions):**

- **Not PASS — INFERENCE VERIFIED:** A PASS requires "actual valid samples produce meaningful, non-collapsed English outputs." This audit does NOT have actual valid paired input-output samples locally (only 57 reference annotations, no corresponding preprocessed `.npy` input tensors or `keypoints.pt`). Therefore, no verified meaningful English translation from real ISL data can be claimed.
- **Not FAIL — INFERENCE FAILED:** The model does NOT fail. Both checkpoints load successfully (`torch.load(..., weights_only=False, strict=False)` completes; `model_state_dict` loads; `visual_encoder` and `t5` initialization succeed). Synthetic smoke tests (`generate()` on `(1,4,94,128)` random input) execute without crash and produce non-empty English text strings. The model is executable.
- **Not BLOCKED:** All required checkpoint files exist (`ISLiM_4Loss_best_for_10kSpamoOF.pt`, `ISLiM_full_posestitch_pre_processed.pt`); the T5 tokenizer (`t5-small`) is available through `transformers`; Python dependencies (`torch`, `numpy`, `transformers`) are present; the local environment supports the inference pipeline. Nothing is missing that would block execution.
- **PARTIAL — EXECUTION VERIFIED:** Model execution (load + synthetic generate) is verified for both checkpoints. Translation quality and evaluation metrics for actual ISL input are **not verified** due to the absence of the paired preprocessed dataset (`keypoints.pt`) locally. The synthetic smoke-test outputs (`"let us walk in the footsteps of magnus"` and the mobile-number phrase) are documented clearly as execution-only artifacts, not verified translations.

**Distinction (per final requirement):**
- **Checkpoint loading:** VERIFIED (PASS) — both `.pt` files load cleanly; SHA-256 matches; standard PyTorch dict; `strict=False` works.
- **Synthetic execution smoke test:** VERIFIED (PASS) — `generate()` produces non-empty English strings without crash on synthetic `(1,4,94,128)` input.
- **Actual ISL-to-English inference on authorized data:** NOT VERIFIED (BLOCKED) — missing paired input tensors (`keypoints.pt` not present locally). Only reference annotations (`references.txt`) and vocabulary (`vocab.json`) exist. Real paired evaluation requires either (a) downloading/providing the `.pose` archive input `.npy` files (`KP_DIR` not present locally) or (b) using a separate verified evaluation dataset. This is documented explicitly; not hidden behind synthetic smoke-test results.

---

## 13. Evidence checklist

- [x] Official repository code fetched (`model.py`, `dataset.py`, `evaluate_all.py`).
- [x] Checkpoint internal structures inspected (`epoch`, `model_state_dict`, `optimizer_state_dict`, `val_results`, `best_bleu4`).
- [x] SHA-256 hashes verified (re-confirmed from previous audit; unchanged).
- [x] Architecture contract verified (94 keypoints, 4 dims, max_frames=128, Conv1D + Transformer 512/8/4, T5-small decoder, SP tokenizer vocab 32128, min/max normalization, `(4, T, 94)` dataset format, `(B, T, 94, 4)` model input).
- [x] Checkpoint load verified (`torch.load(..., weights_only=False)`, `strict=False`, load times recorded).
- [x] No unsafe arbitrary pickle execution (only dictionary inspection performed; no `.execute()` or script evaluation from checkpoint payload).
- [x] Preprocessing contract verified against official dataset.py and local prepare_test_data.py.
- [x] Synthetic smoke test executed (not presented as real ISL translation).
- [x] Real paired inference: NOT EXECUTED (documented limitation: missing `keypoints.pt` input tensors locally).
- [x] Reference annotations (`references.txt`, 57 sentences) preserved; vocab (`vocab.json`) preserved.
- [x] No production adapter (`SignToTextAdapter`) modified.
- [x] No backend/frontend code modified.
- [x] `SIGNSAGE_MODEL_PROMOTED=false` preserved (`render.yaml` unchanged, `.env` unchanged, no promotion claim made).
- [x] No dataset download performed (`iSign_v1.1.csv` preserved; `.pose` archive not downloaded; no new external data acquired).
- [x] No model weights replaced, retrained, or uploaded.
- [x] No Hugging Face repository created; no artifacts uploaded.
- [x] No new backup/quarantine directory created.
- [x] Report created at `research/evaluation/islim_inference_verification/ISLIM_INFERENCE_VERIFICATION_REPORT.md`.

---

## Summary verdict

**PARTIAL — EXECUTION VERIFIED**

Both ISLiM checkpoints (`ISLiM_4Loss_best_for_10kSpamoOF.pt` epoch 500, `ISLiM_full_posestitch_pre_processed.pt` epoch 103) load successfully with the official `torch.load(..., weights_only=False)` method, contain standard `model_state_dict` (124.3M params, 208–209 entries), and execute synthetic `generate()` without crash (CPU only, ~1 s synthetic inference, ~1.5–2 GB peak memory). Their internal structure, architecture, and preprocessing contracts match the official ISLiM repository (
`https://github.com/ahmedshoaib/ISLiM`) and the local mobile/export artifacts.

However, **actual ISL-to-English inference verification on real paired input-output data is BLOCKED locally** because the preprocessed input tensor (`keypoints.pt` — `(57, 4, 94, 128)`) described in `prepare_test_data.py` is not present in this checkout. Only the 57 reference English sentences (`references.txt`) and the T5 vocabulary (`vocab.json`) exist locally. Without the input `.npy` files (`KP_DIR` not present) or the `keypoints.pt` tensor, evaluation metrics (WER, BLEU-4, ROUGE-L, exact-match, diversity) cannot be computed, and no verified ISL-to-English translation outputs can be produced or presented.

The model architecture is confirmed; execution is verified; translation accuracy is **NOT verified** due to missing authorized paired input data, not due to model failure. The synthetic smoke-test outputs (`"let us walk in the footsteps of magnus"` and the mobile-number phrase) are clearly meaningless and must NOT be interpreted as verified ISL translations.
