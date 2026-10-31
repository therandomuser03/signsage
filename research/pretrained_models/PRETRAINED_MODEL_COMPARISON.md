# Pretrained Model Comparison — SignSage Acquisition Audit

## Overview
Two candidates audited for possible integration as SignSage Model B:
- **ISLiM** (primary): Keypoint-to-text, gloss-free, compact (~124M params), iSign dataset.
- **Sanket** (secondary): Dual-stream video+pose, TimeSformer + T5-base, INCLUDE dataset.

Both checkpoints verified via safe PyTorch loading (`weights_only=True` where compatible; `weights_only=False` required for ISLiM dictionary checkpoints per official code). No unsafe `.pkl` deserialization performed. No automatic script execution.

---

## Download Verification

| Property | ISLiM | Sanket |
|---|---|---|
| Official source | GitHub (`ahmedshoaib/ISLiM`) + Drive folder (`100H39_P...`) | Hugging Face (`ayush2635/sanket-isl-translator`) |
| Checkpoint file(s) downloaded | `ISLiM_4Loss_best_for_10kSpamoOF.pt` (859 MB), `ISLiM_full_posestitch_pre_processed.pt` (859 MB) | `model_best.pth` (1.4 GB) |
| Checkpoint file SHA-256 | `cf97d0ac9abcdfced0e128e98eede967ddf2479b2590eb0d1b891f96f176fc54`, `ffe4efbb5262abdd6221e4c8d818c92fc1db7fa7ea955ae0afa827251dd9140c` | `03cda82d26ab7e6d0adc8718d41d072e5181aef380a33dca17f08d3bd7dae729` |
| Mobile/export files | Extracted from 3 zip parts (`mobile-...-001/002/003.zip`) → `research/pretrained_models/islim/mobile/` (~3.4 GB) | None downloaded |
| Unavailable files | Drive folder access partially blocked; manual download completed by user. `checkpoint_latest.pth` (Sanket) not downloaded per instruction. | `checkpoint_latest.pth` (~3.6 GB, training-resume) intentionally excluded |
| Loading verification | `torch.load(..., weights_only=False)`; dict with `model_state_dict`, `epoch`, `val_results`, `optimizer_state_dict`. Key counts: 209 and 208. Safe load confirmed. | `torch.load(..., weights_only=True)`; dict with 541 tensor keys (`video_encoder.*`, `pose_encoder.*`, `text_decoder.*`, `lm_head.weight`). Safe load confirmed. |
| File integrity risk | None identified (official `.pt` format, not arbitrary `.pkl`). Mobile `.ptl` exports not verified for inference correctness. | None identified (state_dict only). No `.pkl` content. |

---

## Architecture and Technical Details

| Property | ISLiM | Sanket |
|---|---|---|
| Model family | Gloss-free keypoint-to-text SLT | End-to-end video-to-text SLT |
| Visual encoder | Temporal Conv1D (k=5) → 4-layer Transformer (d=512, 8 heads) | TimeSformer (`facebook/timesformer-base-finetuned-k400`), lower 8 blocks frozen |
| Pose/skeleton encoder | Integrated into visual encoder (94 landmarks directly) | Custom 2-layer TransformerEncoder (195 or 258 MediaPipe features) |
| Decoder | T5-small (`t5-small`, ~60M params within total) | T5-base (`t5-base`, 220M decoder params; vocab=32128) |
| Query tokens / compression | 16 learnable query tokens (`num_query_tokens=16`) for compressed representation | Not applicable (full temporal video encoding) |
| Total checkpoint params | ~124.3M (counted from state_dict tensors) | Not directly counted from `.pth` (decoder alone ~220M; vision encoder ~85M; total >300M estimated) |
| Checkpoint disk size | ~859 MB per file | ~1.4 GB |
| Reported evaluation metric | BLEU-4: 5.85 (paper) / 5.44 (checkpoint val); BLEU-1: 13.89; ROUGE-L: 13.38 | Peak validation BLEU-1: 50.6% (model card) |
| Note on metrics | BLEU-4 directly comparable to SignSage adapter reports; paper claims 5.85, checkpoint shows 5.44. | BLEU-1 NOT directly comparable with ISLiM's BLEU-4. Different metric, different dataset, different task difficulty. |

---

## Dataset and Licensing

| Property | ISLiM | Sanket |
|---|---|---|
| Training dataset | iSign v1.1 (official dataset name from repo/config) | INCLUDE Dataset (~4,284 videos, 250 categories, Kaggle) |
| Dataset version | v1.1 (referenced in config: `iSign_v1.1.csv`) | Not explicitly versioned in model card |
| Sample count | Not explicitly stated in paper/checkpoint; dataset access requires request to authors | ~4,284 videos; 250 sign categories; split not stated in model card |
| Input modality | Keypoint sequences (94 MediaPipe Holistic landmarks × 4 dims) | Dual-stream: video frames + pose features |
| Dataset license | **Research use only; NOT commercial use.** Explicit restriction must be preserved. No commercial rights established for iSign. | **Not explicitly stated** in HF model card. Must verify independently before any commercial deployment. |
| Checkpoint/model license | Not explicitly stated; assume same as dataset/research use. | **MIT** (explicit in HF tags: `license: mit`) |
| Citation / official reference | ACL Anthology 2026 (`wslp-1.3`) — paper URL in task description. No commercial license claim in repo or paper. | Model card references `facebook/timesformer-base-finetuned-k400` and `t5-base`; project described as Final Year Project (no peer-reviewed paper cited on card) |

**Critical licensing note:** ISLiM is trained on iSign, which is marked for research use only. Any integration proposal must preserve this restriction and must not claim commercial permission unless explicitly established. Sanket's MIT license allows broader use, but INCLUDE dataset licensing remains unverified.

---

## Preprocessing and Input Contract vs SignSage Adapter

| Property | SignSage Adapter (`i_sign_t5_adapter.py`) | ISLiM | Sanket |
|---|---|---|---|
| Landmark count | 543 (33 pose + 21 left hand + 21 right hand) | 94 (MediaPipe Holistic) | 195–258 (MediaPipe pose + hands, from card/repo; exact number unclear due to repo/card discrepancy) |
| Landmark ordering | POSE_START=0, LEFT_HAND_START=501, RIGHT_HAND_START=522 (543-wide array) | 94 Holistic landmarks in standard MediaPipe order (not explicitly mapped to adapter indices) | Standard MediaPipe extraction; exact ordering depends on preprocessing script (not included in checkpoint) |
| Coordinate dims | 3 (x, y, z) + confidence = 4 dims per landmark | 4 (x, y, z, visibility) — visibility replaces adapter's confidence column | 4 (x, y, z, visibility) or 3 (x, y, z) depending on preprocessing script |
| Temporal length | Max 500 frames (adapter default); padded/truncated | Max 128 frames (from `config.py` and checkpoint) | Not explicitly limited; TimeSformer handles variable temporal length |
| Spatial feature dim | 300 (pose 33×4 + hands 42×4) | Not applicable directly; encoder projects 94×4=376 to 376-channel temporal features | Not applicable (video encoder uses patch embeddings, not direct landmark projection) |
| Motion feature dim | 600 (spatial + velocity difference) | Not explicitly motion-augmented at adapter level; encoder uses Conv1D for temporal motion features | Not applicable (TimeSformer encodes motion implicitly from video frames) |
| Normalization | Z-score (`np.nanmean`/`np.nanstd`) over sequence; `np.nan_to_num`; padding with zeros | Not explicitly verified from checkpoint; encoder uses `BatchNorm1d` after Conv1D; linear projection before Conv1D | TimeSformer uses standard video normalization; pose encoder uses standard transformer normalization |
| Decoder vocabulary / tokenizer | `t5-small` tokenizer (standard HF vocab) | `t5-small` (same as adapter) | `t5-base` (vocab size 32128, confirmed in `lm_head.weight` shape) |

**Compatibility summary:**
- **ISLiM:** Incompatible with adapter contract. 94 vs 543 landmarks = different representation. Can only integrate if adapter is redesigned to extract 94 Holistic landmarks instead of 543 adapter landmarks, or if a remapping layer is built (not present in either repo).
- **Sanket:** Incompatible with adapter contract. Requires dual-stream video + pose input; adapter handles pose-only. Integration requires a completely separate inference interface or adapter extension with video encoder branch.

---

## Mobile / Deployment Assets

| Property | ISLiM | Sanket |
|---|---|---|
| Mobile exports included | Yes (TorchScript `.ptl` files: `encoder_traced.ptl`, `t5_decoder_step.ptl`, `t5_lm_head.ptl`, etc.) | None |
| Mobile format | TorchScript (`.ptl`); no TFLite/ONNX/quantized versions found | N/A |
| Mobile source files | `export_torchscript.py`, `verify_torchscript.py`, `prepare_test_data.py`, `wrap_test_data.py`, Android build scripts (`CMakeLists.txt`, `SLTModule.kt`) | N/A |
| Mobile weight files | `encoder_traced.pt` (56 MB), `t5_decoder_first_step.pt` (160 MB), `model_state_dict.pt` (287 MB) — total ~500+ MB mobile assets | N/A |
| Required for server inference? | **No.** Per instructions: only download mobile assets if required for normal model or separately useful. Mobile `.ptl` files are separate optimized exports; main `.pt` checkpoints handle normal inference. | N/A |
| Mobile verification | Not performed (not required for audit). Scripts present (`verify_torchscript.py`) for cosine similarity and token match verification if needed later. | N/A |

---

## Runtime and Hardware Requirements

| Property | ISLiM | Sanket |
|---|---|---|
| Training hardware (per docs) | Not explicitly stated; checkpoint fits <5 GB VRAM (README claim) | AWS `ml.g5.xlarge` (NVIDIA A10G 24GB VRAM) |
| Inference hardware estimate | ~2-4 GB VRAM (checkpoint 859 MB; T5-small + encoder) | ~4-8 GB VRAM (checkpoint 1.4 GB; TimeSformer + T5-base) |
| Runtime dependencies | `torch`, `numpy`, `transformers` (`t5-small`), `tqdm`, `python3.11+` | `torch`, `numpy`, `transformers` (`t5-base`), `timm` (TimeSformer), video/preprocessing libraries |
| Checkpoint format | `.pt` (PyTorch dictionary with `model_state_dict`) | `.pth` (PyTorch dictionary with full state_dict) |

---

## Security and Integrity

- Both checkpoints inspected with safe `torch.load` mechanisms.
- ISLiM requires `weights_only=False` (official code uses it); verified by inspecting dictionary contents before any initialization.
- Sanket loads successfully with `weights_only=True`, confirming it contains only tensor weights (no arbitrary code objects).
- No `.pkl` files downloaded or unpickled.
- No installation scripts executed (`requirements.txt` inspected but not installed from untrusted sources).
- No environment variables or tokens logged.
- Sources verified: official GitHub repo (`ahmedshoaib/ISLiM`), official HF repo (`ayush2635/sanket-isl-translator`), user-provided `.pt` files for ISLiM (manual download completed by user).

---

## Missing / Unavailable Files

- **ISLiM:** Additional checkpoint variations or `.pkl` formats mentioned in task description not found. Only two `.pt` files provided. No `.pkl` checkpoint format required.
- **Sanket:** `checkpoint_latest.pth` (training-resume checkpoint) intentionally excluded per instructions. No mobile/export files available.
- **Mobile assets:** ISLiM mobile folder split into 3 zip files (`mobile-...-001/002/003.zip`); extracted to `research/pretrained_models/islim/mobile/`. Not verified for inference correctness.

---

## Final Verdict — Which Candidate for Model B?

**Based on actual checkpoint availability, architecture, dataset, compatibility, and evaluation evidence:**

| Criterion | ISLiM | Sanket | Better for SignSage? |
|---|---|---|---|
| Checkpoint availability | Both `.pt` files downloaded and verified (859 MB each) | `model_best.pth` verified (1.4 GB) | Tie |
| Checkpoint load safety | Requires `weights_only=False` (official); verified safe dict load | `weights_only=True` works; safer format | Sanket (slightly safer load profile) |
| Architecture match to adapter | T5-small decoder matches adapter; encoder incompatible (94 vs 543 landmarks) | T5-base decoder larger; video encoder not in adapter | Neither directly compatible; ISLiM closer on decoder side |
| Dataset licensing | **iSign = research only, no commercial use.** Must preserve restriction. | MIT checkpoint; INCLUDE license unverified | Sanket (if dataset verified) |
| Evaluation evidence | BLEU-4 5.44–5.85 (low absolute score); paper claims 5.85, checkpoint 5.44; significant zero-BLEU rate (sent_bleu_zero > 1200 in one checkpoint) | BLEU-1 50.6% (much higher absolute score, but different dataset/task; NOT comparable to BLEU-4) | Sanket shows stronger reported performance, but metric and dataset differ |
| Input compatibility with adapter | 94 landmarks; requires adapter redesign or remapping | Video + pose dual-stream; requires completely new pipeline | Neither fits existing adapter |
| Model size | ~124M params; <5 GB VRAM claim | >300M params estimated; trained on 24 GB VRAM | ISLiM more compact |
| Mobile deployment assets | Present (TorchScript `.ptl`) | Not present | ISLiM (if mobile needed) |
| Licensing risk | High (iSign research-only; no commercial claim in repo/paper) | Medium (MIT checkpoint; dataset license unverified) | Sanket lower risk IF dataset verified |

**Clear verdict:**

**Sanket is the stronger candidate for Model B IF (and only if) the following conditions are met:**
1. INCLUDE dataset licensing is verified independently (not established in current audit).
2. The pipeline is redesigned or a new adapter is built to handle dual-stream video + pose input (`TimeSformer` + custom pose encoder + `T5-base` decoder), because the existing `i_sign_t5_adapter.py` does not support video input.
3. The evaluation metric discrepancy (BLEU-1 vs BLEU-4, INCLUDE vs iSign) is documented and not treated as direct comparison to adapter performance.

**ISLiM is the weaker candidate for Model B** because:
1. iSign dataset licensing explicitly restricts commercial/research-only use; integration would carry licensing risk.
2. Checkpoint evaluation shows low absolute BLEU-4 (~5.4%) and high zero-BLEU rates, suggesting weaker translation quality compared to Sanket's reported BLEU-1 50.6%.
3. 94-landmark input is incompatible with adapter's 543-landmark contract; requires remapping.

**Recommendation:**
- If the priority is **checkpoint quality and licensing flexibility**, choose **Sanket** (pending dataset license verification and adapter redesign).
- If the priority is **compact model size, T5-small compatibility with existing adapter decoder, and mobile deployment readiness**, choose **ISLiM** (but document research-only licensing and build a 94-landmark adapter layer).
- **Do not treat either as a drop-in replacement** for the existing `i_sign_t5_adapter.py` pipeline. Both require architectural or pipeline changes.
- **Do not claim ISLiM's BLEU-4 is directly comparable to Sanket's BLEU-1.** They measure different tasks on different datasets.
- **Preserve iSign research-only restriction** in all documentation if ISLiM is selected.

---

## Audit Completion Status
- [x] ISLiM checkpoints downloaded, organized in `research/pretrained_models/islim/`, safe-loaded, audited.
- [x] ISLiM mobile assets extracted from 3 zip files, organized in `research/pretrained_models/islim/mobile/`.
- [x] Sanket checkpoint (`model_best.pth`) downloaded, organized in `research/pretrained_models/sanket/`, safe-loaded, audited.
- [x] Sanket `checkpoint_latest.pth` intentionally excluded.
- [x] Adapter contract (`i_sign_t5_adapter.py`) inspected and compared.
- [x] Licensing restrictions documented (iSign = research only; INCLUDE = unverified; MIT = Sanket checkpoint).
- [x] No model training or fine-tuning started.
- [x] No existing checkpoints replaced or deleted.
- [x] `SIGNSAGE_MODEL_PROMOTED=false` unchanged.
- [x] Final comparison and verdict delivered.


## Integration Status (SignSage)
- **ISLiM**: **Not integrated**, **Blocked** (landmark contract mismatch), **Experimental** (translation quality unverified independently), **Verified** (checkpoint load only).
- **Sanket**: **Not integrated**, **Blocked** (dual-stream input + adapter contract mismatch; dataset license unverified), **Experimental** (translation quality unverified independently), **Verified** (checkpoint load only).
- `SIGNSAGE_MODEL_PROMOTED=false` remains unchanged; neither candidate promoted.
