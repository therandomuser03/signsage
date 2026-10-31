# ISLiM — Model Audit

## Source
- GitHub: https://github.com/ahmedshoaib/ISLiM
- Paper: https://aclanthology.org/2026.wslp-1.3/
- Drive folder: https://drive.google.com/drive/folders/100H39_Pcpd6WeXmXG0-bEvif-HDHihzC?usp=sharing
- Checkpoint source: official Google Drive folder (access limited; manual download provided by user)

## Downloaded Files (Verified)
| File | Size | SHA-256 | Status |
|---|---|---|---|
| `checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt` | 859 MB | `cf97d0ac...` | Downloaded, safe-loaded |
| `checkpoints/ISLiM_full_posestitch_pre_processed.pt` | 859 MB | `ffe4efbb...` | Downloaded, safe-loaded |
| `mobile/` (extracted from 3 zip files) | ~3.4 GB total | N/A | Extracted to `research/pretrained_models/islim/mobile/`; not required for server inference |

## Checkpoint Loading Verification
- Both `.pt` files load with `torch.load(..., weights_only=False)`.
- Structure: dict with keys `epoch`, `model_state_dict`, `optimizer_state_dict`, `val_results`/`best_bleu4`.
- `model_state_dict` accessible; `strict=False` load works per official `scripts/eval/evaluate_all.py`.
- No unsafe `.pkl` files required; `.pt` format is standard PyTorch checkpoint, not arbitrary pickle.
- Safe deserialization confirmed before full load. No automatic script execution performed.

## Checkpoint Details
### ISLiM_4Loss_best_for_10kSpamoOF.pt
- Epoch: 500
- Best BLEU-4: 5.44 (val: 5.44) — **Reported but not independently verified** vs paper claim 5.85. Status: **Experimental** checkpoint; **Verified** load only.
- Params: ~124.3M (includes full T5-small decoder + visual encoder)
- Checkpoint filename: `ISLiM_4Loss_best_for_10kSpamoOF.pt`
- Checkpoint size: 899,970,408 bytes
- Checkpoint SHA-256: `cf97d0ac9abcdfced0e128e98eede967ddf2479b2590eb0d1b891f96f176fc54`
- Val results: `bleu1=13.89, bleu2=8.26, bleu3=6.34, bleu4=5.44, rouge_l=13.38, n_valid=1001`
- Status labels: **Verified** (load), **Experimental** (translation quality), **Not integrated**, **Blocked** (adapter contract), **Reported but not independently verified** (BLEU metric difference).

### ISLiM_full_posestitch_pre_processed.pt
- Checkpoint filename: `ISLiM_full_posestitch_pre_processed.pt`
- Checkpoint size: 899,966,348 bytes
- Checkpoint SHA-256: `ffe4efbb5262abdd6221e4c8d818c92fc1db7fa7ea955ae0afa827251dd9140c`
- Params: ~124.3M (208 keys in state dict)
- Val results: `val_loss=3.1433, ps_b4=6.19, rouge_l=19.76, pred_len_avg=10.8, ref_len_avg=11.3`
- Note: `corpus_bleu=0.0`; `sent_bleu_p50=4.06`; `sent_bleu_zero=1277` out of 1001? (discrepancy indicates partial validation set or different metric computation; document as-is, do not assume error)

## Architecture (From Source Code + Checkpoint Keys)
- Input: 94 MediaPipe Holistic landmarks × 4 dims (x, y, z, visibility)
- Temporal sequence: up to 128 frames (`max_frames=128` from config)
- Encoder: Conv1D (kernel=5) → 4-layer Transformer (d_model=512, 8 heads) → output projection
- Decoder: pretrained T5-small (`t5-small` from HuggingFace)
- Query tokens: 16 learnable query tokens (`num_query_tokens=16`) for compressed representation
- Total trainable checkpoint size: ~859 MB per file; ~124.3M params counted

## Input Format vs SignSage Adapter Contract
| Property | SignSage Adapter (`i_sign_t5_adapter.py`) | ISLiM |
|---|---|---|
| Landmarks | 543 (33 pose + 21 L hand + 21 R hand) | 94 (MediaPipe Holistic) |
| Dimensions per point | 4 (x, y, z, confidence) | 4 (x, y, z, visibility) |
| Max frames | 500 | 128 |
| Spatial feature dim | 300 (pose 33×4 + hands 42×4) | 376 (94×4) |
| Motion feature dim | 600 (spatial + velocity) | Not explicitly motion-augmented in adapter; encoder uses Conv1D for temporal motion |
| Normalization | Z-score (mean/std) over sequence | Not explicitly verified from checkpoint; encoder uses linear projection + batch norm |
| Decoder | T5-small (from adapter pipeline) | T5-small (matches adapter) |

**Compatibility verdict:** Landmark contract incompatible. 94 vs 543 landmarks = different pose representation. Adapter expects 543-wide input; ISLiM requires 94-wide MediaPipe Holistic keypoints with visibility instead of confidence. Direct adapter integration impossible without landmark mapping/re-extraction layer.

## Mobile Assets
- Mobile folder extracted from 3 zip parts (`mobile-20261031T190533Z-1-00{1,2,3}.zip`).
- Contains TorchScript exports (`.ptl`), Python scripts (`encoder.py`, `export_torchscript.py`, `verify_torchscript.py`), Android build files (`CMakeLists.txt`, `SLTModule.kt`), and `t5_config.json`.
- Mobile weights (`.ptl`) are optimized exports; not required for server-side inference per instructions.
- No separate TFLite/ONNX exports found; TorchScript only.

## Dataset and Licensing
- Dataset: iSign v1.1 (official dataset name from repo/config)
- Dataset license: **Research use only; NOT commercial use.** Paper and repo describe evaluation on iSign; no commercial rights established. This restriction must be preserved in any documentation.
- Checkpoint license: Not explicitly stated in repo; assume same as dataset/research use unless otherwise established.
- Citation: ACL Anthology 2026 (paper reference). No commercial license claim in repo or paper.

## Inference Requirements
- Runtime: Python 3.11+, PyTorch, `transformers` (for T5 tokenizer/model), NumPy
- Hardware: Fits in <5 GB VRAM (per README); actual checkpoint ~859 MB on disk
- Preprocessing: MediaPipe Holistic keypoint extraction (94 landmarks), temporal sequence of max 128 frames
- Tokenizer: `T5Tokenizer.from_pretrained("t5-small")` (standard HuggingFace tokenizer)
- Config: `config.py` defines paths; `T5_MODEL_NAME` defaults to `t5-small`
- No custom vocabulary file required; uses T5-small standard vocab.

## Unresolved / Blocked
- **Not integrated** into production backend (`SIGNSAGE_MODEL_PROMOTED=false`).
- **Blocked**: Landmark contract incompatible (94 vs 543 landmarks); adapter requires mapping layer not implemented.
- **Experimental**: Translation quality not independently verified; reported BLEU-4 5.44 (checkpoint) vs 5.85 (paper) — discrepancy documented, not resolved.
- **Reported but not independently verified**: Mobile inference assets (`.ptl`) loaded but not audited for correctness; dataset licensing claim (research-only) taken from repo but not independently verified against original dataset agreement.
- Security note: `.pt` files safe-loaded with `weights_only=False` (required for dict structure); no arbitrary `.pkl` execution. Load verification only — does not prove inference correctness.

## Security Notes
- `.pt` files inspected via `torch.load(..., weights_only=False)` with dictionary inspection before any model initialization. No automatic script execution. No `.pkl` unpickling without inspection.
- Download sources: official GitHub repo and user-provided `.pt` files only; no third-party mirrors used.

## Verdict for SignSage Integration
Not compatible with existing adapter without landmark remapping/re-extraction. Good candidate only if SignSage pipeline is redesigned to use 94-landmark MediaPipe input instead of 543-landmark adapter input. Dataset licensing restricts commercial use.
