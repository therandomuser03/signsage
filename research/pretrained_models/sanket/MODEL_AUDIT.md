# Sanket — Model Audit

## Source
- Hugging Face: https://huggingface.co/ayush2635/sanket-isl-translator
- GitHub (related project): https://github.com/ujjwal-tiwari3039/Sanket-ISL-translator (note: different architecture description from HF card; this audit focuses on HF checkpoint and model card)
- Model card: MIT license tag

## Downloaded Files (Verified)
| File | Size | SHA-256 | Status |
|---|---|---|---|
| `checkpoints/model_best.pth` | 1,438,145,048 bytes (1.4 GB) | `03cda82d26ab7e6d0adc8718d41d072e5181aef380a33dca17f08d3bd7dae729` (full) | Downloaded; **Verified** (`weights_only=True` safe load); **Not integrated**; **Blocked** (adapter contract mismatch) |
| `configs/README.md` (model card) | 2.5 KB | N/A | Downloaded |

Note: `checkpoint_latest.pth` (training-resume checkpoint, ~3.6 GB, requires unsafe unpickling or additional optimizer state) was NOT downloaded per instructions. Only `model_best.pth` obtained.

## Checkpoint Loading Verification
- `torch.load('model_best.pth', weights_only=True, map_location='cpu')` succeeds.
- Checkpoint is a standard state_dict (dict of 541 tensor keys), not an arbitrary pickle object.
- Key categories: `video_encoder.*` (TimeSformer vision encoder), `pose_encoder.*` (custom pose encoder), `text_decoder.*` (T5-base decoder), `text_decoder.lm_head.weight` (output projection).
- No unsafe `.pkl` format; no script execution required.
- Safe deserialization confirmed; no automatic script execution performed.

## Checkpoint Architecture (From State Keys)
- Vision encoder (`video_encoder`): TimeSformer base (`facebook/timesformer-base-finetuned-k400`), lower 8 blocks frozen, embed_dim=768, 197 position embeddings (includes class token + temporal embeddings).
- Pose encoder (`pose_encoder`): Custom TransformerEncoder (2 layers), processes 195 MediaPipe features/frame (per repo docs: 33 pose × 4 + 2 hands × 21 × 3 = 258? Wait: model card says 195 features; checkpoint keys do not reveal exact pose encoder input dimension directly; architecture description from card says 195 features).
- Text decoder (`text_decoder`): T5-base decoder (12 blocks, d_model=768, feedforward=3072, vocab_size=32128), with `lm_head.weight` mapping to vocab.
- Gated cross-modal fusion layers implied by model card description; exact fusion weights embedded in checkpoint.
- Total checkpoint keys: 541; total file size: ~1.4 GB.

## Input Format vs SignSage Adapter Contract
| Property | SignSage Adapter (`i_sign_t5_adapter.py`) | Sanket (HF card) |
|---|---|---|
| Input modality | Pose landmarks only | Video + pose (dual-stream) |
| Video encoder | None (adapter uses landmarks directly) | TimeSformer (video frames) |
| Pose features | 543 landmarks (33 pose + 21 L + 21 R) | 195 MediaPipe features (from card; repo mentions 258 normalized values) |
| Temporal length | Max 500 frames (adapter), 128 (default) | Not explicitly limited in card; TimeSformer handles temporal sequences |
| Decoder | T5-small (adapter) | T5-base (220M decoder params) |
| Checkpoint size | Adapter pipeline uses external `pose_encoder.pt` + `t5-small` | 1.4 GB (single `.pth` file) |
| Dataset | iSign (via adapter) | INCLUDE (~4,284 videos, 250 categories) |
| Reported metric | Not applicable to adapter directly | Peak validation BLEU-1: 50.6% |

**Compatibility verdict:** Incompatible with existing adapter. Adapter expects 543-landmark pose sequences; Sanket expects dual-stream video + pose input. Adapter's `SignST5Adapter` does not handle video frames or TimeSformer embeddings. Integration requires complete pipeline redesign or separate inference interface.

## Dataset and Licensing
- Dataset: INCLUDE Dataset (~4,284 videos across 250 sign categories)
- Dataset source: Kaggle (https://www.kaggle.com/datasets/daskoushik/include/data)
- Dataset license: Not explicitly stated in model card or HF repo. Must be verified before any commercial use.
- Model/checkpoint license: MIT (explicit in HF model card tags: `license: mit`)
- Paper/reference: Final Year Project (no peer-reviewed paper cited on HF card)
- Citation requirement: Model card references `facebook/timesformer-base-finetuned-k400` and `t5-base` as base models.

## Inference Requirements
- Runtime: Python 3.11+, PyTorch, `transformers` (for T5-base), `timm` (for TimeSformer), video/preprocessing libraries.
- Hardware: Trained on AWS `ml.g5.xlarge` (NVIDIA A10G 24GB VRAM). Checkpoint 1.4 GB suggests ~2-4 GB VRAM minimum for inference.
- Tokenizer: `T5Tokenizer.from_pretrained("t5-base")` (standard HuggingFace tokenizer for T5-base, vocab size 32128 confirmed in checkpoint).
- Preprocessing: MediaPipe pose/video feature extraction (195 or 258 features); temporal sequence preparation.
- No custom vocabulary file required.

## Mobile / Deployment Assets
- Not included (user did not provide mobile exports for Sanket; HF repo has only `.pth` checkpoints).
- No TFLite/ONNX/Quantized exports found.

## Unresolved / Blocked
- Status: **Verified** (checkpoint load with `weights_only=True`, 541 state-dict keys); **Not integrated**; **Blocked** (dual-stream video+pose input incompatible with adapter); **Experimental** (translation quality not independently verified).
- `checkpoint_latest.pth` not downloaded (training-resume checkpoint; not needed for inference). **Not integrated** — excluded intentionally.
- INCLUDE dataset license: **Not verified** — unverified license claim; must verify independently before any deployment. **Blocked** on licensing verification.
- **Reported but not independently verified**: Model card claims BLEU-1 50.6% peak validation; no independent reproduction performed. Note: BLEU-1 metric not comparable with ISLiM BLEU-4.
- Input format discrepancy: HF card says 195 features; related GitHub repo (`ujjwal-tiwari3039/Sanket-ISL-translator`) mentions 258 normalized pose/hand values and uses LSTM. Architecture mismatch between HF checkpoint (TimeSformer + T5-base) and GitHub repo description documented. **Blocked** until discrepancy resolved or architecture pinned.

## Security Notes
- Checkpoint loaded with `weights_only=True` first; confirmed safe state_dict.
- No `.pkl` unpickling performed. No script execution.
- Source: official Hugging Face repo only (`ayush2635/sanket-isl-translator`).

## Verdict for SignSage Integration
Not directly compatible with existing adapter. Dual-stream video+pose input and T5-base decoder require different pipeline. MIT license allows broader use than ISLiM, but INCLUDE dataset licensing must be verified. Better candidate for Model B only if pipeline is redesigned to support video input and T5-base decoder, or if adapter is extended with a video encoder branch.
