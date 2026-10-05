# External iSign T5 Model Audit Report

## 1. External Model Architecture

**Repository**: https://huggingface.co/manavdhamecha77/iSign-t5-pose-to-text

**Six trained checkpoints**:
- `t5-small` — T5-Small, spatial pose features only
- `t5-small-motion` — T5-Small, spatial + motion features
- `t5-base` — T5-Base, spatial pose features only
- `t5-base-motion-v2` — T5-Base, spatial + motion features (v2)
- `t5-large` — T5-Large, spatial pose features only
- `t5-large-motion` — T5-Large, spatial + motion features

**Model architecture** (from `inference/model.py`):

- **PoseEncoder**: Simple MLP mapping `pose_dim -> hidden_dim(256) -> embedding_dim(T5 d_model)`
  - `nn.Linear(pose_dim, hidden_dim)` → `nn.ReLU()` → `nn.Dropout(0.1)` → `nn.Linear(hidden_dim, embedding_dim)`
  - Input: `(batch_size, seq_length, pose_dim)`
  - Output: `(batch_size, seq_length, embedding_dim)`

- **T5 component**: `T5ForConditionalGeneration` loaded from checkpoint
  - Pose embeddings substituted for text embeddings via `inputs_embeds=pose_embeddings`
  - Standard T5 encoder-decoder with pose attention mask

- **Pipeline flow**:
  1. Pose input `(batch, seq, pose_dim)` → PoseEncoder → T5 embeddings
  2. T5 encoder processes pose embeddings
  3. T5 decoder generates text auto-regressively
  4. Tokenizer decodes generated IDs to text

**Files in each checkpoint**:
- `pose_encoder.pt` — trained pose encoder weights (`state_dict()`)
- `t5/config.json` — T5 configuration
- `t5/generation_config.json` — generation configuration
- `t5/model.safetensors` — trained T5 parameters

**Dataset**: Subset of iSign dataset
- 18,867 total samples
- 16,979 training
- 1,887 held-out test
- Input: pose files; Target: textual sentence annotations

## 2. External Model Input Contract

### 2.1 Components Used

```python
DEFAULT_COMPONENTS = (
    "POSE_LANDMARKS",
    "LEFT_HAND_LANDMARKS",
    "RIGHT_HAND_LANDMARKS",
)
```

- **3 components**: body pose (33), left hand (21), right hand (21)
- **No face landmarks**

### 2.2 Per-Landmark Format

- `include_confidence=True` by default
- When confidence included: `(x, y, z, confidence)` = 4 values per landmark
- MediaPipe `NormalizedLandmark` format

### 2.3 Per-Frame Dimensionality

| Model | pose_dim | use_motion | Effective per-frame dim |
|-------|----------|------------|------------------------|
| `t5-small` | 300 | False | 300 |
| `t5-small-motion` | 600 | True | 600 |
| `t5-base` | 300 | False | 300 |
| `t5-large` | 300 | False | 300 |
| `t5-large-motion` | 600 | True | 600 |

### 2.4 Sequence Length

- `max_pose_length = 500` frames (default in `inference.py` and `preprocess_pose()`)
- Shorter sequences are zero-padded to 500 frames
- Longer sequences are truncated to 500 frames
- Output shape: `(max_length, pose_dim)` → `(1, 500, pose_dim)` after `unsqueeze(0)`

### 2.4 Normalization

- `normalize_pose(method="zscore")` — default in `preprocess_pose()`
- Computes mean and std across axis=0
- `(pose - mean) / std`, with `np.where(std == 0, 1.0, std)` to avoid division by zero
- `np.nan_to_num()` for NaN handling
- Applied **before** motion feature addition

### 2.5 Motion Feature Construction

From `add_motion_features()`:
- `velocity[1:] = pose[1:] - pose[:-1]`
- `velocity[0] = 0` (first frame velocity is zero)
- `output = np.concatenate([pose, velocity], axis=1)` — doubles the dimension
- Motion models concatenate: `[pose | velocity]`

### 2.6 Padding/Truncation

- Shorter sequences → zero-padded to `(max_length, pose_dim)`
- Longer sequences → truncated to `(max_length, pose_dim)`
- Padding value: `0.0` (zeros)
- Returns both the padded tensor and the actual length

### 2.7 Tokenizer

- `AutoTokenizer.from_pretrained(self.config["base_model"])`
  - `t5-small`: `t5-small` tokenizer
  - `t5-base`: `t5-base` tokenizer
  - `t5-large`: `t5-large` tokenizer

### 2.8 Generation Configuration

- `max_pose_length`: 500 frames
- `max_length`: 128 tokens (output text length)
- `num_beams`: 4 (beam search)
- `length_penalty`: 2.0
- `device`: auto (CUDA if available, else CPU)

### 2.9 Checkpoint Format

- `pose_encoder.pt` — PyTorch `state_dict()`
- `t5/model.safetensors` — T5 weights in safetensors format
- `t5/config.json` — T5 configuration
- `t5/generation_config.json` — generation configuration

**Model loading** (from `model.py` `load_pretrained()`):
```python
self.t5 = T5ForConditionalGeneration.from_pretrained(
    os.path.join(load_path, "t5")
)
self.pose_encoder.load_state_dict(
    torch.load(pose_encoder_path, map_location="cpu")
)
```

### 2.10 Device Requirements

- CUDA automatically selected when available
- Explicit device argument
- `.to(device)` called on model and tensors
- Pose encoder and T5 both moved to same device

## 3. SignSage Current Input Contract

### 3.1 Landmark Contract

```
TOTAL_LANDMARKS = 543
POSE_COUNT = 33
FACE_COUNT = 468 (implicit: 543 - 33 - 21 - 21 = 468)
HAND_COUNT = 21
```

### 3.2 Component Order (Must Match Backend Exactly)

```
POSE_LANDMARKS[0:33] + FACE_LANDMARKS[33:501] + LEFT_HAND_LANDMARKS[501:522] + RIGHT_HAND_LANDMARKS[522:543]
```

### 3.3 Per-Landmark Format

- `LandmarkPoint = [number, number, number]` — only `x, y, z`
- **No confidence value** included
- MediaPipe `NormalizedLandmark` coordinates normalized to image frame

### 3.4 Per-Frame Dimensionality

- 543 landmarks × 3 values (x, y, z) = **1629 values per frame**
- Structure: `[POSE(33×3) | FACE(468×3) | LEFT_HAND(21×3) | RIGHT_HAND(21×3)]`

### 3.4 Output Structure

From `landmarks.ts` `detect()` method:
```typescript
HolisticFrame = {
  landmarks: LandmarkPoint[];    // 543 [x,y,z] points in contract order
  pose: NormalizedLandmark[];
  face: NormalizedLandmark[];
  leftHand: NormalizedLandmark[];
  rightHand: NormalizedLandmark[];
  marks: NormalizedLandmark[] | undefined;
}
```

### 3.4 MediaPipe Detectors Used

- `PoseLandmarker` (33 body pose points)
- `FaceLandmarker` (468 face mesh points)
- `HandLandmarker` (21 points each for left and right hands)

### 3.5 Backend Contract Reference

From comments in `landmarks.ts`:
// SignSage — browser landmark extraction.
// Mirrors backend/app/ml/contracts/landmark_contract.py:
//   543 landmarks per frame in component order
//   POSE(33) + FACE(468) + LEFT_HAND(21) + RIGHT_HAND(21), [x, y, z] each.
// Pose *world* landmarks are excluded (backend mask covers range(543)).

## 4. Exact Compatibility Matrix

| Feature | External Model | SignSage Current | Compatibility |
|---------|---------------|------------------|---------------|
| **Landmark components** | POSE + LEFT_HAND + RIGHT_HAND (3) | POSE + FACE + LEFT_HAND + RIGHT_HAND (4) | ❌ Incompatible |
| **Landmark count per frame** | 75 (33+21+21) | 543 | ❌ Incompatible |
| **Values per landmark** | 4 (x, y, z, confidence) by default | 3 (x, y, z) only | ❌ Incompatible |
| **Values per frame** | 300 (75×4) or 600 (with motion) | 1629 (543×3) | ❌ Incompatible |
| **Normalization** | z-score (per-landmark across sequence) | Not specified/likely different | ❓ Unknown |
| **Motion features** | Explicit: `[pose | velocity]` | Not part of pipeline | ❌ Incompatible |
| **Sequence length** | max 500 frames (padded/truncated) | Unknown | ❓ Unknown |
| **Input format** | `.pose` files via `pose_format` library | MediaPipe landmark enumeration | ❌ Different |
| **Confidence source** | From pose file / `include_confidence` flag | Not provided by SignSage | ❌ Incompatible |
| **Dtype** | `float32` (explicit in code) | Not specified (likely float32) | ✅ Assumed compatible |
| **Batch dimension** | Added by inference code: `unsqueeze(0)` | Not applicable | ✅ Same pattern |
| **Tokenizer** | T5 tokenizer from base model | Not used (different pipeline) | ✅ Same T5 family |

## 5. Incompatibilities

### 5.1 Face Landmarks (Critical)
- **SignSage**: Extracts 468 face mesh landmarks via MediaPipe `FaceLandmarker`
- **External model**: Does not use face landmarks at all (3-component input)
- **Impact**: SignSage output has 1629 values/frame vs external model's 300/600. The face component must be discarded.

### 5.2 Confidence Values
- **External model**: `include_confidence=True` by default; each landmark has 4 values (x,y,z,conf)
- **SignSage**: Landmarks are `[x, y, z]` only — no confidence
- **Impact**: The 4th dimension is missing. Adapter must supply confidence (e.g., all 1.0).

### 5.3 Dimensionality Mismatch
- **SignSage**: 1629 values per frame (543 × 3)
- **External model**: 300 values per frame (75 × 4, spatial) or 600 (75 × 4 + 300 motion)
- **Impact**: Adapter must reduce dimensionality from 1629 → 300/600.

### 5.4 Motion Features
- **External model**: Motion models explicitly concatenate frame-wise velocity
  - `velocity[t] = pose[t] - pose[t-1]`, `velocity[0] = 0`
- **SignSage**: No motion feature construction in the reported pipeline
- **Impact**: To use motion models, adapter must construct velocity features.

### 5.5 Input Format
- **External model**: `.pose` files loaded via `pose_format` library
- **SignSage**: MediaPipe landmark enumeration
- **Impact**: Adapter must convert MediaPipe output to expected format.

### 5.6 Normalization
- **External model**: z-score normalization across the sequence axis
- **SignSage**: No reported normalization step; MediaPipe landmarks are image-normalized
- **Impact**: Adapter must apply z-score normalization matching the training regime.

### 5.7 Sequence Length
- Model trained on sequences up to 500 frames
- SignSage may produce longer/shorter sequences
- **Impact**: Adapter must pad/truncate to exactly 500 frames.

## 6. Required Adapter

### 6.1 Adapter Architecture (summary)

Function `convert_signage_to_hf(holistic_frame, include_confidence=True, use_motion=False, pose_dim=300)`:
1. Extract only components 0-33 (pose), 501-522 (left hand), 522-543 (right hand) — skip face
2. Convert [x,y,z] → [x,y,z,confidence] with confidence=1.0 (or MediaPipe confidence if available)
3. Concatenate: 33 + 21 + 21 = 75 landmarks × 4 = 300 (verified) = 304 values
4. Pad/truncate to `pose_dim` (300 for spatial, 600 for motion)
5. If `use_motion=True`: construct velocity features `[pose | velocity]`, `velocity[t]=pose[t]-pose[t-1]`, `velocity[0]=0`
6. Apply z-score normalization (mean≈0, std≈1 per dimension)
7. Return tensor of shape `(seq_length, pose_dim)`

### 6.2 Adapter Sequence Assembly

1. Collect N frames of SignSage holistic data
2. Process each frame through the adapter
3. Stack frames: shape `(N, pose_dim)`
4. If `use_motion=True`: apply `add_motion_features()` to the stacked sequence
5. Pad/truncate to `max_length=500` frames
6. Convert to `torch.tensor` with dtype `float32`
7. Add batch dimension: `unsqueeze(0)` → `(1, max_length, pose_dim)`
8. Create `pose_length_tensor`: `torch.tensor([actual_length], dtype=torch.long)`

### 6.3 Motion Feature Computation

```python
def add_motion_features(pose_seq: np.ndarray) -> np.ndarray:
    """pose_seq shape: (seq_length, pose_dim // 2) i.e. (N, 300) for motion models"""
    velocity = np.zeros_like(pose_seq, dtype=np.float32)
    velocity[1:] = pose_seq[1:] - pose_seq[:-1]
    return np.concatenate([pose_seq, velocity], axis=1)
```

### 6.4 Z-Score Normalization

Must be applied **before** motion addition:
```python
def zscore_normalize(pose: np.ndarray) -> np.ndarray:
    mean = np.nanmean(pose, axis=0, keepdims=True)
    std = np.nanstd(pose, axis=0, keepdims=True)
    return np.nan_to_num((pose - mean) / np.where(std == 0, 1.0, std))
```

## 7. Recommended Checkpoint

| Checkpoint | Compatibility | Hardware | Recommendation |
|------------|--------------|----------|----------------|
| `t5-small` | Partial — needs adapter | ✅ CPU-inferable | Beginner option |
| `t5-small-motion` | Partial — needs adapter + motion | ✅ CPU-inferable | **Recommended** |
| `t5-base` | Partial — larger model | ⚠️ GPU recommended | Lower priority |
| `t5-base-motion-v2` | Partial — base-sized with motion | ⚠️ GPU recommended | Lower priority |
| `t5-large` / `t5-large-motion` | Partial — significantly larger | ❌ Likely too slow | Not recommended |

**Recommendation**: Start with `t5-small-motion`. Motion features are critical for sign language translation; T5-Small is lightweight (runnable on CPU); adapter for `t5-small` is simpler (pose_dim=300).

## 8. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Model** | `t5-small` (spatial only) | `t5-small-motion` |
| **CPU inference** | Possible (may be slow for real-time) | Possible with optimization |
| **GPU** | Not strictly required for `t5-small` | Recommended for `t5-small-motion` or base+ models |
| **GPU type** | Any CUDA-capable GPU (even integrated) | 4GB VRAM minimum for `t5-small`; 8GB+ for larger |
| **Memory** | ~2GB RAM for model + artifacts | ~4GB RAM+ |
| **Latency** | ~100-500ms per sentence depending on hardware | |
| **Throughput** | ~1-5 sentences/minute (CPU, t5-small) | ~10-30 sentences/minute (GPU, t5-small) |

**Key constraint**: The pose encoder MLP is lightweight (~ few hundred K parameters); the T5 component is the main computational cost. `t5-small` has ~60M parameters total.

## 9. Risks

### 9.1 Distribution Shift
- Model trained on iSign subset; SignSage's MediaPipe landmarks may have different distribution
- Risk: Degraded translation quality or garbled output

### 9.2 Confidence Mismatch
- External model expects confidence values; SignSage does not provide them
- Risk: Using default confidence=1.0 may not match training distribution

### 9.3 Motion Feature Mismanagement
- Motion models require velocity computation; incorrect implementation will produce garbage output
- Risk: Even with correct component selection, bad motion features break the model

### 9.4 Normalization Mismatch
- External model uses z-score normalization; SignSage landmarks are image-normalized differently
- Risk: If adapter doesn't exactly replicate the z-score regime, model performance degrades

### 9.5 Sequence Length Mismatch
- Model trained on average sequence length; SignSage may produce longer/shorter sequences
- Risk: Padding/truncation at 500 frames may cut off valid signs or pad with excessive zeros

### 9.6 T5 Tokenizer Differences
- T5 tokenizers vary by size (t5-small vs t5-base vs t5-large)
- Risk: Using wrong tokenizer could cause encoding/decoding errors

### 9.7 Promotion Gate
- The backend already has a promotion gate: `SIGNSAGE_MODEL_PROMOTED=false` by default
- Even if the model works, it must not be promoted until the promotion gate is manually set
- Risk: If someone promotes prematurely, users see fabricated text

## 10. Exact Integration Plan

### Phase 1: Adapter Development (1-2 weeks)

**1.1** Create adapter module `signage_hf_adapter.py`:
- Function `convert_signage_to_hf(holistic_frame, include_confidence=True, use_motion=False, pose_dim=300)`
- Implements component filtering, confidence addition, dimensionality adjustment, motion feature construction

**1.2** Write unit tests with synthesized landmark data:
- Component filtering test
- Confidence addition test
- Dimensionality test
- Motion feature construction test
- Z-score normalization test
- Padding/truncation test

**1.3** Integration point in the SignSage frontend:
- Where `HolisticFrame.landmarks` is currently assembled for backend transmission
- Instead of sending to backend, run through adapter
- Feed adapter output into the T5 model inference

### Phase 2: Model Loading and Inference (1 week)

**2.1** Load the T5 model using Hugging Face Transformers:
```python
from transformers import T5ForConditionalGeneration, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "manavdhamecha77/iSign-t5-pose-to-text",
    subfolder="t5-small/t5"
)
# Load pose encoder
pose_encoder_state = torch.load(
    "manavdhamecha77/iSign-t5-pose-to-text/t5-small/pose_encoder.pt",
    map_location="cpu"
)
```

**2.2** Implement inference loop:
```python
def translate(adapter_output, model, tokenizer, max_length=128, num_beams=4):
    pose_tensor = torch.from_numpy(adapter_output).unsqueeze(0).float()
    pose_length = torch.tensor([adapter_output.shape[0]], dtype=torch.long)
    with torch.no_grad():
        generated_ids = model.generate(
            pose=pose_tensor,
            pose_length=pose_length,
            max_length=max_length,
            num_beams=num_beams,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

**2.3** Device handling:
- Auto-detect CUDA / CPU
- Move model and tensors to same device

### Phase 3: Evaluation (1 week)

**3.1** Create evaluation script:
- Run adapter + inference on a held-out set of signed gestures
- Compute WER (Word Error Rate) and EM (Exact Match)
- Compare against baseline (currently `MODEL_UNAVAILABLE` → no translation)

**3.2** Human evaluation:
- Present translations to ISL-fluent evaluators
- Score fluency and accuracy on a 1-5 scale

**3.3** Ablation study:
- Test with/without motion features
- Test with/without confidence values
- Test with different normalization schemes

### Phase 4: Decision (1 week)

**4.1** Based on evaluation results, decide one of:
- **Promote model**: Set `SIGNSAGE_MODEL_PROMOTED=true` and `SIGNSAGE_MODEL_BUNDLE_DIR` if quality is acceptable
- **Iterate**: Refine adapter, try different checkpoint, improve data quality
- **Reject**: Abandon external model; pursue training a SignSage-native model

### Phase 5: Deployment (optional, post-approval)

**5.1** Configure backend:
```bash
SIGNSAGE_MODEL_PROMOTED=true
SIGNSAGE_MODEL_BUNDLE_DIR=/path/to/model/bundle
SIGNSAGE_DATASET_VERSION=1.0.0
```

**5.2** Frontend integration:
- Route translation requests through the adapter + model inference
- Handle `MODEL_UNAVAILABLE` gracefully if model bundle missing

**5.3** Monitoring:
- Log translation success/failure rates
- Monitor for distribution drift (changing landmark distributions over time)
- Periodic re-evaluation with new data

### Phase 1 Milestones

| Milestone | Deliverable | Timeline |
|-----------|-------------|----------|
| M1 | Adapter module `signage_hf_adapter.py` with all conversion logic | Week 1 |
| M2 | Unit tests covering all adapter functions | Week 1 |
| M3 | T5 model loading and inference on synthesized data | Week 2 |
| M4 | End-to-end test: MediaPipe → adapter → T5 → text | Week 2 |
| M5 | Evaluation script and preliminary results | Week 3 |
| M6 | Go/No-Go decision meeting | Week 3 |

## 11. Testing Requirements Before Integration

Before integrating the external model into the SignSage pipeline, the following tests must pass:

1. **Component filtering test**: Given SignSage `HolisticFrame.landmarks` (543 × 3), adapter outputs exactly 75 landmarks (33 pose + 21 left hand + 21 right hand) with confidence values appended.

2. **Dimensionality test**: Adapter output tensor has shape `(seq_length, 300)` for spatial models or `(seq_length, 600)` for motion models, matching `POSE_CONFIG[model_id]["pose_dim"]`.

3. **Motion feature test**: For motion models, adapter output includes velocity features: `velocity[t] = pose[t] - pose[t-1]`, `velocity[0] = 0`, and the combined tensor has correct dimension.

4. **Normalization test**: Adapter applies z-score normalization matching `pose_utils.py normalize_pose(method="zscore")` — verified by running on test data and checking mean≈0, std≈1 per dimension.

5. **Padding/truncation test**: Sequences >500 frames truncated; <500 frames zero-padded; actual length correctly reported.

6. **Inference test**: Given a valid adapter output, the T5 model produces text output (even if nonsensical — the test checks that the pipeline runs without error, not that the text is accurate).

7. **Device test**: Pipeline runs on both CPU and CUDA (if available) without errors.

8. **Fallback test**: When adapter or model fails, the pipeline gracefully returns `MODEL_UNAVAILABLE` without crashing or producing garbled text.

9. **Quality test** (optional but recommended): On a small evaluation set, WER < some threshold or human evaluators rate translations ≥3/5 for fluency.

10. **Integration test**: The full SignSage pipeline (camera → MediaPipe landmarks → adapter → T5 inference → text output) works end-to-end in the development environment.
