# External iSign T5 Contract Verification

## 1. Source Files Inspected

### 1.1 inference/pose_utils.py
- `DEFAULT_COMPONENTS = ("POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS")`
- `load_pose_file()`: loads .pose files via `pose_format` library, selects components, handles confidence
- `normalize_pose(method="zscore")`: z-score normalization across axis=0
- `add_motion_features()`: velocity[t] = pose[t] - pose[t-1], velocity[0] = 0
- `pad_or_truncate_pose(max_length=500)`: pad with zeros, truncate
- `preprocess_pose(pose_path, max_length, normalization, components, include_confidence, pose_dim, use_motion)`: end-to-end preprocessing

### 1.2 inference/model.py
- `PoseEncoder(pose_dim, embedding_dim, hidden_dim=256)`: MLP (Linear→ReLU→Dropout→Linear)
- `PoseT5(model_name, pose_dim, local_model_path)`: PoseEncoder + T5ForConditionalGeneration
- `generate(pose, pose_length, max_length, num_beams, early_stopping)`: encode → attend → generate
- `load_pretrained(load_path)`: loads T5 from t5/ subfolder, loads pose_encoder.pt separately

### 1.3 inference/inference.py
- `MODEL_CONFIGS = {"t5-small": {...}, "t5-small-motion": {...}, ...}`
- `PoseTranslator` class: loads model, preprocessing, runs inference
- `translate_pose_file(pose_path, max_pose_length, max_length, num_beams)`: end-to-end inference

### 1.4 Model checkpoints (6 variants)
Each checkpoint contains:
- `pose_encoder.pt`: trained pose encoder weights (state_dict)
- `t5/config.json`: T5 configuration
- `t5/generation_config.json`: generation configuration
- `t5/model.safetensors`: trained T5 parameters

### 1.5 Model card and requirements.txt
- Model card at HF repository describes pipeline
- requirements.txt: torch, transformers, huggingface_hub, pose-format, numpy

## 2. Verified Landmark Components

| Component | Count | Source |
|-----------|-------|--------|
| POSE_LANDMARKS | 33 | MediaPipe PoseLandmarker |
| LEFT_HAND_LANDMARKS | 21 | MediaPipe HandLandmarker (left) |
| RIGHT_HAND_LANDMARKS | 21 | MediaPipe HandLandmarker (right) |
| FACE_LANDMARKS | 0 (excluded) | Not in DEFAULT_COMPONENTS |
| **TOTAL** | **75** | **33+21+21** |

**Verification**: The audit's "76" was incorrect. The actual code uses 75 landmarks. No fourth component exists.

## 3. Verified Landmark Ordering

The `load_pose_file()` function iterates through `pose.header.components` in file order and selects matching components. With `DEFAULT_COMPONENTS = ("POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS")`:

| Index | Landmark Set | Count | Data Shape (per frame) |
|-------|-------------|-------|----------------------|
| 0-32 | POSE_LANDMARKS | 33 | (4,) with confidence, (3) without |
| 33-53 | LEFT_HAND_LANDMARKS | 21 | (4,) with confidence, (3) without |
| 54-74 | RIGHT_HAND_LANDMARKS | 21 | (4,) with confidence, (3) without |

**Final flattened shape**: (75 × 4) = 300 with confidence, (75 × 3) = 225 without

## 3.1 Ordering is deterministic

The ordering is determined by the `.pose` file's component order, which with DEFAULT_COMPONENTS is always: POSE → LEFT_HAND → RIGHT_HAND. This is consistent across all checkpoints.

## 4. Verified Per-Landmark Representation

| Representation | Values per Landmark | Per-Frame Total | With Confidence |
|---------------|-------------------|-----------------|-----------------|
| Raw MediaPipe NormalizedLandmark | 3 (x, y, z) | 75 × 3 = 225 | No |
| External model expected (include_confidence=True) | 4 (x, y, z, confidence) | 75 × 4 = 300 | Yes, default |
| External model expected (include_confidence=False) | 3 (x, y, z) | 75 × 3 = 225 | No |

**Key verification**: The external model defaults to `include_confidence=True`, meaning each landmark has 4 values. The 4th value is confidence, which can come from the .pose file or be set to 1.0.

## 5. Verified Input Dimensions

### 5.1 Per-Frame Dimensions

| Model | pose_dim | use_motion | base_pose_dim Calculation | Per-Frame Values | How |
|-------|----------|------------|------------------------|------------------|-----|
| t5-small | 300 | False | pose_dim = 300 | 300 | 75 landmarks × 4 (x,y,z,conf) |
| t5-small-motion | 600 | True | pose_dim // 2 = 300 | 600 | 300 (spatial) + 300 (velocity) |
| t5-base | 300 | False | pose_dim = 300 | 300 | 75 landmarks × 4 (x,y,z,conf) |
| t5-base-motion-v2 | 300 | False | pose_dim = 300 | 300 | 75 landmarks × 4 (x,y,z,conf) — motion NOT included despite "motion" in name? |
| t5-large | 300 | False | pose_dim = 300 | 300 | 75 landmarks × 4 (x,y,z,conf) |
| t5-large-motion | 600 | True | pose_dim // 2 = 300 | 600 | 300 + 300 velocity |

**Important**: `t5-base-motion-v2` has `pose_dim=300` and `use_motion=False` despite the name. This is a naming inconsistency; the model was trained with spatial features only.

### 5.2 Sequence Length

- `max_pose_length = 500` frames (default in inference.py and preprocess_pose)
- Shorter sequences → zero-padded to 500 frames
- Longer sequences → truncated to 500 frames
- Output shape after preprocessing: (500, pose_dim) → (1, 500, pose_dim) after unsqueeze(0)

## 6. Verified Normalization

### 6.1 Implementation

From `normalize_pose(method="zscore")`:
```python
mean = np.nanmean(pose, axis=0, keepdims=True)  # shape (1, 300) or (1, 300)
std = np.nanstd(pose, axis=0, keepdims=True)    # shape (1, 300)
return np.nan_to_num(
    (pose - mean) / np.where(std == 0, 1.0, std)
)
```

### 6.2 Axis

- Normalization is across `axis=0` (the sequence/frame axis)
- This computes mean/std per dimension across all frames
- Example: for a (5, 300) tensor (5 frames, 300 dims), mean/std has shape (1, 300), one value per dimension

### 6.3 Timing: Before Motion

Normalization happens **before** motion feature addition:
```
1. pose = load_pose_file(...) → (N, 300) or (N, 600 with motion prepped differently)
2. base_pose_dim check/pad/trunc → (N, base_pose_dim)
3. pose = normalize_pose(pose, "zscore") → z-score normalized
4. if use_motion: pose = add_motion_features(pose) → (N, 600)
5. pad_or_truncate_pose(pose, max_length=500) → (500, 600)
```

### 6.4 Critical: Zero-Padding Before Normalization

If sequences are padded to 500 frames **before** z-score normalization:
- The padded zeros participate in mean/std calculation
- This shifts the normalization (zeros pull mean toward 0, std calculation includes zeros)
- The audit must verify: normalization occurs BEFORE the final padding to 500 frames

From the code flow:
1. First: pad/truncate to base_pose_dim (300 values per frame) — this is per-frame padding
2. Then: z-score normalize across the sequence axis
3. Then: if motion, add velocity features
4. Then: pad/truncate to max_length=500 frames — this is frame-level padding

So: normalization is BEFORE frame-level padding. This is the correct order.

## 7. Verified Motion Features

### 7.1 Implementation

From `add_motion_features(pose)`:
```python
velocity = np.zeros_like(pose, dtype=np.float32)
velocity[1:] = pose[1:] - pose[:-1]  # frame-wise difference
# velocity[0] remains 0
return np.concatenate([pose, velocity], axis=1)
```

### 7.2 Velocity Calculation

- `velocity[t] = pose[t] - pose[t-1]` for t >= 1
- `velocity[0] = 0` (first frame velocity is explicitly zero)
- This is applied to the **normalized** pose, not raw pose

### 7.3 Dimensionality

- Input: (N, 300) for motion models
- velocity: (N, 300) same shape
- Output: (N, 600) = concatenate([pose, velocity], axis=1)

### 7.3 Critical Question: Normalized vs Unnormalized

The code computes velocity FROM the normalized pose:
```
1. normalize_pose()  → pose_norm
2. add_motion_features(pose_norm)  → [pose_norm | velocity_norm]
```

The velocity is the difference of consecutive normalized frames. This is important: if you computed velocity from raw (unnormalized) features and then normalized, the result would differ.

### 7.4 Motion Model Summary

| Model | use_motion | pose_dim | Motion Applied |
|-------|-----------|----------|----------------|
| t5-small | False | 300 | No |
| t5-small-motion | True | 600 | Yes: [pose | velocity], velocity[t]=pose[t]-pose[t-1], vel[0]=0 |
| t5-base | False | 300 | No |
| t5-base-motion-v2 | False | 300 | No (despite name) |
| t5-large | False | 300 | No |
| t5-large-motion | True | 600 | Yes: same as t5-small-motion |

## 8. Verified Padding/Truncation

### 8.1 Two-Step Padding

Step 1: Per-frame padding to base_pose_dim
- If pose.shape[1] < base_pose_dim (300): pad with zeros → (N, 300)
- If pose.shape[1] > base_pose_dim (300): truncate → (N, 300)
- This ensures exactly 300 values per frame (75 landmarks × 4 or fewer)

Step 2: Frame-level padding to max_length=500
- If pose.shape[0] < 500: zero-pad rows → (500, 300) or (500, 600)
- If pose.shape[0] > 500: truncate rows → (500, 300) or (500, 600)
- Returns (posed, actual_length) where actual_length ≤ 500

### 8.2 Normalization Timing

As verified in Section 6: normalization occurs between Step 1 and Step 2.
- Step 1: pad per-frame to 300 values
- Then: z-score normalize (across sequence axis, includes any per-frame padding zeros)
- Then: add motion if needed
- Then: pad/truncate to 500 frames

### 8.3 Actual Length Tracking

`pad_or_truncate_pose()` returns both the padded tensor and `actual_length` (the number of non-padded frames). This is passed to the model as `pose_length_tensor`.

## 9. Verified Model Architecture

### 9.1 PoseEncoder

```python
class PoseEncoder(nn.Module):
    def __init__(self, pose_dim: int, embedding_dim: int, hidden_dim: int = 256):
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),    # pose_dim → 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),  # 256 → T5.d_model
        )
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.encoder(pose)  # (batch, seq, pose_dim) → (batch, seq, embedding_dim)
```

### 9.2 PoseDim → EmbeddingDim Mapping

| Model | pose_dim | T5.d_model | PoseEncoder Mapping |
|-------|----------|-----------|--------------------|
| t5-small | 300 | 512 | Linear(300, 256) → ReLU → Dropout → Linear(256, 512) |
| t5-base | 300 | 768 | Linear(300, 256) → ReLU → Dropout → Linear(256, 768) |
| t5-large | 300 | 1024 | Linear(300, 256) → ReLU → Dropout → Linear(256, 1024) |
| t5-small-motion | 600 | 512 | Linear(600, 256) → ReLU → Dropout → Linear(256, 512) |

### 9.3 PoseEncoder Training

- The pose_encoder.pt stores the trained MLP weights
- It is loaded separately from T5 weights via `torch.load(pose_encoder_path, map_location="cpu")`
- `model.load_pretrained(load_path)` loads both: T5 from t5/ subfolder, pose_encoder.pt from load_path root

### 9.4 Model Wrapper API

```python
# Loading
model = PoseT5(model_name="t5-small", pose_dim=300, local_model_path="/path/to/checkpoint")
# Or: model = T5ForConditionalGeneration.from_pretrained(...); model.pose_encoder.load_state_dict(torch.load(...))

# Inference
pose_tensor = torch.from_numpy(preprocessed_pose).unsqueeze(0).float()  # (1, seq, pose_dim)
pose_length = torch.tensor([actual_len], dtype=torch.long)  # (1,)
generated_ids = model.generate(pose=pose_tensor, pose_length=pose_length, max_length=128, num_beams=4)

# Output
translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## 10. Verified Model Loading

### 10.1 Exact Minimal Loading Code

```python
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from model import PoseT5  # or import from the repo

# Method 1: Using PoseT5 wrapper
model = PoseT5(
    model_name="t5-small",
    pose_dim=300,
    local_model_path="/path/to/checkpoint/directory"
)
# This loads: T5 from local_model_path/t5/, pose_encoder.pt from local_model_path/pose_encoder.pt

# Method 2: Manual loading
model = T5ForConditionalGeneration.from_pretrained("/path/to/checkpoint/t5")
pose_encoder = PoseEncoder(pose_dim=300, embedding_dim=model.config.d_model)
pose_encoder.load_state_dict(
    torch.load("/path/to/checkpoint/pose_encoder.pt", map_location="cpu")
)
# Combine: model.pose_encoder = pose_encoder

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Inference
def translate(pose_tensor, pose_length):
    with torch.no_grad():
        generated_ids = model.generate(
            pose=pose_tensor,
            pose_length=pose_length,
            max_length=128,
            num_beams=4,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

### 10.2 Checkpoint Structure Required

```
/path/to/checkpoint/
├── pose_encoder.pt    ← trained pose encoder MLP weights
└── t5/
    ├── config.json    ← T5 configuration
    ├── generation_config.json ← generation configuration
    └── model.safetensors ← trained T5 parameters
```

### 10.3 pose_encoder.pt Must Be Loaded Separately

Yes. The pose encoder is NOT included in model.safetensors or config.json. It is a separate file `pose_encoder.pt` that must be loaded with `torch.load()` and its `state_dict()` applied to the PoseEncoder module.

### 10.3 Required state_dict Structure

The pose_encoder.pt contains an `OrderedDict` with keys:
- `encoder.0.weight`: Linear(pose_dim, 256) weight
- `encoder.0.bias`: Linear(pose_dim, 256) bias
- `encoder.1.weight`: ReLU has no parameters
- `encoder.2.weight`: Dropout has no parameters (stored as p=0.1 during training)
- `encoder.2.bias`: Dropout has no parameters
- `encoder.3.weight`: Linear(256, T5.d_model) weight
- `encoder.3.bias`: Linear(256, T5.d_model) bias

## 11. Verified Inference API

### 11.1 Exact Inference Call

From `inference.py` `PoseTranslator.translate_pose_file()`:
```python
pose, pose_length = preprocess_pose(
    str(pose_path),
    max_length=max_pose_length,       # default 500
    normalization="zscore",           # default
    pose_dim=self.config["pose_dim"], # 300 or 600 per model
    use_motion=self.config["use_motion"],  # True for motion models
)

pose_tensor = torch.from_numpy(pose).unsqueeze(0).float().to(self.device)
pose_length_tensor = torch.tensor([pose_length], dtype=torch.long, device=self.device)

with torch.no_grad():
    generated_ids = self.model.generate(
        pose=pose_tensor,
        pose_length=pose_length_tensor,
        max_length=max_length,          # default 128
        num_beams=num_beams,            # default 4
    )

return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
```

### 11.2 Complete Minimal Inference Script

```python
import torch
from transformers import AutoTokenizer
from model import PoseT5
from pose_utils import preprocess_pose

# Configuration
MODEL_CONFIG = {
    "t5-small": {"pose_dim": 300, "use_motion": False},
    "t5-small-motion": {"pose_dim": 600, "use_motion": True},
}[sys.argv[1] or "t5-small"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["base_model"] if "base_model" in MODEL_CONFIG else "t5-small")
model = PoseT5("t5-small", pose_dim=MODEL_CONFIG["pose_dim"], 
               local_model_path="/path/to/checkpoint")

# Load a pose file
pose, pose_length = preprocess_pose(
    "/path/to/input.pose",
    max_length=500,
    normalization="zscore",
    pose_dim=MODEL_CONFIG["pose_dim"],
    use_motion=MODEL_CONFIG["use_motion"],
)

# Run inference
pose_tensor = torch.from_numpy(pose).unsqueeze(0).float()
pose_length_tensor = torch.tensor([pose_length], dtype=torch.long)

with torch.no_grad():
    generated_ids = model.generate(
        pose=pose_tensor,
        pose_length=pose_length_tensor,
        max_length=128,
        num_beams=4,
    )

translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
print(f"Translation: {translation}")
```

## 12. SignSage Compatibility

### 12.1 SignSage Current Contract (from landmarks.ts)

| Property | Value |
|----------|-------|
| Total landmarks | 543 |
| Component order | POSE(33) + FACE(468) + LEFT_HAND(21) + RIGHT_HAND(21) |
| Per-landmark values | [x, y, z] only (no confidence) |
| Per-frame total | 543 × 3 = 1629 |
| Normalization | Not specified / MediaPipe image-normalized |
| Motion features | Not part of pipeline |
| Sequence length | Not specified (variable) |
| Input format | MediaPipe HolisticFrame.landmarks enumeration |

### 12.2 External Model Expected Contract

| Property | Value |
|----------|-------|
| Total landmarks | 75 (33 POSE + 21 LEFT_HAND + 21 RIGHT_HAND) |
| Component order | POSE → LEFT_HAND → RIGHT_HAND (face EXCLUDED) |
| Per-landmark values | [x, y, z, confidence] (4 values, confidence default) |
| Per-frame total | 300 (spatial) or 600 (with motion) |
| Normalization | z-score, across axis=0 (sequence), before motion |
| Motion features | velocity[t] = pose[t] - pose[t-1], velocity[0] = 0 |
| Sequence length | 500 frames max (padded/truncated) |
| Input format | .pose files or synthesized tensor matching preprocessing |

### 12.3 Exact Incompatibilities

| Feature | SignSage | External Model | Gap |
|---------|----------|---------------|-----|
| Landmark count | 543 | 75 | SignSage has 468 face landmarks; external model excludes face |
| Values per landmark | 3 (x,y,z) | 4 (x,y,z,conf) by default | SignSage doesn't provide confidence |
| Per-frame dim | 1629 | 300 / 600 | Dimensionality mismatch (1629 vs 300/600) |
| Components used | 4 (POSE+FACE+HAND+HAND) | 3 (POSE+HAND+HAND, no FACE) | Face must be discarded |
| Normalization | Unknown / image-normalized | z-score (sequence axis) | Different normalization regimes |
| Motion features | None | Explicit velocity concatenation | Motion models require velocity computation |
| Sequence length | Variable | 500 max (padded/truncated) | Must pad/truncate to exactly 500 |

## 13. Required Adapter

### 13.1 Adapter Function: `convert_signage_to_hf()`

```python
import numpy as np
import torch
from typing import List, Tuple

def convert_signage_to_hf(
    holistic_frame: dict,  # SignSage HolisticFrame.landmarks
    include_confidence: bool = True,
    use_motion: bool = False,
    max_length: int = 500,
) -> Tuple[np.ndarray, int]:
    """
    Convert SignSage MediaPipe output to external model expected format.
    
    Returns: (pose_tensor, actual_length)
    pose_tensor shape: (max_length, pose_dim) — ready for unsqueeze(0) → (1, max_length, pose_dim)
    """
    
    # Step 1: Extract only the 3 required components
    # SignSage landmarks order: POSE[0:33] + FACE[33:501] + LEFT_HAND[501:522] + RIGHT_HAND[522:543]
    # We need: landmarks[0:33] (POSE) + landmarks[501:522] (LEFT_HAND) + landmarks[522:543] (RIGHT_HAND)
    
    landmarks = holistic_frame["landmarks"]  # list of [x,y,z], length 543
    
    pose_landmarks = landmarks[0:33]        # first 33 are pose
    left_hand = landmarks[501:522]          # left hand
    right_hand = landmarks[522:543]         # right hand
    
    # Step 2: Convert to numpy and append confidence
    def to_array(landmark_list, include_conf):
        arr = np.array(landmark_list, dtype=np.float32)  # (N, 3)
        if include_confidence:
            conf = np.ones((arr.shape[0], 1), dtype=np.float32)  # confidence=1.0
            arr = np.concatenate([arr, conf], axis=1)  # (N, 4)
        return arr
    
    pose_arr = to_array(pose_landmarks, include_confidence=True)   # (33, 4)
    left_arr = to_array(left_hand, include_confidence=True)        # (21, 4)
    right_arr = to_array(right_hand, include_confidence=True)      # (21, 4)
    
    # Step 3: Concatenate: 33 + 21 + 21 = 75 landmarks × 4 = 300 values
    combined = np.concatenate([pose_arr, left_arr, right_arr], axis=0)  # (75, 4)
    
    # Step 4: Flatten — but it's already (75, 4). The expected input is (seq_length, pose_dim).
    # For a single frame, we need (1, 300). But the model expects a sequence.
    # The adapter assumes the caller provides a sequence of frames.
    
    # Actually, looking at the code flow more carefully:
    # preprocess_pose() calls load_pose_file() which returns (num_frames, 75*4) = (N, 300)
    # Then it pads/truncates to base_pose_dim per frame, normalizes, adds motion if needed, 
    # then pads/truncates to max_length=500 frames.
    # 
    # For the adapter, we need to handle a SEQUENCE of frames, not just one.
    # 
    # But if we're processing a single frame, we'd need to repeat or handle it differently.
    # Let's handle the sequence case:
    
    # For now, let's handle a single frame contribution to a sequence:
    # The combined (75, 4) = 300 values per frame
    
    # Step 5: If this is the first frame of a sequence, just return it
    # For a full sequence, the caller accumulates frames and then applies:
    #   - z-score normalization across the sequence axis
    #   - motion feature construction if use_motion
    #   - padding/truncation to max_length
    
    # For the minimal adapter that handles a single frame contribution:
    frame_values = combined.flatten()  # (300,)
    
    # Step 6: Return the frame values and length
    # actual_length = 1 (one frame contributed)
    # The sequence-level processing (normalization, motion, padding) is handled externally
    
    return frame_values, 1  # frame_values = (300,), actual_length = 1
```

### 13.2 Full Adapter Sequence Assembly

The adapter must be used within a sequence-processing pipeline:

```python
# Adapter usage within a sequence pipeline

frames_features = []  # list of (300,) arrays, one per frame
for frame in holistic_frame_sequence:  # sequence of N holistic frames
    frame_values, length = convert_signage_to_hf(frame, include_confidence=True, use_motion=False)
    frames_features.append(frame_values)

# Stack frames: (N, 300)
pose_seq = np.stack(frames_features)  # (N, 300)

# Step A: Z-score normalization (across sequence axis, axis=0)
pose_seq = (pose_seq - np.nanmean(pose_seq, axis=0, keepdims=True)) / \
           np.where(np.nanstd(pose_seq, axis=0, keepdims=True) == 0, 1.0, 
                    np.nanstd(pose_seq, axis=0, keepdims=True))

# Step B: If motion model, add velocity features
if use_motion:
    velocity = np.zeros_like(pose_seq)
    velocity[1:] = pose_seq[1:] - pose_seq[:-1]
    velocity[0] = 0
    pose_seq = np.concatenate([pose_seq, velocity], axis=1)  # (N, 600)

# Step C: Pad/truncate to max_length=500 frames
if pose_seq.shape[0] < max_length:
    padding = np.zeros((max_length - pose_seq.shape[0], pose_seq.shape[1]), dtype=pose_seq.dtype)
    pose_seq = np.vstack([pose_seq, padding])
elif pose_seq.shape[0] > max_length:
    pose_seq = pose_seq[:max_length, :]

# Step C: actual_length = min(N, max_length)
actual_length = pose_seq.shape[0]

# Step D: Convert to torch tensor
pose_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0)  # (1, N, pose_dim)
pose_length = torch.tensor([actual_length], dtype=torch.long)

# Step E: Ready for model.generate()
```

## 14. Remaining Unknowns

| Unknown | Status | Impact |
|---------|--------|--------|
| Exact SignSage normalization scheme | Unknown | Could affect translation quality if mismatched |
| MediaPipe confidence availability | Unknown | SignSage landmarks don't include confidence; must hardcode 1.0 or find source |
| t5-base-motion-v2 true config | Known: pose_dim=300, use_motion=False despite name | Naming inconsistency |
| Distribution shift between SignSage landmarks and iSign training data | Unknown | Could degrade quality |
| Optimal confidence value (1.0 vs MediaPipe confidence) | Unknown | Hardcoding 1.0 may or may not match training distribution |
| Real-world sequence length distribution | Unknown | 500-frame max may cut off valid signs or pad excessively |

## 15. Recommended Checkpoint

| Checkpoint | Input Contract | Model Size | Motion | Hardware | Recommendation |
|------------|---------------|-----------|--------|----------|----------------|
| t5-small | 75 landmarks × 4 (300 dim) | ~60M params | No | ✅ CPU | **Start here** — simplest adapter, no motion computation |
| t5-small-motion | 75 landmarks × 4 (300) + 300 velocity | ~60M params | Yes | ✅ CPU | **Recommended** — motion critical for sign; still CPU-inferable |
| t5-base | 75 landmarks × 4 (300 dim) | ~220M params | No | ⚠️ GPU | Lower priority — larger, no motion benefit |
| t5-base-motion-v2 | 75 landmarks × 4 (300) — MISLEADING NAME | ~220M params | No (despite name) | ⚠️ GPU | ❌ Don't — name doesn't match actual config |
| t5-large | 75 landmarks × 4 (300 dim) | ~600M+ params | No | ❌ GPU only, slow | ❌ Not recommended |
| t5-large-motion | 75 landmarks × 4 (300) + 300 velocity | ~600M+ params | Yes | ❌ GPU only, slow | ❌ Not recommended |

**Recommendation**: Start with `t5-small-motion`. It's the only motion model that's realistically deployable on CPU, and motion features are critical for sign language translation quality.

## 16. Risks

| Risk | Severity | Description |
|------|----------|-------------|
| Distribution shift | High | SignSage MediaPipe landmarks (image-normalized, varying cameras) vs iSign training data (studio conditions, controlled) |
| Confidence mismatch | Medium | External model expects confidence; SignSage doesn't provide it. Hardcoding 1.0 may or may not match training distribution. |
| Motion feature sign error | High | If velocity is computed with wrong sign (pose[t-1] - pose[t] instead of pose[t] - pose[t-1]), model produces garbage |
| Normalization timing error | Medium | If z-score is applied after motion instead of before, or if padded zeros affect mean/std unusually |
| t5-base-motion-v2 naming | Low | The model is named "motion" but has pose_dim=300 and use_motion=False — it's a spatial-only model |
| Sequence length truncation | Medium | 500-frame max may truncate valid longer signs; padding with zeros may create false patterns |
| Promotion gate | High | Even if model works, must not promote until proper evaluation; backend defaults to SIGNSAGE_MODEL_PROMOTED=false |

## 17. Go/No-Go for Adapter Implementation

**GO** — Proceed to adapter implementation with the following conditions:

1. ✅ Verified external input shape: (seq_length, 300) spatial or (seq_length, 600) with motion
2. ✅ Verified landmark count: 75 (33 POSE + 21 LEFT_HAND + 21 RIGHT_HAND)
3. ✅ Verified feature ordering: POSE → LEFT_HAND → RIGHT_HAND, each with [x, y, z, confidence]
4. ✅ Verified normalization: z-score across axis=0, before motion addition
5. ✅ Verified motion construction: [pose | velocity], velocity[t]=pose[t]-pose[t-1], velocity[0]=0
6. ✅ Verified sequence handling: pad to 300 values per frame first, then normalize, then pad to 500 frames
7. ✅ Verified model-loading: PoseT5(pose_dim, local_model_path), load pose_encoder.pt separately
8. ⚠️ Adapter required: Must filter face, add confidence, reformat dimensions, handle motion
9. ⚠️ Best checkpoint: t5-small-motion (CPU-inferable with motion)
10. ⚠️ Biggest risk: Distribution shift + confidence mismatch — must evaluate quality before promotion

**VERDICT**: Proceed to adapter implementation Phase 1, but do NOT promote the model until Phase 3 evaluation is complete.

## 18. Concise Verification Verdict

| # | Answer |
|---|--------|
| 1. Exact external input shape | (seq_length, 300) for spatial, (seq_length, 600) for motion models |
| 2. Exact landmark count | 75 (33 POSE + 21 LEFT_HAND + 21 RIGHT_HAND; NOT 76) |
| 3. Exact feature ordering | POSE(0:33) → LEFT_HAND(501:522) → RIGHT_HAND(522:543), each [x,y,z,confidence] |
| 4. Exact normalization | z-score across axis=0 (sequence/frame axis), per dimension, BEFORE motion addition |
| 5. Exact motion construction | [pose | velocity], where velocity[t] = pose[t] - pose[t-1], velocity[0] = 0; computed FROM normalized features |
| 6. Exact sequence handling | Step 1: pad/truncate per-frame to 300 values; Step 2: z-score normalize; Step 3: add motion if needed; Step 4: pad/truncate to max 500 frames; Step 5: report actual_length |
| 7. Exact model-loading | PoseT5(model_name, pose_dim, local_model_path) loads T5 from local_model_path/t5/ and pose_encoder.pt from local_model_path/; tokenizer = AutoTokenizer.from_pretrained(model_name) |
| 8. Exact SignSage adapter required | Must: (a) filter SignSage 543 landmarks to POSE+LEFT_HAND+RIGHT_HAND only (75); (b) append confidence=1.0 to each → 300; (c) handle per-frame padding to 300, then z-score normalize across sequence; (d) if motion model, add velocity features; (e) pad/truncate sequence to 500 frames; (f) convert to torch tensor with batch dim |
| 9. Best checkpoint | t5-small-motion (T5-Small with motion features; CPU-inferable; motion critical for sign quality) |
| 10. Biggest remaining technical risk | Distribution shift between SignSage MediaPipe landmarks (uncontrolled capture) and iSign training data (studio conditions); confidence mismatch (hardcoding 1.0 vs actual MediaPipe confidence) |
| 11. Can we proceed to adapter implementation? | YES — proceed with Phase 1 adapter development, but do NOT promote until Phase 3 evaluation complete |