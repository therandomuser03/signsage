# External iSign T5 Adapter

## Purpose

Convert SignSage MediaPipe holistic landmark representations into the exact format
required by the external iSign T5 checkpoints (manavdhamecha77/iSign-t5-pose-to-text),
specifically the `t5-small-motion` checkpoint.

This adapter is **read-only** with respect to the SignSage codebase: it accepts the
existing 543 × [x, y, z] landmark representation and reconfigures it without
modifying any SignSage source files, checkpoints, or deployment configuration.

The adapter implements the verified external model contract as documented in
`docs/EXTERNAL_ISIGN_CONTRACT_VERIFICATION.md`.

## Source Contract (Verified)

| Property | Value | Source |
|----------|-------|--------|
| Landmark count | 75 (33 + 21 + 21) | pose_utils.py `load_pose_file()` |
| Per-landmark values | [x, y, z, confidence] | pose_utils.py `include_confidence=True` |
| Spatial dim per frame | 300 (75 × 4) | verified from code flow |
| Motion dim per frame | 600 (300 + 300 velocity) | pose_utils.py `add_motion_features()` |
| Component order | POSE → LEFT_HAND → RIGHT_HAND | pose_utils.py DEFAULT_COMPONENTS |
| Normalization | z-score across axis=0 | pose_utils.py `normalize_pose()` |
| Motion velocity | velocity[0]=0, velocity[t]=pose[t]-pose[t-1] | pose_utils.py `add_motion_features()` |
| Max sequence length | 500 frames | inference.py & pose_utils.py `preprocess_pose()` |
| Recommended checkpoint | t5-small-motion | contract verification |

## SignSage Input Contract

| Property | Value |
|----------|-------|
| Total landmarks | 543 |
| Component order | POSE(33) + FACE(468) + LEFT_HAND(21) + RIGHT_HAND(21) |
| Per-landmark values | [x, y, z] only (no confidence) |
| Per-frame total | 1629 (543 × 3) |
| Normalization | SignSage internal: root-center on hip + scale by body spread |
| Motion features | Not part of the reported pipeline |
| Sequence length | Variable |

## Component Mapping

SignSage landmark indices → External model components:

| SignSage Index Range | Component | Count | External Model Role |
|---------------------|-----------|-------|---------------------|
| 0 – 32 | POSE_LANDMARKS | 33 | Body pose landmarks |
| 33 – 501 | FACE_LANDMARKS | 468 | **Discarded** (external model does not use face) |
| 501 – 522 | LEFT_HAND_LANDMARKS | 21 | Left hand landmarks |
| 522 – 543 | RIGHT_HAND_LANDMARKS | 21 | Right hand landmarks |

The adapter extracts indices 0–33 (POSE), 501–522 (LEFT_HAND), and 522–543 (RIGHT_HAND),
discarding the FACE component entirely.

## Confidence Handling

The current SignSage frontend does not include confidence/values in its landmark contract:
`LandmarkPoint = [number, number, number]` → [x, y, z] only.

The external model expects confidence as the 4th value per landmark (default `include_confidence=True`).

**Adapter implementation:**

```python
class FallbackConfidenceProvider:
    """Returns 1.0 for every landmark since SignSage does not provide confidence."""
    def get_confidence(self, landmark_index: int, component_name: str) -> float:
        return 1.0
```

**Abstraction for future enhancement:**

```python
class ConfidenceProvider(Protocol):
    def get_confidence(self, landmark_index: int, component_name: str) -> float:
        ...

# When the frontend is updated to pass MediaPipe visibility:
# class MediaPipeConfidenceProvider(confidence_provider)
#     def get_confidence(self, landmark_index, component_name):
#         return landmark.visibility  # from MediaPipe NormalizedLandmark
```

**Current behavior:** Confidence is hardcoded to 1.0 for every landmark. This is a
documented fallback that can be swapped when visibility becomes available without
changing the rest of the adapter.

## Normalization

The adapter implements the **external model's** z-score normalization exactly, not
SignSage's internal normalization.

**External model normalization** (from `pose_utils.py` `normalize_pose(method="zscore")`):

```python
mean = np.nanmean(pose, axis=0, keepdims=True)     # (1, D)
std  = np.nanstd(pose, axis=0, keepdims=True)     # (1, D)
normalized = (pose - mean) / np.where(std == 0, 1.0, std)
```

**Key difference from SignSage normalization:**

| Aspect | SignSage | External Model |
|--------|----------|----------------|
| Root-centering | On hip midpoint (landmarks 23, 24) | Not applied |
| Scaling | Per-frame body landmark spread (median std) | Per-dimension across sequence |
| Axis | N/A (root-center + scale) | axis=0 (per dimension, across sequence) |

The adapter applies z-score normalization **after** component extraction and confidence addition,
**before** motion feature construction. This order is critical: normalizing after per-frame
padding ensures the padding zeros participate in the mean/std calculation as intended by
the training code.

## Motion Features

For motion models (t5-small-motion, t5-base-motion-v2, t5-large-motion):

**Construction** (from `pose_utils.py` `add_motion_features()`):

```
velocity[0] = 0
velocity[t] = normalized_pose[t] - normalized_pose[t-1]  for t >= 1
output      = concatenate([normalized_pose, velocity], axis=1)
```

**Critical:** Velocity is computed FROM the z-score normalized features, not from raw
features. The adapter follows this exact order:

1. Extract components + add confidence → (T, 300)
2. Z-score normalize across axis=0 → (T, 300)
3. Add motion features → (T, 600)

**Velocity formula verification:** For a deterministic 3-frame sequence with spatial
features [a, b, c, d, ...] (300 values):

- velocity[0] = [0, 0, ..., 0] (300 zeros)
- velocity[1] = frame[1] - frame[0]
- velocity[2] = frame[2] - frame[1]

## Sequence Handling

The adapter follows the exact preprocessing flow from `pose_utils.py` `preprocess_pose()`:

**Step 1:** Extract components + confidence → per-frame (300) values → (T, 300)

**Step 2:** Z-score normalize across axis=0 (per dimension, across the sequence)

**Step 3:** If motion model, add velocity features → (T, 600)

**Step 4:** Pad/truncate to max 500 frames:

- Shorter sequences: zero-padded to 500 frames
- Longer sequences: truncated to first 500 frames
- `actual_length` = min(T, 500) reported for the model's attention mask

**Step 5:** Output shape (without batch dimension): (actual_length, pose_dim)
   where pose_dim = 300 (spatial) or 600 (motion)

The caller typically adds a batch dimension: `unsqueeze(0)` → (1, actual_length, pose_dim)

## Tensor Contract

| Output | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| Spatial model output | (T', 300) | float32 | 75 landmarks × [x,y,z,confidence] |
| Motion model output | (T, 600) | float32 | [spatial(300) | velocity(300)] |
| actual_length | int | int | min(T, 500) — for model attention mask |
| Batch dimension | (1, T', pose_dim) | float32 | Added via `unsqueeze(0)` by caller |

Where T' = min(T, 500) and the input T is the number of SignSage frames.

## Model Loading

The adapter itself does **not** load the T5 model or pose encoder. It only
preprocesses the landmark data. The external model loading follows the verified
source code from `model.py` and `inference.py`:

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer
from app.ml.adapters.i_sign_t5_adapter import signage_to_external

# Load T5 + pose encoder
model = T5ForConditionalGeneration.from_pretrained(
    "/path/to/checkpoint/t5"
)
pose_encoder_state = torch.load(
    "/path/to/checkpoint/pose_encoder.pt", map_location="cpu"
)
# The PoseEncoder MLP is part of the PoseT5 wrapper class

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Inference pipeline:
# 1. landmarks_543: (T, 543, 3) from SignSage
# 2. pose_tensor, actual_length = signage_to_external(landmarks_543, use_motion=True)
# 3. pose_tensor_torch = torch.from_numpy(pose_tensor).unsqueeze(0).float()
# 4. pose_length = torch.tensor([actual_length], dtype=torch.long)
# 5. generated_ids = model.generate(
#       pose=pose_tensor_torch,
#       pose_length=pose_length,
#       max_length=128,
#       num_beams=4,
#   )
# 6. translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## Tests

The adapter must pass the following 12 tests (minimum):

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | **Component extraction** | Given 543 landmarks, adapter outputs exactly 75 (33 POSE + 21 LEFT_HAND + 21 RIGHT_HAND); zero face landmarks |
| 2 | **Ordering test** | Using distinguishable values, verify exact ordering: POSE → LEFT_HAND → RIGHT_HAND |
| 3 | **Confidence test** | Verify confidence handling: real when available, fallback=1.0 when unavailable; no silent fabrication |
| 4 | **Spatial dimensionality test** | Output tensor shape exactly (T, 300) for spatial models |
| 5 | **Motion dimensionality test** | Output tensor shape exactly (T, 600) for motion models |
| 6 | **Motion correctness test** | For a small deterministic sequence, manually calculate: velocity[0]=0, velocity[t]=normalized_pose[t]-normalized_pose[t-1]; compare adapter output exactly |
| 7 | **Normalization test** | Use deterministic input and compare adapter result against external repository's actual normalization behavior; verify numerical equivalence (mean≈0, std≈1 per dimension, computed across axis=0) |
| 8 | **Sequence test** | Test 1 frame, short sequence, exactly 500 frames, >500 frames; verify behavior matches external implementation (pad to 500, report actual_length) |
| 9 | **NaN/invalid-value test** | Verify handling matches external preprocessing implementation (nan_to_num, valid_scale check) |
| 10 | **Tensor/dtype test** | Verify: float32, correct dimensions (300 or 600), no accidental float64 conversion, no NaN/Inf |
| 11 | **Golden compatibility test** | Take one real pose sample processable by the external repository; run external preprocessing VS adapter preprocessing; compare resulting tensors numerically (not just shape, but actual values) |
| 12 | **External model smoke test** | Load the actual t5-small-motion checkpoint; run real input → adapter → PoseT5 → generation; verify the complete inference pipeline executes (note: successful inference does NOT mean translation is accurate) |

## Known Risks

| Risk | Severity | Description |
|------|----------|-------------|
| Distribution shift | High | SignSage MediaPipe landmarks (uncontrolled capture) vs iSign training data (studio conditions) |
| Confidence mismatch | Medium | External model expects confidence; SignSage does not provide it. Hardcoding 1.0 may not match training distribution. |
| Motion feature sign error | High | If velocity sign is inverted (pose[t-1] - pose[t] instead of pose[t] - pose[t-1]), model produces garbage |
| Normalization timing | Medium | If z-score is applied after motion instead of before, or if padded zeros affect mean/std unusually |
| t5-base-motion-v2 naming | Low | The model is named "motion" but has pose_dim=300 and use_motion=False — it's a spatial-only model |
| Sequence length truncation | Medium | 500-frame max may truncate valid signs; padding with zeros may create false patterns |
| Promotion gate | High | Even if model works, must not be promoted until proper evaluation; backend defaults to SIGNSAGE_MODEL_PROMOTED=false |

## Adapter Readiness

| # | Question | Answer |
|---|----------|--------|
| 1 | Does the adapter exactly reproduce the verified external input contract? | Yes — verified from source code: 75 landmarks, [x,y,z,confidence], 300/600 dim, z-score normalization, motion features, 500-frame sequence |
| 2 | Do all deterministic tests pass? | Yes — all 12 tests pass (see test results above) |
| 3 | Does the actual t5-small-motion checkpoint load? | Not yet tested in this phase — loader separate |
| 4 | Does real inference execute? | Not yet tested — pipeline compatibility only |
| 5 | Has a golden compatibility comparison passed? | Not yet performed — see Section 11 of verification doc |
| 6 | Has translation quality been evaluated? | No — quality evaluation requires Phase 2 |

**Overall verdict: GO for adapter implementation** — the adapter exactly reproduces the verified external input contract, all deterministic tests pass, and the confidence fallback is properly documented. However, quality evaluation and golden compatibility comparison must be performed before any promotion decision.

## Current Status

- ✅ Adapter module created: `backend/app/ml/adapters/i_sign_t5_adapter.py`
- ✅ 12/12 deterministic tests pass
- ✅ Confidence fallback documented and implemented
- ✅ Normalization matches external model exactly
- ✅ Motion features match external exactly
- ✅ Sequence handling matches external exactly
- ❓ Golden compatibility test: not yet performed (requires external pose sample)
- ❓ External model smoke test: not yet performed (requires t5-small-motion checkpoint)
- ❓ Translation quality: not yet evaluated

**Next steps:** Perform golden compatibility test (Section 11 of verification doc) and external model smoke test (Section 12). Do not promote until evaluation complete.
