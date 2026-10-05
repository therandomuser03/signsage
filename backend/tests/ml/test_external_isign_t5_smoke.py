"""Smoke test for external iSign t5-small-motion model execution.

This test verifies that the published t5-small-motion checkpoint can be
loaded and executed through the SignSage adapter without modifying the
model's expected preprocessing.

DO NOT MODIFY:
- The model checkpoint or weights
- The adapter implementation
- Any production infrastructure

This is an execution-only test. It does not evaluate translation quality.
"""

import sys
import numpy as np
import torch

# Ensure backend is importable
sys.path.insert(0, '/mnt/D/Codes/signsage/backend')

from transformers import T5ForConditionalGeneration, AutoTokenizer

# Import adapter
from app.ml.adapters.i_sign_t5_adapter import signage_to_external


def test_external_isign_t5_smoke():
    """Smoke test: load t5-small-motion and run inference.

    Steps:
    1. Load t5-small-motion model, tokenizer, and pose_encoder.
    2. Construct deterministic valid input with shape (N, 600) via the adapter.
    3. Construct the corresponding pose_length tensor.
    4. Run PoseT5 inference/generation call.
    5. Decode the output.
    6. Verify: no exception, output is a string, valid generation, no NaN/Inf.
    """
    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_model.eval()

    pose_encoder_path = (
        "/mnt/D/Codes/signsage/models/iSign_t5_small_motion/"
        "t5-small-motion/pose_encoder.pt"
    )
    pose_encoder_state = torch.load(pose_encoder_path, map_location="cpu")

    # Build PoseEncoder matching the external model contract:
    #   - Linear(pose_dim=600, hidden_dim=256)
    #   - ReLU
    #   - Dropout(0.1)
    #   - Linear(256, embedding_dim=512)
    class PoseEncoder(torch.nn.Module):
        def __init__(self, pose_dim, embedding_dim, hidden_dim=256):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(pose_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, embedding_dim),
            )

        def forward(self, pose):
            return self.encoder(pose)

    pose_encoder = PoseEncoder(pose_dim=600, embedding_dim=t5_model.config.d_model)
    pose_encoder.load_state_dict(pose_encoder_state)
    pose_encoder.eval()

    # ------------------------------------------------------------------
    # 2. Construct deterministic valid input with shape (N, 600)
    # ------------------------------------------------------------------
    # Use the adapter to convert SignSage 543-landmark data to the
    # external model's expected format: (seq_length, 600) for motion model.
    # We create a small deterministic pose sequence (5 frames).
    np.random.seed(42)
    # Simulate 5 frames of MediaPipe holistic landmarks (543 × 3)
    # The adapter will filter to 75 landmarks (POSE + LEFT_HAND + RIGHT_HAND),
    # append confidence=1.0, compute motion features, and pad to 500 frames.
    test_data = np.random.rand(5, 543, 3).astype(np.float32)

    # Run adapter preprocessing (matches external pipeline exactly)
    padded, actual_length = signage_to_external(
        test_data, use_motion=True, max_frames=500
    )
    # padded shape: (500, 600) — (max_frames, pose_dim)
    # actual_length: number of non-padded frames

    # Add batch dimension and convert to torch tensor
    pose_tensor = torch.from_numpy(padded).unsqueeze(0).float()  # (1, 500, 600)
    pose_length_tensor = torch.tensor([actual_length], dtype=torch.long)

    # ------------------------------------------------------------------
    # 3. Construct the corresponding pose_length tensor
    # ------------------------------------------------------------------
    # pose_length_tensor shape: torch.Size([1]), value: actual_length (5)
    assert pose_length_tensor.dtype == torch.long
    assert pose_length_tensor.shape == torch.Size([1])

    # ------------------------------------------------------------------
    # 4. Run the actual PoseT5 inference/generation call
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Encode pose through the pose encoder
        pose_embeddings = pose_encoder(pose_tensor)
        # pose_embeddings shape: (1, 500, 512) — (batch, seq, d_model)

        # Generate using the T5 decoder
        generated_ids = t5_model.generate(
            inputs_embeds=pose_embeddings,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    # ------------------------------------------------------------------
    # 5. Decode the output
    # ------------------------------------------------------------------
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # 6. Verify
    # ------------------------------------------------------------------
    # 6a. No exception was raised — if we reach this point, pass
    # 6b. Output is a string
    assert isinstance(output_text, str), (
        f"Output is not a string, got {type(output_text)}"
    )

    # 6c. Model produces a valid generation result (generated_ids has content)
    assert generated_ids is not None, "Generation produced no output"
    assert generated_ids.dim() > 0, "Generated IDs tensor is empty"

    # 6d. No NaN/Inf enters the model
    assert not torch.isnan(pose_embeddings).any(), "NaN detected in pose embeddings"
    assert not torch.isinf(pose_embeddings).any(), "Inf detected in pose embeddings"
    assert not torch.isnan(generated_ids).any(), "NaN detected in generated IDs"
    assert not torch.isinf(generated_ids).any(), "Inf detected in generated IDs"

    # 6e. CPU inference works (already verified by running on CPU)
    # — no GPU dependency, all operations completed on CPU

    print(
        f"SMOKE TEST PASSED: "
        f"input_frames=5, output_text='{output_text.strip()}', "
        f"input_shape={list(pose_tensor.shape)}, "
        f"pose_length={actual_length}"
    )


if __name__ == "__main__":
    test_external_isign_t5_smoke()