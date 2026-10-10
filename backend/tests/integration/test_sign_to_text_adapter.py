"""Integration tests for SignToTextAdapter using a real (but tiny) bundle.

Unlike tests/unit/ml/, these exercise the ACTUAL torch model, the ACTUAL
sentencepiece tokenizer, and the full contract-validation chain -- not just
the pure-python preprocessing math. Requires torch + sentencepiece, which
is why they live under tests/integration/ rather than tests/unit/.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.ml.adapters.sign_to_text_adapter import ModelLoadError, SignToTextAdapter
from app.ml.contracts.model_contract import ModelContractError


def test_adapter_loads_from_a_real_bundle(fake_bundle):
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    assert pipeline.contract.model.max_frames == 160
    assert pipeline.contract.model.special_token_ids == {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


def test_static_landmarks_are_rejected_by_the_activity_gate(fake_bundle):
    """Mirrors the training notebook's own Section 15a self-test: an
    all-static sequence must never be accepted as a translation."""
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    static = np.ones((10, 543, 3), dtype=np.float32) * 0.5
    result = pipeline.transcribe(static)
    assert not result.accepted
    assert result.reason == "insufficient_motion"
    assert result.text == ""


def test_moving_landmarks_run_the_full_pipeline_and_are_accepted(fake_bundle):
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    rng = np.random.default_rng(0)
    moving = rng.normal(size=(40, 543, 3)).astype(np.float32)
    result = pipeline.transcribe(moving, min_confidence=0.0)
    assert result.accepted
    assert result.reason == "ok"
    assert result.motion_energy > 0


def test_wrong_landmark_shape_raises_value_error(fake_bundle):
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    wrong_shape = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        pipeline.transcribe(wrong_shape)


def test_empty_and_single_frame_sequences_raise_value_error(fake_bundle):
    """Empty / single-frame input can never be a translation: fail loudly."""
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    with pytest.raises(ValueError):
        pipeline.transcribe(np.zeros((0, 543, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.transcribe(np.zeros((1, 543, 3), dtype=np.float32))


def test_impossible_confidence_floor_yields_unsupported_result(fake_bundle):
    """No confidence can reach 1.0: the result must be unsupported with NO
    text -- never a plausible-looking sentence."""
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    rng = np.random.default_rng(0)
    moving = rng.normal(size=(40, 543, 3)).astype(np.float32)
    result = pipeline.transcribe(moving, min_confidence=1.0)
    assert result.accepted is False
    assert result.reason == "low_model_confidence"
    assert result.text == ""


def test_repeated_inference_is_deterministic_and_mutates_nothing(fake_bundle):
    """Same input twice must give the same verdict and leave the model in
    eval mode with gradients disabled -- inference holds no per-request state."""
    pipeline = SignToTextAdapter.load(fake_bundle, device="cpu")
    rng = np.random.default_rng(0)
    moving = rng.normal(size=(40, 543, 3)).astype(np.float32)
    before = {k: v.clone() for k, v in pipeline.model.state_dict().items()}
    first = pipeline.transcribe(moving)
    second = pipeline.transcribe(moving)
    assert (first.text, first.accepted, first.reason, first.confidence) == (
        second.text, second.accepted, second.reason, second.confidence)
    assert pipeline.model.training is False
    for k, v in pipeline.model.state_dict().items():
        assert torch.equal(v, before[k]), f"inference mutated parameter {k}"


def test_load_rejects_a_bundle_whose_manifest_was_tampered_with(fake_bundle):
    """A bundle that fails cross-validation must be rejected loudly at load
    time, never silently served -- this is the whole point of
    ml/contracts/."""
    manifest_path = fake_bundle / "sign_to_text_deployment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_dim"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ModelContractError):
        SignToTextAdapter.load(fake_bundle, device="cpu")


def test_load_rejects_a_missing_checkpoint_file(fake_bundle):
    (fake_bundle / "checkpoint" / "signsage_transformer_full.pt").unlink()
    with pytest.raises(ModelLoadError):
        SignToTextAdapter.load(fake_bundle, device="cpu")


def test_load_rejects_checkpoint_with_mismatched_tokenizer_fingerprint(fake_bundle):
    checkpoint = fake_bundle / "checkpoint" / "signsage_transformer_full.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["tokenizer_fingerprint"] = "mismatched"
    torch.save(payload, checkpoint)
    with pytest.raises(ModelLoadError, match="fingerprint"):
        SignToTextAdapter.load(fake_bundle, device="cpu")
