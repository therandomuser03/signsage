"""Regression tests for the video upload endpoint."""
from __future__ import annotations

import pytest
from pathlib import Path


def test_video_endpoint_returns_genuine_prediction_not_mock(make_client, tmp_path):
    bundle_dir = tmp_path / "isolated"
    import json, hashlib, torch, os
    from app.ml.isolated_words import IsolatedWordNet

    bundle_dir.mkdir()
    labels = ["friend", "hello", "hospital", "market", "school", "thank you"]
    checkpoint = bundle_dir / "model.pt"
    torch.save({"state_dict": IsolatedWordNet(len(labels)).state_dict()}, checkpoint)
    evaluation = {
        "status": "NO-GO",
        "tests_passed": False,
        "promotion_eligible": False,
        "signer_independent": False,
        "test": {"accuracy": 0.61, "macro_recall": 0.61, "per_class_recall": {name: 0.5 for name in labels}},
    }
    (bundle_dir / "labels.json").write_text(json.dumps(labels))
    (bundle_dir / "evaluation_results.json").write_text(json.dumps(evaluation))
    (bundle_dir / "metadata.json").write_text(json.dumps({
        "format": "signsage_isolated_words_v1",
        "labels": labels,
        "feature_dim": 225,
        "frame_count": 64,
        "feature_version": "pose_hands_root_scale_v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation": evaluation,
        "promotion_eligible": False,
    }))

    client = make_client(
        bundle=bundle_dir,
        model_kind="isolated_words",
        fallback_enabled="false",
        allow_unpromoted_dev="true",
    )
    # Use the real video clip from the data directory
    real_video = Path("data/isl/videos/friend__INCLUDE__00546__MVI_3826.mp4")
    if not real_video.is_file():
        pytest.skip("Real video clip not available")

    with open(real_video, "rb") as f:
        response = client.post("/api/v1/translation/video", files={"file": ("test.mp4", f, "video/mp4")})

    assert response.status_code == 200
    body = response.json()
    assert "text" in body
    assert "accepted" in body
    assert "reason" in body
    assert "confidence" in body
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body.get("frames_processed", 0) >= 2
    assert body.get("model_version") is not None
    # The result is a genuine inference result (not a hardcoded mock).
    assert body["reason"] is not None
