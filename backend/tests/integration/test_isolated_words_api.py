from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from app.ml.isolated_words import FEATURE_DIM, FEATURE_VERSION, IsolatedWordNet


def _bundle(path):
    path.mkdir()
    labels = ["hello", "school"]
    checkpoint = path / "model.pt"
    torch.save({"state_dict": IsolatedWordNet(len(labels)).state_dict()}, checkpoint)
    evaluation = {
        "status": "NO-GO",
        "signer_independent": False,
        "tests_passed": False,
        "promotion_eligible": False,
    }
    (path / "labels.json").write_text(json.dumps(labels))
    (path / "evaluation_results.json").write_text(json.dumps(evaluation))
    (path / "metadata.json").write_text(json.dumps({
        "format": "signsage_isolated_words_v1",
        "labels": labels,
        "feature_dim": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation": evaluation,
        "promotion_eligible": False,
    }))
    return path


def _payload():
    rng = np.random.default_rng(11)
    return {"frames": [{"landmarks": frame.tolist()} for frame in rng.normal(size=(4, 543, 3))],
            "save_history": False}


def test_isolated_bundle_preserves_rest_response_and_dev_gate(make_client, tmp_path):
    bundle = _bundle(tmp_path / "isolated")
    client = make_client(
        bundle=bundle,
        model_kind="isolated_words",
        fallback_enabled="false",
        allow_unpromoted_dev="true",
    )
    response = client.post("/api/v1/translation/sign-to-text", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "text", "accepted", "reason", "confidence", "motion_energy", "result_type", "history_id"
    }
    assert body["result_type"] in {"model", "failed"}
    assert isinstance(body["confidence"], float)
    assert client.get("/api/v1/capabilities").json()["sign_to_text_source"] == "isolated-word"


def test_isolated_bundle_missing_checkpoint_fails_cleanly(make_client, tmp_path):
    bundle = _bundle(tmp_path / "isolated")
    (bundle / "model.pt").unlink()
    client = make_client(
        bundle=bundle,
        model_kind="isolated_words",
        fallback_enabled="false",
        allow_unpromoted_dev="true",
    )
    response = client.post("/api/v1/translation/sign-to-text", json=_payload())
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


def test_isolated_bundle_cannot_be_promoted_with_no_go_evaluation(make_client, tmp_path):
    bundle = _bundle(tmp_path / "isolated")
    client = make_client(
        bundle=bundle,
        promoted=True,
        model_kind="isolated_words",
        fallback_enabled="false",
    )
    response = client.post("/api/v1/translation/sign-to-text", json=_payload())
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_NOT_PROMOTED"
