"""Tests for the static image classification adapter and API endpoint.

Uses a tiny synthetic bundle so tests run without the full dataset.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.adapters.image_classifier_adapter import (
    ImageClassifierAdapter,
    ImageModelLoadError,
    classify_base64,
)
from app.ml.models.image_cnn import SmallImageCNN


def _make_tiny_bundle(tmp_path: Path, n_classes: int = 4) -> Path:
    """Build a minimal bundle dir with model.pt + labels.json."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model = SmallImageCNN(n_classes=n_classes, img_size=16, n_channels=3)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "n_classes": n_classes,
                "img_size": 16,
                "n_channels": 3,
                "architecture": "SmallCNN",
                "classes": [f"class_{i}" for i in range(n_classes)],
            },
        },
        bundle_dir / "model.pt",
    )
    labels = {
        "label_to_index": {f"class_{i}": i for i in range(n_classes)},
        "index_to_label": {str(i): f"class_{i}" for i in range(n_classes)},
        "classes": [f"class_{i}" for i in range(n_classes)],
        "n_classes": n_classes,
    }
    (bundle_dir / "labels.json").write_text(json.dumps(labels))
    return bundle_dir


def _make_image_bytes(size: int = 16) -> bytes:
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    import PIL.Image as Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_adapter_loads_and_predicts(tmp_path):
    bundle_dir = _make_tiny_bundle(tmp_path, n_classes=4)
    adapter = ImageClassifierAdapter.load(
        bundle_dir, task="alphabet", model_version="test-v1", device="cpu"
    )
    assert adapter.task == "alphabet"
    assert adapter.n_classes == 4
    assert adapter.img_size == 16

    result = adapter.classify(_make_image_bytes(), min_confidence=0.0)
    assert result.task == "alphabet"
    assert result.n_classes == 4
    assert result.label in {f"class_{i}" for i in range(4)}
    assert 0.0 <= result.confidence <= 1.0
    assert result.input_size == (16, 16)


def test_adapter_rejects_low_confidence(tmp_path):
    bundle_dir = _make_tiny_bundle(tmp_path, n_classes=4)
    adapter = ImageClassifierAdapter.load(
        bundle_dir, task="alphabet", model_version="test-v1", device="cpu",
        confidence_threshold=0.99,
    )
    result = adapter.classify(_make_image_bytes(), min_confidence=0.99)
    # With random weights, confidence < 0.99 almost always → not accepted
    assert result.accepted is False
    assert result.label == ""
    assert result.reason == "low_model_confidence"


def test_adapter_rejects_missing_bundle(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ImageModelLoadError):
        ImageClassifierAdapter.load(
            empty_dir, task="alphabet", model_version="test-v1", device="cpu"
        )


def test_classify_base64_data_uri(tmp_path):
    bundle_dir = _make_tiny_bundle(tmp_path, n_classes=4)
    adapter = ImageClassifierAdapter.load(
        bundle_dir, task="word", model_version="test-v1", device="cpu"
    )
    img_bytes = _make_image_bytes()
    import base64
    b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"
    result = classify_base64(adapter, data_uri, min_confidence=0.0)
    assert result.task == "word"
    assert result.label in {f"class_{i}" for i in range(4)}


def test_endpoint_returns_model_unavailable_when_no_bundle(monkeypatch):
    """With no SIGNSAGE_IMAGE_*_BUNDLE_DIR set, the endpoint returns 503."""
    monkeypatch.delenv("SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR", raising=False)
    monkeypatch.delenv("SIGNSAGE_IMAGE_WORD_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("SIGNSAGE_IMAGE_ALPHABET_BUNDLE_DIR", "")
    monkeypatch.setenv("SIGNSAGE_IMAGE_WORD_BUNDLE_DIR", "")
    import importlib
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.api.routes.image_classification as img_mod
    importlib.reload(img_mod)
    from app.ml.adapters.image_classifier_adapter import ImageClassifierAdapter
    # Clear adapter cache
    img_mod._adapters.clear()
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r = client.post(
        "/api/v1/classification/image",
        json={"image": "dGVzdA==", "task": "alphabet"},
    )
    assert r.status_code == 503
    assert r.json()["code"] == "MODEL_UNAVAILABLE"
