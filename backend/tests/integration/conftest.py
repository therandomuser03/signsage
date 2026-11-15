"""Shared fixture: a structurally-real (but tiny, untrained) deployment
bundle -- same file layout, same field names, same special-token IDs as a
real signsage_deployment_bundle/, just small enough to build and load in
under a second. This is what lets tests/integration/ exercise the actual
checkpoint-loading + tokenizer-loading + contract-validation code paths
without needing a real multi-hundred-MB trained checkpoint.
"""
from __future__ import annotations

import json

import pytest
import torch

from app.ml.models.sign_to_text import SignTransformer

VOCAB_TRAINING_CORPUS = (
    "hello world how are you today\n"
    "the cat sat on the mat quietly\n"
    "sign language is a visual way to communicate\n"
    "thank you very much for your help\n"
) * 20


@pytest.fixture(scope="session")
def _bundle_template(tmp_path_factory):
    """Built once per run (training the tokenizer is the slow part); each test gets its own copy."""
    tmp_path = tmp_path_factory.mktemp("bundle_template")
    sentencepiece = pytest.importorskip("sentencepiece")

    bundle = tmp_path / "signsage_deployment_bundle"
    (bundle / "checkpoint").mkdir(parents=True)
    (bundle / "tokenizer").mkdir(parents=True)

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(VOCAB_TRAINING_CORPUS)
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus_path), model_prefix=str(tmp_path / "sp"), vocab_size=40,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, model_type="unigram",
    )
    tokenizer_path = bundle / "tokenizer" / "signsage_en_sp.model"
    tokenizer_path.write_bytes((tmp_path / "sp.model").read_bytes())
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = sp.get_piece_size()

    model = SignTransformer(
        input_dim=3258, vocab_size=vocab_size, max_frames=160, max_tokens=96,
        pad_idx=0, d_model=16, nhead=2, layers=1, ff=32, dropout=0.0,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": 3258, "raw_feature_dim": 1629, "vocab_size": vocab_size,
                "d_model": 16, "nhead": 2, "layers": 1, "ff": 32, "dropout": 0.0,
                "max_frames": 160, "max_tokens": 96,
            },
        },
        bundle / "checkpoint" / "signsage_transformer_full.pt",
    )

    (bundle / "landmark_contract.json").write_text(json.dumps({
        "total_landmark_count": 543, "coordinates_per_landmark": 3, "raw_feature_dim": 1629,
        "component_order": ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        "component_ranges": {
            "POSE_LANDMARKS": [0, 33], "FACE_LANDMARKS": [33, 501],
            "LEFT_HAND_LANDMARKS": [501, 522], "RIGHT_HAND_LANDMARKS": [522, 543],
        },
        "normalization_version": "pose_normalize_v1", "max_frames": 160,
        "final_model_input_dim": 3258, "full_feature_mask": list(range(1629)),
    }))
    (bundle / "sign_to_text_deployment.json").write_text(json.dumps({
        "feature_dim": 1629, "model_input_dim": 3258, "max_frames": 160, "max_tokens": 96,
        "vocab_size": vocab_size, "special_token_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "normalization_version": "pose_normalize_v1",
        "tokenizer_model": "tokenizer/signsage_en_sp.model",
        "checkpoint": "checkpoint/signsage_transformer_full.pt",
        "decoding": {
            "max_decode_tokens": 8, "min_output_tokens": 1,
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.15,
        },
    }))
    return bundle


@pytest.fixture
def fake_bundle(_bundle_template, tmp_path):
    """A private, mutable copy: tests may tamper with or delete files freely."""
    import shutil

    dest = tmp_path / "signsage_deployment_bundle"
    shutil.copytree(_bundle_template, dest)
    return dest


@pytest.fixture
def make_client(monkeypatch):
    """Factory for a TestClient with chosen env. Auth is faked at the Supabase
    boundary: token 'good-<uid>' verifies as user <uid>; anything else is rejected."""
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.core.errors import AuthError
    from app.core.security import AuthenticatedUser, clear_auth_cache
    from app.infrastructure.database.repositories.translation_repository import reset_memory_repository
    from app.ml.models.registry import reset_registry

    async def fake_verify(token, settings, transport=None):
        if token.startswith("good-"):
            return AuthenticatedUser(id=token[5:], email=f"{token[5:]}@example.com", access_token=token)
        raise AuthError("Invalid or expired access token.")

    def _reset():
        get_settings.cache_clear()
        reset_registry()
        reset_memory_repository()
        clear_auth_cache()

    def _make(bundle=None, promoted=False, **env):
        monkeypatch.setenv("SIGNSAGE_SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.setenv("SIGNSAGE_SUPABASE_PUBLISHABLE_KEY", "pub")
        monkeypatch.setenv("SIGNSAGE_HISTORY_BACKEND", "memory")
        # Hermetic: never hit the real Groq API from integration tests.
        monkeypatch.setenv("SIGNSAGE_GROQ_API_KEY", "")
        monkeypatch.setenv("SIGNSAGE_MODEL_PROMOTED", "true" if promoted else "false")
        if bundle is not None:
            monkeypatch.setenv("SIGNSAGE_MODEL_BUNDLE_DIR", str(bundle))
        for k, v in env.items():
            monkeypatch.setenv(f"SIGNSAGE_{k.upper()}", str(v))
        _reset()
        monkeypatch.setattr("app.services.authentication_service.verify_access_token", fake_verify)
        monkeypatch.setattr("app.api.websocket.translation.verify_access_token", fake_verify)
        from app.main import app
        return TestClient(app)

    yield _make
    _reset()
