"""Unit tests for ModelContract validation (app.ml.contracts.model_contract)."""
import json

import pytest

from app.ml.contracts.model_contract import ModelContract, ModelContractError
from app.ml.preprocessing.landmarks import MODEL_INPUT_DIM, NORMALIZATION_VERSION, RAW_FEATURE_DIM

VALID_MANIFEST = {
    "feature_dim": RAW_FEATURE_DIM,
    "model_input_dim": MODEL_INPUT_DIM,
    "max_frames": 160,
    "max_tokens": 96,
    "vocab_size": 16000,
    "special_token_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
    "normalization_version": NORMALIZATION_VERSION,
    "tokenizer_model": "tokenizer/signsage_en_sp.model",
    "checkpoint": "checkpoint/signsage_transformer_full.pt",
    "decoding": {
        "max_decode_tokens": 94,
        "min_output_tokens": 1,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.15,
    },
}


def _write_manifest(tmp_path, overrides=None):
    manifest = {**VALID_MANIFEST, **(overrides or {})}
    path = tmp_path / "sign_to_text_deployment.json"
    path.write_text(json.dumps(manifest))
    return path


def test_loads_valid_manifest(tmp_path):
    path = _write_manifest(tmp_path)
    contract = ModelContract.load(path)
    assert contract.max_frames == 160
    assert contract.special_token_ids == {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
    assert contract.decoding.max_decode_tokens == 94


def test_rejects_wrong_feature_dim(tmp_path):
    path = _write_manifest(tmp_path, {"feature_dim": 999})
    with pytest.raises(ModelContractError):
        ModelContract.load(path)


def test_rejects_wrong_model_input_dim(tmp_path):
    path = _write_manifest(tmp_path, {"model_input_dim": 1629})  # forgot the x2 for deltas
    with pytest.raises(ModelContractError):
        ModelContract.load(path)


def test_rejects_duplicate_special_token_ids(tmp_path):
    path = _write_manifest(
        tmp_path, {"special_token_ids": {"pad": 0, "unk": 0, "bos": 2, "eos": 3}}
    )
    with pytest.raises(ModelContractError):
        ModelContract.load(path)


def test_rejects_wrong_normalization_version(tmp_path):
    path = _write_manifest(tmp_path, {"normalization_version": "some_other_version"})
    with pytest.raises(ModelContractError):
        ModelContract.load(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ModelContractError):
        ModelContract.load(tmp_path / "does_not_exist.json")
