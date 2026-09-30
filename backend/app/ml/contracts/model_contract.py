"""Loader/validator for sign_to_text_deployment.json (the deployment
manifest written by the training notebook, Section 18).

This is the single source of truth for special-token IDs, decoding
hyperparameters, and where the checkpoint/tokenizer live inside the bundle.
Cross-validated against landmark_contract.py at DeploymentContract assembly
time (see ml/contracts/inference_contract.py) so a mismatched pair of files
is caught at load time, never silently served.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.ml.preprocessing.landmarks import MODEL_INPUT_DIM, NORMALIZATION_VERSION, RAW_FEATURE_DIM


class ModelContractError(ValueError):
    """The bundle's deployment manifest is missing, malformed, or
    inconsistent with this package's expectations."""


@dataclass
class DecodingConfig:
    max_decode_tokens: int
    min_output_tokens: int
    no_repeat_ngram_size: int
    repetition_penalty: float


@dataclass
class ModelContract:
    feature_dim: int
    model_input_dim: int
    max_frames: int
    max_tokens: int
    vocab_size: int
    special_token_ids: dict
    normalization_version: str
    tokenizer_model_path: str
    checkpoint_path: str
    decoding: DecodingConfig
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "ModelContract":
        if not path.is_file():
            raise ModelContractError(f"sign_to_text_deployment.json not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        errors: list = []
        if int(manifest.get("feature_dim", -1)) != RAW_FEATURE_DIM:
            errors.append(f"feature_dim={manifest.get('feature_dim')} != expected {RAW_FEATURE_DIM}")
        if int(manifest.get("model_input_dim", -1)) != MODEL_INPUT_DIM:
            errors.append(f"model_input_dim={manifest.get('model_input_dim')} != expected {MODEL_INPUT_DIM}")
        if manifest.get("normalization_version") != NORMALIZATION_VERSION:
            errors.append(
                f"normalization_version={manifest.get('normalization_version')!r} != expected "
                f"{NORMALIZATION_VERSION!r}"
            )
        special = manifest.get("special_token_ids", {})
        try:
            ids = (int(special["pad"]), int(special["unk"]), int(special["bos"]), int(special["eos"]))
            if len(set(ids)) != 4:
                errors.append(f"special_token_ids are not four distinct values: {special}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"special_token_ids is malformed: {special}")
        if "decoding" not in manifest:
            errors.append("manifest is missing the 'decoding' section")
        if errors:
            raise ModelContractError(
                "sign_to_text_deployment.json failed validation:\n  - " + "\n  - ".join(errors)
            )

        decoding = manifest["decoding"]
        return cls(
            feature_dim=int(manifest["feature_dim"]),
            model_input_dim=int(manifest["model_input_dim"]),
            max_frames=int(manifest["max_frames"]),
            max_tokens=int(manifest["max_tokens"]),
            vocab_size=int(manifest["vocab_size"]),
            special_token_ids={k: int(v) for k, v in special.items()},
            normalization_version=str(manifest["normalization_version"]),
            tokenizer_model_path=str(manifest["tokenizer_model"]),
            checkpoint_path=str(manifest["checkpoint"]),
            decoding=DecodingConfig(
                max_decode_tokens=int(decoding["max_decode_tokens"]),
                min_output_tokens=int(decoding["min_output_tokens"]),
                no_repeat_ngram_size=int(decoding["no_repeat_ngram_size"]),
                repetition_penalty=float(decoding["repetition_penalty"]),
            ),
            raw=manifest,
        )
