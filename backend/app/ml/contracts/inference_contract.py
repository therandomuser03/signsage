"""Ties landmark_contract.json and sign_to_text_deployment.json together
into one validated DeploymentContract, cross-checking the two files agree
with each other (not just each being individually valid) -- mirrors the
training notebook's exported SignSageConfig.from_bundle() cross-validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .landmark_contract import LandmarkContract
from .model_contract import ModelContract


class InferenceContractError(ValueError):
    """landmark_contract.json and sign_to_text_deployment.json disagree with
    each other, even though each is individually valid."""


@dataclass
class DeploymentContract:
    landmark: LandmarkContract
    model: ModelContract
    bundle_dir: Path

    @classmethod
    def from_bundle(cls, bundle_dir) -> "DeploymentContract":
        bundle_dir = Path(bundle_dir)
        landmark = LandmarkContract.load(bundle_dir / "landmark_contract.json")
        model = ModelContract.load(bundle_dir / "sign_to_text_deployment.json")

        errors: list = []
        if landmark.raw_feature_dim != model.feature_dim:
            errors.append(
                f"landmark_contract raw_feature_dim={landmark.raw_feature_dim} != "
                f"deployment manifest feature_dim={model.feature_dim}"
            )
        if landmark.final_model_input_dim != model.model_input_dim:
            errors.append(
                f"landmark_contract final_model_input_dim={landmark.final_model_input_dim} != "
                f"deployment manifest model_input_dim={model.model_input_dim}"
            )
        if landmark.normalization_version != model.normalization_version:
            errors.append(
                f"landmark_contract normalization_version={landmark.normalization_version!r} != "
                f"deployment manifest normalization_version={model.normalization_version!r}"
            )
        if landmark.max_frames != model.max_frames:
            errors.append(
                f"landmark_contract max_frames={landmark.max_frames} != "
                f"deployment manifest max_frames={model.max_frames}"
            )
        if errors:
            raise InferenceContractError(
                "landmark_contract.json and sign_to_text_deployment.json disagree:\n  - "
                + "\n  - ".join(errors)
            )

        return cls(landmark=landmark, model=model, bundle_dir=bundle_dir)
