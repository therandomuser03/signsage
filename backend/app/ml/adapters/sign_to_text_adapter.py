"""Sign->Text adapter: the one class services/ should import.

Ties the deployment contract, the SignTransformer checkpoint, and the
SentencePiece tokenizer together behind a single transcribe() call. This is
this project's equivalent of the notebook's own exported
signsage_inference/loader.py -- same validation, same computation, laid out
inside ml/{contracts,models,inference,postprocessing} instead of one flat
package.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from app.ml.contracts.inference_contract import DeploymentContract
from app.ml.inference.sign_to_text import greedy_decode_ids
from app.ml.models.sign_to_text import SignTransformer
from app.ml.postprocessing.confidence import compute_confidence, passes_acceptance_gate
from app.ml.postprocessing.decoding import decode_token_ids
from app.ml.preprocessing.sequence import compute_motion_energy, prepare_live_landmarks


class ModelLoadError(RuntimeError):
    """The checkpoint/tokenizer could not be loaded, or does not match the
    deployment contract it shipped alongside."""


@dataclass
class TranscriptionResult:
    text: str
    accepted: bool
    reason: str
    motion_energy: float
    confidence: float


class SignToTextAdapter:
    def __init__(self, model: SignTransformer, tokenizer, contract: DeploymentContract, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.contract = contract
        self.device = device

    def transcribe(self, landmarks_xyz, min_motion: float = 0.002, min_confidence: float = 0.0) -> TranscriptionResult:
        """landmarks_xyz: raw (T, 543, 3) array from MediaPipe Holistic, in the
        exact component order documented in landmark_contract.json. Do not
        normalize or compute deltas yourself -- this method runs the entire
        preprocessing + model + decode pipeline.
        """
        processed, true_len = prepare_live_landmarks(landmarks_xyz, max_frames=self.contract.model.max_frames)
        motion = compute_motion_energy(landmarks_xyz)

        ids, logps = greedy_decode_ids(
            self.model,
            torch.from_numpy(processed).float(),
            true_len,
            self.contract.model.special_token_ids,
            self.contract.model.decoding,
            self.device,
        )
        text = decode_token_ids(ids[1:], self.tokenizer, self.contract.model.special_token_ids)
        confidence = compute_confidence(logps)

        accepted, reason = passes_acceptance_gate(
            text=text, motion_energy=motion, confidence=confidence,
            min_motion=min_motion, min_confidence=min_confidence,
        )
        return TranscriptionResult(
            text=text if accepted else "",
            accepted=accepted,
            reason=reason,
            motion_energy=motion,
            confidence=confidence,
        )

    @classmethod
    def load(cls, bundle_dir, device: Optional[str] = None) -> "SignToTextAdapter":
        import sentencepiece as spm  # local import: only needed when actually loading a model

        bundle_dir = Path(bundle_dir)
        contract = DeploymentContract.from_bundle(bundle_dir)
        torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = bundle_dir / contract.model.checkpoint_path
        if not checkpoint_path.is_file():
            raise ModelLoadError(f"Checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=torch_device)
        if "state_dict" not in payload or "model_config" not in payload:
            raise ModelLoadError("Checkpoint is missing 'state_dict'/'model_config'.")
        mc = payload["model_config"]

        if int(mc["input_dim"]) != contract.model.model_input_dim:
            raise ModelLoadError(
                f"Checkpoint input_dim={mc['input_dim']} != manifest model_input_dim="
                f"{contract.model.model_input_dim}"
            )
        if int(mc["raw_feature_dim"]) != contract.model.feature_dim:
            raise ModelLoadError(
                f"Checkpoint raw_feature_dim={mc['raw_feature_dim']} != manifest feature_dim="
                f"{contract.model.feature_dim}"
            )

        vocab = payload.get("vocab")
        vocab_size = int(mc["vocab_size"]) if vocab is None else len(vocab)

        model = SignTransformer(
            input_dim=int(mc["input_dim"]),
            vocab_size=vocab_size,
            max_frames=int(mc.get("max_frames", contract.model.max_frames)),
            max_tokens=int(mc.get("max_tokens", contract.model.max_tokens)),
            pad_idx=contract.model.special_token_ids["pad"],
            d_model=int(mc.get("d_model", 192)),
            nhead=int(mc.get("nhead", 4)),
            layers=int(mc.get("layers", 3)),
            ff=int(mc.get("ff", 512)),
            dropout=0.0,  # inference: disable dropout regardless of the training value
            # Pre-352868924 bundles predate this key and are post-norm (False).
            norm_first=bool(mc.get("norm_first", False)),
        ).to(torch_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        tokenizer_path = bundle_dir / contract.model.tokenizer_model_path
        if not tokenizer_path.is_file():
            raise ModelLoadError(f"Tokenizer not found: {tokenizer_path}")
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

        expected = contract.model.special_token_ids
        actual = {"pad": sp.pad_id(), "unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id()}
        if actual != expected:
            raise ModelLoadError(
                f"Tokenizer special-token IDs {actual} do not match manifest {expected}."
            )

        return cls(model=model, tokenizer=sp, contract=contract, device=torch_device)
