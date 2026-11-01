"""ISLiM continuous translation adapter — verified mobile/export architecture.

Uses KeypointToVideoEncoder (full-sequence, no query-token slicing) to match
both downloaded ISLiM checkpoints (cf97d0ac… / ffe4efbb…).

Contract from ISLIM_ARCHITECTURE_COMPATIBILITY_REPORT.md:
- Input: (B, 4, 94, T) float32 or (B, T, 94, 4) — visibility in 4th channel.
- Normalization / padding to 128 frames handled by encoder (sampling if T>128,
  zero-padding if T<128).
- Encoder output: (B, 128, 512) full temporal sequence → T5 inputs_embeds.
- T5 decoder: t5-small from HF with official generate params.
"""
from __future__ import annotations

import sys, os, warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "mobile"))
from encoder import KeypointToVideoEncoder  # type: ignore


class ISLiMAdapterError(RuntimeError):
    pass


class ISLiMAdapter:
    CHECKPOINT_DIR = Path(__file__).resolve().parents[4] / "research" / "pretrained_models" / "islim" / "checkpoints"
    DEFAULT_CKPT = CHECKPOINT_DIR / "ISLiM_4Loss_best_for_10kSpamoOF.pt"
    ALT_CKPT = CHECKPOINT_DIR / "ISLiM_full_posestitch_pre_processed.pt"

    ENCODER_CFG = {
        "keypoint_dim": 4, "num_keypoints": 94,
        "temporal_channels": [376, 256, 256], "temporal_kernel": 5,
        "transformer_embed_dim": 512, "transformer_num_heads": 8,
        "transformer_num_layers": 4, "transformer_dropout": 0.2,
        "output_dim": 512, "max_frames": 128,
    }

    def __init__(self, checkpoint_path: Optional[str | Path] = None,
                 device: str = "cpu", tokenizer_name: str = "t5-small"):
        self.device = torch.device(device if (torch.cuda.is_available() or device == "cpu") or device == "cuda" else "cpu")
        # If GPU unavailable but cuda requested, fall back safely
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; using CPU.")
            self.device = torch.device("cpu")
        visual_cfg = self.ENCODER_CFG.copy()
        visual_cfg["output_dim"] = 512
        self.visual_encoder = KeypointToVideoEncoder(visual_cfg)
        self.visual_encoder.to(self.device)
        self.visual_encoder.eval()

        t5_cfg = T5Config.from_pretrained(tokenizer_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(tokenizer_name, config=t5_cfg)
        self.t5.to(self.device)
        self.t5.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = 128
        self.checkpoint_path = Path(checkpoint_path or self.DEFAULT_CKPT)
        if self.checkpoint_path.exists():
            self.load_checkpoint(self.checkpoint_path)
        else:
            warnings.warn(f"ISLiM checkpoint not found at {self.checkpoint_path}; "
                          "running with uninitialized weights (NOT real translation).")

    def load_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        # Load encoder weights into visual_encoder and t5 weights into t5 separately
        encoder_state = {k: v for k, v in state.items() if k.startswith("visual_encoder.")}
        t5_state = {k: v for k, v in state.items() if k.startswith("t5.")}
        if encoder_state:
            self.visual_encoder.load_state_dict(encoder_state, strict=False)
        if t5_state:
            self.t5.load_state_dict(t5_state, strict=False)
        # Handle any remaining (e.g. logit_scale; not used in generate)
        return payload

    @torch.no_grad()
    def translate(self, keypoints, max_length: Optional[int] = None,
                  num_beams: int = 4, repetition_penalty: float = 1.5,
                  no_repeat_ngram_size: int = 3, do_sample: bool = False,
                  length_penalty: float = 1.2, early_stopping: bool = True) -> List[str]:
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).to(dtype=torch.float32, device=self.device)
        else:
            keypoints = keypoints.to(dtype=torch.float32, device=self.device)

        if keypoints.dim() != 4:
            raise ISLiMAdapterError(f"Expected 4-D input; got {keypoints.dim()}-D shape {tuple(keypoints.shape)}")
        # Accept (B,4,94,T) or (B,T,94,4); reject anything else
        s = tuple(keypoints.shape)
        valid = ((s[1] == 4 and s[2] == 94) or (s[2] == 94 and s[3] == 4))
        if not valid:
            raise ISLiMAdapterError(
                f"Invalid keypoint shape {s}; expected (B,4,94,T) or (B,T,94,4)"
            )

        encoder_output = self.visual_encoder.get_sequence_output(keypoints)
        ml = max_length or self.max_length
        outputs = self.t5.generate(
            inputs_embeds=encoder_output,
            max_length=ml,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )
        return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]

    @torch.no_grad()
    def evaluate_sample(self, keypoints, reference_text: Optional[str] = None) -> dict:
        try:
            predictions = self.translate(keypoints)
        except Exception as exc:
            return {"status": "error", "message": str(exc),
                    "predictions": [], "reference": reference_text}
        unique = len(set(predictions))
        collapsed = unique == 1 and len(predictions) > 1
        return {
            "status": "ok",
            "predictions": predictions,
            "reference": reference_text,
            "prediction_diversity": unique,
            "collapsed": collapsed,
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
        }
