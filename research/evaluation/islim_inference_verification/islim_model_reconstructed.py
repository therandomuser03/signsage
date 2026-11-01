#!/usr/bin/env python3
"""
Reconstructed ISLiM model from official repository code.
Not integrated into production adapter.
Used only for isolated inference verification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the local encoder architecture (matches checkpoint keys)
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/D/Codes/signsage/research/pretrained_models/islim/mobile")
from encoder import KeypointToVideoEncoder

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer


class MinimalISLiMModel(nn.Module):
    """Reconstructed from official evaluate_all.py load_model + export_torchscript.py get_model."""

    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()
        config = {
            "encoder": {
                "keypoint_dim": 4, "num_keypoints": 94,
                "temporal_channels": [376, 256, 256], "temporal_kernel": 5,
                "transformer_embed_dim": 512, "transformer_num_heads": 8,
                "transformer_num_layers": 4, "transformer_dropout": 0.2,
                "output_dim": 512, "max_frames": 128,
                "num_query_tokens": 16,  # official config; checkpoint may omit
            },
            "t5": {
                "t5_model": "t5-small", "d_model": 512,
                "max_target_len": 128, "label_smoothing": 0.0,
            },
        }
        visual_cfg = config["encoder"].copy()
        visual_cfg["output_dim"] = config["t5"]["d_model"]
        self.visual_encoder = KeypointToVideoEncoder(visual_cfg)
        # Note: official model.py creates ISLiMEncoder; but checkpoint keys
        # match KeypointToVideoEncoder (visual_encoder.encoder.*). Using
        # KeypointToVideoEncoder is the minimal reconstruction supported
        # by the downloaded checkpoint's state_dict keys.
        t5_cfg = T5Config.from_pretrained(config["t5"]["t5_model"])
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            config["t5"]["t5_model"], config=t5_cfg
        )
        self.tokenizer = T5Tokenizer.from_pretrained(config["t5"]["t5_model"])
        self.max_length = config["t5"].get("max_target_len", 128)
        self.device = torch.device(device)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        # strict=False per official evaluate_all.py
        self.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint (epoch={payload.get('epoch', '?')}) "
              f"state_dict entries={len(state)} "
              f"best_bleu4={payload.get('best_bleu4', '?')}")
        return payload

    @torch.no_grad()
    def generate(self, keypoints, max_length=None, num_beams=4,
                 repetition_penalty=1.5, no_repeat_ngram_size=3, do_sample=False):
        # keypoints expected format: (B, T, 94, 4) or (B, 4, 94, T)
        encoder_output = self.visual_encoder.get_sequence_output(keypoints)
        # get_sequence_output returns (B, T, 512) per KeypointToVideoEncoder
        # The official generate uses inputs_embeds=encoder_output directly.
        # KeypointToVideoEncoder.get_sequence_output does NOT include the
        # T5 encoder step; the official ISLiMEncoder may include it.
        # For this minimal reconstruction, we pass encoder_output directly.
        ml = max_length or self.max_length
        outputs = self.t5.generate(
            inputs_embeds=encoder_output,
            max_length=ml,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            length_penalty=1.2,
            early_stopping=True,
        )
        return [t.lower() for t in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]


if __name__ == "__main__":
    # Minimal smoke test: synthetic input only
    import numpy as np
    model = MinimalISLiMModel(
        "/mnt/D/Codes/signsage/research/pretrained_models/islim/checkpoints/ISLiM_4Loss_best_for_10kSpamoOF.pt",
        device="cpu"
    )
    # Synthetic only — NOT real ISL data
    B, T = 1, 128
    synthetic_kp = torch.randn(B, 4, 94, T, dtype=torch.float32)
    synthetic_kp[:, :3, :, :] = torch.sigmoid(synthetic_kp[:, :3, :, :])
    synthetic_kp[:, 3, :, :] = torch.sigmoid(synthetic_kp[:, 3, :, :])
    synthetic_kp = synthetic_kp.to(model.device)
    print("Synthetic input shape:", synthetic_kp.shape)
    try:
        out_text = model.generate(synthetic_kp)
        print("Synthetic smoke-test output (NOT a translation):", out_text)
    except Exception as exc:
        print("Smoke-test generate failed:", exc)
