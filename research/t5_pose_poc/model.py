"""T5-small pose-to-text prototype model (RESEARCH ONLY, never serving).

Architecture:
    [B, T, 1629] -> PoseProjection -> [B, T, 512] -> T5 encoder (inputs_embeds)
    -> T5 decoder -> English tokens.

Pose input is the SAME normalized [T, 1629] the production backend uses
(`normalize_pose_sequence` + `resize_sequence` to max 160 frames). Padded
frames are excluded from encoder attention via `pose_mask`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

POSE_FEATURE_DIM = 1629  # 543 landmarks x 3, backend canonical
T5_D_MODEL = 512  # google-t5/t5-small d_model
T5_CHECKPOINT = "google-t5/t5-small"
MAX_FRAMES = 160  # same as backend/SignSage preprocessing convention
MAX_TOKENS = 96  # same as existing V2 training


class PoseProjection(nn.Module):
    """Lightweight pose front-end: Linear(1629->512) + LN + GELU + Linear(512->512)."""

    def __init__(self, in_dim: int = POSE_FEATURE_DIM, d_model: int = T5_D_MODEL,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        return self.net(poses)


class PoseT5(nn.Module):
    """PoseProjection + pretrained T5-small, wired through `inputs_embeds`.

    Pose values NEVER enter the T5 token embedding layer.
    """

    def __init__(self, checkpoint: str = T5_CHECKPOINT, dropout: float = 0.1):
        super().__init__()
        self.projection = PoseProjection(dropout=dropout)
        self.t5 = T5ForConditionalGeneration.from_pretrained(checkpoint)
        assert self.t5.config.d_model == T5_D_MODEL, self.t5.config.d_model

    def encode_poses(self, poses: torch.Tensor, pose_mask: torch.Tensor):
        """poses [B,T,1629] float; pose_mask [B,T] bool, True = real frame."""
        embeds = self.projection(poses)
        return self.t5.encoder(input_ids=None, inputs_embeds=embeds,
                               attention_mask=pose_mask.float())

    def forward(self, poses: torch.Tensor, pose_mask: torch.Tensor,
                labels: torch.Tensor):
        embeds = self.projection(poses)
        return self.t5(inputs_embeds=embeds, attention_mask=pose_mask.float(),
                       labels=labels)

    @torch.no_grad()
    def translate(self, poses: torch.Tensor, pose_mask: torch.Tensor,
                  tokenizer, max_new_tokens: int = 94, num_beams: int = 1) -> list[str]:
        self.eval()
        embeds = self.projection(poses)
        enc = self.t5.encoder(input_ids=None, inputs_embeds=embeds,
                              attention_mask=pose_mask.float())
        out = self.t5.generate(encoder_outputs=enc,
                               attention_mask=pose_mask.float(),
                               max_new_tokens=max_new_tokens, num_beams=num_beams)
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    def set_t5_frozen(self, frozen: bool) -> None:
        for p in self.t5.parameters():
            p.requires_grad = frozen is False

    def trainable_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
