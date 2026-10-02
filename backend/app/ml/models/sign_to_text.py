"""SignSage Sign->Text model definition.

Verbatim port of the training notebook's FrameProjection/PositionalEncoding/
SignTransformer classes (signsage.ipynb, Section 8). max_frames/max_tokens/
pad_idx are explicit constructor arguments (not implicit notebook globals),
so this class has no dependency on training-notebook state and can be
reconstructed from a checkpoint's own model_config in a fresh process. The
forward/encode/decode computation itself is byte-for-byte unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FrameProjection(nn.Module):
    """Project spatial landmarks plus explicit frame-to-frame motion features."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(384, d_model),
            nn.LayerNorm(d_model),
        )
        self.temporal = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)

    def forward(self, x):
        # Explicit velocity is zero for the first frame. This makes the temporal
        # signal available before attention rather than asking the encoder to
        # infer all derivatives from absolute coordinates alone.
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        y = self.net(torch.cat([x, delta], dim=-1))
        y = self.temporal(y.transpose(1, 2)).transpose(1, 2)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SignTransformer(nn.Module):
    """Spatial-temporal Transformer for Sign->Text.

    Same architecture as training. max_frames/max_tokens/pad_idx are
    explicit constructor arguments (previously the notebook globals
    MAX_FRAMES, MAX_TOKENS, PAD_IDX) so this class has no external state
    dependency.
    """

    def __init__(
        self,
        input_dim,
        vocab_size,
        max_frames,
        max_tokens,
        pad_idx,
        d_model=192,
        nhead=4,
        layers=3,
        ff=512,
        dropout=0.15,
        norm_first=False,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.norm_first = norm_first
        self.input_proj = FrameProjection(input_dim, d_model)
        self.src_pos = PositionalEncoding(d_model, max_frames + 4)
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_pos = PositionalEncoding(d_model, max_tokens + 4)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=norm_first)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.scale = d_model ** 0.5

    def encode(self, frames, src_pad):
        return self.encoder(self.src_pos(self.input_proj(frames)), src_key_padding_mask=src_pad)

    def decode(self, tokens, memory, tgt_mask, src_pad, tgt_pad=None):
        x = self.tgt_pos(self.tok_embed(tokens) * self.scale)
        return self.out(
            self.decoder(
                x, memory, tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad,
            )
        )

    def forward(self, frames, src_pad, tokens, tgt_mask):
        return self.decode(
            tokens, self.encode(frames, src_pad), tgt_mask, src_pad,
            tgt_pad=tokens.eq(self.pad_idx),
        )


def padding_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def causal_mask(n, device):
    return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
