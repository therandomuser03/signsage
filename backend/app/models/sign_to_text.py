from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the notebook."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SignTransformer(nn.Module):
    """Serving architecture matching the SignSage research notebook."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        pad_idx: int,
        *,
        max_frames: int = 160,
        max_tokens: int = 40,
        d_model: int = 192,
        nhead: int = 4,
        layers: int = 2,
        ff: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.src_pos = PositionalEncoding(d_model, max_frames + 4)
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_pos = PositionalEncoding(d_model, max_tokens + 4)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.decoder = nn.TransformerDecoder(dec_layer, layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.scale = d_model ** 0.5
        self.pad_idx = pad_idx

    def encode(self, frames: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        x = self.src_pos(self.input_proj(frames))
        return self.encoder(x, src_key_padding_mask=src_pad)

    def decode(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_pad: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_pos(self.tok_embed(tokens) * self.scale)
        y = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad,
        )
        return self.out(y)


class TextToPoseGRU(nn.Module):
    """Serving architecture matching the notebook Text-to-Pose GRU."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        pad_idx: int,
        *,
        hidden: int = 192,
        output_frames: int = 60,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=pad_idx)
        self.context = nn.Linear(hidden, hidden)
        self.time = nn.Embedding(output_frames, hidden)
        self.cell = nn.GRUCell(feature_dim + hidden, hidden)
        self.out = nn.Linear(hidden, feature_dim)
        self.output_frames = output_frames
        self.feature_dim = feature_dim

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != self.embed.padding_idx).float().unsqueeze(-1)
        emb = self.embed(tokens) * mask
        return torch.tanh(
            self.context(emb.sum(1) / mask.sum(1).clamp(min=1.0))
        )

    def forward(
        self,
        tokens: torch.Tensor,
        teacher: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode_text(tokens)
        batch = tokens.size(0)
        prev = torch.zeros(batch, self.feature_dim, device=tokens.device)
        outputs = []

        for t in range(self.output_frames):
            ctx = self.time(torch.tensor(t, device=tokens.device)).expand(batch, -1)
            hidden = self.cell(torch.cat([prev, ctx], dim=-1), hidden)
            pred = self.out(hidden)
            outputs.append(pred)
            prev = teacher[:, t] if teacher is not None else pred

        return torch.stack(outputs, dim=1)
