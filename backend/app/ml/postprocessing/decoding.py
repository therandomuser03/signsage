"""Token-id post-processing: repeated-ngram ban + id-to-text decoding.

Verbatim port of the training notebook's _ban_repeated_ngram() and
decode_ids() (signsage.ipynb, Section 8 / Section 4).
"""
from __future__ import annotations

import torch


def ban_repeated_ngram(logits: torch.Tensor, ids: list, n: int) -> torch.Tensor:
    if n <= 1 or len(ids) < n - 1:
        return logits
    prefix = tuple(ids[-(n - 1):])
    for j in range(len(ids) - n + 1):
        ngram = tuple(ids[j : j + n])
        if ngram[:-1] == prefix:
            logits[ngram[-1]] = -torch.inf
    return logits


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def decode_token_ids(ids: list, sp, special_token_ids: dict) -> str:
    """ids should exclude the leading BOS (pass ids[1:])."""
    eos_id = special_token_ids["eos"]
    bos_id = special_token_ids["bos"]
    pad_id = special_token_ids["pad"]
    clean = []
    for idx in ids:
        idx = int(idx)
        if idx == eos_id:
            break
        if idx not in (bos_id, pad_id):
            clean.append(idx)
    if not clean:
        return ""
    return normalize_text(sp.decode(clean))
