"""Autoregressive greedy decoding loop for the Sign->Text Transformer.

Verbatim port of the training notebook's constrained_greedy_transformer()
(signsage.ipynb, Section 8), split so this module only runs the model and
returns token ids + per-step log-probabilities. Turning ids into text (and
the repeated-ngram ban, which operates on ids) lives in
ml/postprocessing/decoding.py; confidence scoring lives in
ml/postprocessing/confidence.py.
"""
from __future__ import annotations

import torch

from app.ml.contracts.model_contract import DecodingConfig
from app.ml.models.sign_to_text import SignTransformer, causal_mask, padding_mask
from app.ml.postprocessing.decoding import ban_repeated_ngram


@torch.no_grad()
def greedy_decode_ids(
    model: SignTransformer,
    frames: torch.Tensor,
    length: int,
    special_token_ids: dict,
    decoding: DecodingConfig,
    device: torch.device,
):
    """Real autoregressive decoding from BOS, no ground-truth tokens.

    Returns (token_ids_including_bos, per_step_log_probs).
    """
    model.eval()
    pad_id = special_token_ids["pad"]
    unk_id = special_token_ids["unk"]
    bos_id = special_token_ids["bos"]
    eos_id = special_token_ids["eos"]

    frames = frames.unsqueeze(0).to(device)
    length_t = torch.tensor([int(length)], device=device)
    src_pad = padding_mask(length_t, frames.size(1))
    memory = model.encode(frames, src_pad)

    ids = [bos_id]
    logps: list = []
    for step in range(decoding.max_decode_tokens):
        tok = torch.tensor([ids], device=device)
        logits = model.decode(tok, memory, causal_mask(tok.size(1), device), src_pad)[0, -1].float()
        logits[pad_id] = -torch.inf
        logits[bos_id] = -torch.inf
        logits[unk_id] = -torch.inf
        if step < decoding.min_output_tokens:
            logits[eos_id] = -torch.inf
        logits = ban_repeated_ngram(logits, ids, decoding.no_repeat_ngram_size)
        for tid in set(ids[1:]):
            logits[tid] = (
                logits[tid] / decoding.repetition_penalty
                if logits[tid] >= 0
                else logits[tid] * decoding.repetition_penalty
            )
        logp = torch.log_softmax(logits, dim=-1)
        prob, tid = torch.max(logp, dim=-1)
        tid = int(tid)
        logps.append(float(prob))
        ids.append(tid)
        if tid == eos_id:
            break

    return ids, logps
