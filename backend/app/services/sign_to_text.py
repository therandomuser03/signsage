from __future__ import annotations

import time

import numpy as np
import torch

from app.models.sign_to_text import SignTransformer
from app.services.pose_preprocessing import prepare_landmarks
from app.services.vocab import Vocabulary


def padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


def causal_mask(n: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(n, n, device=device, dtype=torch.bool),
        diagonal=1,
    )


def apply_penalties(
    logits: torch.Tensor,
    generated_ids: list[int],
    *,
    repetition_penalty: float = 2.0,
    no_repeat_ngram_size: int = 3,
    temperature: float = 0.8,
) -> torch.Tensor:
    """Score adjustment: suppress specials, penalize repeats, apply temperature."""
    scores = logits.clone()
    # ponytail: hard-mask <pad>(0) and <bos>(1); they should never be generated
    scores[0] = -torch.inf
    scores[1] = -torch.inf

    # Token-level repetition penalty
    for token_id in set(generated_ids):
        if scores[token_id] >= 0:
            scores[token_id] /= repetition_penalty
        else:
            scores[token_id] *= repetition_penalty

    # N-gram blocking
    if len(generated_ids) >= no_repeat_ngram_size - 1:
        prefix = tuple(generated_ids[-(no_repeat_ngram_size - 1) :])
        blocked = {
            ngram[-1]
            for ngram in zip(
                *(generated_ids[index:] for index in range(no_repeat_ngram_size))
            )
            if ngram[:-1] == prefix
        }
        for token_id in blocked:
            scores[token_id] = -torch.inf

    # Temperature scaling sharpens (< 1) or flattens (> 1) the distribution
    scores = scores / max(temperature, 1e-6)
    return scores


class SignToTextService:
    """Beam-search Sign→Text inference.

    Replaces the previous greedy decoder with a simple beam search that
    explores multiple hypothesis paths. This helps a mode-collapsed model
    find more diverse and coherent output than single-path greedy decoding.

    ponytail: no length-normalization or coverage penalty; upgrade path is
    adding those if beam search alone isn't enough after retraining.
    """

    def __init__(
        self,
        *,
        model: SignTransformer,
        vocab: Vocabulary,
        device: torch.device,
        max_frames: int,
        max_tokens: int,
        input_mode: str = "normalized",
        no_face: bool = False,
        beam_width: int = 5,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.input_mode = input_mode
        self.no_face = no_face
        self.beam_width = beam_width

    @torch.inference_mode()
    def predict(self, landmarks: np.ndarray):
        started = time.perf_counter()

        processed = prepare_landmarks(
            landmarks,
            max_frames=self.max_frames,
            no_face=self.no_face,
            input_mode=self.input_mode,
        )

        frames = torch.from_numpy(processed.features).unsqueeze(0).to(self.device)
        lengths = torch.tensor([processed.true_length], device=self.device)
        src_pad = padding_mask(lengths, frames.size(1))
        memory = self.model.encode(frames, src_pad)

        # Beam search
        # Each beam: (cumulative_log_prob, token_ids)
        beams: list[tuple[float, list[int]]] = [(0.0, [self.vocab.bos_id])]
        finished: list[tuple[float, list[int]]] = []

        for _ in range(self.max_tokens):
            candidates: list[tuple[float, list[int]]] = []

            for score, ids in beams:
                tokens = torch.tensor([ids], device=self.device)
                logits = self.model.decode(
                    tokens,
                    memory,
                    causal_mask(tokens.size(1), self.device),
                    src_pad,
                )
                adjusted = apply_penalties(logits[0, -1], ids)
                log_probs = torch.log_softmax(adjusted, dim=-1)

                # Expand top-k candidates from this beam
                topk_probs, topk_ids = log_probs.topk(self.beam_width)
                for prob, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                    new_ids = ids + [tid]
                    new_score = score + prob
                    if tid == self.vocab.eos_id:
                        # Length-normalize: divide by number of generated tokens
                        norm_score = new_score / max(len(new_ids) - 1, 1)
                        finished.append((norm_score, new_ids))
                    else:
                        candidates.append((new_score, new_ids))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[: self.beam_width]

            # Early stop: enough finished hypotheses
            if len(finished) >= self.beam_width:
                break

        # Pick best finished hypothesis, or best beam if none finished
        if finished:
            finished.sort(key=lambda x: x[0], reverse=True)
            best_ids = finished[0][1]
        else:
            beams.sort(key=lambda x: x[0], reverse=True)
            best_ids = beams[0][1]

        elapsed_ms = (time.perf_counter() - started) * 1000
        return self.vocab.decode(best_ids[1:]), processed.true_length, elapsed_ms
