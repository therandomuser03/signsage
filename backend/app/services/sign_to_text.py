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


class SignToTextService:
    """Greedy-decoding Sign->Text inference, matching the notebook's
    `greedy_transformer` exactly.

    A single parameterized class serves both the full-landmark and no-face
    ablation models -- they differ only in whether `prepare_landmarks` masks
    out the face block, never in the decoding logic itself. (Previously this
    was two near-identical classes; collapsed to avoid drift between them.)
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
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.input_mode = input_mode
        self.no_face = no_face

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

        ids = [self.vocab.bos_id]
        for _ in range(self.max_tokens):
            tokens = torch.tensor([ids], device=self.device)
            logits = self.model.decode(
                tokens,
                memory,
                causal_mask(tokens.size(1), self.device),
                src_pad,
            )
            nxt = int(logits[0, -1].argmax())
            if nxt == self.vocab.eos_id:
                break
            ids.append(nxt)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return self.vocab.decode(ids[1:]), processed.true_length, elapsed_ms
