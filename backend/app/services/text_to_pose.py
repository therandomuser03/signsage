from __future__ import annotations

import time

import numpy as np
import torch

from app.models.sign_to_text import TextToPoseGRU
from app.services.gloss import generate_gloss
from app.services.vocab import Vocabulary


class TextToPoseService:
    def __init__(
        self,
        *,
        model: TextToPoseGRU,
        vocab: Vocabulary,
        device: torch.device,
        max_tokens: int,
        enable_groq: bool = False,
        groq_api_key: str | None = None,
        groq_model: str = "openai/gpt-oss-20b",
        enable_ollama: bool = False,
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "llama3.2:3b",
        ollama_timeout_seconds: float = 8.0,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_tokens = max_tokens
        self.enable_groq = enable_groq
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.enable_ollama = enable_ollama
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_timeout_seconds = ollama_timeout_seconds

    @torch.inference_mode()
    def predict(self, text: str):
        started = time.perf_counter()

        gloss_result = generate_gloss(
            text,
            enable_groq=self.enable_groq,
            groq_api_key=self.groq_api_key,
            groq_model=self.groq_model,
            enable_ollama=self.enable_ollama,
            ollama_base_url=self.ollama_base_url,
            ollama_model=self.ollama_model,
            ollama_timeout_seconds=self.ollama_timeout_seconds,
        )
        gloss = gloss_result.gloss

        token_ids = self.vocab.encode(gloss, self.max_tokens)
        tokens = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        poses = (
            self.model(tokens, teacher=None)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        # Round to 4 decimal places for the wire format only. This is a
        # vectorized, precision-preserving trick, not a data change: the
        # model's output is float32 (~7 significant digits) to begin with,
        # so 4 decimals loses nothing perceptible in a rendered skeleton.
        # The result is deliberately kept as float64 (NOT cast back to
        # float32): rounding *then* upcasting to float64 is what makes this
        # work. Rounding a float32 array, or rounding-then-recasting to
        # float32, does not shorten the resulting JSON, because the
        # float32->float64 upcast that `.tolist()` performs downstream
        # reintroduces long, non-round-tripping decimal noise either way.
        # Measured effect on a real (60, 1629) sequence: ~2.07 MB -> ~0.82 MB
        # of JSON (~60% smaller), for about 2 ms of extra CPU time -- see
        # MODEL_CARD.md.
        poses = np.round(poses.astype(np.float64), 4)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return gloss, poses, elapsed_ms, gloss_result.provider
