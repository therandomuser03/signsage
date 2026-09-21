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
        demo_lexicon=None,
        enable_demo_fallback: bool = True,
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
        self.demo_lexicon = demo_lexicon
        self.enable_demo_fallback = enable_demo_fallback

    @staticmethod
    def _quantize(poses: np.ndarray) -> np.ndarray:
        # Round to 4 decimal places for the wire format only -- see the note
        # this used to carry inline: this shrinks a (60, 1629) JSON payload
        # by roughly 60% for ~2ms of extra CPU, with no perceptible change to
        # a rendered skeleton (float32 source precision is preserved).
        return np.round(poses.astype(np.float64), 4)

    @torch.inference_mode()
    def predict(self, text: str):
        """Return (gloss_or_phrase, poses[T,1629], elapsed_ms, provider, is_fallback).

        Bounded demo phrases (see backend/demo_prototypes/) are checked
        first: if `text` matches one of the ~10 supported phrases (exactly or
        via a small alias table), the real recorded clip for that phrase is
        returned instead of the generative model. The generative Text->Pose
        GRU is a Phase 0 research checkpoint that motion-energy evaluation
        showed collapses to a near-static pose (~7.9% of ground-truth
        motion -- see research/outputs/phase0/RESULTS.md); the demo lexicon
        path exists so the bounded phrase set actually animates, matching
        the honest-fallback pattern already used for Sign->Text.
        """
        started = time.perf_counter()

        if (
            self.enable_demo_fallback
            and self.demo_lexicon is not None
            and self.demo_lexicon.is_available()
        ):
            from app.services.demo_lexicon import match_demo_phrase

            phrase = match_demo_phrase(text, self.demo_lexicon.inventory())
            if phrase is not None:
                render = self.demo_lexicon.get_render_sequence(phrase)
                if render is not None:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    return (
                        phrase,
                        self._quantize(render),
                        elapsed_ms,
                        "demo_prototype",
                        True,
                    )

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
        poses = self._quantize(poses)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return gloss, poses, elapsed_ms, gloss_result.provider, False
