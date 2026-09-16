from __future__ import annotations

import time

import numpy as np
import torch

from app.models.sign_to_text import SignTransformer
from app.services.pose_preprocessing import NOFACE_FEATURE_INDICES, prepare_landmarks, validate_landmark_array
from app.services.sentence_refiner import refine_sentence
from app.services.translation_quality import score_text
from app.services.vocab import Vocabulary

try:
    from app.services.demo_lexicon import DemoLexicon
except Exception:  # ponytail: demo lexicon is optional at import time; injected at runtime
    DemoLexicon = None  # type: ignore

HONEST_PLACEHOLDER = "Unrecognized sign — try: Hello / Thank you / Yes / No / Help"


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
    """Beam-search Sign→Text inference with functional fallback for collapsed checkpoint.

    Phase 0 checkpoint is NO-GO (3.9% unique, attractor "this is why sacrifices.").
    This service keeps the checkpoint for mechanical coverage but gates its output
    via translation_quality and, when gated, falls back to demo lexicon DTW.
    ponytail: no length-normalization/coverage penalty; demo lexicon is interim until retrain.
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
        demo_lexicon=None,
        enable_demo_fallback: bool = True,
        demo_match_threshold: float = 6.5,
        enable_llm_refine: bool = False,
        groq_api_key: str | None = None,
        groq_model: str = "openai/gpt-oss-20b",
        enable_ollama: bool = False,
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "llama3.2:3b",
        ollama_timeout_seconds: float = 8.0,
        llm_refine_timeout: float = 8.0,
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.input_mode = input_mode
        self.no_face = no_face
        self.beam_width = beam_width
        self.demo_lexicon = demo_lexicon
        self.enable_demo_fallback = enable_demo_fallback
        self.demo_match_threshold = demo_match_threshold
        self.enable_llm_refine = enable_llm_refine
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.enable_ollama = enable_ollama
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self.llm_refine_timeout = llm_refine_timeout

    def _motion_energy(self, landmarks: np.ndarray) -> float:
        try:
            arr = validate_landmark_array(landmarks, allow_empty=True)
            if arr.shape[0] < 2:
                return 0.0
            # use full landmark diff (hands move even when pose is steady) — frontend poseMotion is pose-only but backend needs hand motion
            flat = arr.reshape(arr.shape[0], -1)
            diffs = np.abs(np.diff(flat, axis=0))
            return float(np.mean(diffs))
        except Exception:
            return 0.0

    def _maybe_refine(self, text: str) -> tuple[str, str]:
        if not self.enable_llm_refine or "Unrecognized sign" in text:
            return text, "naive"
        # demo single-word phrases are already fluent
        if text.lower().strip() in {"hello", "thank you", "yes", "no", "help", "please", "sorry", "good morning", "how are you", "i need help"}:
            return text, "naive"
        try:
            res = refine_sentence(
                text,
                enable_groq=self.groq_api_key is not None and self.enable_llm_refine,
                groq_api_key=self.groq_api_key,
                groq_model=self.groq_model,
                enable_ollama=self.enable_ollama,
                ollama_base_url=self.ollama_base_url,
                ollama_model=self.ollama_model,
                ollama_timeout_seconds=self.ollama_timeout_seconds,
            )
            # if LLM returns same or empty, keep original
            cleaned = res.text.strip()
            if cleaned and len(cleaned) >= 2:
                return cleaned, res.provider
            return text, "naive"
        except Exception:
            return text, "naive"

    @torch.inference_mode()
    def predict(self, landmarks: np.ndarray):
        text, true_len, elapsed, _, _, _ = self.predict_with_meta(landmarks)
        return text, true_len, elapsed

    @torch.inference_mode()
    def predict_with_meta(self, landmarks: np.ndarray):
        """Return (text, true_len, elapsed_ms, is_fallback, confidence, gated_reason)."""
        started = time.perf_counter()

        # keep original for fallback/demo matching before resizing/normalization
        try:
            orig = validate_landmark_array(landmarks, allow_empty=True)
        except Exception:
            orig = np.asarray(landmarks)

        motion_energy = self._motion_energy(landmarks)

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

        raw_text = self.vocab.decode(best_ids[1:])
        elapsed_ms = (time.perf_counter() - started) * 1000

        # Functional demo first: if enabled and we have motion, try lexicon before trusting collapsed model
        # low-motion holder = honest placeholder immediately
        if motion_energy < 0.008:
            # still need elapsed for timing, but return placeholder without model trust
            is_trusted, reason, conf = score_text(raw_text, motion_energy=motion_energy)
            if reason == "low_motion":
                return HONEST_PLACEHOLDER, processed.true_length, elapsed_ms, True, 0.2, reason
            # for attractor with low motion, also placeholder
            if not is_trusted and reason in ("low_motion", "attractor"):
                # check if motion is truly low, placeholder wins over demo
                if motion_energy < 0.008:
                    return HONEST_PLACEHOLDER, processed.true_length, elapsed_ms, True, 0.2, reason

        # Demo lexicon priority when enabled (makes Hi -> Hello functional, not "government")
        if self.enable_demo_fallback and self.demo_lexicon is not None and getattr(self.demo_lexicon, "is_available", lambda: False)():
            if motion_energy >= 0.008:
                try:
                    from app.services.pose_preprocessing import prepare_landmarks as prep

                    seq_processed = prep(landmarks, max_frames=60, no_face=True, input_mode=self.input_mode)
                    seq_noface = seq_processed.features
                    tlen = min(int(seq_processed.true_length), 60)
                    seq_noface = seq_noface[:tlen] if tlen > 0 else seq_noface
                    phrase, dist = self.demo_lexicon.match(seq_noface, threshold=self.demo_match_threshold)
                    if phrase is not None:
                        refined, prov = self._maybe_refine(phrase)
                        return refined, processed.true_length, elapsed_ms, True, 0.88, f"demo:{phrase}:{dist:.2f}:{prov}"
                    # permissive retry for clear motion
                    if motion_energy > 0.015:
                        p2, d2 = self.demo_lexicon.match(seq_noface, threshold=self.demo_match_threshold * 1.6)
                        if p2 is not None:
                            refined2, prov2 = self._maybe_refine(p2)
                            return refined2, processed.true_length, elapsed_ms, True, 0.82, f"demo:{p2}:{d2:.2f}:{prov2}"
                except Exception:
                    pass

        # Quality gate for model output (fallback when demo didn't hit)
        is_trusted, reason, conf = score_text(raw_text, motion_energy=motion_energy)
        if is_trusted:
            refined, prov = self._maybe_refine(raw_text)
            # keep confidence but note provider
            return refined, processed.true_length, elapsed_ms, False, conf, f"{reason}:{prov}"
        if not self.enable_demo_fallback:
            refined, prov = self._maybe_refine(raw_text)
            return refined, processed.true_length, elapsed_ms, False, conf, f"{reason}:{prov}"

        # No demo hit and not trusted -> honest placeholder
        return HONEST_PLACEHOLDER, processed.true_length, elapsed_ms, True, 0.2, reason
