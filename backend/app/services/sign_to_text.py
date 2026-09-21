from __future__ import annotations

import time

import numpy as np
import torch

from app.models.sign_to_text import SignTransformer
from app.services.pose_preprocessing import (
    prepare_landmarks,
    validate_landmark_array,
)
from app.services.sentence_refiner import refine_sentence
from app.services.translation_quality import score_text
from app.services.vocab import Vocabulary

try:
    from app.services.demo_lexicon import DemoLexicon
except Exception:
    # Demo lexicon is optional at import time and may be injected at runtime.
    DemoLexicon = None  # type: ignore


HONEST_PLACEHOLDER = (
    "Unrecognized sign — try: Hello / Thank you / Yes / No / Help"
)


def padding_mask(
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    idx = torch.arange(
        max_len,
        device=lengths.device,
    ).unsqueeze(0)

    return idx >= lengths.unsqueeze(1)


def causal_mask(
    n: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.triu(
        torch.ones(
            n,
            n,
            device=device,
            dtype=torch.bool,
        ),
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
    """
    Apply decoding-time score adjustments.

    This:
    - suppresses <pad> and <bos>
    - applies token-level repetition penalty
    - blocks repeated n-grams
    - applies temperature scaling
    """

    scores = logits.clone()

    # <pad> and <bos> should never be generated.
    scores[0] = -torch.inf
    scores[1] = -torch.inf

    # ---------------------------------------------------------------
    # Token-level repetition penalty
    # ---------------------------------------------------------------

    for token_id in set(generated_ids):
        if scores[token_id] >= 0:
            scores[token_id] /= repetition_penalty
        else:
            scores[token_id] *= repetition_penalty

    # ---------------------------------------------------------------
    # N-gram blocking
    # ---------------------------------------------------------------

    if len(generated_ids) >= (
        no_repeat_ngram_size - 1
    ):
        prefix = tuple(
            generated_ids[
                -(no_repeat_ngram_size - 1) :
            ]
        )

        blocked = {
            ngram[-1]
            for ngram in zip(
                *(
                    generated_ids[index:]
                    for index in range(
                        no_repeat_ngram_size
                    )
                )
            )
            if ngram[:-1] == prefix
        }

        for token_id in blocked:
            scores[token_id] = -torch.inf

    # ---------------------------------------------------------------
    # Temperature
    # ---------------------------------------------------------------

    scores = scores / max(
        temperature,
        1e-6,
    )

    return scores


class SignToTextService:
    """
    Sign → Text inference service.

    IMPORTANT DATA CONTRACT
    -----------------------
    The live browser pipeline sends RAW MediaPipe Holistic landmarks:

        [T, 543, 3]

    The canonical training-time preprocessing is performed exactly once
    by:

        prepare_landmarks(..., input_mode="raw")

    which calls:

        normalize_pose_sequence()

    from app.services.pose_preprocessing.

    This service must therefore NOT assume that live MediaPipe coordinates
    have already been normalized.
    """

    def __init__(
        self,
        *,
        model: SignTransformer,
        vocab: Vocabulary,
        device: torch.device,
        max_frames: int,
        max_tokens: int,

        # -----------------------------------------------------------
        # IMPORTANT:
        # Live MediaPipe input is RAW.
        # -----------------------------------------------------------
        input_mode: str = "raw",

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
        if input_mode not in {
            "raw",
            "normalized",
        }:
            raise ValueError(
                "input_mode must be 'raw' or 'normalized'."
            )

        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_frames = max_frames
        self.max_tokens = max_tokens

        # For the live browser pipeline this should remain "raw".
        self.input_mode = input_mode

        self.no_face = no_face
        self.beam_width = beam_width

        self.demo_lexicon = demo_lexicon
        self.enable_demo_fallback = (
            enable_demo_fallback
        )
        self.demo_match_threshold = (
            demo_match_threshold
        )

        self.enable_llm_refine = (
            enable_llm_refine
        )

        self.groq_api_key = groq_api_key
        self.groq_model = groq_model

        self.enable_ollama = enable_ollama
        self.ollama_base_url = (
            ollama_base_url
        )
        self.ollama_model = ollama_model
        self.ollama_timeout_seconds = (
            ollama_timeout_seconds
        )

        self.llm_refine_timeout = (
            llm_refine_timeout
        )

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _motion_energy(
        self,
        landmarks: np.ndarray,
    ) -> float:
        """
        Estimate motion across the complete landmark sequence.

        Unlike the frontend poseMotion helper, this considers the
        complete landmark vector, including hands.
        """

        try:
            arr = validate_landmark_array(
                landmarks,
                allow_empty=True,
            )

            if arr.shape[0] < 2:
                return 0.0

            flat = arr.reshape(
                arr.shape[0],
                -1,
            )

            diffs = np.abs(
                np.diff(
                    flat,
                    axis=0,
                )
            )

            return float(
                np.mean(diffs)
            )

        except Exception:
            return 0.0

    def _match_demo_phrase(
        self,
        landmarks: np.ndarray,
        motion_energy: float,
    ) -> tuple[str, float] | None:
        """Match bounded recorded phrases before the unreliable checkpoint."""
        if (
            motion_energy < 0.008
            or not self.enable_demo_fallback
            or self.demo_lexicon is None
            or not getattr(self.demo_lexicon, "is_available", lambda: False)()
        ):
            return None

        try:
            processed = prepare_landmarks(
                landmarks,
                max_frames=60,
                no_face=True,
                input_mode=self.input_mode,
            )
            sequence = processed.features[: min(int(processed.true_length), 60)]
            phrase, distance = self.demo_lexicon.match(
                sequence,
                threshold=self.demo_match_threshold,
            )
            if phrase is None:
                return None
            return phrase, distance
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Optional sentence refinement
    # ------------------------------------------------------------------

    def _maybe_refine(
        self,
        text: str,
    ) -> tuple[str, str]:
        """
        Optionally refine model output using an LLM.

        This remains disabled by default.

        The raw Sign→Text output should remain available for research
        evaluation rather than silently replacing model output.
        """

        if (
            not self.enable_llm_refine
            or "Unrecognized sign" in text
        ):
            return text, "naive"

        # Demo phrases are already fluent.
        if text.lower().strip() in {
            "hello",
            "thank you",
            "yes",
            "no",
            "help",
            "please",
            "sorry",
            "good morning",
            "how are you",
            "i need help",
        }:
            return text, "naive"

        try:
            res = refine_sentence(
                text,
                enable_groq=(
                    self.groq_api_key is not None
                    and self.enable_llm_refine
                ),
                groq_api_key=self.groq_api_key,
                groq_model=self.groq_model,
                enable_ollama=self.enable_ollama,
                ollama_base_url=self.ollama_base_url,
                ollama_model=self.ollama_model,
                ollama_timeout_seconds=(
                    self.ollama_timeout_seconds
                ),
            )

            cleaned = res.text.strip()

            if cleaned and len(cleaned) >= 2:
                return (
                    cleaned,
                    res.provider,
                )

            return text, "naive"

        except Exception:
            return text, "naive"

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        landmarks: np.ndarray,
    ):
        (
            text,
            true_len,
            elapsed,
            _,
            _,
            _,
        ) = self.predict_with_meta(
            landmarks
        )

        return (
            text,
            true_len,
            elapsed,
        )

    @torch.inference_mode()
    def predict_with_meta(
        self,
        landmarks: np.ndarray,
    ):
        """
        Return:

            (
                text,
                true_len,
                elapsed_ms,
                is_fallback,
                confidence,
                gated_reason,
            )

        Input:
            RAW MediaPipe landmarks [T, 543, 3]

        Preprocessing:
            prepare_landmarks(..., input_mode="raw")

        Model input:
            [1, max_frames, feature_dim]
        """

        started = time.perf_counter()

        # --------------------------------------------------------------
        # Validate the original sequence.
        #
        # Keep this original sequence for motion measurement and
        # demo-lexicon matching.
        # --------------------------------------------------------------

        try:
            orig = validate_landmark_array(
                landmarks,
                allow_empty=True,
            )
        except Exception:
            orig = np.asarray(
                landmarks,
                dtype=np.float32,
            )

        # --------------------------------------------------------------
        # Motion is calculated from RAW landmarks.
        # --------------------------------------------------------------

        motion_energy = (
            self._motion_energy(orig)
        )

        demo_match = self._match_demo_phrase(orig, motion_energy)
        if demo_match is not None:
            phrase, distance = demo_match
            refined, provider = self._maybe_refine(phrase)
            return (
                refined,
                min(orig.shape[0], self.max_frames),
                (time.perf_counter() - started) * 1000,
                True,
                0.88,
                f"demo:{phrase}:{distance:.2f}:{provider}",
            )

        # --------------------------------------------------------------
        # Canonical preprocessing
        #
        # RAW MediaPipe:
        #
        #   [T, 543, 3]
        #
        #       ↓
        #
        # normalize_pose_sequence()
        #
        #       ↓
        #
        # [T, 1629]
        #
        #       ↓
        #
        # resize_sequence()
        # --------------------------------------------------------------

        processed = prepare_landmarks(
            orig,
            max_frames=self.max_frames,
            no_face=self.no_face,
            input_mode=self.input_mode,
        )

        frames = (
            torch.from_numpy(
                processed.features
            )
            .unsqueeze(0)
            .to(self.device)
        )

        lengths = torch.tensor(
            [processed.true_length],
            device=self.device,
        )

        src_pad = padding_mask(
            lengths,
            frames.size(1),
        )

        memory = self.model.encode(
            frames,
            src_pad,
        )

        # --------------------------------------------------------------
        # Beam search
        # --------------------------------------------------------------

        # Each beam:
        #
        #   (cumulative_log_probability, token_ids)
        #
        beams: list[
            tuple[float, list[int]]
        ] = [
            (
                0.0,
                [self.vocab.bos_id],
            )
        ]

        finished: list[
            tuple[float, list[int]]
        ] = []

        for _ in range(
            self.max_tokens
        ):
            candidates: list[
                tuple[float, list[int]]
            ] = []

            for score, ids in beams:

                tokens = torch.tensor(
                    [ids],
                    device=self.device,
                )

                logits = self.model.decode(
                    tokens,
                    memory,
                    causal_mask(
                        tokens.size(1),
                        self.device,
                    ),
                    src_pad,
                )

                adjusted = apply_penalties(
                    logits[0, -1],
                    ids,
                )

                log_probs = torch.log_softmax(
                    adjusted,
                    dim=-1,
                )

                topk_probs, topk_ids = (
                    log_probs.topk(
                        self.beam_width
                    )
                )

                for prob, tid in zip(
                    topk_probs.tolist(),
                    topk_ids.tolist(),
                ):
                    new_ids = ids + [
                        tid
                    ]

                    new_score = (
                        score + prob
                    )

                    if (
                        tid
                        == self.vocab.eos_id
                    ):
                        norm_score = (
                            new_score
                            / max(
                                len(new_ids)
                                - 1,
                                1,
                            )
                        )

                        finished.append(
                            (
                                norm_score,
                                new_ids,
                            )
                        )

                    else:
                        candidates.append(
                            (
                                new_score,
                                new_ids,
                            )
                        )

            if not candidates:
                break

            candidates.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            beams = candidates[
                : self.beam_width
            ]

            if (
                len(finished)
                >= self.beam_width
            ):
                break

        # --------------------------------------------------------------
        # Select best hypothesis
        # --------------------------------------------------------------

        if finished:
            finished.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            best_ids = finished[0][1]

        else:
            beams.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            best_ids = beams[0][1]

        raw_text = self.vocab.decode(
            best_ids[1:]
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000

        # --------------------------------------------------------------
        # Low-motion gate
        # --------------------------------------------------------------

        if motion_energy < 0.008:

            (
                is_trusted,
                reason,
                conf,
            ) = score_text(
                raw_text,
                motion_energy=motion_energy,
            )

            if reason == "low_motion":
                return (
                    HONEST_PLACEHOLDER,
                    processed.true_length,
                    elapsed_ms,
                    True,
                    0.2,
                    reason,
                )

            if (
                not is_trusted
                and reason
                in (
                    "low_motion",
                    "attractor",
                )
            ):
                return (
                    HONEST_PLACEHOLDER,
                    processed.true_length,
                    elapsed_ms,
                    True,
                    0.2,
                    reason,
                )

        # --------------------------------------------------------------
        # Demo lexicon fallback
        # --------------------------------------------------------------
        #
        # IMPORTANT:
        # The same RAW sequence is passed to prepare_landmarks().
        #
        # This guarantees that the demo path receives the same canonical
        # normalization as the model path.
        # --------------------------------------------------------------

        if (
            self.enable_demo_fallback
            and self.demo_lexicon is not None
            and getattr(
                self.demo_lexicon,
                "is_available",
                lambda: False,
            )()
        ):
            if motion_energy >= 0.008:

                try:
                    seq_processed = (
                        prepare_landmarks(
                            orig,
                            max_frames=60,
                            no_face=True,
                            input_mode=self.input_mode,
                        )
                    )

                    seq_noface = (
                        seq_processed.features
                    )

                    tlen = min(
                        int(
                            seq_processed.true_length
                        ),
                        60,
                    )

                    if tlen > 0:
                        seq_noface = (
                            seq_noface[:tlen]
                        )

                    (
                        phrase,
                        dist,
                    ) = self.demo_lexicon.match(
                        seq_noface,
                        threshold=(
                            self.demo_match_threshold
                        ),
                    )

                    if phrase is not None:

                        refined, prov = (
                            self._maybe_refine(
                                phrase
                            )
                        )

                        return (
                            refined,
                            processed.true_length,
                            elapsed_ms,
                            True,
                            0.88,
                            (
                                f"demo:{phrase}:"
                                f"{dist:.2f}:"
                                f"{prov}"
                            ),
                        )

                    # ------------------------------------------------------
                    # Permissive retry for clear motion.
                    # ------------------------------------------------------

                    if motion_energy > 0.015:

                        (
                            phrase2,
                            dist2,
                        ) = self.demo_lexicon.match(
                            seq_noface,
                            threshold=(
                                self.demo_match_threshold
                                * 1.6
                            ),
                        )

                        if phrase2 is not None:

                            refined2, prov2 = (
                                self._maybe_refine(
                                    phrase2
                                )
                            )

                            return (
                                refined2,
                                processed.true_length,
                                elapsed_ms,
                                True,
                                0.82,
                                (
                                    f"demo:{phrase2}:"
                                    f"{dist2:.2f}:"
                                    f"{prov2}"
                                ),
                            )

                except Exception:
                    # Demo fallback should never bring down the main
                    # Sign → Text inference path.
                    pass

        # --------------------------------------------------------------
        # Model quality gate
        # --------------------------------------------------------------

        (
            is_trusted,
            reason,
            conf,
        ) = score_text(
            raw_text,
            motion_energy=motion_energy,
        )

        if is_trusted:

            refined, prov = (
                self._maybe_refine(
                    raw_text
                )
            )

            return (
                refined,
                processed.true_length,
                elapsed_ms,
                False,
                conf,
                f"{reason}:{prov}",
            )

        # --------------------------------------------------------------
        # If demo fallback is disabled, return model output even when
        # quality gating rejects it.
        # --------------------------------------------------------------

        if not self.enable_demo_fallback:

            refined, prov = (
                self._maybe_refine(
                    raw_text
                )
            )

            return (
                refined,
                processed.true_length,
                elapsed_ms,
                False,
                conf,
                f"{reason}:{prov}",
            )

        # --------------------------------------------------------------
        # Honest fallback
        # --------------------------------------------------------------

        return (
            HONEST_PLACEHOLDER,
            processed.true_length,
            elapsed_ms,
            True,
            0.2,
            reason,
        )
