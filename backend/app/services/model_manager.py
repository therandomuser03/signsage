from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from app.core.config import Settings
from app.core.serialization import load_json
from app.models.sign_to_text import SignTransformer, TextToPoseGRU
from app.services.demo_lexicon import DemoLexicon
from app.services.pose_preprocessing import FULL_FEATURE_DIM, NOFACE_FEATURE_DIM
from app.services.sign_to_text import SignToTextService
from app.services.text_to_pose import TextToPoseService
from app.services.vocab import Vocabulary


# Checkpoints this backend actually serves. `signsage_gru_best.pt` (the
# Sign->Text GRU sanity baseline) is intentionally excluded: the notebook's
# own evaluation found it fully mode-collapsed to a single repeated sentence
# (0.4% unique outputs across 1,500 test rows -- see research/outputs/phase0/
# RESULTS.md), so serving it would be actively misleading rather than merely
# low-quality. It is not loaded, not required at startup, and not reachable
# via any route. If the raw file is still present under weights/ it is kept
# for archival/provenance reasons only -- see MODEL_CARD.md.
CHECKPOINTS = {
    "transformer_full": "signsage_transformer_full.pt",
    "transformer_noface": "signsage_transformer_noface.pt",
    "text_to_pose": "signsage_text_to_pose_best.pt",
}


@dataclass
class ModelManager:
    device: torch.device
    text_vocab: Vocabulary
    gloss_vocab: Vocabulary
    sign_to_text_full: SignToTextService
    sign_to_text_noface: SignToTextService | None
    text_to_pose: TextToPoseService
    manifest: dict
    demo_lexicon: DemoLexicon | None = None

    @classmethod
    def load(cls, settings: Settings) -> "ModelManager":
        model_dir = settings.resolved_model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        device = _resolve_device(settings.device)
        text_vocab = Vocabulary.from_json(settings.resolved_text_vocab)
        gloss_vocab = Vocabulary.from_json(settings.resolved_gloss_vocab)

        full_model = SignTransformer(
            input_dim=FULL_FEATURE_DIM,
            vocab_size=text_vocab.size,
            pad_idx=text_vocab.pad_id,
            max_frames=settings.max_frames,
            max_tokens=settings.max_tokens,
        ).to(device)
        _load_state(
            full_model,
            model_dir / CHECKPOINTS["transformer_full"],
            device,
        )
        full_model.eval()

        noface_service = None
        noface_path = model_dir / CHECKPOINTS["transformer_noface"]
        if settings.allow_noface:
            noface_model = SignTransformer(
                input_dim=NOFACE_FEATURE_DIM,
                vocab_size=text_vocab.size,
                pad_idx=text_vocab.pad_id,
                max_frames=settings.max_frames,
                max_tokens=settings.max_tokens,
            ).to(device)
            _load_state(noface_model, noface_path, device)
            noface_model.eval()
            noface_service = SignToTextService(
                model=noface_model,
                vocab=text_vocab,
                device=device,
                max_frames=settings.max_frames,
                max_tokens=settings.max_tokens,
                input_mode=settings.landmark_input_mode,
                no_face=True,
            )

        pose_model = TextToPoseGRU(
            vocab_size=gloss_vocab.size,
            feature_dim=FULL_FEATURE_DIM,
            pad_idx=gloss_vocab.pad_id,
            output_frames=settings.pose_output_frames,
        ).to(device)
        _load_state(
            pose_model,
            model_dir / CHECKPOINTS["text_to_pose"],
            device,
        )
        pose_model.eval()

        # Demo lexicon for functional fallback (noface DTW)
        demo_lexicon = None
        try:
            demo_dir = settings.resolve(settings.demo_prototypes_dir)
            demo_lexicon = DemoLexicon.load(demo_dir)
        except Exception:
            demo_lexicon = DemoLexicon.load(settings.demo_prototypes_dir)

        manifest = {}
        if settings.resolved_manifest.is_file():
            manifest = load_json(settings.resolved_manifest)

        if settings.require_all_models:
            missing = [
                name for name, filename in CHECKPOINTS.items()
                if not (model_dir / filename).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Required checkpoint(s) missing: {', '.join(missing)}"
                )

        # inject demo lexicon + LLM refine into services
        demo_kwargs = dict(
            demo_lexicon=demo_lexicon,
            enable_demo_fallback=settings.enable_demo_fallback,
            demo_match_threshold=settings.demo_match_threshold,
            enable_llm_refine=settings.enable_llm_refine,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            enable_ollama=settings.enable_ollama_gloss,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            ollama_timeout_seconds=settings.ollama_timeout_seconds,
            llm_refine_timeout=settings.llm_refine_timeout_seconds,
        )
        full_service = SignToTextService(
            model=full_model,
            vocab=text_vocab,
            device=device,
            max_frames=settings.max_frames,
            max_tokens=settings.max_tokens,
            input_mode=settings.landmark_input_mode,
            **demo_kwargs,
        )
        # also inject into noface if present
        if noface_service is not None:
            noface_service.demo_lexicon = demo_lexicon
            noface_service.enable_demo_fallback = settings.enable_demo_fallback
            noface_service.demo_match_threshold = settings.demo_match_threshold
            noface_service.enable_llm_refine = settings.enable_llm_refine
            noface_service.groq_api_key = settings.groq_api_key
            noface_service.groq_model = settings.groq_model
            noface_service.enable_ollama = settings.enable_ollama_gloss
            noface_service.ollama_base_url = settings.ollama_base_url
            noface_service.ollama_model = settings.ollama_model
            noface_service.ollama_timeout_seconds = settings.ollama_timeout_seconds
            noface_service.llm_refine_timeout = settings.llm_refine_timeout_seconds

        return cls(
            device=device,
            text_vocab=text_vocab,
            gloss_vocab=gloss_vocab,
            sign_to_text_full=full_service,
            sign_to_text_noface=noface_service,
            text_to_pose=TextToPoseService(
                model=pose_model,
                vocab=gloss_vocab,
                device=device,
                max_tokens=settings.max_tokens,
                enable_groq=settings.enable_groq_gloss,
                groq_api_key=settings.groq_api_key,
                groq_model=settings.groq_model,
                enable_ollama=settings.enable_ollama_gloss,
                ollama_base_url=settings.ollama_base_url,
                ollama_model=settings.ollama_model,
                ollama_timeout_seconds=settings.ollama_timeout_seconds,
            ),
            manifest=manifest,
            demo_lexicon=demo_lexicon,
        )

    def predict_sign_to_text(self, landmarks, *, no_face: bool = False):
        if no_face:
            if self.sign_to_text_noface is None:
                raise ValueError("The no-face model is disabled.")
            return self.sign_to_text_noface.predict(landmarks)
        return self.sign_to_text_full.predict(landmarks)

    def predict_sign_to_text_with_meta(self, landmarks, *, no_face: bool = False):
        if no_face:
            if self.sign_to_text_noface is None:
                raise ValueError("The no-face model is disabled.")
            return self.sign_to_text_noface.predict_with_meta(landmarks)
        return self.sign_to_text_full.predict_with_meta(landmarks)

    def provenance_summary(self) -> dict | None:
        """Surface the research knowledge-repository manifest (see
        DATA_MODEL.md / DECISIONS.md) via the API.

        The manifest is loaded into memory at startup (see `load()` below)
        specifically so the backend can report "model/version metadata and
        research transparency" -- that is its documented purpose in the
        notebook that produces it. Previously it was loaded and then never
        read again anywhere; this method is what actually fulfills that
        purpose, via `GET /api/v1/models`.
        """
        if not self.manifest:
            return None
        status = self.manifest.get("status", {}) or {}
        return {
            "run_id": self.manifest.get("run_id"),
            "phase_0_status": status.get("phase_0_status"),
            "warning": status.get("warning"),
        }

    def model_inventory(self) -> list[dict]:
        items = [
            {
                "name": "transformer_full",
                "task": "sign_to_text",
                "input_features": FULL_FEATURE_DIM,
                "checkpoint": CHECKPOINTS["transformer_full"],
                "status": "loaded",
                "caveat": (
                    "Phase 0 checkpoint; evaluation documented repetitive/mode-collapsed "
                    "outputs. Mechanical inference is supported; reliability is not established. "
                    "Demo lexicon fallback active."
                ),
            },
            {
                "name": "text_to_pose_gru",
                "task": "text_to_pose",
                "input_features": FULL_FEATURE_DIM,
                "output_frames": 60,
                "checkpoint": CHECKPOINTS["text_to_pose"],
                "status": "loaded",
                "caveat": (
                    "Phase 0 checkpoint; motion-energy evaluation showed near-static output."
                ),
            },
        ]
        if self.sign_to_text_noface is not None:
            items.append(
                {
                    "name": "transformer_no_face",
                    "task": "sign_to_text",
                    "input_features": NOFACE_FEATURE_DIM,
                    "checkpoint": CHECKPOINTS["transformer_noface"],
                    "status": "loaded",
                    "caveat": "Phase 0 non-manual-marker/face ablation checkpoint.",
                }
            )
        # demo lexicon is not a checkpoint but expose availability
        if self.demo_lexicon is not None and self.demo_lexicon.is_available():
            items.append(
                {
                    "name": "demo_lexicon",
                    "task": "sign_to_text_fallback",
                    "input_features": NOFACE_FEATURE_DIM,
                    "checkpoint": "demo_prototypes/*.json",
                    "status": f"loaded ({len(self.demo_lexicon.inventory())} phrases)",
                    "caveat": "DTW noface fallback for collapsed outputs; honest placeholder when no match.",
                }
            )
        return items


def _resolve_device(value: str) -> torch.device:
    value = value.lower().strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SIGNSAGE_DEVICE requests CUDA, but CUDA is unavailable.")
    return torch.device(value)


def _load_state(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a state_dict: {path}")

    model.load_state_dict(state, strict=True)
