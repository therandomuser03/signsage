from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.services.demo_lexicon import DemoLexicon


@dataclass
class ModelManager:
    """Live lightweight recognizer.

    Recognition uses actual camera landmarks matched against stored landmark
    templates. Research PyTorch checkpoints remain archived, but are not part
    of live application startup or inference.
    """

    device: str
    lexicon: DemoLexicon
    template_match_threshold: float
    sign_to_text_full: None = None
    sign_to_text_noface: None = None
    text_to_pose: None = None
    manifest: dict | None = None

    @classmethod
    def load(cls, settings: Settings) -> "ModelManager":
        directory = settings.resolve(settings.demo_prototypes_dir)
        lexicon = DemoLexicon.load(directory)
        if not lexicon.is_available():
            raise FileNotFoundError(
                f"No supported landmark templates found in {directory}"
            )
        return cls(
            device="cpu",
            lexicon=lexicon,
            template_match_threshold=settings.template_match_threshold,
            manifest={},
        )

    def match_demo_landmarks(self, landmarks):
        return self.lexicon.match_raw(
            landmarks,
            threshold=self.template_match_threshold,
        )

    def provenance_summary(self) -> dict | None:
        return None

    def model_inventory(self) -> list[dict]:
        from app.services.pose_preprocessing import NOFACE_FEATURE_DIM

        return [
            {
                "name": "controlled_landmark_matcher",
                "task": "sign_to_text",
                "input_features": NOFACE_FEATURE_DIM,
                "checkpoint": "demo_prototypes/*.json",
                "status": f"loaded ({len(self.lexicon.inventory())} templates)",
                "caveat": (
                    "Live MediaPipe landmarks matched with DTW. "
                    "Templates are one reference sequence per supported sign."
                ),
            }
        ]
