"""Resolves where the Sign->Text deployment bundle actually lives on disk.

Deliberately the ONLY place that knows about SIGNSAGE_MODEL_BUNDLE_DIR / the
research/artifacts layout, so the rest of the backend never touches
filesystem paths directly.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


class ModelBundleNotConfiguredError(RuntimeError):
    pass


class ModelBundleNotFoundError(RuntimeError):
    pass


def resolve_bundle_dir(settings: Settings) -> Path:
    if not settings.model_bundle_dir:
        raise ModelBundleNotConfiguredError(
            "SIGNSAGE_MODEL_BUNDLE_DIR is not set. Point it at a "
            "signsage_deployment_bundle/ directory (see "
            "research/manifests/models/*.yaml for available runs)."
        )
    bundle_dir = Path(settings.model_bundle_dir)
    if not bundle_dir.is_dir():
        raise ModelBundleNotFoundError(f"Model bundle directory not found: {bundle_dir}")
    if settings.model_kind == "isolated_words":
        required = ["model.pt", "metadata.json", "labels.json", "evaluation_results.json"]
    else:
        required = ["checkpoint", "tokenizer", "landmark_contract.json", "sign_to_text_deployment.json"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise ModelBundleNotFoundError(
            f"Model bundle at {bundle_dir} is missing required entries: {missing}"
        )
    return bundle_dir
