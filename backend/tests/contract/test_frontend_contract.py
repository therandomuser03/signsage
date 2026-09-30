"""Backend <-> frontend contract. Fails when either side drifts from the other.

Skipped when the frontend/ folder isn't next to backend/ (e.g. backend-only CI).
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"
pytestmark = pytest.mark.skipif(not FRONTEND.is_dir(), reason="frontend/ not present next to backend/")


def _read(rel: str) -> str:
    return (FRONTEND / rel).read_text(encoding="utf-8")


def test_avatar_vocabulary_matches_the_frontend_manifest():
    from app.services.avatar_gloss_service import load_vocabulary

    manifest = json.loads(_read("public/models/avatar/manifest.json"))
    assert set(manifest) == set(load_vocabulary()), "sign_vocabulary.json and the avatar manifest must list the same signs"
    for gloss_id, entry in manifest.items():
        assert entry["id"] == gloss_id


def test_every_frontend_endpoint_exists_in_the_openapi_schema():
    from app.main import app

    openapi_paths = set(app.openapi()["paths"])
    literals = re.findall(r"[\"'`](/api/v1[^\"'`$]*)", _read("src/services/api/endpoints.ts"))
    assert literals, "no endpoints found in endpoints.ts"
    for path in literals:
        if path.endswith("/"):  # template like /api/v1/history/${id}
            assert any(p.startswith(path) and "{" in p for p in openapi_paths), f"{path} has no parametrised backend route"
        else:
            assert path in openapi_paths, f"frontend calls {path} but the backend has no such route"


def test_websocket_path_matches():
    from app.main import app

    ws_paths = {getattr(r, "path", None) for r in app.routes}
    ws_path = re.search(r'WS_PATH\s*=\s*"([^"]+)"', _read("src/services/api/endpoints.ts")).group(1)
    assert ws_path == "/ws/translate"
    assert any("/ws/translate" in str(r) for r in app.routes) or ws_path in ws_paths


def test_websocket_message_types_and_error_codes_match():
    from app.schemas import websocket as backend

    proto = _read("src/services/websocket/protocol.ts")
    backend_types = {m.model_fields["type"].default for m in (
        backend.AuthMessage, backend.LandmarksFrameMessage, backend.SentenceBoundaryMessage,
        backend.AuthOkMessage, backend.TranslationTextMessage, backend.ErrorMessage)}
    for t in backend_types:
        assert f'"{t}"' in proto, f"protocol.ts is missing message type {t}"
    codes = {v for k, v in vars(backend.ErrorCode).items() if not k.startswith("_")}
    for code in codes:
        assert f'"{code}"' in proto, f"protocol.ts is missing error code {code}"


def test_landmark_constants_match():
    from app.ml.preprocessing.landmarks import COORDINATES_PER_LANDMARK, TOTAL_LANDMARKS

    proto = _read("src/services/websocket/protocol.ts")
    assert int(re.search(r"LANDMARK_COUNT\s*=\s*(\d+)", proto).group(1)) == TOTAL_LANDMARKS
    assert int(re.search(r"COORDINATES_PER_LANDMARK\s*=\s*(\d+)", proto).group(1)) == COORDINATES_PER_LANDMARK


@pytest.mark.parametrize("model_path", [
    "app.schemas.common:HealthResponse", "app.schemas.common:CapabilitiesResponse",
    "app.schemas.common:ModelInfo", "app.schemas.auth:MeResponse",
    "app.schemas.avatar_gloss:AvatarGlossRequest", "app.schemas.avatar_gloss:AvatarGlossResponse",
    "app.schemas.avatar_gloss:AvatarVocabularyResponse",
    "app.schemas.translation:SignToTextRequest", "app.schemas.translation:SignToTextResponse",
    "app.schemas.translation:LandmarkFramePayload",
    "app.schemas.history:HistoryItem", "app.schemas.history:HistoryListResponse", "app.schemas.history:DeletedResponse",
    "app.schemas.common:ErrorResponse",
])
def test_every_backend_schema_field_is_declared_in_frontend_types(model_path):
    import importlib

    module_name, class_name = model_path.split(":")
    model = getattr(importlib.import_module(module_name), class_name)
    api_ts = _read("src/types/api.ts")
    for field in model.model_fields:
        assert re.search(rf"\b{field}\??:", api_ts), f"{class_name}.{field} is missing from frontend/src/types/api.ts"
