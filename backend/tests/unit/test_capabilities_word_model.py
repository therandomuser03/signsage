"""/capabilities: word model version comes from meta.json, never a hardcoded literal."""
import json

from app.api.routes.capabilities import _word_model_version, capabilities
from app.core.config import Settings


def _settings(tmp_path=None, **overrides):
    base = dict(
        word_mode_enabled=True,
        word_model_dir=str(tmp_path) if tmp_path is not None else "models/word_v3",
    )
    base.update(overrides)
    return Settings(**base)


def _write_meta(tmp_path, payload):
    (tmp_path / "meta.json").write_text(json.dumps(payload), encoding="utf-8")


def test_model_version_is_read_from_meta_json(tmp_path):
    _write_meta(tmp_path, {"model_version": "word_vX_test_s000"})
    assert _word_model_version(_settings(tmp_path)) == "word_vX_test_s000"


def test_capabilities_reports_the_meta_json_version_not_a_literal(tmp_path):
    _write_meta(tmp_path, {"model_version": "word_vX_test_s000"})
    caps = capabilities(_settings(tmp_path))
    assert caps.word_recognition_model_version == "word_vX_test_s000"
    assert caps.word_recognition_model_version != "word_v3_8w_s123"


def test_model_version_none_when_word_mode_disabled(tmp_path):
    _write_meta(tmp_path, {"model_version": "word_vX_test_s000"})
    assert _word_model_version(_settings(tmp_path, word_mode_enabled=False)) is None
    assert capabilities(_settings(tmp_path, word_mode_enabled=False)).word_recognition_model_version is None


def test_model_version_none_when_meta_missing_or_invalid(tmp_path):
    assert _word_model_version(_settings(tmp_path)) is None  # no meta.json at all
    (tmp_path / "meta.json").write_text("{not json", encoding="utf-8")
    assert _word_model_version(_settings(tmp_path)) is None
    _write_meta(tmp_path, {"note": "no model_version field"})
    assert _word_model_version(_settings(tmp_path)) is None
