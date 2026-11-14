"""Tests for resolve_word_model_dir (word model directory resolution).

The resolver lives in app.ml.word_classifier_adapter so both the REST word
routes and the model registry can use it. (The routes keep a thin alias
`_resolve_word_model_dir` for compatibility.)

Resolution order:
  1. absolute SIGNSAGE_WORD_MODEL_DIR
  2. <backend root>/word_model      (baked into the image next to app/)
  3. <repo root>/models/word_v3     (local dev layout)
  4. <cwd>/<word_model_dir>         (custom relative staging)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.ml.word_classifier_adapter import resolve_word_model_dir
import app.ml.word_classifier_adapter as wca


def _fake_module_file(monkeypatch, path: str) -> None:
    monkeypatch.setattr(wca, "__file__", path, raising=False)


def test_absolute_env_var_wins(monkeypatch, tmp_path):
    target = tmp_path / "staged" / "word_v3"
    target.mkdir(parents=True)
    monkeypatch.setenv("SIGNSAGE_WORD_MODEL_DIR", str(target))
    _fake_module_file(monkeypatch, "/no/such/repo/backend/app/ml/word_classifier_adapter.py")
    assert resolve_word_model_dir(Settings()) == str(target)


def test_backend_word_model_wins_over_repo_and_cwd(monkeypatch, tmp_path):
    # Image layout: file under tmp/backend/app/..., model baked at
    # tmp/backend/word_model (next to app/). Neither repo-root models/ nor
    # cwd have it; backend/word_model must win.
    backend_root = tmp_path / "backend"   # == parents[3] of the faked __file__
    (backend_root / "app" / "ml").mkdir(parents=True)
    (backend_root / "word_model").mkdir()
    monkeypatch.delenv("SIGNSAGE_WORD_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # cwd has no models/word_v3
    _fake_module_file(monkeypatch, str(backend_root / "app" / "ml" / "word_classifier_adapter.py"))
    assert resolve_word_model_dir(Settings()) == str(backend_root / "word_model")


def test_repo_root_models_beats_cwd(monkeypatch, tmp_path):
    # Local dev layout: repo-root models/word_v3 exists, cwd is empty.
    repo_root = tmp_path
    (repo_root / "models" / "word_v3").mkdir(parents=True)
    (repo_root / "backend" / "app" / "ml").mkdir(parents=True)
    empty_cwd = tmp_path / "empty_cwd"
    empty_cwd.mkdir()
    monkeypatch.delenv("SIGNSAGE_WORD_MODEL_DIR", raising=False)
    monkeypatch.chdir(empty_cwd)  # cwd has no models/word_v3
    _fake_module_file(monkeypatch, str(repo_root / "backend" / "app" / "ml" / "word_classifier_adapter.py"))
    assert resolve_word_model_dir(Settings()) == str(repo_root / "models" / "word_v3")


def test_cwd_candidate_when_nothing_else_staged(monkeypatch, tmp_path):
    # Custom relative staging at the working dir wins as last resort.
    target = tmp_path / "models" / "word_v3"
    target.mkdir(parents=True)
    (tmp_path / "backend" / "app" / "ml").mkdir(parents=True)
    monkeypatch.delenv("SIGNSAGE_WORD_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _fake_module_file(monkeypatch, str(tmp_path / "backend" / "app" / "ml" / "word_classifier_adapter.py"))
    assert resolve_word_model_dir(Settings()) == str(target)


def test_nothing_staged_falls_back_to_backend_word_model(monkeypatch, tmp_path):
    # Nothing staged anywhere: deterministic fallback path (startup then
    # fails loudly with a clear missing-file error).
    repo_root = tmp_path
    (repo_root / "backend" / "app" / "ml").mkdir(parents=True)
    monkeypatch.delenv("SIGNSAGE_WORD_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _fake_module_file(monkeypatch, str(repo_root / "backend" / "app" / "ml" / "word_classifier_adapter.py"))
    assert resolve_word_model_dir(Settings()) == str(repo_root / "backend" / "word_model")


def test_default_value_is_models_word_v3(monkeypatch):
    monkeypatch.delenv("SIGNSAGE_WORD_MODEL_DIR", raising=False)
    assert Settings().word_model_dir == "models/word_v3"
