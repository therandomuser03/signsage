"""Session + adapter entry validation: incompatible representations must raise
clear ValueErrors (mapped to INVALID_MESSAGE at the transport boundary),
never flow silently into preprocessing or the model.
"""
import numpy as np
import pytest

from app.core.config import Settings
from app.ml.preprocessing.landmarks import TOTAL_LANDMARKS
from app.services.session_service import SequenceTooLongError, TranslationSession


def _settings() -> Settings:
    return Settings()


def _frame_rows(n: int) -> list:
    return [[[0.1, 0.2, 0.3]] * TOTAL_LANDMARKS for _ in range(n)]


def test_take_sequence_returns_canonical_shape():
    session = TranslationSession(session_id="s", settings=_settings())
    for frame in _frame_rows(5):
        session.add_frame(frame)
    arr = session.take_sequence()
    assert arr.shape == (5, TOTAL_LANDMARKS, 3)
    assert arr.dtype == np.float32


def test_take_sequence_rejects_empty_buffer_loudly():
    session = TranslationSession(session_id="s", settings=_settings())
    with pytest.raises(ValueError):
        session.take_sequence()


def test_window_rejects_ragged_buffer_loudly():
    session = TranslationSession(session_id="s", settings=_settings())
    for frame in _frame_rows(3):
        session.add_frame(frame)
    session._frames.append([[0.0, 0.0]])  # corrupt a frame post-validation
    with pytest.raises(ValueError):
        session.take_sequence()


def test_add_frame_enforces_buffer_ceiling():
    settings = _settings()
    session = TranslationSession(session_id="s", settings=settings)
    cap = settings.max_frames_per_sequence * 4
    for frame in _frame_rows(cap):
        session.add_frame(frame)
    with pytest.raises(SequenceTooLongError):
        session.add_frame(_frame_rows(1)[0])


def test_rest_schema_rejects_too_many_frames():
    from app.schemas.translation import SignToTextRequest
    from pydantic import ValidationError

    frames = [{"landmarks": [[0.0, 0.0, 0.0]] * TOTAL_LANDMARKS} for _ in range(641)]
    with pytest.raises(ValidationError):
        SignToTextRequest(frames=frames)


def test_rest_schema_rejects_too_few_frames():
    from app.schemas.translation import SignToTextRequest
    from pydantic import ValidationError

    frames = [{"landmarks": [[0.0, 0.0, 0.0]] * TOTAL_LANDMARKS}]
    with pytest.raises(ValidationError):
        SignToTextRequest(frames=frames)


def test_adapter_transcribe_rejects_incompatible_input():
    from types import SimpleNamespace

    from app.ml.adapters.sign_to_text_adapter import SignToTextAdapter

    contract = SimpleNamespace(model=SimpleNamespace(max_frames=160))
    adapter = SignToTextAdapter(model=None, tokenizer=None, contract=contract, device="cpu")
    with pytest.raises(ValueError):
        adapter.transcribe(np.zeros((5, 10, 3), dtype=np.float32))  # wrong count
    with pytest.raises(ValueError):
        adapter.transcribe(np.zeros((1, TOTAL_LANDMARKS, 3), dtype=np.float32))  # 1 frame
    with pytest.raises(ValueError):
        adapter.transcribe(np.zeros((5, TOTAL_LANDMARKS, 3), dtype=bool))  # bool dtype
