import numpy as np
import pytest

from app.services.session import SessionError, TranslationSession


def _frame(value: float = 0.0) -> np.ndarray:
    return np.full((543, 3), value, dtype=np.float32)


def test_session_id_locks_in_on_first_use():
    session = TranslationSession()
    session.bind_session_id("abc")
    assert session.session_id == "abc"


def test_session_id_change_is_rejected():
    session = TranslationSession()
    session.bind_session_id("abc")
    with pytest.raises(SessionError):
        session.bind_session_id("xyz")


def test_empty_session_id_is_rejected():
    session = TranslationSession()
    with pytest.raises(SessionError):
        session.bind_session_id("")


def test_take_sentence_stacks_buffered_frames_in_order():
    session = TranslationSession()
    session.add_frame(_frame(1.0), max_buffered_frames=10)
    session.add_frame(_frame(2.0), max_buffered_frames=10)
    sequence = session.take_sentence()
    assert sequence.shape == (2, 543, 3)
    assert np.allclose(sequence[0], 1.0)
    assert np.allclose(sequence[1], 2.0)


def test_take_sentence_clears_the_buffer():
    session = TranslationSession()
    session.add_frame(_frame(), max_buffered_frames=10)
    session.take_sentence()
    assert session.buffered_frame_count == 0
    with pytest.raises(SessionError):
        session.take_sentence()


def test_take_sentence_without_frames_is_rejected():
    session = TranslationSession()
    with pytest.raises(SessionError):
        session.take_sentence()


def test_frame_buffer_overflow_is_rejected():
    session = TranslationSession()
    session.add_frame(_frame(), max_buffered_frames=1)
    with pytest.raises(SessionError):
        session.add_frame(_frame(), max_buffered_frames=1)


def test_turn_counters_increment_independently():
    """Person A (signing) and Person B (typing) share one connection but
    are independent turn streams -- see the class docstring."""
    session = TranslationSession()
    session.add_frame(_frame(), max_buffered_frames=10)
    session.take_sentence()
    session.record_text_to_pose_turn()
    session.record_text_to_pose_turn()

    assert session.sign_to_text_turns == 1
    assert session.text_to_pose_turns == 2


def test_snapshot_is_plain_data():
    session = TranslationSession()
    session.bind_session_id("abc")
    snap = session.snapshot()
    assert snap["session_id"] == "abc"
    assert snap["buffered_frames"] == 0
