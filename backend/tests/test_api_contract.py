import numpy as np
from pydantic import ValidationError

from app.schemas.api import (
    LandmarksFrame,
    SentenceBoundary,
    TextInput,
    TranslationPose,
    TranslationText,
    ErrorMessage,
)


def test_landmarks_frame_contract():
    msg = LandmarksFrame(
        type="landmarks.frame",
        session_id="abc",
        timestamp_ms=123,
        landmarks=[0.0] * 1629,
    )
    assert len(msg.landmarks) == 1629


def test_boundary_contract():
    msg = SentenceBoundary(
        type="sentence.boundary",
        session_id="abc",
        trigger="manual",
    )
    assert msg.trigger == "manual"


def test_text_input_contract():
    msg = TextInput(type="text.input", session_id="abc", text="hello")
    assert msg.text == "hello"


def test_server_contracts():
    text = TranslationText(session_id="abc", text="hello", latency_ms=12.3)
    pose = TranslationPose(
        session_id="abc",
        gloss="HELLO",
        frames=[[0.0] * 1629] * 60,
        fps=25,
        latency_ms=45.6,
    )
    err = ErrorMessage(
        session_id="abc",
        message="bad",
        code="INVALID_MESSAGE",
    )
    assert text.type == "translation.text"
    assert pose.type == "translation.pose"
    assert err.type == "error"


def test_bad_landmark_length_rejected():
    try:
        LandmarksFrame(
            type="landmarks.frame",
            session_id="abc",
            timestamp_ms=0,
            landmarks=[0.0] * 1628,
        )
    except ValidationError:
        return
    raise AssertionError("Expected ValidationError")
