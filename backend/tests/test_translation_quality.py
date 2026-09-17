from app.services.translation_quality import is_attractor, score_text, ATTRACTOR_PHRASES


def test_attractor_phrases_detected():
    """All known collapsed attractor outputs should be detected."""
    for phrase in ATTRACTOR_PHRASES:
        assert is_attractor(phrase), f"Expected {phrase!r} to be detected as attractor"


def test_non_attractor_passes_gate():
    """Normal sign outputs should not be attractors."""
    assert not is_attractor("hello")
    assert not is_attractor("thank you")
    assert not is_attractor("how are you")


def test_attractor_rejected_by_score_text():
    """score_text should reject attractor outputs."""
    for phrase in ATTRACTOR_PHRASES:
        trusted, reason, conf = score_text(phrase)
        assert not trusted, f"Expected attractor {phrase!r} to be untrusted"
        assert reason == "attractor"


def test_normal_text_trusted():
    """Normal sign language outputs should be trusted."""
    trusted, reason, conf = score_text("hello there")
    assert trusted is True
    assert reason == "ok"


def test_low_motion_rejected():
    """Very low motion energy should reject the output."""
    trusted, reason, conf = score_text("hello", motion_energy=0.001)
    assert trusted is False
    assert reason == "low_motion"


def test_empty_text():
    """Empty text should be rejected."""
    trusted, reason, conf = score_text("")
    assert not trusted
    assert reason == "empty"


def test_unk_rate_too_high():
    """High unk_rate should reject the output."""
    trusted, reason, conf = score_text("_TYPES _TYPES _TYPES _TYPES", unk_token="_TYPES")
    assert trusted is False
    assert reason == "unk_rate"