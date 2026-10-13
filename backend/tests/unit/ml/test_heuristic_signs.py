"""Unit tests for the heuristic fallback recognizer -- pure numpy, no checkpoint."""
import numpy as np

from app.ml.inference.heuristic_signs import (
    FALLBACK_VOCABULARY,
    GLOSS_TEXT,
    HeuristicSignPipeline,
    classify_frame,
    classify_hand_shape,
    recognize_sequence,
)


def _make_hand(extended: set) -> np.ndarray:
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.5, 0.8, 0.0]
    defs = {
        "thumb": (2, 3, 4, 0.60, 0.70),
        "index": (5, 6, 8, 0.45, 0.70),
        "middle": (9, 10, 12, 0.50, 0.70),
        "ring": (13, 14, 16, 0.55, 0.70),
        "pinky": (17, 18, 20, 0.60, 0.72),
    }
    for name, (b, j, t, bx, by) in defs.items():
        if name in extended:
            hand[b] = [bx, by - 0.05, 0.0]
            hand[j] = [bx, by - 0.15, 0.0]
            hand[t] = [bx, by - 0.35, 0.0]
        else:
            hand[b] = [bx, by - 0.05, 0.0]
            hand[j] = [bx, by - 0.08, 0.0]
            hand[t] = [0.52, 0.76, 0.0]
    return hand


def _sequence(extended: set, n: int = 30) -> np.ndarray:
    frames = np.zeros((n, 543, 3), dtype=np.float32)
    frames[:, 522:543] = _make_hand(extended)
    return frames


def test_all_five_shapes_classify():
    cases = {
        "HELLO": {"thumb", "index", "middle", "ring", "pinky"},
        "NO": set(),
        "YES": {"thumb"},
        "YOU": {"index"},
        "HELP": {"index", "middle"},
    }
    for gloss, ext in cases.items():
        assert classify_hand_shape(_make_hand(ext)) == gloss


def test_missing_and_ambiguous_hands_rejected():
    assert classify_hand_shape(np.zeros((21, 3), dtype=np.float32)) is None
    assert classify_hand_shape(_make_hand({"ring"})) is None
    assert classify_hand_shape(_make_hand({"thumb", "pinky"})) is None


def _frame_with_hands(left_ext, right_ext) -> np.ndarray:
    frame = np.zeros((543, 3), dtype=np.float32)
    if left_ext is not None:
        frame[501:522] = _make_hand(left_ext)
    if right_ext is not None:
        frame[522:543] = _make_hand(right_ext)
    return frame


def test_supported_left_hand_shape_not_shadowed_by_unrecognized_right():
    # Right hand present but unclassifiable (ring only) must not veto a
    # supported shape on the left hand (fist = NO).
    frame = _frame_with_hands(set(), {"ring"})
    assert classify_frame(frame) == "NO"


def test_right_hand_keeps_priority_when_both_hands_supported():
    # Right fist + left open palm: right wins (existing priority preserved).
    frame = _frame_with_hands(
        {"thumb", "index", "middle", "ring", "pinky"}, set()
    )
    assert classify_frame(frame) == "NO"


def test_single_hand_frames_unchanged():
    assert classify_frame(_frame_with_hands(None, set())) == "NO"
    assert classify_frame(_frame_with_hands(set(), None)) == "NO"
    assert classify_frame(_frame_with_hands(None, None)) is None


def test_sequence_majority_vote_per_shape():
    for gloss, ext in {
        "HELLO": {"thumb", "index", "middle", "ring", "pinky"},
        "NO": set(),
        "YES": {"thumb"},
        "YOU": {"index"},
        "HELP": {"index", "middle"},
    }.items():
        label, agreement, _ = recognize_sequence(_sequence(ext))
        assert label == gloss
        assert agreement >= 0.6


def test_empty_sequence_reports_no_hand():
    label, agreement, reason = recognize_sequence(np.zeros((30, 543, 3), dtype=np.float32))
    assert label is None
    assert reason == "no_hand_detected"


def test_mixed_shapes_rejected_as_unrecognized():
    mixed = np.concatenate([
        _sequence({"thumb", "index", "middle", "ring", "pinky"}, 15),
        _sequence(set(), 15),
    ])
    label, agreement, reason = recognize_sequence(mixed)
    assert label is None
    assert reason == "unrecognized_shape"


def test_pipeline_transcribe_matches_adapter_shape():
    pipeline = HeuristicSignPipeline()
    result = pipeline.transcribe(_sequence({"thumb"}))
    assert result.accepted and result.text == "Yes"
    assert 0.0 <= result.confidence <= 1.0
    rejected = pipeline.transcribe(np.zeros((30, 543, 3), dtype=np.float32))
    assert not rejected.accepted and rejected.text == ""


def test_fallback_vocab_is_in_avatar_vocabulary():
    from app.services.avatar_gloss_service import load_vocabulary

    assert set(FALLBACK_VOCABULARY) <= set(load_vocabulary())
    assert set(FALLBACK_VOCABULARY) == set(GLOSS_TEXT)


def test_build_translation_service_uses_fallback_when_transformer_gated():
    from app.core.config import Settings
    from app.core.dependencies import build_translation_service
    from app.ml.inference.heuristic_signs import HeuristicSignPipeline

    settings = Settings(model_promoted=False, allow_unpromoted_dev=False)
    service = build_translation_service(settings)
    assert isinstance(service._pipeline, HeuristicSignPipeline)
    result = service.translate_sign_to_text(_sequence(set()))
    assert result.accepted and result.text == "No"
