import numpy as np
from pathlib import Path
from app.services.demo_lexicon import DemoLexicon, match_demo_phrase
from app.services.pose_preprocessing import FULL_FEATURE_DIM, TRUE_HAND_FEATURE_START, TRUE_HAND_FEATURE_DIM


ROOT = Path(__file__).parents[1]


def test_lexicon_loads():
    """Demo lexicon should load from prototypes directory."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    assert lex.is_available()
    assert len(lex.inventory()) > 0


def test_lexicon_inventory_phrases():
    """Lexicon should contain expected phrases."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    phrases = lex.inventory()
    expected = {"Hello", "Thank you", "Yes", "No", "Help", "Please", "Sorry", "Good morning",
                "How are you", "I need help"}
    for phrase in expected:
        assert phrase in phrases, f"Expected {phrase!r} in lexicon, got {phrases}"


def test_dtw_match_same_phrase():
    """DTW should match a prototype to itself (distance = 0)."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    hello_proto = lex.prototypes["Hello"]
    # Perturb slightly and should still match
    seq = hello_proto.copy()
    seq[0, 0] += 0.1
    matched, dist = lex.match(seq, threshold=420.0)
    assert matched == "Hello"
    assert dist < 420.0


def test_dtw_match_hello():
    """DTW should match hello prototype correctly."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    # Exact match should work
    matched, dist = lex.match(lex.prototypes["Hello"].copy(), threshold=420.0)
    assert matched == "Hello"


def test_dtw_unmatched_returns_none():
    """DTW should return None for completely unmatched sequences when above threshold."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    seq = np.ones((60, 225), dtype=np.float64) * 100.0
    matched, dist = lex.match(seq, threshold=1.0)  # very strict threshold
    assert matched is None


def test_dtw_distance_scale():
    """Different phrases should have different DTW distances."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    # Use threshold 420 for both calls
    d1, _ = lex.match(lex.prototypes["Hello"].astype(np.float64), threshold=420.0)
    # Get distance by matching against hello prototype
    hello_proto = lex.prototypes["Hello"].astype(np.float64)
    good_morning_proto = lex.prototypes["Good morning"].astype(np.float64)
    # Compute DTW manually for distance comparison
    a = hello_proto
    b = good_morning_proto
    n, m = a.shape[0], b.shape[0]
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    dist_hello_gm = float(dtw[n, m] / (n + m))
    
    # Same phrase distance
    a = hello_proto
    b = hello_proto
    n, m = a.shape[0], b.shape[0]
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    dist_hello_hello = float(dtw[n, m] / (n + m))
    
    # Same-phrase distance should be much smaller
    assert dist_hello_hello < dist_hello_gm

def test_match_demo_phrase_exact_and_alias():
    """English text should map onto the bounded phrase set, exact or via alias."""
    available = ["Hello", "Thank you", "Yes", "No", "Help", "Please", "Sorry",
                 "Good morning", "How are you", "I need help"]
    assert match_demo_phrase("Hello", available) == "Hello"
    assert match_demo_phrase("hi", available) == "Hello"
    assert match_demo_phrase("Hi!", available) == "Hello"
    assert match_demo_phrase("thanks", available) == "Thank you"
    assert match_demo_phrase("How are you?", available) == "How are you"
    assert match_demo_phrase("good morning", available) == "Good morning"
    assert match_demo_phrase("I need help", available) == "I need help"


def test_match_demo_phrase_no_match_for_arbitrary_text():
    """Arbitrary/unsupported English text must not silently match a phrase."""
    available = ["Hello", "Thank you", "Yes", "No", "Help"]
    assert match_demo_phrase("Where is the nearest hospital", available) is None
    assert match_demo_phrase("", available) is None


def test_get_render_sequence_shape_and_hand_placement():
    """Render sequence embeds real hand motion at the correct feature offset
    and leaves pose/face at zero (see AUDIT_NOTES.md)."""
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    render = lex.get_render_sequence("Hello")
    assert render is not None
    assert render.shape == (60, FULL_FEATURE_DIM)
    # Pose/face region (everything before the hand block) must be zero.
    assert np.allclose(render[:, :TRUE_HAND_FEATURE_START], 0.0)
    # Hand region must be non-trivial (real captured motion).
    hand_block = render[:, TRUE_HAND_FEATURE_START:TRUE_HAND_FEATURE_START + TRUE_HAND_FEATURE_DIM]
    assert np.abs(hand_block).sum() > 0.0
    # Nothing populated past the hand block.
    assert np.allclose(render[:, TRUE_HAND_FEATURE_START + TRUE_HAND_FEATURE_DIM:], 0.0)


def test_get_render_sequence_unknown_phrase_returns_none():
    lex = DemoLexicon.load(ROOT / "demo_prototypes")
    assert lex.get_render_sequence("Not A Real Phrase") is None
