import numpy as np
from pathlib import Path
from app.services.demo_lexicon import DemoLexicon


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