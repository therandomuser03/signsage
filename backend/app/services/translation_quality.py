from __future__ import annotations

import re


# Exact collapsed attractors observed from Phase 0 + Image 1. Lowercased, stripped punctuation.
ATTRACTOR_PHRASES = {
    "this is why sacrifices.",
    "this is why sacrifices",
    "this is why they were very dangerous.",
    "this is why they were very dangerous",
    "let me tell you about it.",
    "let me tell you about it",
}

# Extra junk that tokenizer may produce when collapsed
_ATTRACTOR_TOKENS = {"sacrifices.", "sacrifices"}

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def is_attractor(text: str) -> bool:
    n = _normalize(text)
    if n in ATTRACTOR_PHRASES:
        return True
    # also block any output that contains the attractor token alone as strong signal
    # e.g. "this is why sacrifices. this is why sacrifices." would have been trigram-blocked but still check
    tokens = n.split()
    if tokens and tokens[-1] in _ATTRACTOR_TOKENS and len(tokens) <= 5 and "this is why" in n:
        return True
    return False

def repetition_density(text: str) -> float:
    tokens = _normalize(text).split()
    if len(tokens) < 3:
        return 0.0
    # fraction of trigrams that are duplicate
    seen = set()
    dups = 0
    for i in range(len(tokens) - 2):
        tri = tuple(tokens[i : i + 3])
        if tri in seen:
            dups += 1
        else:
            seen.add(tri)
    return dups / max(1, len(tokens) - 2)

def unk_rate(text: str, unk_token: str = "<unk>") -> float:
    tokens = text.split()
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if t == unk_token) / len(tokens)

def score_text(
    text: str,
    *,
    unk_token: str = "<unk>",
    motion_energy: float | None = None,
) -> tuple[bool, str, float]:
    """Return (is_trusted, reason, confidence).

    is_trusted False means caller should fallback to demo lexicon / honest placeholder.
    Confidence is in [0,1] for UI.
    """
    n = text.strip()
    if not n:
        return False, "empty", 0.0
    if is_attractor(n):
        return False, "attractor", 0.05
    # very short single-word outputs from collapsed model are not useful for continuous ISL
    tokens = _normalize(n).split()
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False, "too_short", 0.2
    if unk_rate(n, unk_token) > 0.3:
        return False, "unk_rate", 0.3
    if repetition_density(n) > 0.25:
        return False, "repetition", 0.25
    # low motion energy -> holder not signing, don't trust spurious output
    if motion_energy is not None and motion_energy < 0.005:
        return False, "low_motion", 0.2
    # otherwise trusted, confidence heuristic
    # longer diverse phrase gets higher confidence
    conf = min(0.95, 0.55 + 0.06 * len(tokens) - 0.4 * repetition_density(n))
    return True, "ok", float(max(0.0, min(1.0, conf)))
