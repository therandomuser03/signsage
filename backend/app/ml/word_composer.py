"""Rule-based sentence composer + LLM-output guard for word mode.

Rules run FIRST and are deterministic. The vocabulary is read from the
model's meta.json (passed in by the caller) and is never hardcoded here.

Groq is used only as a refinement for 2+ word sequences that match no
explicit rule, and its output is strictly guarded; anything that fails the
guard falls back to a plain comma-join of the recognised words.
"""
from __future__ import annotations

import re
from typing import Iterable

_FUNCTION_WORDS = frozenset({"is", "are", "am", "the", "a", "i"})
# Forbidden adjacent token pairs / endings the LLM must never produce.
_FORBIDDEN_BIGRAMS = {("is", "a"), ("a", "the"), ("the", "a"), ("is", "is")}
_MAX_WORDS = 8  # cap on recognised words fed to the composer
_MAX_OUTPUT_WORDS = None  # set to len(input) + 3 at guard time
_HAS_DIGIT = re.compile(r"\d")
_MARKDOWN_OR_QUOTES = re.compile(r"[\"'`*_#\[\]\(\)<>\\/@$%&+={}|~]")
_WORD_TOKENS = re.compile(r"[A-Za-z]+")
_A_DOT_AT_END = re.compile(r"\ba\.\s*$")


def _first_cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _clean_words(words: Iterable[str], vocab: set[str]) -> list[str]:
    """Drop ``rest``, unknown words and consecutive duplicates; cap length."""
    cleaned: list[str] = []
    for word in words:
        if word == "rest" or word not in vocab:
            continue
        if cleaned and cleaned[-1] == word:
            continue
        cleaned.append(word)
        if len(cleaned) >= _MAX_WORDS:
            break
    return cleaned


def _rule_sentence(cleaned: list[str]) -> str | None:
    """Return the deterministic sentence for an explicit pattern, else None.

    Patterns stay faithful to the signed words and add only function words
    (is, are, am, the, a) plus punctuation. Never new facts or content verbs.
    """
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f"{_first_cap(cleaned[0])}."
    w = cleaned
    if w == ["where", "water"]:
        return "Where is the water?"
    if w == ["where", "you"]:
        return "Where are you?"
    if w == ["where", "me"]:
        return "Where am I?"
    if w == ["help", "me"]:
        return "Help me."
    if len(w) == 2 and w[0] == "hello":
        return f"Hello, {w[1]}."  # X stays as in the vocabulary (mid-sentence)
    return None


def _comma_join(cleaned: list[str]) -> str:
    """Words joined by commas, capitalised first word, final period.

    Adds NO function words. Used for any combination without an explicit
    pattern, and as the LLM guard fallback.
    """
    if not cleaned:
        return ""
    sentence = _first_cap(", ".join(cleaned))
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def compose_fallback(words: list[str], vocab: set[str]) -> str:
    """Deterministic sentence from recognised words (no LLM). Rules first."""
    cleaned = _clean_words(words, vocab)
    ruled = _rule_sentence(cleaned)
    if ruled is not None:
        return ruled
    return _comma_join(cleaned)


def guard_llm_output(output: str, input_words: list[str]) -> str | None:
    """Return `output` if safe to present, else None (use the comma-join).

    All must hold, else None:
      * non-empty; no digits, quotes or markdown
      * every output token is an input word or a function word
        ({is, are, am, the, a, I})
      * every input word appears in the output, in the same order and with
        the same count
      * at most len(input) + 3 output words
      * no word immediately repeated in the output
      * none of: "is a", "a the", "the a", "is is", "a ."
    """
    text = (output or "").strip()
    if not text:
        return None
    if _HAS_DIGIT.search(text) or _MARKDOWN_OR_QUOTES.search(text):
        return None
    tokens = _WORD_TOKENS.findall(text)
    if not tokens:
        return None
    low = [t.lower() for t in tokens]
    allowed = {w.lower() for w in input_words} | _FUNCTION_WORDS
    if any(t not in allowed for t in low):
        return None
    # Input word budget.
    if len(low) > len(input_words) + 3:
        return None
    # Every input word in the output, same order, same count.
    input_low = [w.lower() for w in input_words]
    from collections import Counter

    input_count = Counter(input_low)
    output_count = Counter(low)
    for word, needed in input_count.items():
        if output_count.get(word, 0) != needed:
            return None
    # Order: input must be a subsequence of the output tokens.
    it = iter(low)
    if not all(any(t == w for t in it) for w in input_low):
        return None
    # No word immediately repeated.
    if any(low[i] == low[i + 1] for i in range(len(low) - 1)):
        return None
    # Forbidden bigrams / endings.
    for i in range(len(low) - 1):
        if (low[i], low[i + 1]) in _FORBIDDEN_BIGRAMS:
            return None
    if _A_DOT_AT_END.search(text):
        return None
    return text
