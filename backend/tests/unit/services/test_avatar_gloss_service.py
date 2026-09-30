"""Unit tests for the deterministic avatar-gloss stage (no network, no LLM keys)."""
import pytest

from app.core.config import Settings
from app.services.avatar_gloss_service import _deterministic_gloss, generate_gloss, load_vocabulary


def _settings():
    return Settings(groq_api_key=None, ollama_base_url=None)


async def test_multiword_sign_matches_as_one_gloss():
    r = await generate_gloss("Hello Thank You", _settings())
    assert r.gloss == ["HELLO", "THANK_YOU"]
    assert r.source == "deterministic"


async def test_unknown_words_are_reported_not_guessed():
    r = await generate_gloss("Hello Xyzzy World", _settings())
    assert r.gloss == ["HELLO"]
    assert r.unmatched_words == ["Xyzzy", "World"]


async def test_single_word_sign_still_matches_outside_a_phrase():
    r = await generate_gloss("How are you", _settings())
    assert r.gloss == ["HOW", "YOU"]
    assert r.unmatched_words == ["are"]


async def test_nothing_matches_gives_empty_gloss():
    r = await generate_gloss("Xyzzy Qwerty", _settings())
    assert r.gloss == [] and r.unmatched_words == ["Xyzzy", "Qwerty"]


def test_vocabulary_loads_from_packaged_data_file():
    vocab = load_vocabulary(_settings())
    assert {"HELLO", "THANK_YOU", "YOU"} <= vocab


def test_custom_vocabulary_is_respected():
    gloss, unmatched = _deterministic_gloss("good morning friend", frozenset({"GOOD_MORNING"}))
    assert gloss == ["GOOD_MORNING"] and unmatched == ["friend"]
