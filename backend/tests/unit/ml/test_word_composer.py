"""Unit tests for the word-mode rule-based composer and LLM-output guard."""
from __future__ import annotations

from collections import Counter

import pytest

from app.ml.word_composer import (
    compose_fallback,
    guard_llm_output,
)

VOCAB = {
    "hello", "me", "you", "yes", "no", "help", "water", "where", "rest",
}


# --- Explicit patterns (rules first, deterministic) -----------------------

@pytest.mark.parametrize(
    "words, expected",
    [
        (["hello"], "Hello."),
        (["yes"], "Yes."),
        (["no"], "No."),
        (["water"], "Water."),
        (["where", "water"], "Where is the water?"),
        (["where", "you"], "Where are you?"),
        (["where", "me"], "Where am I?"),
        (["help", "me"], "Help me."),
        (["hello", "help"], "Hello, help."),
        (["hello", "water"], "Hello, water."),
    ],
)
def test_compose_fallback_explicit_patterns(words, expected):
    assert compose_fallback(words, VOCAB) == expected


def test_compose_fallback_single_word_any_vocab_word():
    assert compose_fallback(["help"], VOCAB) == "Help."
    assert compose_fallback(["where"], VOCAB) == "Where."


# --- Comma-join for anything without an explicit pattern -----------------

@pytest.mark.parametrize(
    "words, expected",
    [
        (["hello", "help", "water"], "Hello, help, water."),
        (["you", "help"], "You, help."),
        (["no", "water"], "No, water."),
        (["yes", "help", "me"], "Yes, help, me."),
        (["me", "where", "water"], "Me, where, water."),
        (["water", "hello", "yes"], "Water, hello, yes."),
    ],
)
def test_compose_fallback_comma_join(words, expected):
    assert compose_fallback(words, VOCAB) == expected


# --- Cleaning: rest, unknown words, consecutive duplicates, cap ----------

def test_compose_fallback_drops_rest_and_duplicates():
    assert compose_fallback(["rest", "water"], VOCAB) == "Water."
    assert compose_fallback(["hello", "hello", "water"], VOCAB) == "Hello, water."
    assert compose_fallback(["yes", "yes", "yes"], VOCAB) == "Yes."


def test_compose_fallback_filters_unknown():
    assert compose_fallback(["hello", "banana"], VOCAB) == "Hello."
    assert compose_fallback(["banana"], VOCAB) == ""


def test_compose_fallback_empty():
    assert compose_fallback([], VOCAB) == ""
    assert compose_fallback(["rest", "rest"], VOCAB) == ""


def test_compose_fallback_caps_at_eight_words():
    many = ["hello", "where", "you", "help", "me", "yes", "no", "water", "where"]
    sentence = compose_fallback(many, VOCAB)
    assert len(sentence.split(", ")) <= 8  # at most 8 recognised words


# --- Guard rejects the four previously-bad LLM outputs -----------------

@pytest.mark.parametrize(
    "output, words",
    [
        ("Hello is a.", ["hello"]),
        ("Yes is yes.", ["yes"]),
        ("You are you.", ["you"]),
        ("No help is the a.", ["no", "help"]),
    ],
)
def test_guard_rejects_previously_bad_outputs(output, words):
    assert guard_llm_output(output, words) is None


# --- Guard accepts faithful comma-join / rule output --------------------

@pytest.mark.parametrize(
    "output, words",
    [
        ("Hello, help, water.", ["hello", "help", "water"]),
        ("Where is the water?", ["where", "water"]),
        ("Help me.", ["help", "me"]),
        ("You, help.", ["you", "help"]),
    ],
)
def test_guard_accepts_faithful_output(output, words):
    assert guard_llm_output(output, words) == output


# --- Guard rejects: added content words, wrong counts/order, too long ---

@pytest.mark.parametrize(
    "output, words",
    [
        ("Hello, I need help getting water.", ["hello", "help", "water"]),  # need/getting
        ("There is no water available.", ["no", "water"]),  # there/available
        ("I am where water is.", ["me", "where", "water"]),  # me missing, order
        ("one two three four five six seven eight nine", ["hello"]),  # too long
        ("hello hello water.", ["hello", "water"]),  # immediate repeat
        ("Help the a.", ["help"]),  # "the a" forbidden bigram
        ("Water is a.", ["water"]),  # "a ." at end
        ("2 water.", ["water"]),  # digit
        ('He said "hi".', ["hello"]),  # quote
        ("# Water", ["water"]),  # markdown
    ],
)
def test_guard_rejects(output, words):
    assert guard_llm_output(output, words) is None


# --- Consecutive duplicates collapse before composing --------------------

def test_compose_fallback_collapses_consecutive_duplicates():
    assert compose_fallback(["water", "water"], VOCAB) == "Water."
    assert compose_fallback(["hello", "hello", "help"], VOCAB) == "Hello, help."


def test_gloss_composer_collapses_duplicates_rule_and_llm_paths(monkeypatch):
    """compose_words (REST /sentence/compose, auto mode, WebSocket) dedupes
    before the rule path AND before the LLM path + guard."""
    from app.core.config import Settings
    from app.services import gloss_composer

    settings = Settings(_env_file=None, groq_api_key=None)
    assert gloss_composer.compose_words(["water", "water"], VOCAB, settings) == ("Water.", "fallback")
    assert gloss_composer.compose_words(["hello", "hello", "help"], VOCAB, settings) == ("Hello, help.", "fallback")

    # LLM path: guard rejects "Water is water." ('water is water'), comma-join wins.
    monkeypatch.setenv("SIGNSAGE_GROQ_API_KEY", "test-key")
    monkeypatch.setattr(gloss_composer, "_groq_chat", lambda *a, **k: "Water is water.")
    llm_settings = Settings(_env_file=None, groq_api_key="test-key", groq_model="test-model")
    sentence, method = gloss_composer.compose_words(["water", "where"], VOCAB, llm_settings)
    assert sentence == "Water, where."
    assert method == "fallback"


@pytest.mark.parametrize(
    "output, words",
    [
        ("Water is water.", ["water", "where"]),
        ("water is water", ["water", "where"]),
        ("Where are where?", ["where", "water"]),
        ("water am water", ["water"]),
    ],
)
def test_guard_rejects_x_is_x(output, words):
    assert guard_llm_output(output, words) is None


def test_guard_rejects_content_word_count_overflow_vs_cleaned_input():
    # Input is cleaned upstream; even a legal-count-looking duplicate is rejected.
    assert guard_llm_output("Water is water.", ["water"]) is None
    assert guard_llm_output("Help help me.", ["help", "me"]) is None
