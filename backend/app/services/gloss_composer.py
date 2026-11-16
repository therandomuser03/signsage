"""Shared gloss→sentence composer for ordered word sequences.

Used by:
  * the /api/v1/translation/compose route (REST)
  * the continuous sign pipeline output (WS + /api/v1/translation/sentence)

Order: deterministic rules -> Groq refinement (only when no rule matches and
Groq is configured) -> deterministic comma-join fallback.

The LLM is constrained to the signed words: every input word must appear in
its output exactly once, only a small set of function words may be added,
and anything else is rejected by the guard (app.ml.word_composer) and falls
back to the comma-join. Contents of the request are never logged.
"""
from __future__ import annotations

import logging

import httpx

from app.ml.word_composer import clean_words, compose_fallback, guard_llm_output

logger = logging.getLogger(__name__)

# Retry budget: one attempt with the configured model, and one with the
# module default documented as working on this account, on either a hard
# failure (e.g. the configured SIGNSAGE_GROQ_MODEL was removed or the account
# lost access) or a guard rejection.
_GROQ_FALLBACK_MODEL = "allam-2-7b"

# gpt-oss / qwen chat models are reasoning models: they consume most of a
# small max_tokens budget on hidden reasoning and return empty content.
# Ask for minimal reasoning and leave headroom.
_REASONING_TAGS = ("gpt-oss", "qwen")

_SYSTEM_PROMPT = (
    "You are an English sentence composer for Indian Sign Language gloss sequences. "
    "The input is a comma-separated list of isolated signed words, in the order the "
    "signer produced them. Write ONE short English sentence. Rules: "
    "(1) Every signed word must appear in your sentence exactly once, spelled the same "
    "(case-insensitive). "
    "(2) You may add ONLY these function words: is, are, am, the, a, i, can, could. "
    "(3) You may reorder words only as needed for grammatical English. "
    "(4) Do not add any other words, names, facts or details. "
    "(5) If the sequence has no coherent literal reading, output the signed words "
    "separated by commas, unchanged. "
    "Reply with only the sentence, nothing else."
)


def _groq_chat(settings, model: str, system: str, user: str) -> str:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        # Reasoning models need headroom: reasoning tokens count toward
        # completion tokens. 256 covers the reasoning + a short sentence.
        "max_tokens": 256,
    }
    if any(tag in model for tag in _REASONING_TAGS):
        body["reasoning_effort"] = "low"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=4.0) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
    return (resp.json()["choices"][0]["message"].get("content") or "").strip()


def compose_words(words: list[str], vocab: set[str], settings) -> tuple[str, str]:
    """Compose a sentence from recognised words.

    Returns (sentence, method) where method is "llm" or "fallback".
    """
    # Collapse consecutive duplicate words BEFORE any path runs — the rule
    # path, the LLM path and the guard all see the deduped sequence.
    cleaned = clean_words(words, vocab)
    if not cleaned:
        return "", "fallback"

    # --- Rules first: explicit patterns (deterministic, never Groq) ---
    if _rule_matched(cleaned):
        return compose_fallback(cleaned, vocab), "fallback"

    # --- LLM path (Groq): only 2+ words with no explicit pattern ---
    if len(cleaned) >= 2 and settings.groq_api_key:
        joined = ", ".join(cleaned)
        for model in dict.fromkeys([settings.groq_model, _GROQ_FALLBACK_MODEL]):
            try:
                content = _groq_chat(settings, model, _SYSTEM_PROMPT, joined)
            except Exception:
                logger.info("Groq compose failed on model %r; trying next.", model)
                continue
            guarded = guard_llm_output(content, cleaned)
            if guarded is not None:
                return guarded, "llm"
            logger.info("Groq compose output from model %r rejected by the guard.", model)

    # --- Rule-based fallback: comma-join, no function words added ---
    return compose_fallback(cleaned, vocab), "fallback"


def _rule_matched(cleaned: list[str]) -> bool:
    """True when `cleaned` matches an explicit deterministic pattern."""
    if not cleaned:
        return True  # empty -> handled by caller
    if len(cleaned) == 1:
        return True
    w = cleaned
    if w in (["where", "water"], ["where", "you"], ["where", "me"], ["help", "me"]):
        return True
    if len(w) == 2 and w[0] == "hello":
        return True
    return False
