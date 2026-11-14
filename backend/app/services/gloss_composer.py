"""Shared gloss→sentence composer for ordered word sequences.

Used by:
  * the /api/v1/translation/compose route (REST)
  * the continuous sign pipeline output (WS + /api/v1/translation/sentence)

Order: deterministic rules -> Groq refinement (only when no rule matches and
Groq is configured) -> deterministic comma-join fallback.
"""
from __future__ import annotations

import logging

import httpx

from app.ml.word_composer import compose_fallback, guard_llm_output

logger = logging.getLogger(__name__)

# Retry budget: one attempt with the configured model, and if it answers 404
# (the configured SIGNSAGE_GROQ_MODEL was removed or the account lost access),
# one attempt with the module default documented as working on this account.
_GROQ_FALLBACK_MODEL = "allam-2-7b"


def _groq_chat(settings, model: str, system: str, user: str) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=4.0) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def compose_words(words: list[str], vocab: set[str], settings) -> tuple[str, str]:
    """Compose a sentence from recognised words.

    Returns (sentence, method) where method is "llm" or "fallback".
    """
    cleaned = [w for w in words if w != "rest" and w in vocab]
    if not cleaned:
        return "", "fallback"

    # --- Rules first: explicit patterns (deterministic, never Groq) ---
    if _rule_matched(cleaned):
        return compose_fallback(cleaned, vocab), "fallback"

    # --- LLM path (Groq): only 2+ words with no explicit pattern ---
    if len(cleaned) >= 2 and settings.groq_api_key:
        try:
            joined = ", ".join(cleaned)
            system = (
                "Compose ONE short English sentence using only these words: "
                f"{joined}. You may add only the function words is, are, "
                "am, the, a, I, and punctuation. Do not add any other words, "
                "names, facts, or details. Reply with only the sentence, "
                "nothing else."
            )
            model = settings.groq_model
            try:
                content = _groq_chat(settings, model, system, joined)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404 and not settings.groq_model.endswith(_GROQ_FALLBACK_MODEL):
                    logger.warning(
                        "Groq model %r returned 404; retrying with %r",
                        settings.groq_model,
                        _GROQ_FALLBACK_MODEL,
                    )
                    content = _groq_chat(settings, _GROQ_FALLBACK_MODEL, system, joined)
                    model = _GROQ_FALLBACK_MODEL
                else:
                    raise
            guarded = guard_llm_output(content, cleaned)
            if guarded is not None:
                return guarded, "llm"
        except Exception:
            # fall through to deterministic comma-join fallback
            logger.info("Groq compose failed; using deterministic fallback.")

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
