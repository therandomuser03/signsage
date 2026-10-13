"""English -> ISL gloss generation for the avatar pipeline.

Groq -> Ollama -> deterministic fallback chain (see MODEL_CARD.md). Both LLM
stages are constrained to the SAME fixed vocabulary the deterministic stage
uses -- they choose which of your existing pre-authored signs to play, they
do NOT invent new sign IDs (that would need new animation clips, a manual
step). Their value is handling phrasing the rigid word-lookup can't: "could
you help me find water" has no exact-word overlap with HELP/WATER's
one-word-per-gloss lookup, but an LLM reads intent, not just tokens.

Any single point of failure (missing key, network error, rate limit,
malformed output) falls straight through -- the deterministic stage always
succeeds, so this feature degrades, it never breaks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")
_DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "sign_vocabulary.json"


@dataclass
class GlossResult:
    gloss: List[str]
    unmatched_words: List[str]
    source: str  # "llm:groq" | "llm:ollama" | "deterministic"


@lru_cache(maxsize=8)
def _load_vocabulary(path: str) -> FrozenSet[str]:
    with open(path, "r", encoding="utf-8") as f:
        return frozenset(json.load(f)["gloss_ids"])


def load_vocabulary(settings: Optional[Settings] = None) -> FrozenSet[str]:
    settings = settings or get_settings()
    return _load_vocabulary(settings.avatar_vocabulary_path or str(_DEFAULT_VOCAB_PATH))


def _deterministic_gloss(text: str, vocabulary: FrozenSet[str]) -> Tuple[List[str], List[str]]:
    """Greedy longest-match lookup; returns (gloss_ids, unmatched_words).

    Multi-word signs (THANK_YOU) match by joining consecutive words with "_",
    longest span first -- otherwise "thank you" would degrade to just YOU.
    Unmatched words are reported, never guessed.
    """
    originals = _WORD_RE.findall(text)
    words = [w.upper() for w in originals]
    max_span = max((g.count("_") + 1 for g in vocabulary), default=1)
    gloss: List[str] = []
    unmatched: List[str] = []
    i = 0
    while i < len(words):
        for span in range(min(max_span, len(words) - i), 0, -1):
            candidate = "_".join(words[i : i + span])
            if candidate in vocabulary:
                gloss.append(candidate)
                i += span
                break
        else:
            unmatched.append(originals[i])
            i += 1
    return gloss, unmatched


def _build_prompt(text: str, vocabulary: FrozenSet[str]) -> Tuple[str, str]:
    vocab_list = ", ".join(sorted(vocabulary))
    system = (
        "You translate short English sentences into a sequence of sign-language "
        "gloss IDs for a 3D avatar. You may ONLY use IDs from this exact list, "
        f"exactly as spelled: {vocab_list}. "
        "Pick the IDs that best carry the sentence's meaning, in natural signing "
        "order. Skip any concept with no matching ID -- never invent an ID that "
        "is not in the list. "
        "Reply with ONLY the chosen IDs separated by single spaces, uppercase, "
        "nothing else -- no punctuation, no explanation, no markdown. "
        "If nothing in the sentence matches any ID, reply with an empty string."
    )
    return system, text.strip()


def _parse_llm_gloss(raw_content: str, vocabulary: FrozenSet[str]) -> List[str]:
    """Never trust the model's output blindly: keep only tokens that are
    actually in the vocabulary, in the order the model returned them."""
    tokens = re.findall(r"[A-Za-z_]+", raw_content.upper())
    return [t for t in tokens if t in vocabulary]


async def _groq_gloss(text: str, settings: Settings, vocabulary: FrozenSet[str]) -> Optional[List[str]]:
    if not settings.groq_api_key:
        return None

    system, user = _build_prompt(text, vocabulary)
    body = {
        "model": settings.groq_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 64,
    }
    headers = {"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"}
    url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=settings.groq_timeout_seconds) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code == 429:
                retry_after_header = resp.headers.get("retry-after")
                retry_after = float(retry_after_header) if retry_after_header is not None else None
                if retry_after is not None and retry_after <= settings.groq_max_retry_wait_seconds:
                    if retry_after > 0:
                        await asyncio.sleep(retry_after)
                    resp = await client.post(url, json=body, headers=headers)
                else:
                    logger.info("Groq rate-limited (retry-after=%s); falling through to Ollama.", retry_after_header)
                    return None
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("Groq gloss request failed (%s); falling through to Ollama.", type(exc).__name__)
        return None

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Groq returned an unexpected response shape: %s", exc)
        return None

    return _parse_llm_gloss(content, vocabulary)


async def _ollama_gloss(text: str, settings: Settings, vocabulary: FrozenSet[str]) -> Optional[List[str]]:
    if not settings.ollama_base_url:
        return None

    system, user = _build_prompt(text, vocabulary)
    body = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0},
    }
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("Ollama gloss request failed (%s); falling through to deterministic.", type(exc).__name__)
        return None

    try:
        content = resp.json()["message"]["content"]
    except (KeyError, ValueError) as exc:
        logger.warning("Ollama returned an unexpected response shape: %s", exc)
        return None

    return _parse_llm_gloss(content, vocabulary)


async def generate_gloss(text: str, settings: Optional[Settings] = None) -> GlossResult:
    settings = settings or get_settings()
    vocabulary = load_vocabulary(settings)

    for fn, source in ((_groq_gloss, "llm:groq"), (_ollama_gloss, "llm:ollama")):
        try:
            result = await fn(text, settings, vocabulary)
        except Exception:  # noqa: BLE001 -- an LLM stage must never break translation
            logger.exception("%s gloss stage raised unexpectedly; falling through.", source)
            result = None
        if result:
            return GlossResult(gloss=result, unmatched_words=[], source=source)

    gloss, unmatched = _deterministic_gloss(text, vocabulary)
    return GlossResult(gloss=gloss, unmatched_words=unmatched, source="deterministic")
