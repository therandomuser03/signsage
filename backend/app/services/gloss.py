from __future__ import annotations

import re
from dataclasses import dataclass

import httpx


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am",
    "to", "of", "in", "on", "and", "but", "it", "this",
    "that", "for", "with", "as", "at", "by",
}

GLOSS_ALIASES = {
    "hi": "hello",
}

CONTRACTIONS = {
    "how's": "how is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "it's": "it is",
    "that's": "that is",
    "you're": "you are",
    "your": "your",
}


@dataclass(frozen=True)
class GlossResult:
    gloss: str
    provider: str


def naive_gloss(text: str) -> str:
    normalized = str(text).lower()
    for contraction, expansion in CONTRACTIONS.items():
        normalized = re.sub(rf"\b{re.escape(contraction)}\b", expansion, normalized)
    words = re.findall(r"[a-z0-9]+", normalized)
    kept = [w for w in words if w not in STOPWORDS]
    gloss = kept if kept else words
    gloss = [GLOSS_ALIASES.get(word, word) for word in gloss]
    # ponytail: minimal ISL ordering — questions last, time first (cheap heuristic, LLM does full)
    question_words = {"where", "what", "when", "why", "how", "who", "whom", "which"}
    q = [w for w in gloss if w in question_words]
    rest = [w for w in gloss if w not in question_words]
    if q:
        # move question to end: "where is hospital" -> "hospital where"
        gloss = rest + q
    gloss = list(dict.fromkeys(gloss))
    # keep as is otherwise (SOV approx via stopword removal already)
    return " ".join(gloss)


def _clean_gloss(value: str) -> str:
    value = re.sub(r"[\r\n]+", " ", str(value)).strip()
    value = re.sub(r"^(gloss|isl gloss)\s*:\s*", "", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9' _-]", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def groq_gloss(
    text: str,
    *,
    api_key: str,
    model: str,
    timeout_seconds: float = 15.0,
) -> str:
    from groq import Groq

    client = Groq(api_key=api_key, timeout=timeout_seconds)
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="low",
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert English to concise Indian Sign Language (ISL) gloss. "
                    "Rules: SOV/topic-comment order, drop articles (a/an/the), copulas (is/are), "
                    "keep content words, fingerspell names. Example: 'Where is the hospital?' -> 'hospital where'. "
                    "Reply only with gloss words, lowercase, no punctuation, no explanation."
                ),
            },
            {"role": "user", "content": str(text)},
        ],
    )
    content = response.choices[0].message.content or ""
    cleaned = _clean_gloss(content)
    if not cleaned:
        raise RuntimeError("Groq returned an empty gloss.")
    return cleaned


def ollama_gloss(
    text: str,
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> str:
    prompt = (
        "Convert English to concise Indian Sign Language (ISL) gloss. "
        "Rules: SOV/topic-comment, drop articles/copulas, keep content words. "
        "Example: 'Where is the hospital?' -> 'hospital where'\n\n"
        f"English: {text}\nGloss:"
    )
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        data = response.json()

    cleaned = _clean_gloss(data.get("response", ""))
    if not cleaned:
        raise RuntimeError("Ollama returned an empty gloss.")
    return cleaned


def generate_gloss(
    text: str,
    *,
    enable_groq: bool,
    groq_api_key: str | None,
    groq_model: str,
    enable_ollama: bool,
    ollama_base_url: str,
    ollama_model: str,
    ollama_timeout_seconds: float,
) -> GlossResult:
    errors: list[str] = []

    # Ollama is the primary local model. Groq is the immediate remote
    # fallback when Ollama is unavailable or returns an unusable response.
    if enable_ollama:
        try:
            return GlossResult(
                ollama_gloss(
                    text,
                    base_url=ollama_base_url,
                    model=ollama_model,
                    timeout_seconds=ollama_timeout_seconds,
                ),
                "ollama",
            )
        except Exception as exc:
            errors.append(f"ollama:{type(exc).__name__}")

    if enable_groq and groq_api_key:
        try:
            return GlossResult(
                groq_gloss(
                    text,
                    api_key=groq_api_key,
                    model=groq_model,
                ),
                "groq",
            )
        except Exception as exc:
            errors.append(f"groq:{type(exc).__name__}")

    # The notebook's deterministic fallback.
    return GlossResult(naive_gloss(text), "naive")
