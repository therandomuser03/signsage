from __future__ import annotations

import re
from dataclasses import dataclass

import httpx


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am",
    "to", "of", "in", "on", "and", "but", "it", "this",
    "that", "for", "with", "as", "at", "by",
}


@dataclass(frozen=True)
class GlossResult:
    gloss: str
    provider: str


def naive_gloss(text: str) -> str:
    words = re.findall(r"[a-z0-9']+", str(text).lower())
    kept = [w for w in words if w not in STOPWORDS]
    return " ".join(kept) if kept else " ".join(words)


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
                    "Convert the English sentence to concise Indian Sign Language "
                    "gloss ordering. Reply only with the gloss words, no explanation."
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
        "Convert this English sentence to concise Indian Sign Language gloss "
        "ordering. Reply only with gloss words, no explanation.\n\n"
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

    # The notebook's deterministic fallback.
    return GlossResult(naive_gloss(text), "naive")
