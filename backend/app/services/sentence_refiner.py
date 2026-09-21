from __future__ import annotations

import re
import httpx
from dataclasses import dataclass


@dataclass(frozen=True)
class RefineResult:
    text: str
    provider: str  # groq | ollama | naive

# keep naive as fallback — deterministic, no network
def _naive_refine(text: str) -> str:
    # basic sentence cleanup: trim, capitalize first letter, ensure period
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return t
    # if already looks like honest placeholder, don't refine
    if "Unrecognized sign" in t or "try:" in t.lower():
        return t
    # capitalize first char
    t = t[0].upper() + t[1:] if len(t) > 1 else t.upper()
    if not t.endswith((".", "!", "?")):
        t += "."
    return t

def _clean_llm(value: str) -> str:
    v = re.sub(r"[\r\n]+", " ", str(value)).strip()
    # strip leading prefixes like "Refined:" or quotes
    v = re.sub(r"^(refined|corrected|english)\s*:\s*", "", v, flags=re.I)
    v = v.strip('\"\' ')
    # remove extra explanations after newline
    if "\n" in v:
        v = v.split("\n")[0]
    v = re.sub(r"\s+", " ", v).strip()
    return v

def groq_refine(text: str, *, api_key: str, model: str, timeout_seconds: float = 8.0) -> str:
    from groq import Groq
    client = Groq(api_key=api_key, timeout=timeout_seconds)
    resp = client.chat.completions.create(
        model=model,
        reasoning_effort="low",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that corrects English sentences from ISL translation. "
                    "Fix grammar, add articles where needed, keep meaning, keep it concise (one sentence). "
                    "Reply only with the corrected sentence, no explanation."
                ),
            },
            {"role": "user", "content": f"Fix this sentence: {text}"},
        ],
    )
    content = resp.choices[0].message.content or ""
    cleaned = _clean_llm(content)
    if not cleaned:
        raise RuntimeError("Groq returned empty refine")
    return cleaned

def ollama_refine(text: str, *, base_url: str, model: str, timeout_seconds: float) -> str:
    prompt = (
        "Fix grammar and make this ISL-translated English sentence fluent. Reply only with the corrected sentence.\n\n"
        f"Sentence: {text}\nCorrected:"
    )
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        data = r.json()
    cleaned = _clean_llm(data.get("response", ""))
    if not cleaned:
        raise RuntimeError("Ollama returned empty refine")
    return cleaned

def refine_sentence(
    text: str,
    *,
    enable_groq: bool,
    groq_api_key: str | None,
    groq_model: str,
    enable_ollama: bool,
    ollama_base_url: str,
    ollama_model: str,
    ollama_timeout_seconds: float,
) -> RefineResult:
    # don't refine honest placeholders or very short demo phrases
    if not text or "Unrecognized sign" in text:
        return RefineResult(text, "naive")
    # demo phrases like "Hello", "Thank you" are already fluent — just naive capitalize
    if text.lower().strip() in {"hello", "thank you", "yes", "no", "help", "please", "sorry", "good morning", "how are you", "i need help"}:
        return RefineResult(_naive_refine(text), "naive")

    # Prefer the configured local Ollama model, then immediately fall back to
    # Groq if Ollama is unavailable or fails.
    if enable_ollama:
        try:
            return RefineResult(
                ollama_refine(
                    text,
                    base_url=ollama_base_url,
                    model=ollama_model,
                    timeout_seconds=ollama_timeout_seconds,
                ),
                "ollama",
            )
        except Exception:
            pass

    if enable_groq and groq_api_key:
        try:
            return RefineResult(
                groq_refine(text, api_key=groq_api_key, model=groq_model),
                "groq",
            )
        except Exception:
            pass
    return RefineResult(_naive_refine(text), "naive")
