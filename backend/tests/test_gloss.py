from app.services import gloss
from app.services.gloss import naive_gloss


def test_naive_gloss_is_deterministic():
    assert naive_gloss("This is a test for the system") == "test system"


def test_naive_gloss_maps_unseen_greeting_to_vocabulary_word():
    assert naive_gloss("Hi") == "hello"


def test_naive_gloss_expands_contractions_before_stopword_removal():
    assert naive_gloss("Hi. How are you? How's your health?") == "hello you your health how"


def test_ollama_is_primary_provider(monkeypatch):
    calls = []

    monkeypatch.setattr(gloss, "ollama_gloss", lambda *args, **kwargs: calls.append("ollama") or "hello")
    monkeypatch.setattr(gloss, "groq_gloss", lambda *args, **kwargs: calls.append("groq") or "hello")

    result = gloss.generate_gloss(
        "Hello",
        enable_groq=True,
        groq_api_key="test-key",
        groq_model="test-model",
        enable_ollama=True,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        ollama_timeout_seconds=1,
    )

    assert result.provider == "ollama"
    assert calls == ["ollama"]


def test_groq_is_immediate_fallback_when_ollama_fails(monkeypatch):
    calls = []

    def fail_ollama(*args, **kwargs):
        calls.append("ollama")
        raise RuntimeError("offline")

    monkeypatch.setattr(gloss, "ollama_gloss", fail_ollama)
    monkeypatch.setattr(gloss, "groq_gloss", lambda *args, **kwargs: calls.append("groq") or "hello")

    result = gloss.generate_gloss(
        "Hello",
        enable_groq=True,
        groq_api_key="test-key",
        groq_model="test-model",
        enable_ollama=True,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
        ollama_timeout_seconds=1,
    )

    assert result.provider == "groq"
    assert calls == ["ollama", "groq"]
