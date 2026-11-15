"""Groq/Ollama gloss stages against mocked HTTP -- no real network, no API keys."""
import httpx
import pytest

from app.core.config import Settings
from app.services.avatar_gloss_service import generate_gloss, load_vocabulary

VOCAB = load_vocabulary(Settings(groq_api_key=None, ollama_base_url=None))


def _groq_settings(transport, **kw):
    s = Settings(groq_api_key="gsk_test", ollama_base_url=None, **kw)
    return s, transport


def _mount(monkeypatch, transport):
    """Patch httpx.AsyncClient so our service's `async with httpx.AsyncClient(...)`
    uses this MockTransport instead of touching the network."""
    import app.services.avatar_gloss_service as svc

    orig = httpx.AsyncClient
    monkeypatch.setattr(svc.httpx, "AsyncClient", lambda **kw: orig(**{**kw, "transport": transport}))


def _groq_response(content, status=200, headers=None):
    def handler(request: httpx.Request):
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]}, headers=headers or {})
    return httpx.MockTransport(handler)


async def test_groq_success_is_used_and_request_is_well_formed(monkeypatch):
    seen = {}
    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = httpx.Request("POST", "x", content=request.content).content  # keep bytes accessible
        import json as _json
        seen["json"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "HELLO THANK_YOU"}}]})
    _mount(monkeypatch, httpx.MockTransport(handler))

    r = await generate_gloss("hello, thank you so much", Settings(groq_api_key="gsk_test"))
    assert r.gloss == ["HELLO", "THANK_YOU"] and r.source == "llm:groq"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"
    assert seen["json"]["model"] == "openai/gpt-oss-20b"
    assert VOCAB <= VOCAB  # sanity


async def test_groq_output_is_filtered_against_the_vocabulary_not_trusted_blindly(monkeypatch):
    """The model hallucinating an ID outside the list must never reach the avatar."""
    _mount(monkeypatch, _groq_response("HELLO BANANA MADE_UP_SIGN THANK_YOU"))
    r = await generate_gloss("hi", Settings(groq_api_key="gsk_test"))
    assert r.gloss == ["HELLO", "THANK_YOU"]


async def test_groq_empty_reply_is_a_valid_no_signs_result_not_a_failure(monkeypatch):
    _mount(monkeypatch, _groq_response(""))
    r = await generate_gloss("xyzzy qwerty", Settings(groq_api_key="gsk_test"))
    assert r.gloss == [] and r.unmatched_words == ["xyzzy", "qwerty"] and r.source == "deterministic"
    # empty LLM result correctly falls through to deterministic, which also finds nothing


async def test_groq_short_retry_after_is_honoured_then_succeeds(monkeypatch):
    calls = []
    def handler(request: httpx.Request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "YES"}}]})
    _mount(monkeypatch, httpx.MockTransport(handler))
    r = await generate_gloss("yes", Settings(groq_api_key="gsk_test", groq_max_retry_wait_seconds=3))
    assert r.gloss == ["YES"] and r.source == "llm:groq" and len(calls) == 2


async def test_groq_long_retry_after_falls_through_immediately_to_deterministic(monkeypatch):
    def handler(request: httpx.Request):
        return httpx.Response(429, headers={"retry-after": "30"})
    _mount(monkeypatch, httpx.MockTransport(handler))
    r = await generate_gloss("yes", Settings(groq_api_key="gsk_test", groq_max_retry_wait_seconds=3))
    assert r.gloss == ["YES"] and r.source == "deterministic"  # "yes" matches the dictionary too


async def test_groq_network_failure_falls_through(monkeypatch):
    def handler(request: httpx.Request):
        raise httpx.ConnectError("down")
    _mount(monkeypatch, httpx.MockTransport(handler))
    r = await generate_gloss("hello", Settings(groq_api_key="gsk_test"))
    assert r.source == "deterministic" and r.gloss == ["HELLO"]


async def test_groq_malformed_response_shape_falls_through(monkeypatch):
    def handler(request: httpx.Request):
        return httpx.Response(200, json={"unexpected": "shape"})
    _mount(monkeypatch, httpx.MockTransport(handler))
    r = await generate_gloss("hello", Settings(groq_api_key="gsk_test"))
    assert r.source == "deterministic"


async def test_no_groq_key_skips_straight_to_ollama(monkeypatch):
    seen = {}
    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        import json as _json
        seen["json"] = _json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "HOW YOU"}})
    _mount(monkeypatch, httpx.MockTransport(handler))
    r = await generate_gloss("how are you", Settings(groq_api_key=None, ollama_base_url="http://localhost:11434"))
    assert r.gloss == ["HOW", "YOU"] and r.source == "llm:ollama"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["json"]["model"] == "llama3.2" and seen["json"]["stream"] is False


async def test_ollama_failure_falls_through_to_deterministic(monkeypatch):
    def handler(request: httpx.Request):
        raise httpx.ConnectError("down")
    _mount(monkeypatch, httpx.MockTransport(handler))
    r = await generate_gloss("please help", Settings(groq_api_key=None, ollama_base_url="http://localhost:11434"))
    assert r.gloss == ["PLEASE", "HELP"] and r.source == "deterministic"


async def test_neither_llm_configured_uses_deterministic_directly(monkeypatch):
    called = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: (_ for _ in ()).throw(AssertionError("should not touch the network")))
    r = await generate_gloss("hello", Settings(groq_api_key=None, ollama_base_url=None))
    assert r.gloss == ["HELLO"] and r.source == "deterministic"


async def test_llm_handles_paraphrased_input_the_dictionary_lookup_cannot(monkeypatch):
    """The value-add isn't just vocabulary overlap (short pronouns like ME/YOU
    incidentally appear in plain English too) -- it's that "assist ... finding"
    has NO literal overlap with HELP at all, which word-for-word lookup can
    never bridge no matter how the vocabulary is phrased."""
    from app.services.avatar_gloss_service import _deterministic_gloss
    plain_gloss, _ = _deterministic_gloss("could you assist me in finding some water please", VOCAB)
    assert "HELP" not in plain_gloss  # "assist" is not the literal word "help"

    _mount(monkeypatch, _groq_response("PLEASE HELP ME WATER"))
    r = await generate_gloss("could you assist me in finding some water please", Settings(groq_api_key="gsk_test"))
    assert r.gloss == ["PLEASE", "HELP", "ME", "WATER"] and r.source == "llm:groq"
