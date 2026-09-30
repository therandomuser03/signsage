"""verify_access_token against a mocked Supabase (httpx.MockTransport)."""
import httpx
import pytest

from app.core.config import Settings
from app.core.errors import AuthError, AuthNotConfiguredError, AuthUnavailableError
from app.core.security import clear_auth_cache, extract_bearer_token, verify_access_token


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_auth_cache()
    yield
    clear_auth_cache()


def _settings(**kw):
    return Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="pub-key", **kw)


def _transport(status, body=None, calls=None):
    def handler(request: httpx.Request):
        if calls is not None:
            calls.append(request)
        return httpx.Response(status, json=body or {})
    return httpx.MockTransport(handler)


async def test_valid_token_returns_user_and_sends_expected_request():
    calls = []
    user = await verify_access_token(
        "tok", _settings(), transport=_transport(200, {"id": "u1", "email": "a@b.c"}, calls)
    )
    assert (user.id, user.email, user.access_token) == ("u1", "a@b.c", "tok")
    req = calls[0]
    assert str(req.url) == "https://x.supabase.co/auth/v1/user"
    assert req.headers["apikey"] == "pub-key" and req.headers["authorization"] == "Bearer tok"


async def test_token_is_not_in_repr():
    user = await verify_access_token("secret-tok", _settings(), transport=_transport(200, {"id": "u1"}))
    assert "secret-tok" not in repr(user)


async def test_rejected_token_raises_auth_error():
    with pytest.raises(AuthError):
        await verify_access_token("bad", _settings(), transport=_transport(401))


async def test_provider_outage_is_503_not_401():
    with pytest.raises(AuthUnavailableError):
        await verify_access_token("t", _settings(), transport=_transport(500))


async def test_network_failure_is_auth_unavailable():
    def boom(request):
        raise httpx.ConnectError("down")
    with pytest.raises(AuthUnavailableError):
        await verify_access_token("t", _settings(), transport=httpx.MockTransport(boom))


async def test_unconfigured_auth_raises():
    with pytest.raises(AuthNotConfiguredError):
        await verify_access_token("t", Settings(supabase_url=None, supabase_publishable_key=None))


async def test_successful_verification_is_cached():
    calls = []
    t = _transport(200, {"id": "u1"}, calls)
    await verify_access_token("tok", _settings(), transport=t)
    await verify_access_token("tok", _settings(), transport=t)
    assert len(calls) == 1


async def test_cache_disabled_when_ttl_zero():
    calls = []
    t = _transport(200, {"id": "u1"}, calls)
    await verify_access_token("tok", _settings(auth_cache_ttl_seconds=0), transport=t)
    await verify_access_token("tok", _settings(auth_cache_ttl_seconds=0), transport=t)
    assert len(calls) == 2


def test_bearer_extraction():
    assert extract_bearer_token(None) is None
    assert extract_bearer_token("Bearer abc") == "abc"
    assert extract_bearer_token("bearer abc") == "abc"
    for bad in ("Basic abc", "Bearer", "abc"):
        with pytest.raises(AuthError):
            extract_bearer_token(bad)
