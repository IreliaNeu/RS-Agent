import asyncio

import httpx
import pytest

from rs_agent.core.config import ProviderConfig
from rs_agent.providers.base import ProviderError
from rs_agent.providers.openai_compatible import OpenAICompatibleProvider


def make_provider(monkeypatch, handler, max_retries=3):
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        name="test",
        config=ProviderConfig(
            base_url="https://example.test/v1",
            api_key_env="TEST_API_KEY",
            max_retries=max_retries,
        ),
        client=client,
    )
    return provider, client


def test_non_retryable_400_is_called_once(monkeypatch) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "invalid request"})

    async def run() -> None:
        provider, client = make_provider(monkeypatch, handler, max_retries=3)
        with pytest.raises(ProviderError, match="HTTP 400"):
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="model-a",
                temperature=0.0,
                max_tokens=20,
            )
        await client.aclose()

    asyncio.run(run())
    assert calls == 1


def test_metadata_cannot_override_core_request_fields(monkeypatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be sent")

    async def run() -> None:
        provider, client = make_provider(monkeypatch, handler)
        with pytest.raises(ValueError, match="cannot override"):
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="model-a",
                temperature=0.0,
                max_tokens=20,
                metadata={"model": "unexpected-model"},
            )
        await client.aclose()

    asyncio.run(run())

