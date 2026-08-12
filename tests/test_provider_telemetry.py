import asyncio

import httpx

from rs_agent.core.config import ProviderConfig
from rs_agent.providers.openai_compatible import OpenAICompatibleProvider


def test_provider_records_retry_statuses_and_usage(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "try again"})
        return httpx.Response(
            200,
            json={
                "id": "ok",
                "model": "model-a",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            },
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatibleProvider(
            "test",
            ProviderConfig(
                base_url="https://example.test/v1",
                api_key_env="TEST_API_KEY",
                max_retries=1,
            ),
            client,
        )
        response = await provider.complete([], "model-a", 0.0, 20)
        await client.aclose()
        return response

    response = asyncio.run(run())
    assert response.telemetry.attempts == 2
    assert response.telemetry.status_code == 200
    assert response.telemetry.attempt_status_codes == [503, 200]
    assert response.telemetry.latency_ms >= 0
    assert response.usage.total_tokens == 5
