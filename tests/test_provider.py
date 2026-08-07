import asyncio

import httpx

from rs_agent.core.config import ProviderConfig
from rs_agent.providers.openai_compatible import (
    OpenAICompatibleProvider,
    chat_completions_url,
)


def test_chat_completions_url_normalization() -> None:
    assert (
        chat_completions_url("https://openrouter.ai/api/v1")
        == "https://openrouter.ai/api/v1/chat/completions"
    )
    assert (
        chat_completions_url("https://api.siliconflow.cn")
        == "https://api.siliconflow.cn/v1/chat/completions"
    )


def test_openai_compatible_provider_parses_response(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-value"
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "model-a",
                "choices": [{"message": {"content": "  completion text  "}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatibleProvider(
            name="test",
            config=ProviderConfig(
                base_url="https://example.test/v1",
                api_key_env="TEST_API_KEY",
                max_retries=0,
            ),
            client=client,
        )
        response = await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="model-a",
            temperature=0.0,
            max_tokens=20,
        )
        await client.aclose()
        return response

    response = asyncio.run(run())
    assert response.content == "completion text"
    assert response.usage.total_tokens == 5

