import asyncio
import json

import httpx
import pytest

from rs_agent.core.config import ProviderConfig
from rs_agent.providers.base import ProviderError
from rs_agent.providers.openai_compatible import (
    OpenAICompatibleProvider,
    assistant_message_for_followup,
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


def test_openai_compatible_provider_preserves_reasoning_details(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    reasoning_details = [{"type": "reasoning.text", "text": "private reasoning token"}]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-value"
        payload = json.loads(request.content)
        assert payload["reasoning"] == {"enabled": True}
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "model-a",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "  completion text  ",
                            "reasoning_details": reasoning_details,
                        }
                    }
                ],
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
            metadata={"reasoning": {"enabled": True}},
        )
        await client.aclose()
        return response

    response = asyncio.run(run())
    assert response.content == "completion text"
    assert response.usage.total_tokens == 5
    assert response.assistant_message["reasoning_details"] == reasoning_details
    assert assistant_message_for_followup(response) == {
        "role": "assistant",
        "content": "  completion text  ",
        "reasoning_details": reasoning_details,
    }


def test_reasoning_only_response_can_be_continued(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "model-a",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_details": [{"type": "reasoning.text", "text": "step"}],
                        }
                    }
                ],
            },
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatibleProvider(
            "test",
            ProviderConfig(
                base_url="https://example.test/v1",
                api_key_env="TEST_API_KEY",
                max_retries=0,
            ),
            client,
        )
        response = await provider.complete([], "model-a", 0.0, 20)
        await client.aclose()
        return response

    response = asyncio.run(run())
    assert response.content == ""
    assert assistant_message_for_followup(response)["reasoning_details"][0]["text"] == "step"


def test_http_200_choice_error_is_not_accepted_as_completion(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "model-a",
                "choices": [
                    {
                        "finish_reason": "error",
                        "error": {"code": 500, "message": "upstream failed"},
                        "message": {"role": "assistant", "content": "partial text"},
                    }
                ],
            },
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatibleProvider(
            "test",
            ProviderConfig(
                base_url="https://example.test/v1",
                api_key_env="TEST_API_KEY",
                max_retries=0,
            ),
            client,
        )
        with pytest.raises(ProviderError):
            await provider.complete([], "model-a", 0.0, 20)
        await client.aclose()

    asyncio.run(run())
