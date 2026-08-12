"""Async adapter for OpenAI-compatible chat-completion APIs."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from time import perf_counter
from typing import Any, Dict, List, Optional

import httpx

from rs_agent.core.config import ProviderConfig
from rs_agent.core.schemas import ModelResponse, RequestTelemetry, TokenUsage
from rs_agent.providers.base import ChatMessage, ProviderError

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RESERVED_PAYLOAD_KEYS = {"model", "messages", "temperature", "max_tokens"}


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def assistant_message_for_followup(response: ModelResponse) -> ChatMessage:
    """Return the assistant turn with OpenRouter reasoning fields unchanged."""
    if response.assistant_message:
        message = deepcopy(response.assistant_message)
        message["role"] = "assistant"
        message.setdefault("content", response.content)
        return message
    return {"role": "assistant", "content": response.content}


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        value = item.get("text") if item.get("type") == "text" else item.get("content")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.name = name
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatibleProvider":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    @staticmethod
    def _telemetry(
        attempts: int,
        started: float,
        status_code: Optional[int],
        attempt_status_codes: List[int],
    ) -> RequestTelemetry:
        return RequestTelemetry(
            attempts=attempts,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            status_code=status_code,
            attempt_status_codes=list(attempt_status_codes),
        )

    async def complete(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if metadata:
            reserved = RESERVED_PAYLOAD_KEYS.intersection(metadata)
            if reserved:
                raise ValueError(
                    "metadata cannot override request fields: {}".format(sorted(reserved))
                )
            payload.update(metadata)

        headers = {
            "Authorization": "Bearer {}".format(self.config.resolve_api_key()),
            "Content-Type": "application/json",
            **self.config.headers,
        }

        last_error: Optional[Exception] = None
        attempts = 0
        started = perf_counter()
        attempt_status_codes: List[int] = []
        final_status: Optional[int] = None
        async with self._semaphore:
            for attempt in range(self.config.max_retries + 1):
                attempts = attempt + 1
                try:
                    response = await self._client.post(
                        chat_completions_url(self.config.base_url),
                        headers=headers,
                        json=payload,
                    )
                except httpx.HTTPError as exc:
                    last_error = exc
                else:
                    final_status = response.status_code
                    attempt_status_codes.append(response.status_code)
                    if response.status_code >= 400:
                        message = "HTTP {}: {}".format(
                            response.status_code, response.text[:300]
                        )
                        if response.status_code not in RETRYABLE_STATUS_CODES:
                            raise ProviderError(
                                message,
                                telemetry=self._telemetry(
                                    attempts,
                                    started,
                                    final_status,
                                    attempt_status_codes,
                                ),
                            )
                        last_error = ProviderError(message)
                    else:
                        try:
                            parsed = self._parse_response(
                                response.json(), requested_model=model
                            )
                            return parsed.model_copy(
                                update={
                                    "telemetry": self._telemetry(
                                        attempts,
                                        started,
                                        final_status,
                                        attempt_status_codes,
                                    )
                                }
                            )
                        except (
                            KeyError,
                            IndexError,
                            TypeError,
                            ValueError,
                            ProviderError,
                        ) as exc:
                            last_error = exc

                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2**attempt, 8))

        raise ProviderError(
            "provider {} failed after {} attempts: {}".format(
                self.name, attempts, last_error
            ),
            telemetry=self._telemetry(
                attempts, started, final_status, attempt_status_codes
            ),
        )

    def _parse_response(self, data: Dict[str, Any], requested_model: str) -> ModelResponse:
        choice = data["choices"][0]
        if not isinstance(choice, dict):
            raise ProviderError("provider returned an invalid completion choice")
        choice_error = choice.get("error")
        if choice.get("finish_reason") == "error" or choice_error:
            raise ProviderError(
                "provider returned a completion error: {}".format(choice_error or choice)
            )
        message = choice["message"]
        if not isinstance(message, dict):
            raise ProviderError("provider returned an invalid assistant message")
        content = _text_content(message.get("content"))
        has_reasoning = bool(message.get("reasoning_details") or message.get("reasoning"))
        if not content and not has_reasoning:
            raise ProviderError("provider returned empty completion content")
        usage_data = data.get("usage") or {}
        return ModelResponse(
            provider=self.name,
            model=str(data.get("model") or requested_model),
            content=content,
            response_id=data.get("id"),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens"),
                completion_tokens=usage_data.get("completion_tokens"),
                total_tokens=usage_data.get("total_tokens"),
            ),
            assistant_message=message,
            raw_response=data,
        )
