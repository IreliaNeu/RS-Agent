"""Provider interfaces used by domain agents and evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from rs_agent.core.schemas import ModelResponse, RequestTelemetry

ChatMessage = Dict[str, Any]


class ChatProvider(Protocol):
    async def complete(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        ...


class ProviderError(RuntimeError):
    """Raised when a provider request cannot produce a valid completion."""

    def __init__(
        self, message: str, telemetry: Optional[RequestTelemetry] = None
    ) -> None:
        super().__init__(message)
        self.telemetry = telemetry
