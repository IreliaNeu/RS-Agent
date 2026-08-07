"""Five-model RS-CC candidate generation used by the paper benchmark."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List
from uuid import uuid4

from pydantic import Field, model_validator

from rs_agent.core.config import ModelConfig
from rs_agent.core.schemas import CaptionCandidate, ModelRef, StrictModel, utc_now
from rs_agent.domains.remote_sensing.caption_prompts import (
    caption_enrichment_messages,
    clean_caption_output,
    prompt_version,
)
from rs_agent.domains.remote_sensing.config import RSCCExperimentConfig
from rs_agent.providers.base import ChatMessage
from rs_agent.providers.registry import ProviderRegistry


class RSCCRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    item_id: str
    original_caption: str

    @model_validator(mode="after")
    def validate_caption(self) -> "RSCCRequest":
        if not self.original_caption.strip():
            raise ValueError("RS-CC requires one original Change-Agent caption")
        return self


class CaptionGenerationBatch(StrictModel):
    request_id: str
    item_id: str
    original_caption: str
    profile: str
    prompt_version: str
    candidates: List[CaptionCandidate]
    errors: Dict[str, str] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime

    @property
    def successful_candidates(self) -> List[CaptionCandidate]:
        return [candidate for candidate in self.candidates if not candidate.error]


class RSCCAgent:
    def __init__(self, config: RSCCExperimentConfig, providers: ProviderRegistry):
        self.config = config
        self.providers = providers

    async def generate(self, request: RSCCRequest) -> CaptionGenerationBatch:
        started_at = utc_now()
        messages = caption_enrichment_messages(
            request.original_caption, self.config.prompt_profile
        )
        tasks = [
            self._generate_one(index, model, messages)
            for index, model in enumerate(self.config.caption_generators)
        ]
        candidates = await asyncio.gather(*tasks)
        errors = {
            candidate.label: candidate.error
            for candidate in candidates
            if candidate.error is not None
        }
        return CaptionGenerationBatch(
            request_id=request.request_id,
            item_id=request.item_id,
            original_caption=request.original_caption.strip(),
            profile=self.config.profile,
            prompt_version=prompt_version(self.config.prompt_profile),
            candidates=candidates,
            errors=errors,
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def _generate_one(
        self, index: int, model: ModelConfig, messages: List[ChatMessage]
    ) -> CaptionCandidate:
        label = chr(ord("A") + index)
        model_ref = ModelRef(name=model.name, provider=model.provider, model=model.model)
        try:
            provider = self.providers.get(model.provider)
            response = await provider.complete(
                messages=messages,
                model=model.model,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
            )
            caption = clean_caption_output(response.content)
            if not caption:
                raise ValueError("model returned an empty caption")
            return CaptionCandidate(
                label=label,
                model=model_ref,
                text=caption,
                response=response,
            )
        except Exception as exc:
            return CaptionCandidate(
                label=label,
                model=model_ref,
                text="",
                error="{}: {}".format(type(exc).__name__, exc),
            )
