"""Five-model visual answer generation for the paper-aligned RS-VQA stage."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from rs_agent.core.config import ModelConfig
from rs_agent.core.schemas import (
    ImagePair,
    ModelRef,
    RequestTelemetry,
    StrictModel,
    VQAAnswerCandidate,
    VQAQuestion,
    utc_now,
)
from rs_agent.domains.remote_sensing.image_inputs import EncodedImagePair
from rs_agent.domains.remote_sensing.vqa_config import RSVQAExperimentConfig
from rs_agent.domains.remote_sensing.vqa_prompts import answer_messages
from rs_agent.experiments.protocol import ExperimentProtocol
from rs_agent.providers.base import ChatMessage, ProviderError
from rs_agent.providers.registry import ProviderRegistry


class RSVQARequest(StrictModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    item_id: str
    images: ImagePair
    user_questions: List[str] = Field(default_factory=list)
    knowledge_caption: Optional[str] = None


class VQAGenerationBatch(StrictModel):
    request_id: str
    item_id: str
    profile: str
    protocol: ExperimentProtocol
    questions: List[VQAQuestion]
    candidates: List[VQAAnswerCandidate]
    errors: Dict[str, str] = Field(default_factory=dict)
    image_sha256: Dict[str, str]
    knowledge_caption: Optional[str] = None
    started_at: datetime
    completed_at: datetime


class RSVQAAgent:
    def __init__(self, config: RSVQAExperimentConfig, providers: ProviderRegistry):
        self.config = config
        self.providers = providers

    async def generate(
        self,
        request: RSVQARequest,
        questions: List[VQAQuestion],
        images: EncodedImagePair,
    ) -> VQAGenerationBatch:
        started_at = utc_now()
        candidates: List[VQAAnswerCandidate] = []
        for question in questions:
            messages = answer_messages(
                question.text,
                images,
                knowledge_caption=request.knowledge_caption,
            )
            tasks = [
                self._answer_one(question, index, model, messages)
                for index, model in enumerate(self.config.answer_models)
            ]
            candidates.extend(await asyncio.gather(*tasks))
        errors = {
            "{}:{}".format(candidate.question_id, candidate.label): candidate.error
            for candidate in candidates
            if candidate.error is not None
        }
        return VQAGenerationBatch(
            request_id=request.request_id,
            item_id=request.item_id,
            profile=self.config.profile,
            protocol=self.config.protocol,
            questions=questions,
            candidates=candidates,
            errors=errors,
            image_sha256={
                "before": images.before_sha256,
                "after": images.after_sha256,
            },
            knowledge_caption=request.knowledge_caption,
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def _answer_one(
        self,
        question: VQAQuestion,
        index: int,
        model: ModelConfig,
        messages: List[ChatMessage],
    ) -> VQAAnswerCandidate:
        label = chr(ord("A") + index)
        model_ref = ModelRef(name=model.name, provider=model.provider, model=model.model)
        try:
            response = await self.providers.get(model.provider).complete(
                messages=messages,
                model=model.model,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
                metadata=model.request_options or None,
            )
            text = response.content.strip()
            if not text:
                raise ValueError("model returned an empty VQA answer")
            return VQAAnswerCandidate(
                question_id=question.question_id,
                label=label,
                text=text,
                model=model_ref,
                response=response,
                telemetry=response.telemetry,
            )
        except Exception as exc:
            telemetry = (
                exc.telemetry
                if isinstance(exc, ProviderError)
                else RequestTelemetry()
            )
            return VQAAnswerCandidate(
                question_id=question.question_id,
                label=label,
                text="",
                model=model_ref,
                telemetry=telemetry,
                error="{}: {}".format(type(exc).__name__, exc),
            )
