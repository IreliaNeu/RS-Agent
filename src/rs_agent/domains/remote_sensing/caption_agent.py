"""Five-model capability-aware RS-CC candidate generation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import Field, model_validator

from rs_agent.core.schemas import (
    CaptionCandidate,
    ImagePair,
    ModelRef,
    RequestTelemetry,
    StrictModel,
    utc_now,
)
from rs_agent.domains.remote_sensing.caption_prompts import (
    IMAGE_TEXT_PROMPT_VERSION,
    caption_enrichment_messages,
    clean_caption_output,
    image_text_caption_messages,
    prompt_version,
)
from rs_agent.domains.remote_sensing.config import (
    CaptionInputMode,
    CaptionModelConfig,
    RSCCExperimentConfig,
)
from rs_agent.domains.remote_sensing.image_inputs import (
    EncodedImagePair,
    encode_original_image_pair,
)
from rs_agent.experiments.protocol import ExperimentProtocol
from rs_agent.providers.base import ChatMessage, ProviderError
from rs_agent.providers.registry import ProviderRegistry


class CaptionReplayCandidate(StrictModel):
    label: str
    model: ModelRef
    input_mode: str
    text: str
    source_candidate_id: str
    source_artifact: str


class RSCCRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    item_id: str
    original_caption: str
    images: Optional[ImagePair] = None
    replayed_candidates: Dict[str, CaptionReplayCandidate] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_caption(self) -> "RSCCRequest":
        if not self.original_caption.strip():
            raise ValueError("RS-CC requires one original Change-Agent caption")
        for label, candidate in self.replayed_candidates.items():
            if label != label.upper() or label != candidate.label:
                raise ValueError("replayed candidate keys must match uppercase labels")
        return self


class CaptionGenerationBatch(StrictModel):
    request_id: str
    item_id: str
    original_caption: str
    profile: str
    protocol: ExperimentProtocol
    prompt_version: str
    input_modes: Dict[str, str]
    image_sha256: Dict[str, str] = Field(default_factory=dict)
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
        encoded: Optional[EncodedImagePair] = None
        if self.config.requires_images:
            if request.images is None:
                raise ValueError(
                    "image-text RS-CC configuration requires the original image pair"
                )
            encoded = encode_original_image_pair(request.images)
        tasks = []
        for index, model in enumerate(self.config.caption_generators):
            label = chr(ord("A") + index)
            replay = request.replayed_candidates.get(label)
            if replay is not None:
                tasks.append(self._replay_one(index, model, replay))
            else:
                tasks.append(
                    self._generate_one(index, model, request.original_caption, encoded)
                )
        candidates = await asyncio.gather(*tasks)
        errors = {
            candidate.label: candidate.error
            for candidate in candidates
            if candidate.error is not None
        }
        modes = {
            chr(ord("A") + index): model.input_mode.value
            for index, model in enumerate(self.config.caption_generators)
        }
        versions = {prompt_version(self.config.prompt_profile)}
        if self.config.requires_images:
            versions.add(IMAGE_TEXT_PROMPT_VERSION)
        return CaptionGenerationBatch(
            request_id=request.request_id,
            item_id=request.item_id,
            original_caption=request.original_caption.strip(),
            profile=self.config.profile,
            protocol=self.config.protocol,
            prompt_version="+".join(sorted(versions)),
            input_modes=modes,
            image_sha256=(
                {"before": encoded.before_sha256, "after": encoded.after_sha256}
                if encoded
                else {}
            ),
            candidates=candidates,
            errors=errors,
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def _replay_one(
        self,
        index: int,
        model: CaptionModelConfig,
        replay: CaptionReplayCandidate,
    ) -> CaptionCandidate:
        label = chr(ord("A") + index)
        expected_model = ModelRef(
            name=model.name,
            provider=model.provider,
            model=model.model,
        )
        if replay.label != label:
            raise ValueError(
                "replayed candidate label {} does not match slot {}".format(
                    replay.label, label
                )
            )
        if replay.model != expected_model:
            raise ValueError(
                "replayed candidate {} model identity does not match config".format(label)
            )
        if replay.input_mode != model.input_mode.value:
            raise ValueError(
                "replayed candidate {} input mode does not match config".format(label)
            )
        if not replay.text.strip():
            raise ValueError("replayed candidate {} is empty".format(label))
        return CaptionCandidate(
            label=label,
            model=expected_model,
            input_mode=replay.input_mode,
            text=replay.text.strip(),
            generation_mode="replayed",
            source_candidate_id=replay.source_candidate_id,
            source_artifact=replay.source_artifact,
        )

    async def _generate_one(
        self,
        index: int,
        model: CaptionModelConfig,
        original_caption: str,
        images: Optional[EncodedImagePair],
    ) -> CaptionCandidate:
        label = chr(ord("A") + index)
        model_ref = ModelRef(
            name=model.name, provider=model.provider, model=model.model
        )
        messages: List[ChatMessage]
        if model.input_mode == CaptionInputMode.IMAGE_TEXT:
            if images is None:
                raise ValueError("image-text candidate requires encoded images")
            messages = image_text_caption_messages(original_caption, images)
        else:
            messages = caption_enrichment_messages(
                original_caption, self.config.prompt_profile
            )
        try:
            response = await self.providers.get(model.provider).complete(
                messages=messages,
                model=model.model,
                temperature=model.temperature,
                max_tokens=model.max_tokens,
                metadata=model.request_options or None,
            )
            caption = clean_caption_output(response.content)
            if not caption:
                raise ValueError("model returned an empty caption")
            return CaptionCandidate(
                label=label,
                model=model_ref,
                input_mode=model.input_mode.value,
                text=caption,
                response=response,
                telemetry=response.telemetry,
            )
        except Exception as exc:
            telemetry = (
                exc.telemetry if isinstance(exc, ProviderError) else RequestTelemetry()
            )
            return CaptionCandidate(
                label=label,
                model=model_ref,
                input_mode=model.input_mode.value,
                text="",
                telemetry=telemetry,
                error="{}: {}".format(type(exc).__name__, exc),
            )
