"""Validated configuration for paper and capability-aware RS-CC experiments."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import Field, model_validator

from rs_agent.core.config import ModelConfig, ModelRegistryConfig, ProviderConfig
from rs_agent.core.schemas import StrictModel
from rs_agent.experiments.protocol import (
    ExperimentProtocol,
    ExperimentTrack,
)


class CaptionPromptProfile(str, Enum):
    BASE = "base"
    COT_WITHOUT_BACKGROUND = "cot_without_background"


class CaptionInputMode(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_TEXT = "image_text"


class CaptionModelConfig(ModelConfig):
    input_mode: CaptionInputMode = CaptionInputMode.TEXT_ONLY


def _legacy_protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        track=ExperimentTrack.OPERATIONAL,
        baseline_id="legacy_unspecified",
        method_variant="text_only",
        substitutes_paper_models=True,
    )


class RSCCExperimentConfig(StrictModel):
    profile: str
    protocol: ExperimentProtocol = Field(default_factory=_legacy_protocol)
    providers: Dict[str, ProviderConfig]
    caption_generators: List[CaptionModelConfig] = Field(min_length=5, max_length=5)
    selector: ModelConfig
    evaluator: ModelConfig
    prompt_profile: CaptionPromptProfile = CaptionPromptProfile.BASE
    minimum_successful_candidates: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def validate_profile(self) -> "RSCCExperimentConfig":
        models = list(self.caption_generators) + [self.selector, self.evaluator]
        unknown = sorted(
            {model.provider for model in models if model.provider not in self.providers}
        )
        if unknown:
            raise ValueError(
                "unknown providers referenced by RS-CC models: {}".format(unknown)
            )
        names = [model.name for model in self.caption_generators]
        model_ids = [model.model for model in self.caption_generators]
        if len(names) != len(set(names)):
            raise ValueError("caption generator names must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("RS-CC requires five distinct caption models")
        modes = {model.input_mode for model in self.caption_generators}
        if self.protocol.track == ExperimentTrack.PAPER and modes != {
            CaptionInputMode.TEXT_ONLY
        }:
            raise ValueError("paper RS-CC track must remain text-only")
        if (
            CaptionInputMode.IMAGE_TEXT in modes
            and self.protocol.track != ExperimentTrack.ENHANCEMENT
        ):
            raise ValueError("image-text RS-CC must use the enhancement track")
        return self

    @property
    def requires_images(self) -> bool:
        return any(
            model.input_mode == CaptionInputMode.IMAGE_TEXT
            for model in self.caption_generators
        )

    def model_registry(self) -> ModelRegistryConfig:
        return ModelRegistryConfig(
            providers=self.providers,
            caption_generators=self.caption_generators,
            selector=self.selector,
            evaluator=self.evaluator,
        )


def load_rs_cc_config(path: Path) -> RSCCExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("RS-CC config must be a YAML object")
    return RSCCExperimentConfig.model_validate(data)
