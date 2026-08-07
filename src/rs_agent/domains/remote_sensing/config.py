"""Validated configuration for the paper RS-CC pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import Field, model_validator

from rs_agent.core.config import ModelConfig, ModelRegistryConfig, ProviderConfig
from rs_agent.core.schemas import StrictModel


class CaptionPromptProfile(str, Enum):
    BASE = "base"
    COT_WITHOUT_BACKGROUND = "cot_without_background"


class RSCCExperimentConfig(StrictModel):
    profile: str
    providers: Dict[str, ProviderConfig]
    caption_generators: List[ModelConfig] = Field(min_length=5, max_length=5)
    selector: ModelConfig
    evaluator: ModelConfig
    prompt_profile: CaptionPromptProfile = CaptionPromptProfile.BASE
    minimum_successful_candidates: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def validate_profile(self) -> "RSCCExperimentConfig":
        models = list(self.caption_generators) + [self.selector, self.evaluator]
        unknown = sorted({model.provider for model in models if model.provider not in self.providers})
        if unknown:
            raise ValueError("unknown providers referenced by RS-CC models: {}".format(unknown))
        names = [model.name for model in self.caption_generators]
        model_ids = [model.model for model in self.caption_generators]
        if len(names) != len(set(names)):
            raise ValueError("caption generator names must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("paper profile requires five distinct caption models")
        return self

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