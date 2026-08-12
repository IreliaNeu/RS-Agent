"""Validated configuration for paper and operational RS-VQA experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import Field, model_validator

from rs_agent.core.config import ModelConfig, ProviderConfig, ProviderRegistryConfig
from rs_agent.core.schemas import StrictModel
from rs_agent.experiments.protocol import ExperimentProtocol, ExperimentTrack


def _legacy_protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        track=ExperimentTrack.OPERATIONAL,
        baseline_id="legacy_unspecified",
        method_variant="image_pair_vqa",
        substitutes_paper_models=True,
    )


class RSVQAExperimentConfig(StrictModel):
    profile: str
    protocol: ExperimentProtocol = Field(default_factory=_legacy_protocol)
    providers: Dict[str, ProviderConfig]
    answer_models: List[ModelConfig] = Field(min_length=5, max_length=5)
    selector: ModelConfig
    evaluator: ModelConfig
    default_template_ids: List[str] = Field(
        default_factory=lambda: [
            "change_summary",
            "change_presence",
            "spatial_location",
        ]
    )
    minimum_successful_answers_per_question: int = Field(default=5, ge=1, le=5)

    @model_validator(mode="after")
    def validate_profile(self) -> "RSVQAExperimentConfig":
        models = list(self.answer_models) + [self.selector, self.evaluator]
        unknown = sorted(
            {model.provider for model in models if model.provider not in self.providers}
        )
        if unknown:
            raise ValueError(
                "unknown providers referenced by RS-VQA models: {}".format(unknown)
            )
        names = [model.name for model in self.answer_models]
        model_ids = [model.model for model in self.answer_models]
        if len(names) != len(set(names)):
            raise ValueError("RS-VQA answer model names must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("RS-VQA requires five distinct answer models")
        return self

    def provider_registry(self) -> ProviderRegistryConfig:
        return ProviderRegistryConfig(providers=self.providers)


def load_rs_vqa_config(path: Path) -> RSVQAExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("RS-VQA config must be a YAML object")
    return RSVQAExperimentConfig.model_validate(data)
