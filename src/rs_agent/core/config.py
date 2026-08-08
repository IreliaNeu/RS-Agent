"""Configuration models and YAML loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import Field, model_validator

from rs_agent.core.schemas import StrictModel


class ProviderConfig(StrictModel):
    api_style: str = "openai_compatible"
    base_url: str
    api_key_env: str
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_concurrency: int = Field(default=5, ge=1)
    headers: Dict[str, str] = Field(default_factory=dict)

    def resolve_api_key(self) -> str:
        value = os.getenv(self.api_key_env, "").strip()
        if not value:
            raise ValueError(
                "missing API key environment variable: {}".format(self.api_key_env)
            )
        return value


class ModelConfig(StrictModel):
    name: str
    provider: str
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    request_options: Dict[str, Any] = Field(default_factory=dict)


class ProviderRegistryConfig(StrictModel):
    providers: Dict[str, ProviderConfig]


class ModelRegistryConfig(ProviderRegistryConfig):
    caption_generators: List[ModelConfig] = Field(min_length=1)
    selector: ModelConfig
    evaluator: ModelConfig

    @model_validator(mode="after")
    def validate_provider_references(self) -> "ModelRegistryConfig":
        models = list(self.caption_generators) + [self.selector, self.evaluator]
        unknown = sorted(
            {model.provider for model in models if model.provider not in self.providers}
        )
        if unknown:
            raise ValueError("unknown providers referenced by models: {}".format(unknown))
        return self


def load_model_registry(path: Path) -> ModelRegistryConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("model registry must be a YAML object")
    return ModelRegistryConfig.model_validate(data)
