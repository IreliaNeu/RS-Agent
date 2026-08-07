from pathlib import Path

import pytest
from pydantic import ValidationError

from rs_agent.core.config import ModelRegistryConfig, ProviderConfig, load_model_registry


def test_provider_reads_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    provider = ProviderConfig(
        base_url="https://example.test/v1", api_key_env="TEST_API_KEY"
    )
    assert provider.resolve_api_key() == "secret-value"


def test_model_registry_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        ModelRegistryConfig.model_validate(
            {
                "providers": {
                    "known": {
                        "base_url": "https://example.test/v1",
                        "api_key_env": "TEST_API_KEY",
                    }
                },
                "caption_generators": [
                    {"name": "candidate", "provider": "missing", "model": "model-a"}
                ],
                "selector": {"name": "selector", "provider": "known", "model": "model-b"},
                "evaluator": {
                    "name": "evaluator",
                    "provider": "known",
                    "model": "model-c",
                },
            }
        )


def test_example_model_registry_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_model_registry(root / "configs" / "models.example.yaml")
    assert len(registry.caption_generators) == 5

