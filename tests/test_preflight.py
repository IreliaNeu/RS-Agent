import asyncio

import pytest

from rs_agent.core.config import ModelConfig, ProviderConfig
from rs_agent.providers.preflight import (
    CheckStatus,
    collect_provider_models,
    models_url,
    run_preflight,
)


def test_models_url_handles_v1_and_completion_urls() -> None:
    assert models_url("https://example.test/v1") == "https://example.test/v1/models"
    assert (
        models_url("https://example.test/v1/chat/completions")
        == "https://example.test/v1/models"
    )


def test_credential_only_preflight_never_returns_key(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "super-secret-value")
    provider = ProviderConfig(
        base_url="https://example.test/v1",
        api_key_env="TEST_API_KEY",
    )
    model = ModelConfig(name="model-a", provider="fake", model="vendor/model-a")
    collected = collect_provider_models([({"fake": provider}, [model])])

    report = asyncio.run(run_preflight(collected, check_network=False))
    serialized = report.model_dump_json()

    assert report.ok is True
    assert report.providers[0].credential_status == CheckStatus.OK
    assert report.providers[0].endpoint_status == CheckStatus.SKIPPED
    assert "super-secret-value" not in serialized


def test_missing_credential_fails_preflight(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    provider = ProviderConfig(
        base_url="https://example.test/v1",
        api_key_env="MISSING_TEST_KEY",
    )
    model = ModelConfig(name="model-a", provider="fake", model="vendor/model-a")

    report = asyncio.run(
        run_preflight(
            collect_provider_models([({"fake": provider}, [model])]),
            check_network=False,
        )
    )

    assert report.ok is False
    assert report.providers[0].credential_status == CheckStatus.ERROR


def test_conflicting_provider_configs_are_rejected() -> None:
    first = ProviderConfig(base_url="https://one.test/v1", api_key_env="KEY")
    second = ProviderConfig(base_url="https://two.test/v1", api_key_env="KEY")
    model = ModelConfig(name="model-a", provider="same", model="vendor/model-a")

    with pytest.raises(ValueError, match="conflicting"):
        collect_provider_models(
            [
                ({"same": first}, [model]),
                ({"same": second}, [model]),
            ]
        )
