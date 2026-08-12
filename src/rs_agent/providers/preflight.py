"""Credential-safe configuration and endpoint preflight checks."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from pydantic import Field

from rs_agent.core.config import ModelConfig, ProviderConfig
from rs_agent.core.schemas import StrictModel


class CheckStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class ModelVisibility(StrictModel):
    name: str
    model: str
    visible: Optional[bool] = None


class ProviderPreflightResult(StrictModel):
    provider: str
    base_url: str
    api_key_env: str
    credential_status: CheckStatus
    endpoint_status: CheckStatus
    http_status: Optional[int] = None
    models: List[ModelVisibility] = Field(default_factory=list)
    message: str = ""


class PreflightReport(StrictModel):
    network_checked: bool
    providers: List[ProviderPreflightResult]
    ok: bool


def models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def _endpoint_identity(config: ProviderConfig) -> Tuple[Any, ...]:
    return (
        config.api_style,
        config.base_url.rstrip("/"),
        config.api_key_env,
        tuple(sorted(config.headers.items())),
    )


def collect_provider_models(
    groups: Iterable[Tuple[Dict[str, ProviderConfig], Iterable[ModelConfig]]]
) -> Dict[str, Tuple[ProviderConfig, List[ModelConfig]]]:
    collected: Dict[str, Tuple[ProviderConfig, List[ModelConfig]]] = {}
    for providers, models in groups:
        for model in models:
            config = providers[model.provider]
            if model.provider in collected:
                existing, existing_models = collected[model.provider]
                if _endpoint_identity(existing) != _endpoint_identity(config):
                    raise ValueError(
                        "provider {} has conflicting endpoint or credential settings".format(
                            model.provider
                        )
                    )
                if all(item.model != model.model for item in existing_models):
                    existing_models.append(model)
            else:
                collected[model.provider] = (config, [model])
    return collected


async def run_preflight(
    provider_models: Dict[str, Tuple[ProviderConfig, List[ModelConfig]]],
    *,
    check_network: bool,
) -> PreflightReport:
    async def inspect(
        name: str, config: ProviderConfig, models: List[ModelConfig]
    ) -> ProviderPreflightResult:
        visibility = [ModelVisibility(name=item.name, model=item.model) for item in models]
        try:
            api_key = config.resolve_api_key()
        except ValueError:
            return ProviderPreflightResult(
                provider=name,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                credential_status=CheckStatus.ERROR,
                endpoint_status=CheckStatus.SKIPPED,
                models=visibility,
                message="required API key environment variable is missing",
            )
        if not check_network:
            return ProviderPreflightResult(
                provider=name,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                credential_status=CheckStatus.OK,
                endpoint_status=CheckStatus.SKIPPED,
                models=visibility,
                message="credential is present; network check was not requested",
            )

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Accept": "application/json",
            **config.headers,
        }
        try:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                response = await client.get(models_url(config.base_url), headers=headers)
        except httpx.HTTPError as exc:
            return ProviderPreflightResult(
                provider=name,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                credential_status=CheckStatus.OK,
                endpoint_status=CheckStatus.ERROR,
                models=visibility,
                message="network request failed: {}".format(type(exc).__name__),
            )
        if response.status_code >= 400:
            return ProviderPreflightResult(
                provider=name,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                credential_status=(
                    CheckStatus.ERROR
                    if response.status_code in {401, 403}
                    else CheckStatus.OK
                ),
                endpoint_status=CheckStatus.ERROR,
                http_status=response.status_code,
                models=visibility,
                message="model-list endpoint returned HTTP {}".format(response.status_code),
            )
        try:
            payload = response.json()
            entries = payload.get("data", []) if isinstance(payload, dict) else []
            visible_ids = {
                str(entry.get("id"))
                for entry in entries
                if isinstance(entry, dict) and entry.get("id")
            }
        except (ValueError, TypeError):
            visible_ids = set()
        if not visible_ids:
            return ProviderPreflightResult(
                provider=name,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                credential_status=CheckStatus.OK,
                endpoint_status=CheckStatus.WARNING,
                http_status=response.status_code,
                models=visibility,
                message="endpoint reachable but no model IDs could be parsed",
            )
        checked = [
            ModelVisibility(
                name=item.name,
                model=item.model,
                visible=item.model in visible_ids,
            )
            for item in models
        ]
        missing = [item.model for item in checked if item.visible is False]
        return ProviderPreflightResult(
            provider=name,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            credential_status=CheckStatus.OK,
            endpoint_status=CheckStatus.WARNING if missing else CheckStatus.OK,
            http_status=response.status_code,
            models=checked,
            message=(
                "endpoint reachable; configured models not listed: {}".format(missing)
                if missing
                else "endpoint reachable and configured models are visible"
            ),
        )

    results = await asyncio.gather(
        *(
            inspect(name, config, models)
            for name, (config, models) in sorted(provider_models.items())
        )
    )
    ok = all(
        result.credential_status == CheckStatus.OK
        and result.endpoint_status != CheckStatus.ERROR
        for result in results
    )
    return PreflightReport(network_checked=check_network, providers=results, ok=ok)
