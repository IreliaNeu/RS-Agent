"""Provider construction from validated model registry configuration."""

from __future__ import annotations

from typing import Dict

from rs_agent.core.config import ModelRegistryConfig
from rs_agent.providers.openai_compatible import OpenAICompatibleProvider


class ProviderRegistry:
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self._providers: Dict[str, OpenAICompatibleProvider] = {}

    def get(self, name: str) -> OpenAICompatibleProvider:
        if name not in self.config.providers:
            raise KeyError("provider is not configured: {}".format(name))
        if name not in self._providers:
            config = self.config.providers[name]
            if config.api_style != "openai_compatible":
                raise ValueError("unsupported API style: {}".format(config.api_style))
            self._providers[name] = OpenAICompatibleProvider(name=name, config=config)
        return self._providers[name]

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()

