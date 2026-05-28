"""
Provider router — selects the right LLM provider based on model name.
"""

from __future__ import annotations

import logging

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.providers.base import BaseProvider

_log = logging.getLogger(__name__)


def get_provider(config: LLMConfig) -> BaseProvider:
    """
    Return the appropriate provider for a given LLMConfig.

    Routing rules (checked in order):
      - SMART_GATEWAY_URL set → GatewayProvider (all models route through gateway)
      - gemini/ or google/  → GeminiProvider (OpenAI compat endpoint)
      - claude/ or anthropic/ or starts with "claude-" → AnthropicProvider
      - everything else (gpt-*, azure/, etc.) → OpenAIProvider
    """
    from orchestrator.config import settings

    if settings.smart_gateway_url:
        from orchestrator.llm.providers.gateway_provider import GatewayProvider

        mode = config.gateway_router_mode or settings.smart_gateway_default_mode
        from orchestrator.llm.providers.gateway_provider import _MODE_TO_TIER

        tier = _MODE_TO_TIER.get(mode, "mid")
        routed_model = (
            config.model
            if ("/" in config.model or config.model.startswith("auto"))
            else f"auto/{tier}"
        )
        _log.info(
            "🔀 Smart Gateway routing: model=%s mode=%s url=%s",
            routed_model,
            mode,
            settings.smart_gateway_url,
        )
        return GatewayProvider(
            gateway_url=settings.smart_gateway_url,
            api_key=settings.smart_gateway_api_key,
            router_mode=mode,
        )

    model = config.model.lower()

    if any(model.startswith(p) for p in ("gemini/", "google/")):
        from orchestrator.llm.providers.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=settings.gemini_api_key)

    if any(model.startswith(p) for p in ("claude/", "anthropic/", "claude-")):
        from orchestrator.llm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=settings.anthropic_api_key)

    # Default: OpenAI (handles gpt-*, azure/, openai/, etc.)
    from orchestrator.llm.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(
        api_key=config.api_key or settings.openai_api_key,
        organization=settings.openai_organization,
        api_base=config.api_base,
        api_version=config.api_version,
    )


__all__ = ["get_provider", "BaseProvider"]
