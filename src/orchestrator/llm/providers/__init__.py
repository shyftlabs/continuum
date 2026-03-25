"""
Provider router — selects the right LLM provider based on model name.
"""

from __future__ import annotations

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.providers.base import BaseProvider


def get_provider(config: LLMConfig) -> BaseProvider:
    """
    Return the appropriate provider for a given LLMConfig.

    Routing rules (checked in order):
      - gemini/ or google/  → GeminiProvider (OpenAI compat endpoint)
      - claude/ or anthropic/ or starts with "claude-" → AnthropicProvider
      - everything else (gpt-*, azure/, etc.) → OpenAIProvider
    """
    from orchestrator.config import settings

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
