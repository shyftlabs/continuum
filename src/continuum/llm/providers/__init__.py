"""
Provider router — selects the right LLM provider based on model name.

Routing is an open registry rather than a hardcoded chain: built-in providers
register their model-name prefixes at import, and third parties can add their own
backend without editing this module:

    from continuum.llm.providers import register_provider, BaseProvider

    class MyProvider(BaseProvider): ...

    register_provider("myco/", lambda config, settings: MyProvider(api_key=...))
    # now any model named "myco/..." routes to MyProvider

Resolution order in get_provider():
    1. Smart Gateway short-circuit — if SMART_GATEWAY_URL is set, ALL models go
       through GatewayProvider (registry is bypassed by design).
    2. First registered prefix whose lowercased model name matches (longest
       prefix wins, so more specific registrations take precedence).
    3. The default factory (OpenAI) — handles gpt-*, azure/, openai/, etc.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from continuum.llm.config import LLMConfig
from continuum.llm.providers.base import BaseProvider

if TYPE_CHECKING:
    from continuum.config import Settings

_log = logging.getLogger(__name__)

# A factory takes the per-call LLMConfig and the global Settings and returns a
# ready provider. Keeping construction in the factory lets each provider pull the
# exact args it needs (api_base/version for OpenAI, just api_key for the others)
# without get_provider knowing those details.
ProviderFactory = Callable[[LLMConfig, "Settings"], BaseProvider]

# prefix (lowercased) -> factory. Order is not relied upon; longest-prefix match
# wins in get_provider so a more specific prefix beats a more general one.
_PROVIDERS: dict[str, ProviderFactory] = {}
_default_factory: ProviderFactory | None = None

# Set after the built-in registrations at the bottom of this module; anything
# registered beyond these is a deliberate custom provider. Used to warn (once)
# when the Smart Gateway short-circuit silently shadows custom registrations.
_BUILTIN_PREFIXES: frozenset[str] = frozenset()
_gateway_shadow_warned = False


def register_provider(prefix: str, factory: ProviderFactory) -> None:
    """Register a provider factory for model names starting with ``prefix``.

    ``prefix`` is matched case-insensitively against the model name (e.g.
    ``"claude/"``, ``"gemini/"``, or a bare vendor tag like ``"claude-"``).
    Re-registering an existing prefix overrides it, so projects can swap a
    built-in provider for their own implementation. The Smart Gateway, when
    configured, takes precedence over every registration.
    """
    _PROVIDERS[prefix.lower()] = factory


def register_default_provider(factory: ProviderFactory) -> None:
    """Set the fallback factory used when no registered prefix matches."""
    global _default_factory
    _default_factory = factory


def get_provider(config: LLMConfig) -> BaseProvider:
    """
    Return the appropriate provider for a given LLMConfig.

    Routing rules (checked in order):
      - SMART_GATEWAY_URL set → GatewayProvider (all models route through gateway)
      - longest registered model-name prefix that matches → its provider
      - no match → the default provider (OpenAI: gpt-*, azure/, openai/, etc.)
    """
    from continuum.config import settings

    if settings.smart_gateway_url:
        from continuum.llm.providers.gateway_provider import _MODE_TO_TIER, GatewayProvider

        # The gateway short-circuit shadows every registration. Built-ins are
        # expected to be shadowed (that's the feature), but a CUSTOM provider a
        # developer registered deliberately going silently inert is a footgun —
        # say so once, loudly.
        global _gateway_shadow_warned
        custom = set(_PROVIDERS) - _BUILTIN_PREFIXES
        if custom and not _gateway_shadow_warned:
            _gateway_shadow_warned = True
            _log.warning(
                "SMART_GATEWAY_URL is set: ALL models route through the Smart "
                "Gateway, so custom provider registrations %s are bypassed. "
                "Unset SMART_GATEWAY_URL (set it to '') to use them.",
                sorted(custom),
            )

        mode = config.gateway_router_mode or settings.smart_gateway_default_mode
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
            max_retries=config.max_retries,
        )

    model = config.model.lower()

    # Longest matching prefix wins so a specific registration ("claude-3-5/")
    # beats a broad one ("claude-") if both are present.
    match = max(
        (p for p in _PROVIDERS if model.startswith(p)),
        key=len,
        default=None,
    )
    if match is not None:
        return _PROVIDERS[match](config, settings)

    if _default_factory is not None:
        return _default_factory(config, settings)

    # Should never happen — the default is registered at import below.
    raise RuntimeError(f"No provider registered for model '{config.model}' and no default set.")


# ---------------------------------------------------------------------------
# Built-in registrations — identical routing/behavior to the previous hardcoded
# chain. Imports are deferred inside the factories so importing this module does
# not pull in every provider SDK eagerly.
# ---------------------------------------------------------------------------


def _make_gemini(config: LLMConfig, settings: Settings) -> BaseProvider:
    from continuum.llm.providers.gemini_provider import GeminiProvider

    return GeminiProvider(api_key=settings.gemini_api_key, max_retries=config.max_retries)


def _make_anthropic(config: LLMConfig, settings: Settings) -> BaseProvider:
    from continuum.llm.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(api_key=settings.anthropic_api_key, max_retries=config.max_retries)


def _make_openai(config: LLMConfig, settings: Settings) -> BaseProvider:
    from continuum.llm.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(
        api_key=config.api_key or settings.openai_api_key,
        organization=settings.openai_organization,
        api_base=config.api_base,
        api_version=config.api_version,
        max_retries=config.max_retries,
    )


register_provider("gemini/", _make_gemini)
register_provider("google/", _make_gemini)
register_provider("claude/", _make_anthropic)
register_provider("anthropic/", _make_anthropic)
register_provider("claude-", _make_anthropic)
register_default_provider(_make_openai)

# Everything registered above is a built-in; later registrations are custom.
_BUILTIN_PREFIXES = frozenset(_PROVIDERS)


__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "get_provider",
    "register_provider",
    "register_default_provider",
]
