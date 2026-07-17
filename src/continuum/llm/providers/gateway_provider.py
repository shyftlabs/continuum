"""
Smart Gateway provider — routes all LLM calls through the Continuum Smart Gateway.

Drop-in replacement for the per-provider classes when SMART_GATEWAY_URL is set.
Routing mode is encoded in the OpenAI `model` field using the gateway's native
auto-routing format: `auto/<tier>` (cheap | mid | quality).

Model fidelity: an explicit tier (``auto/<tier>``) and provider-qualified names
(``claude/claude-haiku-4-5``, ``gemini/gemini-2.5-flash``) pass through to the
gateway verbatim — naming a model means you get that model. Only bare,
single-segment names (``gpt-4o-mini``) are treated as routable placeholders and
replaced with ``auto/<tier>``, and that substitution is logged as a WARNING so
it is never silent.
"""

from __future__ import annotations

from typing import Any, ClassVar

from continuum.llm.config import LLMConfig
from continuum.llm.providers.openai_provider import OpenAIProvider
from continuum.logging import get_logger

logger = get_logger(__name__)

# Continuum mode → gateway tier (encoded in model field, e.g. "auto/mid")
_MODE_TO_TIER: dict[str, str] = {
    "quality": "quality",
    "modest": "mid",
    "strict": "cheap",
}


class GatewayProvider(OpenAIProvider):
    """Routes all LLM calls through the Smart Gateway at SMART_GATEWAY_URL."""

    # Errors from this provider are the gateway's, not api.openai.com's —
    # attribute them accordingly so a gateway 401 isn't blamed on OpenAI.
    _provider_label = "gateway"

    # Substitution warn-once bookkeeping (per requested model name).
    _warned_models: ClassVar[set[str]] = set()

    def __init__(
        self,
        gateway_url: str,
        api_key: str | None,
        router_mode: str | None,
        max_retries: int | None = None,
    ) -> None:
        self._router_mode = router_mode or "modest"
        self._gateway_url = gateway_url
        super().__init__(api_key=api_key, api_base=gateway_url, max_retries=max_retries)

    def _build_kwargs(
        self,
        config: LLMConfig,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs = super()._build_kwargs(config, tools, tool_choice)
        kwargs.pop("temperature", None)
        return kwargs

    def _normalize_model(self, model: str) -> str:
        # Explicit tier request — the gateway's native auto-routing format.
        if model.startswith("auto/"):
            return model
        # Provider-qualified names pass through verbatim: the caller named a
        # specific model, so the gateway must serve exactly that (or fail
        # loudly) rather than silently substituting a tier.
        if "/" in model:
            return model
        # Bare single-segment names are routable placeholders — replace with
        # the mode's tier, but never silently.
        tier = _MODE_TO_TIER.get(self._router_mode, "mid")
        routed = f"auto/{tier}"
        if model not in self._warned_models:
            self._warned_models.add(model)
            logger.warning(
                "Smart Gateway: replacing requested model '%s' with '%s' "
                "(router mode '%s'). To pin a specific model through the "
                "gateway, use a provider-qualified name (e.g. "
                "'claude/claude-haiku-4-5') or request a tier explicitly with "
                "'auto/<tier>'.",
                model,
                routed,
                self._router_mode,
            )
        return routed
