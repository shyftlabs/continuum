"""
Smart Gateway provider — routes all LLM calls through the Continuum Smart Gateway.

Drop-in replacement for the per-provider classes when SMART_GATEWAY_URL is set.
Routing mode is encoded in the OpenAI `model` field using the gateway's native
auto-routing format: `auto/<tier>` (cheap | mid | quality).
"""

from __future__ import annotations

from orchestrator.llm.providers.openai_provider import OpenAIProvider

# Continuum mode → gateway tier (encoded in model field, e.g. "auto/mid")
_MODE_TO_TIER: dict[str, str] = {
    "quality": "quality",
    "modest": "mid",
    "strict": "cheap",
}


class GatewayProvider(OpenAIProvider):
    """Routes all LLM calls through the Smart Gateway at SMART_GATEWAY_URL."""

    def __init__(self, gateway_url: str, api_key: str | None, router_mode: str | None) -> None:
        self._router_mode = router_mode or "modest"
        super().__init__(api_key=api_key, api_base=gateway_url)

    def _build_kwargs(self, config, tools, tool_choice):
        kwargs = super()._build_kwargs(config, tools, tool_choice)
        kwargs.pop("temperature", None)
        return kwargs

    def _normalize_model(self, model: str) -> str:
        # Already in gateway auto-routing format — pass through unchanged.
        if model.startswith("auto/"):
            return model
        # Everything else (single-segment or provider-qualified like
        # "gemini/gemini-2.5-flash") gets routed by the gateway.
        tier = _MODE_TO_TIER.get(self._router_mode, "mid")
        return f"auto/{tier}"
