"""
Smart Gateway provider — routes all LLM calls through the Continuum Smart Gateway.

Drop-in replacement for the per-provider classes when SMART_GATEWAY_URL is set.
Routing mode is encoded in the OpenAI `model` field using the gateway's native
auto-routing format: `auto/<tier>` (cheap | mid | quality).

Model fidelity: an explicit tier (``auto/<tier>``) and provider-qualified names
pass through to the gateway verbatim — naming a model means you get that model.
The prefix must be the GATEWAY's provider id (``anthropic/claude-opus-4-8``,
``openai/gpt-4o``, ``google/gemini-2.5-flash``), not Continuum's routing prefix
(``claude/…`` is not a gateway provider id and returns a 400). Only bare,
single-segment names (``gpt-4o-mini``) are treated as routable placeholders and
replaced with ``auto/<tier>``, and that substitution is logged as a WARNING so
it is never silent.
"""

from __future__ import annotations

from typing import Any, ClassVar

from continuum.llm.config import LLMConfig
from continuum.llm.providers.openai_provider import OpenAIProvider
from continuum.llm.structured_output import openai_schema_tool, schema_from_response_format
from continuum.logging import get_logger

logger = get_logger(__name__)

# Continuum mode → gateway tier (encoded in model field, e.g. "auto/mid")
_MODE_TO_TIER: dict[str, str] = {
    "quality": "quality",
    "modest": "mid",
    "strict": "cheap",
}

# Gateway provider ids whose request translation has no `response_format` entry,
# so the parameter is dropped before it reaches the real backend — silently, and
# for every model under that provider.
#
# Bedrock: the gateway builds its Converse request from a fixed parameter table
# (src/providers/bedrock/chatComplete.ts). That table maps `tools` → toolConfig
# and `tool_choice` → toolChoice, including the forced form, but contains no
# `response_format` key — so a schema sent that way never reaches AWS. Sending
# it as a forced tool instead uses translation the gateway already performs.
_SCHEMA_VIA_TOOL_PROVIDERS: frozenset[str] = frozenset({"bedrock"})


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
        *,
        enforce_schema: bool = False,
    ) -> dict[str, Any]:
        kwargs = super()._build_kwargs(config, tools, tool_choice, enforce_schema=enforce_schema)
        kwargs.pop("temperature", None)
        return kwargs

    def _schema_delivery(
        self,
        config: LLMConfig,
        tools: list[dict[str, Any]] | None,
        *,
        enforce_schema: bool,
    ) -> dict[str, Any] | None:
        """Send the schema as a forced tool for backends that drop response_format.

        Only for a model pinned to such a provider: with ``auto/<tier>`` the
        gateway chooses the model *after* this request is built, so the target is
        unknowable here and guessing would degrade every tier request.

        Skipped when the caller has tools of its own — forcing the synthetic tool
        would take those off the table — and when streaming, where a forced tool
        yields tool-call deltas and no text.
        """
        if not enforce_schema or tools:
            return None
        if self._target_provider(config.model) not in _SCHEMA_VIA_TOOL_PROVIDERS:
            return None
        schema = schema_from_response_format(config.response_format)
        if schema is None:
            # Bare json_object names no shape, so there is nothing to build a
            # tool from. It is dropped for these providers too, but an empty
            # tool would not help and would change the answer's form.
            return None
        logger.debug(
            "Smart Gateway: sending the output schema for %s as a forced tool "
            "(this backend's request translation drops response_format).",
            config.model,
        )
        return openai_schema_tool(schema)

    @staticmethod
    def _target_provider(model: str) -> str:
        """The gateway provider id a model is pinned to ("" when unpinned).

        Mirrors _normalize_model: only a provider-qualified name names its
        backend. ``auto/<tier>`` is deliberately excluded — "auto" is a routing
        instruction, not a provider.
        """
        if "/" not in model or model.startswith("auto/"):
            return ""
        return model.split("/", 1)[0].lower()

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
                "gateway, use a gateway-provider-qualified name (e.g. "
                "'anthropic/claude-opus-4-8', 'openai/gpt-4o') or request a "
                "tier explicitly with 'auto/<tier>'.",
                model,
                routed,
                self._router_mode,
            )
        return routed
