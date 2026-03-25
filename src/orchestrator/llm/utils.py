"""
LLM utility functions for model support checking and validation.

Uses hardcoded capability sets per provider — no LiteLLM dependency.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent

# Models that support response_format (JSON mode)
_RESPONSE_FORMAT_SUPPORTED: set[str] = {
    "gpt-4o", "gpt-4o-mini", "gpt-4o-turbo", "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-1.5-pro", "gemini-1.5-flash",
}

# Models that support strict JSON schema (Pydantic / json_schema type)
_JSON_SCHEMA_SUPPORTED: set[str] = {
    "gpt-4o", "gpt-4o-mini", "gpt-4o-turbo", "gpt-4-turbo",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-1.5-pro", "gemini-1.5-flash",
}

# Providers that do NOT support tools + JSON mode simultaneously
_NO_TOOLS_WITH_JSON_PROVIDERS: set[str] = {"gemini", "google", "vertex_ai"}


def _base_model(model: str) -> str:
    """Strip provider prefix and return bare model name."""
    return model.split("/")[-1].lower()


def check_response_format_support(model: str, custom_llm_provider: str | None = None) -> bool:
    """Check if a model supports response_format (JSON mode)."""
    base = _base_model(model)
    # Anthropic doesn't support response_format natively
    if "claude" in base or (custom_llm_provider or "").lower() == "anthropic":
        return False
    return any(supported in base for supported in _RESPONSE_FORMAT_SUPPORTED)


def check_json_schema_support(model: str, custom_llm_provider: str | None = None) -> bool:
    """Check if a model supports strict JSON schema (structured outputs)."""
    base = _base_model(model)
    if "claude" in base or (custom_llm_provider or "").lower() == "anthropic":
        return False
    return any(supported in base for supported in _JSON_SCHEMA_SUPPORTED)


def validate_json_schema_config(agent: "BaseAgent") -> tuple[bool, str | None]:
    """
    Validate an agent's JSON schema configuration.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not agent.enable_json_mode:
        return True, None

    if not check_response_format_support(model=agent.model):
        return (
            False,
            f"Model '{agent.model}' does not support response_format. "
            "JSON mode requires a model that supports structured outputs.",
        )

    if agent.json_schema is not None:
        try:
            from pydantic import BaseModel
            is_pydantic_model = isinstance(agent.json_schema, type) and issubclass(agent.json_schema, BaseModel)
        except Exception:
            is_pydantic_model = False

        if is_pydantic_model or isinstance(agent.json_schema, dict):
            if not check_json_schema_support(model=agent.model):
                return (
                    False,
                    f"Model '{agent.model}' does not support JSON schema. "
                    "Use a model that supports structured outputs with schema validation "
                    "(e.g., gpt-4o, gemini-2.5-pro) or use simple JSON mode "
                    "by setting json_schema=None.",
                )

    return True, None


def supports_tools_with_json_mode(model: str, custom_llm_provider: str | None = None) -> bool:
    """
    Check if a model supports function calling (tools) with JSON mode simultaneously.

    Gemini does not support both at once.
    """
    if custom_llm_provider and custom_llm_provider.lower() in _NO_TOOLS_WITH_JSON_PROVIDERS:
        return False
    model_lower = model.lower()
    if any(p in model_lower for p in ("gemini", "vertex")):
        return False
    return True
