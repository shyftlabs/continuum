"""
LLM utility functions for model capability checks.

Whether a provider can *enforce* an output schema is answered by the provider
itself — ``BaseProvider.supports_native_schema()`` — not by a name allowlist
here. An allowlist of model names goes stale the week a new model ships, and
this module previously carried one that no code consulted and that still
believed gpt-4o was the newest thing OpenAI made.
"""

# Providers that do NOT support tools + JSON mode simultaneously
_NO_TOOLS_WITH_JSON_PROVIDERS: set[str] = {"gemini", "google", "vertex_ai"}


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
