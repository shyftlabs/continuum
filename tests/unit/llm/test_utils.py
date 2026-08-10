"""Unit tests for LLM utils.

The model-name allowlists that used to live here (check_response_format_support /
check_json_schema_support / validate_json_schema_config) are gone: nothing in the
codebase consulted them, and they had gone stale — gpt-5 was absent and every
claude model was hardcoded to False. Whether a provider can enforce a schema is
now answered by the provider itself, and covered in test_provider_native_schema.py.
"""

import logging

from continuum.llm.utils import supports_tools_with_json_mode

logger = logging.getLogger(__name__)


class TestSupportsToolsWithJsonMode:
    def test_openai_supported(self):
        logger.info("SupportsToolsWithJsonMode: openai supported")
        assert supports_tools_with_json_mode("gpt-4o") is True

    def test_gemini_not_supported(self):
        logger.info("SupportsToolsWithJsonMode: gemini not supported")
        assert supports_tools_with_json_mode("gemini/gemini-2.5-flash") is False

    def test_vertex_not_supported(self):
        logger.info("SupportsToolsWithJsonMode: vertex not supported")
        assert supports_tools_with_json_mode("vertex_ai/gemini-pro") is False

    def test_custom_provider_gemini(self):
        logger.info("SupportsToolsWithJsonMode: custom provider gemini")
        assert supports_tools_with_json_mode("model", custom_llm_provider="gemini") is False

    def test_anthropic_supported(self):
        logger.info("SupportsToolsWithJsonMode: anthropic supported")
        assert supports_tools_with_json_mode("claude-3-opus") is True
