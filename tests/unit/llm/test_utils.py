"""Unit tests for LLM utils."""

import logging
from unittest.mock import MagicMock, patch  # patch used in TestValidateJsonSchemaConfig

from orchestrator.llm.utils import (
    check_json_schema_support,
    check_response_format_support,
    supports_tools_with_json_mode,
    validate_json_schema_config,
)

logger = logging.getLogger(__name__)


class TestCheckResponseFormatSupport:
    def test_supported_model(self):
        logger.info("CheckResponseFormatSupport: supported model")
        assert check_response_format_support("gpt-4o") is True

    def test_unsupported_model(self):
        logger.info("CheckResponseFormatSupport: unsupported model")
        assert check_response_format_support("old-model") is False

    def test_anthropic_unsupported(self):
        logger.info("CheckResponseFormatSupport: anthropic unsupported")
        assert check_response_format_support("claude-3-opus") is False


class TestCheckJsonSchemaSupport:
    def test_supported(self):
        logger.info("CheckJsonSchemaSupport: supported")
        assert check_json_schema_support("gpt-4o") is True

    def test_unsupported_model(self):
        logger.info("CheckJsonSchemaSupport: unsupported model")
        assert check_json_schema_support("old-model") is False


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


class TestValidateJsonSchemaConfig:
    def test_json_mode_disabled(self):
        logger.info("ValidateJsonSchemaConfig: json mode disabled")
        agent = MagicMock()
        agent.enable_json_mode = False
        valid, err = validate_json_schema_config(agent)
        assert valid is True
        assert err is None

    def test_json_mode_no_response_format_support(self):
        logger.info("ValidateJsonSchemaConfig: json mode no response format support")
        agent = MagicMock()
        agent.enable_json_mode = True
        agent.model = "old-model"
        agent.json_schema = None
        with patch("orchestrator.llm.utils.check_response_format_support", return_value=False):
            valid, err = validate_json_schema_config(agent)
        assert valid is False
        assert "does not support" in err

    def test_json_schema_no_schema_support(self):
        logger.info("ValidateJsonSchemaConfig: json schema no schema support")
        agent = MagicMock()
        agent.enable_json_mode = True
        agent.model = "gpt-4"
        agent.json_schema = {"name": "test"}
        with (
            patch("orchestrator.llm.utils.check_response_format_support", return_value=True),
            patch("orchestrator.llm.utils.check_json_schema_support", return_value=False),
        ):
            valid, err = validate_json_schema_config(agent)
        assert valid is False

    def test_json_schema_fully_supported(self):
        logger.info("ValidateJsonSchemaConfig: json schema fully supported")
        agent = MagicMock()
        agent.enable_json_mode = True
        agent.model = "gpt-4o-2024-08-06"
        agent.json_schema = {"name": "test"}
        with (
            patch("orchestrator.llm.utils.check_response_format_support", return_value=True),
            patch("orchestrator.llm.utils.check_json_schema_support", return_value=True),
        ):
            valid, err = validate_json_schema_config(agent)
        assert valid is True
