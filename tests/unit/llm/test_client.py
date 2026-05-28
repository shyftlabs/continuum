"""Unit tests for LLM client."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
)
from orchestrator.llm.types import ChatMessage, LLMResponse, ToolDefinition, Usage

logger = logging.getLogger(__name__)


def _make_llm_response(**kwargs) -> LLMResponse:
    defaults = dict(
        model="gpt-4",
        content="Hello!",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    defaults.update(kwargs)
    return LLMResponse(**defaults)


class TestLLMClientInit:
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_client_initialization(self, mock_setup):
        logger.info("LLMClientInit: client initialization")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        assert client.default_config is not None
        assert client._langfuse_enabled is False

    @patch("orchestrator.llm.client.setup_langfuse", side_effect=Exception("no langfuse"))
    def test_client_initialization_langfuse_fails(self, mock_setup):
        logger.info("LLMClientInit: client initialization langfuse fails")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=True)
        assert client._langfuse_enabled is True

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_client_with_rate_limiter(self, mock_setup):
        logger.info("LLMClientInit: client with rate limiter")
        from orchestrator.llm.client import LLMClient

        config = LLMConfig(rate_limit_rpm=60)
        client = LLMClient(config=config, enable_langfuse=False)
        assert client._rate_limiter is not None


class TestLLMClientConversions:
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_convert_messages_from_chat_message(self, mock_setup):
        logger.info("LLMClientConversions: convert messages from chat message")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        msgs = [ChatMessage(role="user", content="hello")]
        result = client._convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_convert_messages_from_dict(self, mock_setup):
        logger.info("LLMClientConversions: convert messages from dict")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        msgs = [{"role": "user", "content": "hello"}]
        result = client._convert_messages(msgs)
        assert result[0] == msgs[0]

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_convert_tools(self, mock_setup):
        logger.info("LLMClientConversions: convert tools")
        from orchestrator.llm.client import LLMClient
        from orchestrator.llm.types import FunctionDefinition

        client = LLMClient(enable_langfuse=False)
        tools = [ToolDefinition(function=FunctionDefinition(name="fn", description="d"))]
        result = client._convert_tools(tools)
        assert result[0]["type"] == "function"

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_convert_tools_none(self, mock_setup):
        logger.info("LLMClientConversions: convert tools none")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        assert client._convert_tools(None) is None


class TestLLMClientMetadata:
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_build_metadata_without_langfuse(self, mock_setup):
        logger.info("LLMClientMetadata: build metadata without langfuse")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        metadata = client._build_metadata(trace_metadata={"key": "val"})
        assert metadata == {"key": "val"}

    @patch("orchestrator.llm.client.get_langfuse_metadata", return_value={"trace_id": "t1"})
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_build_metadata_with_langfuse(self, mock_setup, mock_meta):
        logger.info("LLMClientMetadata: build metadata with langfuse")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._langfuse_enabled = True
        metadata = client._build_metadata()
        assert "trace_id" in metadata


class TestLLMClientExceptionHandling:
    """Exceptions are now raised by providers — test they propagate through the client."""

    @patch("orchestrator.llm.client.get_provider")
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_provider_auth_error_propagates(self, mock_setup, mock_get_provider):
        logger.info("LLMClientExceptionHandling: auth error propagates from provider")
        from orchestrator.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = LLMAuthenticationError(
            "auth failed", model="gpt-4", provider="openai"
        )
        mock_get_provider.return_value = mock_provider
        client = LLMClient(enable_langfuse=False)
        with pytest.raises(LLMAuthenticationError):
            client.chat_sync([ChatMessage(role="user", content="hi")])

    @patch("orchestrator.llm.client.get_provider")
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_provider_rate_limit_propagates(self, mock_setup, mock_get_provider):
        logger.info("LLMClientExceptionHandling: rate limit propagates from provider")
        from orchestrator.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = LLMRateLimitError(
            "rate limited", model="gpt-4", provider="openai"
        )
        mock_get_provider.return_value = mock_provider
        client = LLMClient(enable_langfuse=False)
        with pytest.raises(LLMRateLimitError):
            client.chat_sync([ChatMessage(role="user", content="hi")])

    @patch("orchestrator.llm.client.get_provider")
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_provider_generic_error_propagates(self, mock_setup, mock_get_provider):
        logger.info("LLMClientExceptionHandling: generic error propagates from provider")
        from orchestrator.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = LLMError(
            "something broke", model="gpt-4", provider="openai"
        )
        mock_get_provider.return_value = mock_provider
        client = LLMClient(enable_langfuse=False)
        with pytest.raises(LLMError):
            client.chat_sync([ChatMessage(role="user", content="hi")])


class TestLLMClientProvider:
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_get_provider_from_model(self, mock_setup):
        logger.info("LLMClientProvider: get provider from model")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        assert client._get_provider_from_model("gpt-4") == "openai"
        assert client._get_provider_from_model("claude-3-opus") == "anthropic"
        assert client._get_provider_from_model("gemini/gemini-pro") == "gemini"
        assert client._get_provider_from_model("unknown-model") == "unknown"


class TestLLMClientJsonHelpers:
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_log_json_mode_status_json_mode(self, mock_setup):
        logger.info("LLMClientJsonHelpers: log json mode status")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._log_json_mode_status(LLMConfig(json_mode=True))
        client._log_json_mode_status(LLMConfig(json_mode=False))

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_validate_json_response_valid(self, mock_setup):
        logger.info("LLMClientJsonHelpers: validate json response valid")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._validate_json_response('{"key": "val"}', LLMConfig(json_mode=True))
        client._validate_json_response(None, LLMConfig(json_mode=True))
        client._validate_json_response("hello", LLMConfig(json_mode=False))

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_validate_json_response_invalid(self, mock_setup):
        logger.info("LLMClientJsonHelpers: validate json response invalid")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._validate_json_response("{invalid", LLMConfig(json_mode=True))
        client._validate_json_response("not json", LLMConfig(json_mode=True))

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_apply_json_mode_compat_gemini(self, mock_setup):
        logger.info("LLMClientJsonHelpers: apply json mode compat disables for gemini + tools")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        config = LLMConfig(json_mode=True, model="gemini/gemini-2.5-flash")
        tools = [{"type": "function", "function": {"name": "fn"}}]
        result = client._apply_json_mode_compat(config, tools)
        assert result.json_mode is False
        assert result.response_format is None

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_apply_json_mode_compat_openai_unchanged(self, mock_setup):
        logger.info("LLMClientJsonHelpers: apply json mode compat does not change openai")
        from orchestrator.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        config = LLMConfig(json_mode=True, model="gpt-4o")
        tools = [{"type": "function", "function": {"name": "fn"}}]
        result = client._apply_json_mode_compat(config, tools)
        assert result.json_mode is True


class TestLLMClientChatSync:
    @patch("orchestrator.llm.client.get_provider")
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_chat_sync(self, mock_setup, mock_get_provider):
        logger.info("LLMClientChatSync: chat sync")
        from orchestrator.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.return_value = _make_llm_response(content="Hello!", model="gpt-4")
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        result = client.chat_sync([ChatMessage(role="user", content="hi")])
        assert result.content == "Hello!"
        assert result.model == "gpt-4"
        mock_provider.complete.assert_called_once()


class TestLLMClientChatAsync:
    @patch("orchestrator.llm.client.get_provider")
    @patch("orchestrator.llm.client.setup_langfuse")
    @pytest.mark.asyncio
    async def test_chat_async(self, mock_setup, mock_get_provider):
        logger.info("LLMClientChatAsync: chat async")
        from orchestrator.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(
            return_value=_make_llm_response(content="Hello async!", model="gpt-4")
        )
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        result = await client.chat(
            [ChatMessage(role="user", content="hi")],
            auto_session=False,
        )
        assert result.content == "Hello async!"
        mock_provider.acomplete.assert_called_once()
