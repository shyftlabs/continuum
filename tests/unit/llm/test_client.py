"""Unit tests for LLM client."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.llm.config import LLMConfig
from continuum.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
)
from continuum.llm.types import ChatMessage, LLMResponse, ToolDefinition, Usage

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
    @patch("continuum.llm.client.setup_langfuse")
    def test_client_initialization(self, mock_setup):
        logger.info("LLMClientInit: client initialization")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        assert client.default_config is not None
        assert client._langfuse_enabled is False

    @patch("continuum.llm.client.setup_langfuse", side_effect=Exception("no langfuse"))
    def test_client_initialization_langfuse_fails(self, mock_setup):
        logger.info("LLMClientInit: client initialization langfuse fails")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=True)
        assert client._langfuse_enabled is True

    @patch("continuum.llm.client.setup_langfuse")
    def test_client_with_rate_limiter(self, mock_setup):
        logger.info("LLMClientInit: client with rate limiter")
        from continuum.llm.client import LLMClient

        config = LLMConfig(rate_limit_rpm=60)
        client = LLMClient(config=config, enable_langfuse=False)
        assert client._rate_limiter is not None


class TestLLMClientConversions:
    @patch("continuum.llm.client.setup_langfuse")
    def test_convert_messages_from_chat_message(self, mock_setup):
        logger.info("LLMClientConversions: convert messages from chat message")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        msgs = [ChatMessage(role="user", content="hello")]
        result = client._convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    @patch("continuum.llm.client.setup_langfuse")
    def test_convert_messages_from_dict(self, mock_setup):
        logger.info("LLMClientConversions: convert messages from dict")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        msgs = [{"role": "user", "content": "hello"}]
        result = client._convert_messages(msgs)
        assert result[0] == msgs[0]

    @patch("continuum.llm.client.setup_langfuse")
    def test_convert_tools(self, mock_setup):
        logger.info("LLMClientConversions: convert tools")
        from continuum.llm.client import LLMClient
        from continuum.llm.types import FunctionDefinition

        client = LLMClient(enable_langfuse=False)
        tools = [ToolDefinition(function=FunctionDefinition(name="fn", description="d"))]
        result = client._convert_tools(tools)
        assert result[0]["type"] == "function"

    @patch("continuum.llm.client.setup_langfuse")
    def test_convert_tools_none(self, mock_setup):
        logger.info("LLMClientConversions: convert tools none")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        assert client._convert_tools(None) is None


class TestLLMClientMetadata:
    @patch("continuum.llm.client.setup_langfuse")
    def test_build_metadata_without_langfuse(self, mock_setup):
        logger.info("LLMClientMetadata: build metadata without langfuse")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        metadata = client._build_metadata(trace_metadata={"key": "val"})
        assert metadata == {"key": "val"}

    @patch("continuum.llm.client.get_langfuse_metadata", return_value={"trace_id": "t1"})
    @patch("continuum.llm.client.setup_langfuse")
    def test_build_metadata_with_langfuse(self, mock_setup, mock_meta):
        logger.info("LLMClientMetadata: build metadata with langfuse")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._langfuse_enabled = True
        metadata = client._build_metadata()
        assert "trace_id" in metadata


class TestLLMClientExceptionHandling:
    """Exceptions are now raised by providers — test they propagate through the client."""

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    def test_provider_auth_error_propagates(self, mock_setup, mock_get_provider):
        logger.info("LLMClientExceptionHandling: auth error propagates from provider")
        from continuum.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = LLMAuthenticationError(
            "auth failed", model="gpt-4", provider="openai"
        )
        mock_get_provider.return_value = mock_provider
        client = LLMClient(enable_langfuse=False)
        with pytest.raises(LLMAuthenticationError):
            client.chat_sync([ChatMessage(role="user", content="hi")])

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    def test_provider_rate_limit_propagates(self, mock_setup, mock_get_provider):
        logger.info("LLMClientExceptionHandling: rate limit propagates from provider")
        from continuum.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = LLMRateLimitError(
            "rate limited", model="gpt-4", provider="openai"
        )
        mock_get_provider.return_value = mock_provider
        client = LLMClient(enable_langfuse=False)
        with pytest.raises(LLMRateLimitError):
            client.chat_sync([ChatMessage(role="user", content="hi")])

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    def test_provider_generic_error_propagates(self, mock_setup, mock_get_provider):
        logger.info("LLMClientExceptionHandling: generic error propagates from provider")
        from continuum.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = LLMError(
            "something broke", model="gpt-4", provider="openai"
        )
        mock_get_provider.return_value = mock_provider
        client = LLMClient(enable_langfuse=False)
        with pytest.raises(LLMError):
            client.chat_sync([ChatMessage(role="user", content="hi")])


class TestLLMClientProvider:
    @patch("continuum.llm.client.setup_langfuse")
    def test_get_provider_from_model(self, mock_setup):
        logger.info("LLMClientProvider: get provider from model")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        assert client._get_provider_from_model("gpt-4") == "openai"
        assert client._get_provider_from_model("claude-3-opus") == "anthropic"
        assert client._get_provider_from_model("gemini/gemini-pro") == "gemini"
        assert client._get_provider_from_model("unknown-model") == "unknown"


class TestLLMClientJsonHelpers:
    @patch("continuum.llm.client.setup_langfuse")
    def test_log_json_mode_status_json_mode(self, mock_setup):
        logger.info("LLMClientJsonHelpers: log json mode status")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._log_json_mode_status(LLMConfig(json_mode=True))
        client._log_json_mode_status(LLMConfig(json_mode=False))

    @patch("continuum.llm.client.setup_langfuse")
    def test_validate_json_response_valid(self, mock_setup):
        logger.info("LLMClientJsonHelpers: validate json response valid")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._validate_json_response('{"key": "val"}', LLMConfig(json_mode=True))
        client._validate_json_response(None, LLMConfig(json_mode=True))
        client._validate_json_response("hello", LLMConfig(json_mode=False))

    @patch("continuum.llm.client.setup_langfuse")
    def test_validate_json_response_invalid(self, mock_setup):
        logger.info("LLMClientJsonHelpers: validate json response invalid")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        client._validate_json_response("{invalid", LLMConfig(json_mode=True))
        client._validate_json_response("not json", LLMConfig(json_mode=True))

    @patch("continuum.llm.client.setup_langfuse")
    def test_apply_json_mode_compat_gemini(self, mock_setup):
        logger.info("LLMClientJsonHelpers: apply json mode compat disables for gemini + tools")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        config = LLMConfig(json_mode=True, model="gemini/gemini-2.5-flash")
        tools = [{"type": "function", "function": {"name": "fn"}}]
        result = client._apply_json_mode_compat(config, tools)
        assert result.json_mode is False
        assert result.response_format is None

    @patch("continuum.llm.client.setup_langfuse")
    def test_apply_json_mode_compat_openai_unchanged(self, mock_setup):
        logger.info("LLMClientJsonHelpers: apply json mode compat does not change openai")
        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        config = LLMConfig(json_mode=True, model="gpt-4o")
        tools = [{"type": "function", "function": {"name": "fn"}}]
        result = client._apply_json_mode_compat(config, tools)
        assert result.json_mode is True


class TestLLMClientChatSync:
    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    def test_chat_sync(self, mock_setup, mock_get_provider):
        logger.info("LLMClientChatSync: chat sync")
        from continuum.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.complete.return_value = _make_llm_response(content="Hello!", model="gpt-4")
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        result = client.chat_sync([ChatMessage(role="user", content="hi")])
        assert result.content == "Hello!"
        assert result.model == "gpt-4"
        mock_provider.complete.assert_called_once()


class TestLLMClientChatAsync:
    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    @pytest.mark.asyncio
    async def test_chat_async(self, mock_setup, mock_get_provider):
        logger.info("LLMClientChatAsync: chat async")
        from continuum.llm.client import LLMClient

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


class TestLLMTotalDeadline:
    """A hung/endlessly-retried LLM call must be hard-bounded.

    Regression: llm_request_timeout is a per-attempt timeout, so with SDK
    retries the real worst case is several × timeout (observed ~295s at
    llm_request_timeout=60). client.chat() now wraps the completion in an
    overall deadline derived from timeout × (max_retries + 1) + backoff.
    """

    def test_deadline_derivation(self):
        from continuum.llm.client import LLMClient

        # timeout × (retries+1) + retries × 8s backoff cap
        assert LLMClient._total_llm_deadline(LLMConfig(model="m", timeout=60, max_retries=3)) == 264.0
        assert LLMClient._total_llm_deadline(LLMConfig(model="m", timeout=60, max_retries=0)) == 60.0
        # Disabled when timeout is unset/non-positive.
        assert LLMClient._total_llm_deadline(LLMConfig(model="m", timeout=0)) is None

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    @pytest.mark.asyncio
    async def test_hanging_call_raises_timeout_not_hangs(self, mock_setup, mock_get_provider):
        import asyncio

        from continuum.llm.client import LLMClient
        from continuum.llm.exceptions import LLMTimeoutError

        async def _hang(*args, **kwargs):
            await asyncio.sleep(3600)  # would hang forever without the deadline

        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(side_effect=_hang)
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        # timeout=1, retries=0 -> 1s hard ceiling; must raise quickly, not hang.
        with pytest.raises(LLMTimeoutError):
            await asyncio.wait_for(
                client.chat(
                    [ChatMessage(role="user", content="hi")],
                    config=LLMConfig(model="gpt-4o-mini", timeout=1, max_retries=0),
                    auto_session=False,
                ),
                timeout=30,  # test-level safety net; the deadline should fire at ~1s
            )

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    @pytest.mark.asyncio
    async def test_fast_call_unaffected(self, mock_setup, mock_get_provider):
        from continuum.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(
            return_value=_make_llm_response(content="ok", model="gpt-4")
        )
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        result = await client.chat(
            [ChatMessage(role="user", content="hi")],
            config=LLMConfig(model="gpt-4o-mini", timeout=60, max_retries=3),
            auto_session=False,
        )
        assert result.content == "ok"
