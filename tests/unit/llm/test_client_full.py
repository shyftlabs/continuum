"""Comprehensive tests for llm/client.py."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.client import LLMClient
from orchestrator.llm.types import ChatMessage, LLMResponse

logger = logging.getLogger(__name__)


class TestLLMClientInit:
    def test_default_init(self):
        logger.info("LLMClientInit: default init")
        client = LLMClient()
        assert client is not None
        assert client.default_config is not None

    def test_custom_config(self):
        logger.info("LLMClientInit: custom config")
        from orchestrator.llm.config import LLMConfig

        config = LLMConfig(model="gpt-4", temperature=0.5)
        client = LLMClient(config=config)
        assert client.default_config.model == "gpt-4"
        assert client.default_config.temperature == 0.5

    def test_langfuse_enabled(self):
        logger.info("LLMClientInit: langfuse enabled")
        client = LLMClient(enable_langfuse=False)
        assert client._langfuse_enabled is False


class TestLLMClientProperties:
    def test_default_config(self):
        logger.info("LLMClientProperties: default config")
        client = LLMClient()
        assert client.default_config is not None


def _make_mock_response(content="Hello!", has_tool_calls=False):
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.role = "assistant"
    mock_message.function_call = None
    mock_message.audio = None
    mock_message.tool_calls = None

    if has_tool_calls:
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.type = "function"
        mock_tc.function = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"q": "test"}'
        mock_message.tool_calls = [mock_tc]

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"
    mock_choice.index = 0

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-4o"
    mock_response.id = "chatcmpl-123"
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": content, "role": "assistant"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return mock_response


class TestLLMClientChat:
    @pytest.mark.asyncio
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.get_provider")
    async def test_chat_basic(self, mock_get_provider, mock_setup):
        logger.info("LLMClientChat: chat basic")
        from orchestrator.llm.types import Usage

        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(
            return_value=LLMResponse(
                model="gpt-4o",
                content="Hello!",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        )
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        messages = [ChatMessage(role="user", content="Hi")]
        result = await client.chat(messages, auto_session=False)
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.get_provider")
    async def test_chat_with_model_override(self, mock_get_provider, mock_setup):
        logger.info("LLMClientChat: chat with model override")
        from orchestrator.llm.types import Usage

        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(
            return_value=LLMResponse(
                model="gpt-4",
                content="response",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        )
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        messages = [ChatMessage(role="user", content="Hi")]
        result = await client.chat(messages, model="gpt-4", auto_session=False)
        assert result.content == "response"

    @pytest.mark.asyncio
    @patch("orchestrator.llm.client.setup_langfuse")
    @patch("orchestrator.llm.client.get_provider")
    async def test_chat_exception(self, mock_get_provider, mock_setup):
        logger.info("LLMClientChat: chat exception")
        mock_provider = MagicMock()
        mock_provider.acomplete = AsyncMock(side_effect=Exception("API Error"))
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        messages = [ChatMessage(role="user", content="Hi")]
        with pytest.raises(Exception):
            await client.chat(messages, auto_session=False)


class TestLLMClientConvertMessages:
    def test_convert_messages_chat_message(self):
        logger.info("LLMClientConvertMessages: convert messages chat message")
        client = LLMClient()
        messages = [ChatMessage(role="user", content="hello")]
        result = client._convert_messages(messages)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_convert_messages_dict(self):
        logger.info("LLMClientConvertMessages: convert messages dict")
        client = LLMClient()
        messages = [{"role": "user", "content": "hello"}]
        result = client._convert_messages(messages)
        assert isinstance(result, list)

    def test_convert_messages_mixed(self):
        logger.info("LLMClientConvertMessages: convert messages mixed")
        client = LLMClient()
        messages = [
            ChatMessage(role="user", content="first"),
            {"role": "assistant", "content": "second"},
        ]
        result = client._convert_messages(messages)
        assert len(result) == 2
