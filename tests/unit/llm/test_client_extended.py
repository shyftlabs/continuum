"""Extended tests for LLM client - covering more methods and edge cases."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.types import ChatMessage, LLMResponse, StreamChunk, Usage

logger = logging.getLogger(__name__)


def _make_llm_response(**kwargs) -> LLMResponse:
    defaults = dict(model="gpt-4", content="Hello!", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    defaults.update(kwargs)
    return LLMResponse(**defaults)


class TestLLMClientSetup:
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_client_initializes_with_providers(self, mock_setup):
        logger.info("LLMClientSetup: client initializes with provider layer")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        assert client.default_config is not None

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_json_mode_pydantic_log(self, mock_setup):
        logger.info("LLMClientSetup: json mode pydantic log")
        from pydantic import BaseModel
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)

        class MyModel(BaseModel):
            name: str

        # Should not raise
        client._log_json_mode_status(LLMConfig(response_format=MyModel))

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_validate_json_response_array(self, mock_setup):
        logger.info("LLMClientSetup: validate json response array")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response('[1, 2, 3]', LLMConfig(json_mode=True))

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_validate_json_response_invalid(self, mock_setup):
        logger.info("LLMClientSetup: validate json response invalid")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response("{invalid", LLMConfig(json_mode=True))

    @patch("orchestrator.llm.client.setup_langfuse")
    def test_validate_json_response_no_format(self, mock_setup):
        logger.info("LLMClientSetup: validate json response no format")
        from orchestrator.llm.client import LLMClient
        client = LLMClient(enable_langfuse=False)
        client._validate_json_response("hello", LLMConfig(json_mode=False))


class TestLLMClientSyncStream:
    @patch("orchestrator.llm.client.get_provider")
    @patch("orchestrator.llm.client.setup_langfuse")
    def test_chat_stream_sync(self, mock_setup, mock_get_provider):
        logger.info("LLMClientSyncStream: chat stream sync")
        from orchestrator.llm.client import LLMClient

        mock_provider = MagicMock()
        mock_provider.stream.return_value = iter([
            StreamChunk(id="c1", model="gpt-4", content="Hello", is_finished=False),
            StreamChunk(id="c2", model="gpt-4", content=" world", finish_reason="stop", is_finished=True),
        ])
        mock_get_provider.return_value = mock_provider

        client = LLMClient(enable_langfuse=False)
        chunks = list(client.chat_stream_sync([ChatMessage(role="user", content="hi")]))
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        mock_provider.stream.assert_called_once()


class TestLLMRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        logger.info("LLMRateLimiter: rate limiter acquire")
        from orchestrator.llm.client import _LLMRateLimiter
        rl = _LLMRateLimiter(rpm=60)
        await rl.acquire()
        assert rl.tokens < 60

    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_acquires(self):
        logger.info("LLMRateLimiter: rate limiter multiple acquires")
        from orchestrator.llm.client import _LLMRateLimiter
        rl = _LLMRateLimiter(rpm=1000)
        for _ in range(5):
            await rl.acquire()
        assert rl.tokens < 1000
