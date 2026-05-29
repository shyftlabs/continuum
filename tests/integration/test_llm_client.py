"""
Integration tests for LLM Client — real LLM API calls.

Tests chat completion, streaming, token counting with real models.
Uses the configured DEFAULT_LLM_MODEL from .env.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


def _skip_if_no_api_key():
    """Skip if no LLM API key is configured."""
    has_key = any(
        os.getenv(k)
        for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]
    )
    if not has_key:
        pytest.skip("No LLM API key configured")


class TestLLMClientIntegration:
    async def test_chat_returns_response(self, real_llm_client):
        """Test real LLM chat completion."""
        _skip_if_no_api_key()

        messages = [
            {"role": "user", "content": "Reply with exactly one word: hello"},
        ]
        response = await real_llm_client.chat(messages=messages, auto_session=False)
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    async def test_chat_stream_yields_chunks(self, real_llm_client):
        """Test real LLM streaming returns chunks."""
        _skip_if_no_api_key()

        messages = [
            {"role": "user", "content": "Count from 1 to 3"},
        ]
        chunks = []
        try:
            async for chunk in real_llm_client.chat_stream(messages=messages, auto_session=False):
                chunks.append(chunk)
        except Exception as e:
            if "expired" in str(e).lower() or "invalid" in str(e).lower():
                pytest.skip(f"API key issue: {e}")
            raise

        assert len(chunks) > 0
        # At least some chunks should have content
        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) > 0

    def test_count_tokens_real_model(self, real_llm_client):
        """Test token counting with real model."""
        messages = [
            {"role": "user", "content": "Hello, how are you doing today?"},
        ]
        count = real_llm_client.count_tokens(messages)
        assert count > 0
        assert count < 100  # Simple message shouldn't be > 100 tokens

    def test_count_tokens_fallback_for_unknown_model(self, real_llm_client):
        """Test fallback estimation for unknown model."""
        messages = [
            {"role": "user", "content": "x" * 300},
        ]
        count = real_llm_client.count_tokens(messages, model="fake-model-999")
        # Should use fallback, return > 0
        assert count > 0

    async def test_chat_with_system_message(self, real_llm_client):
        """Test chat with system + user messages."""
        _skip_if_no_api_key()

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        response = await real_llm_client.chat(messages=messages, auto_session=False)
        assert "4" in response.content

    async def test_chat_usage_returned(self, real_llm_client):
        """Test that token usage is returned from real API."""
        _skip_if_no_api_key()

        messages = [{"role": "user", "content": "Say hi"}]
        response = await real_llm_client.chat(messages=messages, auto_session=False)
        assert response.usage is not None
        assert response.usage.total_tokens > 0
