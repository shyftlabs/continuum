"""Seam tests — Headroom compression wired into LLMClient.chat()/chat_stream().

Asserts:
- headroom_enabled=False (default) → zero behavior change, compressor never built.
- headroom_enabled=True → the provider receives the compressor's output list.
- Headroom runs BEFORE the summarizer (plan decision #2 ordering).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.config import settings
from continuum.llm.types import ChatMessage, LLMResponse

COMPRESSED_SENTINEL = [{"role": "user", "content": "compressed!"}]


def _mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.acomplete = AsyncMock(
        return_value=LLMResponse(content="ok", model="gpt-4o")
    )
    return provider


def _stub_compressor() -> MagicMock:
    compressor = MagicMock()
    compressor.apply = AsyncMock(return_value=COMPRESSED_SENTINEL)
    return compressor


@pytest.fixture(autouse=True)
def _disable_context_management(monkeypatch):
    """Isolate the Headroom seam from the summarizer."""
    monkeypatch.setattr(settings, "context_management_enabled", False)


class TestChatSeam:
    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    async def test_disabled_by_default_provider_gets_original(
        self, _langfuse, mock_get_provider, monkeypatch
    ):
        monkeypatch.setattr(settings, "headroom_enabled", False)
        provider = _mock_provider()
        mock_get_provider.return_value = provider

        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        with patch(
            "continuum.llm.headroom.compressor.get_headroom_compressor"
        ) as mock_get_compressor:
            await client.chat([ChatMessage(role="user", content="hi")], auto_session=False)
            mock_get_compressor.assert_not_called()

        sent_messages = provider.acomplete.call_args.args[0]
        assert sent_messages == [{"role": "user", "content": "hi"}]

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    async def test_enabled_provider_gets_compressed_list(
        self, _langfuse, mock_get_provider, monkeypatch
    ):
        monkeypatch.setattr(settings, "headroom_enabled", True)
        provider = _mock_provider()
        mock_get_provider.return_value = provider
        compressor = _stub_compressor()

        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        with patch(
            "continuum.llm.headroom.compressor.get_headroom_compressor",
            return_value=compressor,
        ):
            await client.chat([ChatMessage(role="user", content="hi")], auto_session=False)

        compressor.apply.assert_awaited_once()
        # apply() received the assembled messages_dict + model
        applied_messages, applied_model = compressor.apply.call_args.args
        assert applied_messages == [{"role": "user", "content": "hi"}]
        assert applied_model  # effective model string
        # provider received the compressor's output (rebind semantics)
        sent_messages = provider.acomplete.call_args.args[0]
        assert sent_messages == COMPRESSED_SENTINEL

    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    async def test_headroom_runs_before_summarizer(
        self, _langfuse, mock_get_provider, monkeypatch
    ):
        """Decision #2 ordering: summarizer sees POST-Headroom messages."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(settings, "context_management_enabled", True)
        provider = _mock_provider()
        mock_get_provider.return_value = provider
        compressor = _stub_compressor()

        seen_by_summarizer: dict = {}

        class StubManager:
            class config:
                enabled = True

            async def compress_if_needed(self, messages, model):
                seen_by_summarizer["messages"] = messages
                result = MagicMock()
                result.was_compressed = False
                return messages, result

        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        with (
            patch(
                "continuum.llm.headroom.compressor.get_headroom_compressor",
                return_value=compressor,
            ),
            patch(
                "continuum.llm.context_management.get_progressive_context_manager",
                return_value=StubManager(),
            ),
        ):
            await client.chat([ChatMessage(role="user", content="hi")], auto_session=False)

        assert seen_by_summarizer["messages"] == COMPRESSED_SENTINEL


class TestChatStreamSeam:
    @patch("continuum.llm.client.get_provider")
    @patch("continuum.llm.client.setup_langfuse")
    async def test_stream_enabled_provider_gets_compressed_list(
        self, _langfuse, mock_get_provider, monkeypatch
    ):
        monkeypatch.setattr(settings, "headroom_enabled", True)
        provider = MagicMock()

        async def fake_astream(messages, config, tools, tool_choice):
            fake_astream.seen = messages
            if False:  # pragma: no cover - make this an async generator
                yield None

        provider.astream = fake_astream
        mock_get_provider.return_value = provider
        compressor = _stub_compressor()

        from continuum.llm.client import LLMClient

        client = LLMClient(enable_langfuse=False)
        with patch(
            "continuum.llm.headroom.compressor.get_headroom_compressor",
            return_value=compressor,
        ):
            async for _ in client.chat_stream([ChatMessage(role="user", content="hi")]):
                pass

        assert fake_astream.seen == COMPRESSED_SENTINEL
