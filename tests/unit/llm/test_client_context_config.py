"""The compression pass inside LLMClient.chat()/chat_stream() follows the agent's config.

That pass runs on every executor turn. It used the process-wide manager's config
(the first agent's, or defaults), so an agent with enabled=False was still
compressed there with another agent's settings. The executor now publishes the
running agent's ContextManagementConfig; a direct call can pass context_config=.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.config import settings
from continuum.llm.context_management import (
    CompressionResult,
    ContextManagementConfig,
    agent_context_config,
    get_active_context_config,
    use_context_config,
)
from continuum.llm.types import ChatMessage, LLMResponse, StreamChunk


def _result() -> CompressionResult:
    return CompressionResult(
        original_token_count=1,
        compressed_token_count=1,
        messages_before=1,
        messages_after=1,
        was_compressed=False,
        strategy_used="none",
        compression_ratio=1.0,
        latency_ms=0.0,
    )


@pytest.fixture
def shared_manager():
    """The process-wide manager, configured by some earlier agent (enabled, caching on)."""
    manager = MagicMock()
    manager.config = ContextManagementConfig(enabled=True, keep_recent_messages=7)
    manager.compress_if_needed = AsyncMock(
        side_effect=lambda messages, model, config=None: (messages, _result())
    )
    with patch(
        "continuum.llm.context_management.get_progressive_context_manager",
        return_value=manager,
    ):
        yield manager


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "headroom_enabled", False)
    provider = MagicMock()
    provider.acomplete = AsyncMock(return_value=LLMResponse(content="ok", model="gpt-4o"))

    async def _stream(*_a, **_k):
        yield StreamChunk(content="ok")

    provider.astream = _stream
    with (
        patch("continuum.llm.client.get_provider", return_value=provider),
        patch("continuum.llm.client.setup_langfuse"),
    ):
        from continuum.llm.client import LLMClient

        yield LLMClient(enable_langfuse=False)


MESSAGES = [ChatMessage(role="user", content="hi")]


async def test_without_an_agent_config_the_shared_managers_config_is_used(client, shared_manager):
    await client.chat(MESSAGES, auto_session=False)

    assert shared_manager.compress_if_needed.await_args.kwargs["config"] is shared_manager.config


async def test_published_agent_config_disables_compression(client, shared_manager):
    with use_context_config(ContextManagementConfig(enabled=False)):
        await client.chat(MESSAGES, auto_session=False)

    shared_manager.compress_if_needed.assert_not_awaited()


async def test_published_agent_config_is_passed_to_the_manager(client, shared_manager):
    mine = ContextManagementConfig(enabled=True, enable_caching=False, keep_recent_messages=2)
    with use_context_config(mine):
        await client.chat(MESSAGES, auto_session=False)

    assert shared_manager.compress_if_needed.await_args.kwargs["config"] is mine


async def test_explicit_context_config_wins(client, shared_manager):
    published = ContextManagementConfig(enabled=True)
    explicit = ContextManagementConfig(enabled=False)
    with use_context_config(published):
        await client.chat(MESSAGES, auto_session=False, context_config=explicit)

    shared_manager.compress_if_needed.assert_not_awaited()


async def test_chat_stream_honours_context_config(client, shared_manager):
    mine = ContextManagementConfig(enabled=True, keep_recent_messages=3)
    async for _ in client.chat_stream(MESSAGES, context_config=mine):
        pass

    assert shared_manager.compress_if_needed.await_args.kwargs["config"] is mine

    shared_manager.compress_if_needed.reset_mock()
    async for _ in client.chat_stream(
        MESSAGES, context_config=ContextManagementConfig(enabled=False)
    ):
        pass
    shared_manager.compress_if_needed.assert_not_awaited()


def test_use_context_config_restores_the_previous_value():
    outer = ContextManagementConfig()
    with use_context_config(outer):
        with use_context_config(None):
            assert get_active_context_config() is None
        assert get_active_context_config() is outer
    assert get_active_context_config() is None


def test_agent_context_config_reads_the_agents_setting():
    mine = ContextManagementConfig()
    assert agent_context_config(MagicMock(config=MagicMock(context_management=mine))) is mine
    assert agent_context_config(MagicMock(config=None)) is None
    assert agent_context_config(object()) is None


async def test_executor_publishes_the_agents_config_for_its_loop():
    from continuum.agent.execution.executor import Executor

    mine = ContextManagementConfig(enabled=False)
    seen = {}

    async def _impl(agent, messages, context, run_state):
        seen["config"] = get_active_context_config()

    executor = MagicMock(_execute_loop_impl=_impl)
    agent = MagicMock(policy_store=None, config=MagicMock(context_management=mine))
    agent.name = "a"
    await Executor.execute_loop(executor, agent, [], MagicMock(), MagicMock())

    assert seen["config"] is mine
    assert get_active_context_config() is None
