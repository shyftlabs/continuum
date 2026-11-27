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

    # No config= at all: the manager applies its own, as on dev.
    shared_manager.compress_if_needed.assert_awaited_once()
    assert "config" not in shared_manager.compress_if_needed.await_args.kwargs


async def test_a_manager_override_without_a_config_parameter_still_compresses(client):
    """An override whose compress_if_needed(messages, model) takes no config used to
    raise TypeError there, which was caught and logged, so compression was skipped."""
    manager = MagicMock()
    manager.config = ContextManagementConfig(enabled=True)

    async def compress_if_needed(messages, model):
        return messages, _result()

    manager.compress_if_needed = AsyncMock(side_effect=compress_if_needed)
    with patch(
        "continuum.llm.context_management.get_progressive_context_manager",
        return_value=manager,
    ):
        await client.chat(MESSAGES, auto_session=False)
        async for _ in client.chat_stream(MESSAGES):
            pass

    assert manager.compress_if_needed.await_count == 2
    for call in manager.compress_if_needed.await_args_list:
        assert "config" not in call.kwargs


def test_context_config_type_hints_resolve_at_runtime():
    import typing

    from continuum.llm.client import LLMClient

    for method in (LLMClient.chat, LLMClient.chat_stream):
        hint = typing.get_type_hints(method)["context_config"]
        assert ContextManagementConfig in typing.get_args(hint)


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


# --- run_stream with an injected client (round 2) ---------------------------------


def _stream_agent(context_management):
    agent = MagicMock()
    agent.name = "streamer"
    agent.model = "gpt-4o-mini"
    agent.temperature = 0.7
    agent.max_tokens = 256
    agent.gateway_mode = None
    agent.extra_body = None
    agent.enable_json_mode = False
    agent.json_schema = None
    agent.json_strict = False
    agent.metadata = {}
    agent.on_end = None
    agent.output_schema = None
    agent.config.output_scanners = None
    agent.config.context_management = context_management
    agent.is_handoff_tool_call.return_value = (False, None)
    agent.get_tools_for_llm.return_value = []
    return agent


async def _run_stream(llm_client, agent):
    from continuum.agent.runner import AgentRunner
    from continuum.agent.types import PrepareRunResult

    with patch("continuum.agent.runner.get_container", return_value=MagicMock()):
        runner = AgentRunner(llm_client=llm_client)
    ctx = MagicMock(max_turns=2, session_id=None, priority=5, metadata={}, recorder=None)
    ctx.run_id, ctx.trace_id = "run-1", "trace-1"
    runner._prepare_run = AsyncMock(
        return_value=PrepareRunResult(success=True, context=ctx, run_state=MagicMock(messages=[]))
    )
    runner._finalizer.finalize = AsyncMock()
    return [e async for e in runner.run_stream(agent, "hi")]


class _StrictStreamClient:
    """An injected client whose chat_stream takes no **kwargs (worked on dev)."""

    def __init__(self):
        self.calls = 0

    async def chat_stream(self, messages, tools=None, config=None, trace_metadata=None):
        self.calls += 1
        yield StreamChunk(content="streamed answer", model="m")


async def test_run_stream_works_with_a_client_that_takes_no_context_config():
    from continuum.agent.types import EventType

    llm = _StrictStreamClient()
    events = await _run_stream(llm, _stream_agent(context_management=None))

    assert llm.calls == 1
    assert not [e for e in events if e.type == EventType.RUN_ERROR]
    deltas = [e.data["content"] for e in events if e.type == EventType.CONTENT_DELTA]
    assert "streamed answer" in "".join(deltas)


async def test_run_stream_publishes_the_agents_own_config():
    """run_stream publishes the agent's config as run() does; no new kwarg reaches
    chat_stream."""
    mine = ContextManagementConfig(keep_recent_messages=4)
    seen = {}

    class _Client:
        async def chat_stream(self, **kwargs):
            seen.update(kwargs)
            seen["active"] = get_active_context_config()
            yield StreamChunk(content="ok", model="m")

    await _run_stream(_Client(), _stream_agent(context_management=mine))

    assert seen["active"] is mine
    assert "context_config" not in seen
    assert get_active_context_config() is None


# --- run_stream publishes the agent's config (round 3) -----------------------------


async def test_run_stream_works_with_a_strict_client_for_an_agent_with_a_config():
    """A chat_stream with fixed parameters worked on dev for every agent; it must
    keep working when the agent sets context_management."""
    from continuum.agent.types import EventType

    llm = _StrictStreamClient()
    events = await _run_stream(
        llm, _stream_agent(context_management=ContextManagementConfig(enabled=False))
    )

    assert llm.calls == 1
    assert not [e for e in events if e.type == EventType.RUN_ERROR]


async def test_config_less_streamed_child_under_a_parent_run_gets_the_process_wide_config():
    """A parent run() publishes its config; a tool streams a child that sets none.
    The child publishes None, as a non-streamed child does, so LLMClient falls
    back to the process-wide config instead of the parent's."""
    from continuum.agent.execution.executor import Executor

    parent_config = ContextManagementConfig(enabled=False)
    seen = {}

    class _Client:
        async def chat_stream(self, **kwargs):
            seen["child"] = get_active_context_config()
            yield StreamChunk(content="ok", model="m")

    async def _parent_loop(agent, messages, context, run_state):
        seen["parent_before"] = get_active_context_config()
        await _run_stream(_Client(), _stream_agent(context_management=None))
        seen["parent_after"] = get_active_context_config()

    executor = MagicMock(_execute_loop_impl=_parent_loop)
    parent = MagicMock(policy_store=None, config=MagicMock(context_management=parent_config))
    parent.name = "parent"
    await Executor.execute_loop(executor, parent, [], MagicMock(), MagicMock())

    assert seen["parent_before"] is parent_config
    assert seen["child"] is None
    assert seen["parent_after"] is parent_config


async def test_run_stream_structured_output_formatting_call_follows_the_agents_config():
    """The blocking formatting chat() after a streamed answer that does not validate
    runs inside run_stream's scope, so it uses the agent's config as in run()."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    mine = ContextManagementConfig(compression_threshold=0.99)
    seen = {}

    class _Client:
        async def chat_stream(self, **kwargs):
            yield StreamChunk(content="prose, not JSON", model="m")

        async def chat(self, **kwargs):
            seen["formatting"] = get_active_context_config()
            return LLMResponse(content='{"value": 1}', model="m")

    agent = _stream_agent(context_management=mine)
    agent.output_schema = Answer
    agent.output_schema_strict = False
    await _run_stream(_Client(), agent)

    assert seen["formatting"] is mine


async def test_stream_executor_publishes_the_agents_config():
    from continuum.agent.execution.stream_executor import StreamExecutor

    mine = ContextManagementConfig(keep_recent_messages=3)
    seen = {}

    class _Client:
        async def chat_stream(self, messages, tools=None, config=None, trace_metadata=None):
            seen["active"] = get_active_context_config()
            yield StreamChunk(content="ok", model="m")

    ctx = MagicMock(max_turns=1, session_id=None, metadata={})
    ctx.run_id, ctx.trace_id = "run-1", "trace-1"
    with pytest.warns(DeprecationWarning):
        async for _ in StreamExecutor(_Client()).execute_stream(
            _stream_agent(context_management=mine), [], ctx, MagicMock()
        ):
            pass

    assert seen["active"] is mine
    assert get_active_context_config() is None


# --- the message-builder pass (round 2) --------------------------------------------


def _builder_agent(context_management=None):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="builder-agent",
        instructions="You are helpful.",
        config=AgentConfig(
            input_sanitization=False,
            injection_detection=False,
            context_management=context_management,
        ),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


async def _prepare(agent):
    from continuum.agent.execution.message_builder import MessageBuilder
    from continuum.agent.utils.context_utils import create_run_context

    session = MagicMock(get_conversation_history=AsyncMock(return_value=[]))
    memory = MagicMock(retrieve_memories=AsyncMock(return_value=[]))
    builder = MessageBuilder(memory_service=memory, session_service=session)
    return await builder.prepare_messages(agent, "hello", create_run_context())


async def test_message_builder_without_an_agent_config_uses_the_managers_config(shared_manager):
    """A process-wide enable_caching=False (or TTL) set through
    get_progressive_context_manager(config=...) applies to this pass again; it was
    replaced by a default config, so summaries were cached anyway."""
    shared_manager.config = ContextManagementConfig(enabled=True, enable_caching=False)

    await _prepare(_builder_agent())

    assert shared_manager.compress_if_needed.await_args.kwargs["config"] is shared_manager.config


async def test_message_builder_does_not_make_an_agents_config_the_process_default(monkeypatch):
    """The first agent through the builder created the process-wide manager with its
    own config, so its settings became every config-less caller's fallback."""
    import continuum.llm.context_management as cm

    monkeypatch.setattr(cm, "_global_context_manager", None)
    mine = ContextManagementConfig(keep_recent_messages=3, enable_caching=False)

    await _prepare(_builder_agent(context_management=mine))

    assert cm.get_progressive_context_manager().config is not mine
    assert cm.get_progressive_context_manager().config == ContextManagementConfig()
