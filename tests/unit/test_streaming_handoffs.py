"""
Unit tests for streaming handoff execution (Fix 5).

Tests the handoff branch inside run_stream:
- No HandoffExecutor → HANDOFF_END with success=False, stream ends
- HandoffExecutor returns failure → HANDOFF_END with error, stream ends
- HandoffExecutor returns success → correct event sequence emitted
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agent.types import (
    AgentEvent,
    AgentResponse,
    EventType,
    HandoffResult,
    PrepareRunResult,
    ResponseStatus,
    RunContext,
    RunState,
    generate_handoff_id,
    generate_run_id,
)
from orchestrator.llm.types import FunctionCall, StreamChunk, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_context(run_id: str = "run-test") -> RunContext:
    return RunContext(run_id=run_id, max_turns=5)


def _make_run_state(run_id: str = "run-test") -> RunState:
    return RunState(run_id=run_id)


def _make_prepare_run_result() -> PrepareRunResult:
    ctx = _make_run_context()
    result = PrepareRunResult(
        success=True,
        context=ctx,
        run_state=_make_run_state(),
        user_message_index=0,
    )
    return result


def _make_handoff_tool_call(target: str) -> ToolCall:
    return ToolCall(
        id="tc-handoff-1",
        type="function",
        function=FunctionCall(
            name=f"transfer_to_{target}",
            arguments='{"reason": "test handoff"}',
        ),
    )


def _make_agent(name: str = "source-agent", handoff_target: str = "target-agent"):
    """Fake BaseAgent that reports one handoff tool call."""
    agent = MagicMock()
    agent.name = name
    agent.get_tools_for_llm = MagicMock(return_value=[])
    agent.on_end = None

    # Real values for everything LLMConfig.from_agent_config() feeds into pydantic
    # on the streaming-handoff path. The mock predates the Smart-Gateway
    # `gateway_mode` field, so these must be concrete (not auto-MagicMock).
    agent.model = "gpt-4o-mini"
    agent.temperature = 0.7
    agent.max_tokens = 1024
    agent.gateway_mode = None
    agent.enable_json_mode = False
    agent.json_schema = None
    agent.json_strict = False

    agent.is_handoff_tool_call = MagicMock(
        side_effect=lambda tool_name: (
            (True, handoff_target)
            if tool_name == f"transfer_to_{handoff_target}"
            else (False, None)
        )
    )
    return agent


async def _run_stream_events(runner, agent, input_text: str) -> list[AgentEvent]:
    """Collect all events from run_stream into a list."""
    events = []
    async for event in runner.run_stream(agent, input_text):
        events.append(event)
    return events


def _event_types(events: list[AgentEvent]) -> list[str]:
    return [e.type for e in events]


# ---------------------------------------------------------------------------
# Fixtures: minimal AgentRunner with mocked internals
# ---------------------------------------------------------------------------


def _make_runner(handoff_executor=None):
    """Build an AgentRunner with all heavy dependencies mocked out."""
    from orchestrator.agent.runner import AgentRunner

    runner = AgentRunner.__new__(AgentRunner)

    # LLM client — default: yields one tool call chunk then stops
    mock_llm = MagicMock()
    runner._llm_client = mock_llm

    # Other services — mocked to no-ops
    runner._memory_client = None
    runner._session_client = None
    runner._tool_executor = MagicMock()
    runner._tracing_manager = None
    runner._state_manager = None
    runner._agent_registry = {}
    runner._config = MagicMock()
    runner._config.circuit_breaker_threshold = 5
    runner._config.circuit_breaker_cooldown = 60

    from orchestrator.agent.utils.circuit_breaker import CircuitBreaker

    runner._circuit_breaker = CircuitBreaker(threshold=5, cooldown=60)

    runner._handoff_executor = handoff_executor

    # Services used by run_stream
    runner._tool_service = MagicMock()
    runner._finalizer = MagicMock()
    runner._finalizer.finalize = AsyncMock()
    runner._finalizer.handle_error = AsyncMock()

    return runner


def _patch_prepare_run(runner, prepare_result: PrepareRunResult):
    """Patch _prepare_run to return a fixed result."""
    runner._prepare_run = AsyncMock(return_value=prepare_result)


def _make_stream_with_handoff_call(target: str):
    """LLM stream that yields one chunk with a handoff tool call."""
    tc = _make_handoff_tool_call(target)
    chunk = StreamChunk(tool_calls=[tc], is_finished=True)

    async def _stream(*args, **kwargs):
        yield chunk

    return _stream


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingHandoffNoExecutor:
    @pytest.mark.asyncio
    async def test_yields_handoff_end_with_failure_when_no_executor(self):
        """When _handoff_executor is None, HANDOFF_END has success=False."""
        runner = _make_runner(handoff_executor=None)
        prepare_result = _make_prepare_run_result()
        _patch_prepare_run(runner, prepare_result)

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")
        types = _event_types(events)

        assert EventType.HANDOFF_START in types
        assert EventType.HANDOFF_END in types

        handoff_end = next(e for e in events if e.type == EventType.HANDOFF_END)
        assert handoff_end.data.get("success") is False
        assert handoff_end.data.get("error") is not None

    @pytest.mark.asyncio
    async def test_stream_ends_after_handoff_failure_no_executor(self):
        """Stream ends cleanly after HANDOFF_END when no executor — no RUN_ERROR."""
        runner = _make_runner(handoff_executor=None)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")
        types = _event_types(events)

        assert EventType.RUN_ERROR not in types


class TestStreamingHandoffExecutorFails:
    @pytest.mark.asyncio
    async def test_yields_handoff_end_with_error_on_executor_failure(self):
        """When executor.execute_handoff returns success=False, HANDOFF_END carries the error."""
        failed_result = HandoffResult(
            handoff_id=generate_handoff_id(),
            from_agent="source-agent",
            to_agent="target-agent",
            success=False,
            error="Target agent not found",
        )
        mock_executor = MagicMock()
        mock_executor.execute_handoff = AsyncMock(return_value=failed_result)

        runner = _make_runner(handoff_executor=mock_executor)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")

        handoff_end = next(e for e in events if e.type == EventType.HANDOFF_END)
        assert handoff_end.data.get("success") is False
        assert "Target agent not found" in handoff_end.data.get("error", "")

    @pytest.mark.asyncio
    async def test_stream_ends_after_executor_failure(self):
        """Stream ends cleanly after executor failure — no further content events."""
        failed_result = HandoffResult(
            handoff_id=generate_handoff_id(),
            from_agent="source-agent",
            to_agent="target-agent",
            success=False,
            error="failed",
        )
        mock_executor = MagicMock()
        mock_executor.execute_handoff = AsyncMock(return_value=failed_result)

        runner = _make_runner(handoff_executor=mock_executor)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")
        types = _event_types(events)

        assert EventType.CONTENT_COMPLETE not in types
        assert EventType.RUN_ERROR not in types


class TestStreamingHandoffSuccess:
    def _make_success_result(self, content: str = "target response") -> HandoffResult:
        response = AgentResponse(
            content=content,
            run_id=generate_run_id(),
            agent_name="target-agent",
            status=ResponseStatus.SUCCESS,
        )
        return HandoffResult(
            handoff_id=generate_handoff_id(),
            from_agent="source-agent",
            to_agent="target-agent",
            success=True,
            response=response,
        )

    @pytest.mark.asyncio
    async def test_yields_handoff_end_with_success_true(self):
        mock_executor = MagicMock()
        mock_executor.execute_handoff = AsyncMock(return_value=self._make_success_result())

        runner = _make_runner(handoff_executor=mock_executor)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")

        handoff_end = next(e for e in events if e.type == EventType.HANDOFF_END)
        assert handoff_end.data.get("success") is True

    @pytest.mark.asyncio
    async def test_yields_handoff_return_event_with_content(self):
        mock_executor = MagicMock()
        mock_executor.execute_handoff = AsyncMock(
            return_value=self._make_success_result("answer from target")
        )

        runner = _make_runner(handoff_executor=mock_executor)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")
        types = _event_types(events)

        assert EventType.HANDOFF_RETURN in types
        handoff_return = next(e for e in events if e.type == EventType.HANDOFF_RETURN)
        assert handoff_return.data.get("content") == "answer from target"

    @pytest.mark.asyncio
    async def test_yields_content_complete_with_target_response(self):
        mock_executor = MagicMock()
        mock_executor.execute_handoff = AsyncMock(
            return_value=self._make_success_result("final answer")
        )

        runner = _make_runner(handoff_executor=mock_executor)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")

        content_complete = next((e for e in events if e.type == EventType.CONTENT_COMPLETE), None)
        assert content_complete is not None
        assert content_complete.data.get("content") == "final answer"

    @pytest.mark.asyncio
    async def test_event_order_on_success(self):
        """Events must arrive in the correct order on successful handoff."""
        mock_executor = MagicMock()
        mock_executor.execute_handoff = AsyncMock(return_value=self._make_success_result("done"))

        runner = _make_runner(handoff_executor=mock_executor)
        _patch_prepare_run(runner, _make_prepare_run_result())

        agent = _make_agent(handoff_target="target-agent")
        runner._llm_client.chat_stream = _make_stream_with_handoff_call("target-agent")

        events = await _run_stream_events(runner, agent, "hello")
        types = _event_types(events)

        hs = types.index(EventType.HANDOFF_START)
        he = types.index(EventType.HANDOFF_END)
        hr = types.index(EventType.HANDOFF_RETURN)
        cc = types.index(EventType.CONTENT_COMPLETE)

        assert hs < he < hr <= cc
