"""
Tests for HandoffExecutor context propagation.
Verifies that conversation_id is preserved and is_handoff=True is set
on the target RunContext when executing a handoff.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agent.types import RunContext, RunState, ResponseStatus, AgentResponse
from orchestrator.agent.utils.context_utils import create_run_context


def _make_agent(name="agent-a"):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
    )


def _make_tool_call(target_name: str):
    tc = MagicMock()
    tc.function.name = f"transfer_to_{target_name}"
    tc.function.arguments = '{"reason": "test handoff"}'
    tc.id = "tc-1"
    return tc


def _make_run_state(source_agent_name="agent-a"):
    from orchestrator.agent.persistence.state import RunStateManager
    rs = RunState(run_id="run-1")
    rs.push_agent(source_agent_name)
    return rs


def _make_executor(target_agent=None, captured_contexts=None):
    """Build a HandoffExecutor that captures the target RunContext."""
    from orchestrator.agent.execution.handoff_executor import HandoffExecutor
    from orchestrator.agent.handoff.manager import HandoffManager

    hm = MagicMock(spec=HandoffManager)
    hm._max_depth = 10
    hm.detect_cycle = MagicMock(return_value=False)
    hm.prepare_handoff = AsyncMock(return_value=MagicMock(
        handoff_id="h1",
        to_dict=MagicMock(return_value={"to_agent": "agent-b"}),
    ))
    hm.build_handoff_messages = MagicMock(return_value=[{"role": "user", "content": "hand me off"}])
    hm.trace_handoff = AsyncMock()

    captured = captured_contexts if captured_contexts is not None else []

    async def fake_execute_loop(agent, messages, context, run_state):
        captured.append(context)
        return AgentResponse(content="handoff response", agent_name=agent.name, status=ResponseStatus.SUCCESS)

    inner_executor = MagicMock()
    inner_executor.execute_loop = fake_execute_loop

    he = HandoffExecutor(
        handoff_manager=hm,
        agent_registry={},
        executor=inner_executor,
    )

    ta = target_agent or _make_agent("agent-b")
    he.register_agent(ta)

    return he, captured


class TestHandoffSetsIsHandoffFlag:
    async def test_target_context_has_is_handoff_true(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert result.success is True
        assert len(captured) == 1
        assert captured[0].is_handoff is True

    async def test_source_context_is_handoff_remains_false(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-1")
        assert ctx.is_handoff is False

        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        # Original context must not be mutated
        assert ctx.is_handoff is False


class TestHandoffPreservesConversationId:
    async def test_conversation_id_copied_to_target_context(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-1", conversation_id="conv-999")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert captured[0].conversation_id == "conv-999"

    async def test_none_conversation_id_propagates(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-1")  # no conversation_id
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert captured[0].conversation_id is None

    async def test_session_id_also_preserved(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-abc", conversation_id="conv-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert captured[0].session_id == "sess-abc"


class TestHandoffFailures:
    async def test_returns_failure_when_target_not_registered(self):
        from orchestrator.agent.execution.handoff_executor import HandoffExecutor
        from orchestrator.agent.handoff.manager import HandoffManager

        hm = MagicMock(spec=HandoffManager)
        hm._max_depth = 10
        hm.detect_cycle = MagicMock(return_value=False)

        source = _make_agent("agent-a")
        source.get_handoff = MagicMock(return_value=None)

        he = HandoffExecutor(handoff_manager=hm, agent_registry={}, executor=MagicMock())

        ctx = create_run_context()
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("orchestrator.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert result.success is False
        assert "agent-b" in result.error
