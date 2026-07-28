"""
Tests for HandoffExecutor context propagation.
Verifies that conversation_id is preserved and is_handoff=True is set
on the target RunContext when executing a handoff.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from continuum.agent.types import AgentResponse, ResponseStatus, RunState
from continuum.agent.utils.context_utils import create_run_context


def _make_agent(name="agent-a"):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
    )


def _make_tool_call(target_name: str):
    from continuum.agent.types import HANDOFF_TOOL_PREFIX

    tc = MagicMock()
    # execute_handoff() is called directly below with an explicit target, so this
    # name is not what routes the call -- but it must still be the prefix the SDK
    # actually emits, or the fixture teaches a name that does not exist.
    tc.function.name = f"{HANDOFF_TOOL_PREFIX}{target_name}"
    tc.function.arguments = '{"reason": "test handoff"}'
    tc.id = "tc-1"
    return tc


def _make_run_state(source_agent_name="agent-a"):
    rs = RunState(run_id="run-1")
    rs.push_agent(source_agent_name)
    return rs


def _make_executor(target_agent=None, captured_contexts=None):
    """Build a HandoffExecutor that captures the target RunContext."""
    from continuum.agent.execution.handoff_executor import HandoffExecutor
    from continuum.agent.handoff.manager import HandoffManager

    hm = MagicMock(spec=HandoffManager)
    hm._max_depth = 10
    hm.detect_cycle = MagicMock(return_value=False)
    hm.prepare_handoff = AsyncMock(
        return_value=MagicMock(
            handoff_id="h1",
            to_dict=MagicMock(return_value={"to_agent": "agent-b"}),
        )
    )
    hm.build_handoff_messages = MagicMock(return_value=[{"role": "user", "content": "hand me off"}])
    hm.trace_handoff = AsyncMock()

    captured = captured_contexts if captured_contexts is not None else []

    async def fake_execute_loop(agent, messages, context, run_state):
        captured.append(context)
        return AgentResponse(
            content="handoff response", agent_name=agent.name, status=ResponseStatus.SUCCESS
        )

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

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
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

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
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

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert captured[0].conversation_id == "conv-999"

    async def test_none_conversation_id_propagates(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-1")  # no conversation_id
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert captured[0].conversation_id is None

    async def test_session_id_also_preserved(self):
        source = _make_agent("agent-a")
        captured = []
        he, captured = _make_executor(captured_contexts=captured)

        ctx = create_run_context(session_id="sess-abc", conversation_id="conv-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert captured[0].session_id == "sess-abc"


class TestHandoffFailures:
    async def test_returns_failure_when_target_not_registered(self):
        from continuum.agent.execution.handoff_executor import HandoffExecutor
        from continuum.agent.handoff.manager import HandoffManager

        hm = MagicMock(spec=HandoffManager)
        hm._max_depth = 10
        hm.detect_cycle = MagicMock(return_value=False)

        source = _make_agent("agent-a")
        source.get_handoff = MagicMock(return_value=None)

        he = HandoffExecutor(handoff_manager=hm, agent_registry={}, executor=MagicMock())

        ctx = create_run_context()
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(source, "agent-b", tc, [], ctx, rs)

        assert result.success is False
        assert "agent-b" in result.error


def _make_failing_executor(target_agent, exc: Exception):
    """HandoffExecutor whose target execute_loop raises ``exc``."""
    from continuum.agent.execution.handoff_executor import HandoffExecutor
    from continuum.agent.handoff.manager import HandoffManager

    hm = MagicMock(spec=HandoffManager)
    hm._max_depth = 10
    hm.detect_cycle = MagicMock(return_value=False)
    hm.prepare_handoff = AsyncMock(
        return_value=MagicMock(
            handoff_id="h1", to_dict=MagicMock(return_value={"to_agent": "agent-b"})
        )
    )
    hm.build_handoff_messages = MagicMock(return_value=[{"role": "user", "content": "x"}])
    hm.trace_handoff = AsyncMock()

    async def failing_execute_loop(agent, messages, context, run_state):
        raise exc

    inner = MagicMock()
    inner.execute_loop = failing_execute_loop

    he = HandoffExecutor(handoff_manager=hm, agent_registry={}, executor=inner)
    he.register_agent(target_agent)
    return he


class TestHandoffRecipientLifecycleHooks:
    """Recipient agent's lifecycle hooks must fire (audit-trail completeness)."""

    async def test_recipient_on_start_and_on_end_fire(self):
        target = _make_agent("agent-b")
        target.on_start = MagicMock()
        target.on_end = MagicMock()
        he, _ = _make_executor(target_agent=target)

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(_make_agent("agent-a"), "agent-b", tc, [], ctx, rs)

        assert result.success is True
        # on_start fired with the recipient agent as first arg
        target.on_start.assert_called_once()
        assert target.on_start.call_args.args[0] is target
        # on_end fired with the recipient agent and the response
        target.on_end.assert_called_once()
        assert target.on_end.call_args.args[0] is target
        assert target.on_end.call_args.args[1]["response"] is result.response

    async def test_recipient_on_start_fires_before_on_end(self):
        target = _make_agent("agent-b")
        order: list[str] = []
        target.on_start = MagicMock(side_effect=lambda *a, **k: order.append("start"))
        target.on_end = MagicMock(side_effect=lambda *a, **k: order.append("end"))
        he, _ = _make_executor(target_agent=target)

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            await he.execute_handoff(_make_agent("agent-a"), "agent-b", tc, [], ctx, rs)

        assert order == ["start", "end"]

    async def test_recipient_on_error_fires_on_failure(self):
        target = _make_agent("agent-b")
        target.on_start = MagicMock()
        target.on_end = MagicMock()
        target.on_error = MagicMock()
        boom = RuntimeError("loop blew up")
        he = _make_failing_executor(target, boom)

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(_make_agent("agent-a"), "agent-b", tc, [], ctx, rs)

        assert result.success is False
        target.on_start.assert_called_once()  # started before failing
        target.on_error.assert_called_once()
        assert target.on_error.call_args.args[0] is target
        assert target.on_error.call_args.args[1] is boom
        target.on_end.assert_not_called()  # never completed

    async def test_raising_on_end_does_not_mask_success_as_failure(self):
        # on_end runs OUTSIDE the execute_loop try, so a hook that raises must
        # propagate rather than be caught and reported as an execution failure
        # (which would also wrongly fire on_error after on_end already ran).
        target = _make_agent("agent-b")
        target.on_end = MagicMock(side_effect=RuntimeError("hook blew up"))
        target.on_error = MagicMock()
        he, _ = _make_executor(target_agent=target)

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(_make_agent("agent-a"), "agent-b", tc, [], ctx, rs)

        # The outer execute_handoff handler catches the propagated hook error and
        # surfaces it as a handoff failure — but on_error must NOT have fired (the
        # execution itself succeeded; only the post-success hook raised).
        assert result.success is False
        target.on_end.assert_called_once()
        target.on_error.assert_not_called()

    async def test_no_hooks_configured_is_safe(self):
        # Recipient with no hooks set must not error.
        he, _ = _make_executor(target_agent=_make_agent("agent-b"))
        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state("agent-a")
        tc = _make_tool_call("agent-b")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            result = await he.execute_handoff(_make_agent("agent-a"), "agent-b", tc, [], ctx, rs)

        assert result.success is True
