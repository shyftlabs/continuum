"""
Tests for the MaxTurnsExceededError partial-response fix.

Covers:
- runner.run() still raises MaxTurnsExceededError (non-breaking)
- exception carries partial_response with correct status and run_artifacts
- _finalizer.finalize() is called (not handle_error) so tracing/metrics fire
- agent.on_end hook fires before raise
- circuit breaker is NOT penalised for a max-turns outcome
- save_session_data is skipped (no dangling user message in Redis)
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agent.exceptions import MaxTurnsExceededError
from orchestrator.agent.types import AgentResponse, ResponseStatus, RunState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_agent(name="test-agent", log_to_session=True):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(log_to_session=log_to_session),
        memory_config=AgentMemoryConfig(),
    )


def _make_run_state(messages=None):
    rs = RunState(run_id="run-test")
    rs.push_agent("test-agent")
    rs.messages = messages or []
    return rs


def _make_finalizer():
    from orchestrator.agent.execution.run_finalizer import RunFinalizer
    from orchestrator.agent.services.session_service import SessionService
    from orchestrator.agent.services.context_service import ContextService
    from orchestrator.agent.execution.run_lifecycle import RunLifecycle

    sess_svc = MagicMock(spec=SessionService)
    sess_svc.save_messages = AsyncMock()
    sess_svc.save_tool_context_state = AsyncMock()

    ctx_svc = MagicMock(spec=ContextService)
    ctx_svc.save_run_state = AsyncMock()

    lifecycle = MagicMock(spec=RunLifecycle)
    lifecycle.report_metrics = AsyncMock()
    lifecycle.end_trace = AsyncMock()

    sc = MagicMock()
    sc.is_enabled = True

    finalizer = RunFinalizer(
        session_service=sess_svc,
        context_service=ctx_svc,
        lifecycle=lifecycle,
        session_client=sc,
    )
    return finalizer, sess_svc, ctx_svc, lifecycle


def _metrics_patch():
    m = MagicMock()
    m.record_latency = MagicMock()
    m.track_tokens = MagicMock()
    return patch("orchestrator.observability.metrics.get_metrics_collector", return_value=m)


def _make_runner_with_executor(executor_mock, finalizer_mock=None):
    from orchestrator.agent.runner import AgentRunner

    llm = MagicMock()
    llm.is_enabled = True
    sc = MagicMock()
    sc.is_enabled = True

    with patch("orchestrator.agent.runner.get_container") as mock_container:
        container = MagicMock()
        container.llm_client = llm
        container.memory_client = MagicMock()
        container.session_client = sc
        container.tool_executor = MagicMock()
        mock_container.return_value = container

        runner = AgentRunner(
            llm_client=llm,
            session_client=sc,
            memory_client=MagicMock(),
            tool_executor=MagicMock(),
        )

    runner._executor = executor_mock
    if finalizer_mock:
        runner._finalizer = finalizer_mock
    return runner


def _stub_prepare_run(runner, run_state):
    from orchestrator.agent.types import PrepareRunResult
    from orchestrator.agent.utils.context_utils import create_run_context

    ctx = create_run_context(session_id="sess-1", max_turns=4)
    result = PrepareRunResult(
        success=True,
        context=ctx,
        run_state=run_state,
        user_message_index=1,
        tool_context_state=None,
    )
    runner._prepare_run = AsyncMock(return_value=result)
    return ctx


def _make_executor_that_raises():
    executor = MagicMock()
    executor.execute_loop = AsyncMock(
        side_effect=MaxTurnsExceededError(max_turns=4, current_turn=4, agent_name="test-agent")
    )
    return executor


def _make_noop_finalizer():
    finalizer = MagicMock()
    finalizer.finalize = AsyncMock()
    finalizer.handle_error = AsyncMock()
    return finalizer


# ---------------------------------------------------------------------------
# Tests for RunFinalizer session save guard
# ---------------------------------------------------------------------------

class TestFinalizerSessionGuard:
    async def test_skips_save_messages_on_max_turns_reached(self):
        finalizer, sess_svc, ctx_svc, _ = _make_finalizer()
        agent = _make_agent()
        from orchestrator.agent.utils.context_utils import create_run_context

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = AgentResponse(
            content="",
            agent_name="test-agent",
            status=ResponseStatus.MAX_TURNS_REACHED,
            error="max turns hit",
        )

        with _metrics_patch():
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(), [])

        sess_svc.save_messages.assert_not_called()

    async def test_run_state_still_marked_completed_on_max_turns(self):
        finalizer, _, ctx_svc, _ = _make_finalizer()
        agent = _make_agent()
        from orchestrator.agent.utils.context_utils import create_run_context
        from orchestrator.agent.types import RunStatus

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = AgentResponse(
            content="",
            agent_name="test-agent",
            status=ResponseStatus.MAX_TURNS_REACHED,
        )

        with _metrics_patch():
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(), [])

        assert rs.status == RunStatus.COMPLETED
        ctx_svc.save_run_state.assert_called_once()

    async def test_tracing_ends_on_max_turns(self):
        finalizer, _, _, lifecycle = _make_finalizer()
        agent = _make_agent()
        from orchestrator.agent.utils.context_utils import create_run_context

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = AgentResponse(
            content="",
            agent_name="test-agent",
            status=ResponseStatus.MAX_TURNS_REACHED,
        )

        with _metrics_patch():
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(), [])

        lifecycle.end_trace.assert_called_once()

    async def test_save_messages_called_normally_on_success(self):
        finalizer, sess_svc, _, _ = _make_finalizer()
        agent = _make_agent()
        from orchestrator.agent.utils.context_utils import create_run_context

        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = AgentResponse(
            content="done",
            agent_name="test-agent",
            status=ResponseStatus.SUCCESS,
        )

        with _metrics_patch():
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(),
                                     [{"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "done"}])

        sess_svc.save_messages.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for AgentRunner.run() on MaxTurnsExceededError
# ---------------------------------------------------------------------------

class TestRunnerMaxTurnsExceededError:
    async def test_still_raises_max_turns_exceeded_error(self):
        run_state = _make_run_state()
        runner = _make_runner_with_executor(_make_executor_that_raises(), _make_noop_finalizer())
        _stub_prepare_run(runner, run_state)
        agent = _make_agent()
        runner._circuit_breaker = MagicMock()
        runner._circuit_breaker.check = MagicMock()

        with pytest.raises(MaxTurnsExceededError):
            await runner.run(agent, "q")

    async def test_exception_carries_partial_response(self):
        run_state = _make_run_state()
        runner = _make_runner_with_executor(_make_executor_that_raises(), _make_noop_finalizer())
        _stub_prepare_run(runner, run_state)
        agent = _make_agent()
        runner._circuit_breaker = MagicMock()
        runner._circuit_breaker.check = MagicMock()

        with pytest.raises(MaxTurnsExceededError) as exc_info:
            await runner.run(agent, "q")

        assert exc_info.value.partial_response is not None
        assert exc_info.value.partial_response.status == ResponseStatus.MAX_TURNS_REACHED

    async def test_partial_response_has_correct_run_artifacts(self):
        run_state = _make_run_state()
        runner = _make_runner_with_executor(_make_executor_that_raises(), _make_noop_finalizer())
        _stub_prepare_run(runner, run_state)
        agent = _make_agent()
        runner._circuit_breaker = MagicMock()
        runner._circuit_breaker.check = MagicMock()

        with pytest.raises(MaxTurnsExceededError) as exc_info:
            await runner.run(agent, "q")

        artifacts = exc_info.value.partial_response.run_artifacts
        assert artifacts["stopped_reason"] == "max_turns"
        assert artifacts["turns_used"] == 4

    async def test_finalize_called_not_handle_error(self):
        run_state = _make_run_state()
        finalizer_mock = _make_noop_finalizer()
        runner = _make_runner_with_executor(_make_executor_that_raises(), finalizer_mock)
        _stub_prepare_run(runner, run_state)
        agent = _make_agent()
        runner._circuit_breaker = MagicMock()
        runner._circuit_breaker.check = MagicMock()

        with pytest.raises(MaxTurnsExceededError):
            await runner.run(agent, "q")

        finalizer_mock.finalize.assert_called_once()
        finalizer_mock.handle_error.assert_not_called()

    async def test_on_end_hook_called_before_raise(self):
        run_state = _make_run_state()
        runner = _make_runner_with_executor(_make_executor_that_raises(), _make_noop_finalizer())
        _stub_prepare_run(runner, run_state)

        on_end = MagicMock()
        agent = _make_agent()
        agent.on_end = on_end
        runner._circuit_breaker = MagicMock()
        runner._circuit_breaker.check = MagicMock()

        with pytest.raises(MaxTurnsExceededError):
            await runner.run(agent, "q")

        on_end.assert_called_once()

    async def test_circuit_breaker_not_penalised(self):
        run_state = _make_run_state()
        runner = _make_runner_with_executor(_make_executor_that_raises(), _make_noop_finalizer())
        _stub_prepare_run(runner, run_state)
        agent = _make_agent()
        cb = MagicMock()
        cb.check = MagicMock()
        cb.record_failure = MagicMock()
        runner._circuit_breaker = cb

        with pytest.raises(MaxTurnsExceededError):
            await runner.run(agent, "q")

        cb.record_failure.assert_not_called()
