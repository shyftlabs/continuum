"""
Tests for RunFinalizer.finalize() with the new user_message_index parameter
and updated save_session_data() behavior.
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agent.types import AgentResponse, ResponseStatus, RunState
from orchestrator.agent.utils.context_utils import create_run_context


def _make_agent(name="finalizer-agent", log_to_session=True):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(log_to_session=log_to_session),
        memory_config=AgentMemoryConfig(),
    )


def _make_finalizer(save_messages_mock=None):
    from orchestrator.agent.execution.run_finalizer import RunFinalizer
    from orchestrator.agent.services.session_service import SessionService
    from orchestrator.agent.services.context_service import ContextService
    from orchestrator.agent.execution.run_lifecycle import RunLifecycle

    sess_svc = MagicMock(spec=SessionService)
    sess_svc.save_messages = save_messages_mock or AsyncMock()
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
    return finalizer, sess_svc


def _make_run_state():
    rs = RunState(run_id="run-1")
    rs.push_agent("finalizer-agent")
    return rs


def _make_response():
    return AgentResponse(
        content="final answer",
        agent_name="finalizer-agent",
        status=ResponseStatus.SUCCESS,
    )


class TestUserMessageIndexPassthrough:
    async def test_index_passed_to_save_messages(self):
        finalizer, sess_svc = _make_finalizer()
        agent = _make_agent()
        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = _make_response()

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": "sys2"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "final answer"},
        ]

        with patch("orchestrator.observability.metrics.get_metrics_collector", return_value=MagicMock(
            record_latency=MagicMock(), track_tokens=MagicMock(),
        )):
            await finalizer.finalize(agent, ctx, rs, response, 2, None, time.time(), messages)

        sess_svc.save_messages.assert_called_once()
        call_kwargs = sess_svc.save_messages.call_args.kwargs
        assert call_kwargs["user_message_index"] == 2

    async def test_index_zero_passes_through(self):
        finalizer, sess_svc = _make_finalizer()
        agent = _make_agent()
        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = _make_response()
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        with patch("orchestrator.observability.metrics.get_metrics_collector", return_value=MagicMock(
            record_latency=MagicMock(), track_tokens=MagicMock(),
        )):
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(), messages)

        call_kwargs = sess_svc.save_messages.call_args.kwargs
        assert call_kwargs["user_message_index"] == 0


class TestSaveSessionDataGuards:
    async def test_skips_save_when_log_to_session_false(self):
        finalizer, sess_svc = _make_finalizer()
        agent = _make_agent(log_to_session=False)
        ctx = create_run_context(session_id="sess-1")
        rs = _make_run_state()
        response = _make_response()

        with patch("orchestrator.observability.metrics.get_metrics_collector", return_value=MagicMock(
            record_latency=MagicMock(), track_tokens=MagicMock(),
        )):
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(), [])

        sess_svc.save_messages.assert_not_called()

    async def test_skips_save_when_no_session_id(self):
        finalizer, sess_svc = _make_finalizer()
        agent = _make_agent()
        ctx = create_run_context()  # no session_id
        rs = _make_run_state()
        response = _make_response()

        with patch("orchestrator.observability.metrics.get_metrics_collector", return_value=MagicMock(
            record_latency=MagicMock(), track_tokens=MagicMock(),
        )):
            await finalizer.finalize(agent, ctx, rs, response, 0, None, time.time(), [])

        sess_svc.save_messages.assert_not_called()

    async def test_session_id_passed_to_save_messages(self):
        finalizer, sess_svc = _make_finalizer()
        agent = _make_agent()
        ctx = create_run_context(session_id="my-session")
        rs = _make_run_state()
        response = _make_response()

        with patch("orchestrator.observability.metrics.get_metrics_collector", return_value=MagicMock(
            record_latency=MagicMock(), track_tokens=MagicMock(),
        )):
            await finalizer.finalize(agent, ctx, rs, response, 1, None, time.time(),
                                     [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])

        call_kwargs = sess_svc.save_messages.call_args.kwargs
        assert call_kwargs["session_id"] == "my-session"
