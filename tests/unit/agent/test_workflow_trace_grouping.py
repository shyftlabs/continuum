"""Regression tests: per-request trace grouping for workflow agents.

Before the fix, ``_run_workflow_agent`` dispatched straight to ``execute()``
without touching the trace lifecycle, so a planner/pool/drafter workflow never
created a request-level parent trace. Each nested ``runner.run()`` then created
its own sessionless top-level ``agent-run-<role>`` trace, and a failing step —
with no trace context — spawned an orphan ``error-UNKNOWN_ERROR`` trace.

These tests pin the fix:
- the workflow wrapper starts the parent trace before dispatch and ends it
  after (owning it), and reports failures against it;
- ``start_trace`` reports ownership correctly (creates vs reuses);
- a non-owning (nested) run does NOT tear down the shared parent trace, so
  sibling/subsequent steps stay grouped under it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.execution.run_lifecycle import RunLifecycle
from continuum.agent.types import AgentResponse, ResponseStatus, RunContext
from continuum.observability.trace_context import (
    clear_trace_context,
    get_current_trace_id,
    set_trace_context,
)

# ---------------------------------------------------------------------------
# Wrapper wiring: _run_workflow_agent participates in the trace lifecycle
# ---------------------------------------------------------------------------


@dataclass
class _StubWorkflowAgent(BaseAgent):
    """Minimal workflow agent whose orchestration lives in execute()."""

    fail: bool = False
    execute_calls: list = field(default_factory=list)

    async def execute(self, input_text, runner, context, llm_client=None):
        self.execute_calls.append(input_text)
        if self.fail:
            raise RuntimeError("boom in a workflow step")
        return AgentResponse(
            content=f"ORCHESTRATED:{input_text}",
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
        )


def _make_runner():
    from continuum.agent.runner import AgentRunner

    llm = MagicMock()
    llm.is_enabled = True
    container = MagicMock()
    container.llm_client = llm
    container.memory_client = None
    container.session_client = None
    container.tool_executor = None

    with patch("continuum.agent.runner.get_container", return_value=container):
        runner = AgentRunner(
            llm_client=llm, memory_client=None, session_client=None, tool_executor=None
        )
    return runner


def _mock_lifecycle(owns: bool = True):
    lifecycle = MagicMock()
    lifecycle.start_trace = AsyncMock(return_value=owns)
    lifecycle.end_trace = AsyncMock()
    lifecycle.report_error = AsyncMock()
    return lifecycle


class TestWorkflowWrapperTraceLifecycle:
    async def test_success_starts_and_ends_parent_trace(self):
        runner = _make_runner()
        runner._lifecycle = _mock_lifecycle(owns=True)
        wf = _StubWorkflowAgent(name="planner", instructions="orchestrate")

        resp = await runner.run(wf, "do the thing")

        assert resp.status == ResponseStatus.SUCCESS
        assert wf.execute_calls == ["do the thing"]
        # Parent trace created before dispatch, ended after — owned by the wrapper.
        runner._lifecycle.start_trace.assert_awaited_once()
        runner._lifecycle.end_trace.assert_awaited_once()
        assert runner._lifecycle.end_trace.await_args.kwargs["owns_trace"] is True
        runner._lifecycle.report_error.assert_not_awaited()

    async def test_start_precedes_dispatch(self):
        runner = _make_runner()
        order: list[str] = []
        lc = _mock_lifecycle(owns=True)
        lc.start_trace = AsyncMock(side_effect=lambda *a, **k: order.append("start") or True)
        lc.end_trace = AsyncMock(side_effect=lambda *a, **k: order.append("end"))
        runner._lifecycle = lc

        wf = _StubWorkflowAgent(name="planner", instructions="orchestrate")
        wf.execute = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda *a, **k: order.append("execute")
            or AgentResponse(content="ok", agent_name="planner", status=ResponseStatus.SUCCESS)
        )

        await runner.run(wf, "x")

        assert order == ["start", "execute", "end"]  # trace opened BEFORE the steps run

    async def test_failing_step_is_reported_against_parent_trace(self):
        runner = _make_runner()
        runner._lifecycle = _mock_lifecycle(owns=True)
        wf = _StubWorkflowAgent(name="drafter", instructions="orchestrate", fail=True)

        with pytest.raises(Exception):
            await runner.run(wf, "will fail")

        # Failure routed through report_error (attached to the request trace),
        # not left to escape untraced into an orphan error-* trace.
        runner._lifecycle.report_error.assert_awaited_once()
        assert runner._lifecycle.report_error.await_args.kwargs["owns_trace"] is True
        runner._lifecycle.end_trace.assert_not_awaited()

    async def test_nested_workflow_does_not_own_trace(self):
        """A workflow reusing an outer trace passes owns_trace=False downstream."""
        runner = _make_runner()
        runner._lifecycle = _mock_lifecycle(owns=False)  # start_trace reused a parent
        wf = _StubWorkflowAgent(name="inner", instructions="orchestrate")

        await runner.run(wf, "nested")

        assert runner._lifecycle.end_trace.await_args.kwargs["owns_trace"] is False


# ---------------------------------------------------------------------------
# RunLifecycle ownership semantics (contextvar-level, no Langfuse needed)
# ---------------------------------------------------------------------------


def _agent():
    return BaseAgent(name="a", instructions="t")


def _ctx():
    return RunContext(run_id="r1")


class TestStartTraceOwnership:
    async def test_reuses_existing_trace_and_reports_not_owned(self):
        clear_trace_context()
        set_trace_context(trace_id="parent-trace")
        try:
            ctx = _ctx()
            owns = await RunLifecycle().start_trace(_agent(), ctx, None, "hi")
            assert owns is False  # reused the parent — must not own/tear it down
            assert ctx.trace_id == "parent-trace"
        finally:
            clear_trace_context()

    async def test_creates_and_owns_trace_when_none_exists(self):
        clear_trace_context()
        fake_trace = MagicMock()
        fake_trace.id = "new-trace"
        fake_trace.langfuse_trace = MagicMock()
        manager = MagicMock()
        manager.create_trace.return_value = fake_trace
        try:
            with patch("continuum.observability.TracingManager", return_value=manager):
                ctx = _ctx()
                owns = await RunLifecycle().start_trace(_agent(), ctx, None, "hi")
            assert owns is True  # created the trace -> owns it
            assert ctx.trace_id == "new-trace"
        finally:
            clear_trace_context()

    async def test_does_not_own_when_no_trace_gets_created(self):
        clear_trace_context()
        manager = MagicMock()
        manager.create_trace.return_value = None  # provider disabled / not sampled
        try:
            with patch("continuum.observability.TracingManager", return_value=manager):
                owns = await RunLifecycle().start_trace(_agent(), _ctx(), None, "hi")
            assert owns is False
        finally:
            clear_trace_context()


class TestTeardownOwnership:
    async def test_non_owner_end_trace_preserves_parent_trace(self):
        clear_trace_context()
        set_trace_context(trace_id="parent-trace")
        try:
            ctx = _ctx()
            ctx._langfuse_trace = MagicMock()
            resp = AgentResponse(content="ok", agent_name="a", status=ResponseStatus.SUCCESS)
            await RunLifecycle().end_trace(_agent(), ctx, resp, owns_trace=False)
            # Shared parent trace still active for the next workflow step, and a
            # nested run does NOT finalize (update) the shared trace.
            assert get_current_trace_id() == "parent-trace"
            ctx._langfuse_trace.update.assert_not_called()
        finally:
            clear_trace_context()

    async def test_owner_end_trace_finalizes_trace(self):
        clear_trace_context()
        set_trace_context(trace_id="my-trace")
        try:
            ctx = _ctx()
            ctx._langfuse_trace = MagicMock()
            resp = AgentResponse(content="ok", agent_name="a", status=ResponseStatus.SUCCESS)
            await RunLifecycle().end_trace(_agent(), ctx, resp, owns_trace=True)
            # The owner writes the final output/metadata onto its trace.
            ctx._langfuse_trace.update.assert_called_once()
        finally:
            clear_trace_context()

    async def test_non_owner_report_error_preserves_parent_trace(self):
        clear_trace_context()
        set_trace_context(trace_id="parent-trace")
        try:
            await RunLifecycle().report_error(
                _agent(), _ctx(), RuntimeError("step failed"), None, owns_trace=False
            )
            assert get_current_trace_id() == "parent-trace"
        finally:
            clear_trace_context()
