"""A run's trace is closed when the run ends, and its id is the provider's trace id.

Found running a research agent against a real Langfuse:

* ``end_trace`` cleared the trace context inside its own ``trace_run_end`` span,
  and a span puts back the context it was opened in when it closes, so the clear
  was undone. Every later run in the same task joined the first run's trace: a
  batch job looping over companies produced one trace for the whole batch.
* The id returned as ``AgentResponse.trace_id`` was a random id, not the Langfuse
  trace's, so the model calls sat in one trace while tool spans (looked up by id)
  went to a second, orphan trace, and the run could not be linked to its trace.
"""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock, patch

from continuum.agent.base import BaseAgent
from continuum.agent.execution.run_lifecycle import RunLifecycle
from continuum.agent.types import (
    AgentResponse,
    PrepareRunResult,
    ResponseStatus,
    RunContext,
    RunState,
)
from continuum.observability.trace_context import (
    clear_trace_context,
    get_current_session_id,
    get_current_trace_id,
    get_current_user_id,
    set_trace_context,
)
from continuum.observability.tracing import TracingManager

_ids = itertools.count(1)


class _Obs:
    def end(self, **kw):
        pass

    def update(self, **kw):
        pass


class _ProviderTrace(_Obs):
    def __init__(self):
        self.id = f"provider-trace-{next(_ids)}"

    def span(self, **kw):
        return _Obs()

    def generation(self, **kw):
        return _Obs()


def _manager():
    manager = MagicMock(is_enabled=True)
    manager.trace.side_effect = lambda **kw: _ProviderTrace()
    return manager


def _agent():
    return BaseAgent(name="ask", instructions="x", model="m")


async def _one_run(lifecycle, *, error=None):
    ctx = RunContext(run_id="r")
    owns = await lifecycle.start_trace(_agent(), ctx, None, "q")
    trace_during = get_current_trace_id()
    if error:
        await lifecycle.report_error(_agent(), ctx, error, None, owns_trace=owns)
    else:
        response = AgentResponse(content="a", status=ResponseStatus.SUCCESS, trace_id=ctx.trace_id)
        await lifecycle.end_trace(_agent(), ctx, response, owns_trace=owns)
    return ctx.trace_id, trace_during, owns


async def test_trace_id_is_the_providers_and_context_is_cleared_after_the_run():
    clear_trace_context()
    with patch.object(TracingManager, "_get_provider_manager", lambda self: _manager()):
        lifecycle = RunLifecycle()
        first, during, owns = await _one_run(lifecycle)
        assert owns
        assert first.startswith("provider-trace-") and during == first
        assert get_current_trace_id() is None

        second, _, owns2 = await _one_run(lifecycle)
        assert owns2 and second != first
        assert get_current_trace_id() is None


async def test_context_is_cleared_after_a_failed_run():
    clear_trace_context()
    with patch.object(TracingManager, "_get_provider_manager", lambda self: _manager()):
        lifecycle = RunLifecycle()
        with patch("continuum.agent.execution.run_lifecycle.report_error"):
            await _one_run(lifecycle, error=RuntimeError("boom"))
        assert get_current_trace_id() is None
        second, _, owns = await _one_run(lifecycle)
        assert owns and get_current_trace_id() is None


# The run puts back the trace context it found, not an empty one: a caller that
# set user_id/session_id ambiently (``set_trace_context``) keeps them, so a later
# ``llm_client.chat()`` without ``session_id=`` still loads and saves history.


def _ambient():
    clear_trace_context()
    set_trace_context(user_id="u-ambient", session_id="s-ambient")


def _assert_ambient_back():
    assert get_current_trace_id() is None
    assert get_current_user_id() == "u-ambient"
    assert get_current_session_id() == "s-ambient"


async def test_ambient_user_and_session_survive_a_run():
    _ambient()
    with patch.object(TracingManager, "_get_provider_manager", lambda self: _manager()):
        lifecycle = RunLifecycle()
        _, during, owns = await _one_run(lifecycle)
        assert owns and during is not None
        _assert_ambient_back()
        # and a second run in the same task opens its own trace again
        _, _, owns2 = await _one_run(lifecycle)
        assert owns2
        _assert_ambient_back()


async def test_ambient_user_and_session_survive_a_failed_run():
    _ambient()
    with patch.object(TracingManager, "_get_provider_manager", lambda self: _manager()):
        lifecycle = RunLifecycle()
        with patch("continuum.agent.execution.run_lifecycle.report_error"):
            await _one_run(lifecycle, error=RuntimeError("boom"))
        _assert_ambient_back()


async def test_nested_run_on_shared_context_does_not_replace_the_owners_snapshot():
    # Workflow steps reuse the parent's RunContext. The step takes the
    # "existing trace" branch, so the owner's snapshot (taken before the trace
    # existed) is what gets restored, not the workflow's trace id.
    _ambient()
    with patch.object(TracingManager, "_get_provider_manager", lambda self: _manager()):
        lifecycle = RunLifecycle()
        ctx = RunContext(run_id="r")
        assert await lifecycle.start_trace(_agent(), ctx, None, "q")
        snapshot = ctx.trace_context_snapshot

        step_owns = await lifecycle.start_trace(_agent(), ctx, None, "step")
        assert not step_owns and ctx.trace_context_snapshot is snapshot
        response = AgentResponse(content="a", status=ResponseStatus.SUCCESS)
        await lifecycle.end_trace(_agent(), ctx, response, owns_trace=step_owns)
        assert get_current_trace_id() == ctx.trace_id  # step leaves the trace alone

        await lifecycle.end_trace(_agent(), ctx, response, owns_trace=True)
        _assert_ambient_back()


def _runner():
    from continuum.agent.runner import AgentRunner

    with patch("continuum.agent.runner.get_container") as mock_gc:
        mock_gc.return_value = MagicMock()
        return AgentRunner(llm_client=MagicMock())


async def test_run_stream_closed_early_still_closes_its_trace():
    _ambient()
    with patch.object(TracingManager, "_get_provider_manager", lambda self: _manager()):
        runner = _runner()
        agent = _agent()
        ctx = RunContext(run_id="r")
        run_state = RunState(run_id="r")
        run_state.owns_trace = await runner._lifecycle.start_trace(agent, ctx, run_state, "q")
        assert run_state.owns_trace and get_current_trace_id() == ctx.trace_id
        runner._prepare_run = AsyncMock(
            return_value=PrepareRunResult(success=True, context=ctx, run_state=run_state)
        )

        stream = runner.run_stream(agent, "q")
        await stream.__anext__()  # RUN_START, then the caller walks away
        await stream.aclose()

        _assert_ambient_back()


async def test_fork_does_not_own_the_callers_trace(monkeypatch):
    # fork() never calls start_trace, so its run must not tear down the trace
    # the caller opened around it.
    from continuum.agent.trace import config as trace_config
    from continuum.agent.trace.types import DecisionStep, DecisionTrace, StepKind
    from continuum.config import settings

    monkeypatch.setattr(settings, "decision_trace_store", "memory")
    monkeypatch.setattr(settings, "decision_trace_enabled", True)
    trace_config.get_trace_store.cache_clear()
    try:
        store = trace_config.get_trace_store()
        parent = DecisionTrace(run_id="p1", root_agent="ask")
        parent.steps.append(
            DecisionStep(
                step_id="s1",
                kind=StepKind.LLM_CALL,
                agent_name="ask",
                messages_snapshot=[{"role": "user", "content": "hi"}],
            )
        )
        await store.save(parent)

        runner = _runner()
        seen: dict = {}

        async def fake_loop(agent, messages, context, run_state):
            seen["owns_trace"] = run_state.owns_trace
            return AgentResponse(content="ok", agent_name=agent.name, status=ResponseStatus.SUCCESS)

        monkeypatch.setattr(runner._executor, "execute_loop", fake_loop)

        clear_trace_context()
        set_trace_context(trace_id="caller-trace", user_id="u-ambient", session_id="s-ambient")
        await runner.fork("p1", "s1", agent=_agent())

        assert seen["owns_trace"] is False
        assert get_current_trace_id() == "caller-trace"
        assert get_current_session_id() == "s-ambient"
    finally:
        clear_trace_context()
        trace_config.get_trace_store.cache_clear()
