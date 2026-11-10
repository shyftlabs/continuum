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
from unittest.mock import MagicMock, patch

from continuum.agent.base import BaseAgent
from continuum.agent.execution.run_lifecycle import RunLifecycle
from continuum.agent.types import AgentResponse, ResponseStatus, RunContext
from continuum.observability.trace_context import clear_trace_context, get_current_trace_id
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
