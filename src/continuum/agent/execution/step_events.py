"""Step events for streamed workflow runs.

``runner.run_stream(workflow, ..., stream_steps=True)`` opens a sink in a
context variable before it starts the workflow. The nested ``runner.run()``
calls the workflow makes (one per step) and the tools those steps call push
events into it, so the caller sees each step start, call tools and finish
while the workflow is still running instead of one block at the end.

Without an open sink every ``emit`` is a no-op, so plain runs are unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from continuum.agent.types import AgentEvent, EventType


@dataclass
class StepSink:
    """Collects step events for one streamed workflow run."""

    workflow_name: str
    run_id: str
    root_agent_id: int = 0
    trace_id: str | None = None
    queue: asyncio.Queue[AgentEvent] = field(default_factory=asyncio.Queue)

    def emit(self, type: EventType, agent_name: str, data: dict[str, Any] | None = None) -> None:
        payload = {"workflow": self.workflow_name, **(data or {})}
        self.queue.put_nowait(
            AgentEvent(
                type=type,
                agent_name=agent_name,
                run_id=self.run_id,
                data=payload,
                trace_id=self.trace_id,
            )
        )


_STEP_SINK: ContextVar[StepSink | None] = ContextVar("continuum_step_sink", default=None)


def current_step_sink() -> StepSink | None:
    """The sink of the streamed workflow this code runs under, if any."""
    return _STEP_SINK.get()


@contextmanager
def open_step_sink(sink: StepSink) -> Iterator[StepSink]:
    """Make ``sink`` the current sink; tasks created inside inherit it."""
    token = _STEP_SINK.set(sink)
    try:
        yield sink
    finally:
        _STEP_SINK.reset(token)


def emit_step_event(type: EventType, agent_name: str, data: dict[str, Any] | None = None) -> None:
    """Push a step event if a streamed workflow is listening; otherwise do nothing."""
    sink = _STEP_SINK.get()
    if sink is not None:
        sink.emit(type, agent_name, data)
