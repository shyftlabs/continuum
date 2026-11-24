"""run_stream(workflow, stream_steps=True) sends step events while the workflow runs.

Covers:
- each step's AGENT_START / AGENT_END (with its output) arrive in order, under
  the workflow's run_id, before the final result
- events arrive live: a step's AGENT_END is received while the workflow is
  still blocked on its next step
- default (stream_steps=False) output is unchanged: no step events
- a failing step sends AGENT_END with status error and the error propagates
- tool calls inside a step send TOOL_CALL_START / TOOL_CALL_END
- without an open sink, emit_step_event and execute_tool_call send nothing
- cancelling or closing the stream early leaves no pending tasks
- a workflow that raises still ends with RUN_ERROR / RUN_END on its run_id
- two concurrent steps of one agent get distinct step_ids, carried on their
  tool events
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.exceptions import AgentExecutionError
from continuum.agent.execution.step_events import (
    StepSink,
    current_step_sink,
    emit_step_event,
    open_step_sink,
)
from continuum.agent.types import AgentResponse, EventType, ResponseStatus


@dataclass
class TwoStepWorkflow(BaseAgent):
    """Runs two plain steps through runner.run(), like SequentialAgent does."""

    steps: list = field(default_factory=list)
    gate: asyncio.Event | None = None

    async def execute(self, input_text, runner, context, llm_client=None):
        out = input_text
        for i, step in enumerate(self.steps):
            if i == 1 and self.gate is not None:
                await self.gate.wait()
            resp = await runner.run(step, out, context=context)
            out = resp.content
        return AgentResponse(content=f"FINAL:{out}", agent_name=self.name)


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
        return AgentRunner(
            llm_client=llm, memory_client=None, session_client=None, tool_executor=None
        )


def _fake_steps(runner, fail: str | None = None):
    """Replace the plain-agent body so steps answer without an LLM."""
    real = runner._run_impl

    async def fake(agent, input, **kw):
        if callable(getattr(type(agent), "execute", None)):
            return await real(agent, input, **kw)
        if agent.name == fail:
            raise RuntimeError(f"{agent.name} broke")
        return AgentResponse(content=f"{input}>{agent.name}", agent_name=agent.name, turn_count=1)

    runner._run_impl = fake


def _workflow(**kw):
    steps = [BaseAgent(name="research"), BaseAgent(name="write")]
    return TwoStepWorkflow(name="pipeline", steps=steps, **kw)


async def test_step_events_arrive_in_order_before_final_result():
    runner = _make_runner()
    _fake_steps(runner)

    events = [e async for e in runner.run_stream(_workflow(), "q", stream_steps=True)]

    seq = [(e.type, e.agent_name) for e in events]
    assert seq == [
        (EventType.RUN_START, "pipeline"),
        (EventType.AGENT_START, "pipeline"),
        (EventType.AGENT_START, "research"),
        (EventType.AGENT_END, "research"),
        (EventType.AGENT_START, "write"),
        (EventType.AGENT_END, "write"),
        (EventType.CONTENT_DELTA, "pipeline"),
        (EventType.CONTENT_COMPLETE, "pipeline"),
        (EventType.AGENT_END, "pipeline"),
        (EventType.RUN_END, "pipeline"),
    ]
    steps = [e for e in events if e.data.get("step")]
    assert len(steps) == 4
    assert all(e.data["workflow"] == "pipeline" for e in steps)
    assert steps[1].data["content"] == "q>research"
    assert steps[1].data["status"] == ResponseStatus.SUCCESS.value
    assert steps[3].data["content"] == "q>research>write"
    assert len({e.run_id for e in events}) == 1
    assert events[-1].data["content"] == "FINAL:q>research>write"


async def test_step_events_are_live():
    runner = _make_runner()
    _fake_steps(runner)
    gate = asyncio.Event()
    seen = []

    async def consume():
        async for e in runner.run_stream(_workflow(gate=gate), "q", stream_steps=True):
            seen.append(e)
            if e.type == EventType.AGENT_END and e.agent_name == "research":
                gate.set()  # the workflow waits on this before step 2

    await asyncio.wait_for(consume(), timeout=5)
    assert seen[-1].type == EventType.RUN_END


async def test_default_stream_has_no_step_events():
    runner = _make_runner()
    _fake_steps(runner)

    events = [e async for e in runner.run_stream(_workflow(), "q")]

    assert not [e for e in events if e.data.get("step")]
    assert [e.type for e in events][:2] == [EventType.RUN_START, EventType.AGENT_START]
    assert events[-1].data["content"] == "FINAL:q>research>write"


async def test_failing_step_reports_error_and_raises():
    runner = _make_runner()
    _fake_steps(runner, fail="write")
    seen = []

    with pytest.raises(Exception, match="write broke"):
        async for e in runner.run_stream(_workflow(), "q", stream_steps=True):
            seen.append(e)

    end = [e for e in seen if e.type == EventType.AGENT_END and e.agent_name == "write"]
    assert end and end[0].data["status"] == ResponseStatus.ERROR.value
    assert "write broke" in end[0].data["error"]


def _tool_call(name="search"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = "{}"
    tc.id = "c1"
    return tc


async def test_tool_calls_send_step_events():
    from continuum.agent.services.tool_service import ToolService

    service = ToolService.__new__(ToolService)
    service._execute_tool_call = AsyncMock(
        side_effect=[
            ({"role": "tool", "content": "ok"}, {"tool_name": "search"}),
            ({"role": "tool", "content": "x"}, {"tool_name": "sql", "error": "timeout"}),
        ]
    )
    agent = BaseAgent(name="research")
    sink = StepSink(workflow_name="pipeline", run_id="r1")

    with open_step_sink(sink):
        await service.execute_tool_call(agent, _tool_call("search"), MagicMock())
        await service.execute_tool_call(agent, _tool_call("sql"), MagicMock())

    events = [sink.queue.get_nowait() for _ in range(sink.queue.qsize())]
    assert [(e.type, e.data["tool_name"]) for e in events] == [
        (EventType.TOOL_CALL_START, "search"),
        (EventType.TOOL_CALL_END, "search"),
        (EventType.TOOL_CALL_START, "sql"),
        (EventType.TOOL_CALL_END, "sql"),
    ]
    assert events[1].data["success"] is True
    assert events[3].data["success"] is False and events[3].data["error"] == "timeout"
    assert all(e.agent_name == "research" and e.run_id == "r1" for e in events)


async def test_nothing_is_sent_without_a_sink():
    from continuum.agent.services.tool_service import ToolService

    assert current_step_sink() is None
    emit_step_event(EventType.AGENT_START, "x")  # no-op, no error

    service = ToolService.__new__(ToolService)
    service._execute_tool_call = AsyncMock(return_value=({"role": "tool"}, {}))
    result = await service.execute_tool_call(BaseAgent(name="a"), _tool_call(), MagicMock())
    assert result == ({"role": "tool"}, {})


def _leftover_tasks():
    me = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not me and not t.done()]


async def test_cancel_while_waiting_leaves_no_pending_tasks():
    runner = _make_runner()
    _fake_steps(runner)
    gate = asyncio.Event()  # never set: the workflow blocks before step 2
    first_step_done = asyncio.Event()

    async def consume():
        async for e in runner.run_stream(_workflow(gate=gate), "q", stream_steps=True):
            if e.type == EventType.AGENT_END and e.agent_name == "research":
                first_step_done.set()

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(first_step_done.wait(), timeout=5)
    await asyncio.sleep(0)  # let the stream go back to waiting for the next event
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert _leftover_tasks() == []


async def test_close_stream_early_stops_the_workflow():
    from contextlib import aclosing

    runner = _make_runner()
    _fake_steps(runner)
    gate = asyncio.Event()  # never set

    async with aclosing(runner.run_stream(_workflow(gate=gate), "q", stream_steps=True)) as s:
        async for e in s:
            if e.type == EventType.AGENT_END and e.agent_name == "research":
                break

    assert _leftover_tasks() == []


async def test_raising_workflow_ends_with_run_error_and_run_end():
    runner = _make_runner()
    _fake_steps(runner, fail="write")
    seen = []

    with pytest.raises(AgentExecutionError, match="write broke") as raised:
        async for e in runner.run_stream(_workflow(), "q", stream_steps=True):
            seen.append(e)

    assert [e.type for e in seen[-2:]] == [EventType.RUN_ERROR, EventType.RUN_END]
    assert seen[-2].data["error_type"] == type(raised.value).__name__
    assert "write broke" in seen[-2].data["error"]
    assert seen[0].type == EventType.RUN_START
    assert len({e.run_id for e in seen}) == 1


@dataclass
class TwinWorkflow(BaseAgent):
    """Runs the same agent twice at once, like a DAG with two stages on one agent."""

    step: BaseAgent | None = None

    async def execute(self, input_text, runner, context, llm_client=None):
        a, b = await asyncio.gather(
            runner.run(self.step, "a", context=context),
            runner.run(self.step, "b", context=context),
        )
        return AgentResponse(content=f"{a.content}|{b.content}", agent_name=self.name)


async def test_concurrent_steps_of_one_agent_have_distinct_step_ids():
    from continuum.agent.services.tool_service import ToolService

    runner = _make_runner()
    service = ToolService.__new__(ToolService)
    service._execute_tool_call = AsyncMock(return_value=({"role": "tool"}, {}))
    real = runner._run_impl

    async def fake(agent, input, **kw):
        if callable(getattr(type(agent), "execute", None)):
            return await real(agent, input, **kw)
        await asyncio.sleep(0)  # let the other step start, so the two interleave
        await service.execute_tool_call(agent, _tool_call(f"tool_{input}"), MagicMock())
        return AgentResponse(content=input, agent_name=agent.name, turn_count=1)

    runner._run_impl = fake
    wf = TwinWorkflow(name="twins", step=BaseAgent(name="worker"))

    events = [e async for e in runner.run_stream(wf, "q", stream_steps=True)]

    steps = [e for e in events if e.data.get("step")]
    starts = [e for e in steps if e.type == EventType.AGENT_START]
    assert [e.agent_name for e in starts] == ["worker", "worker"]
    ids = {e.data["step_id"] for e in starts}
    assert len(ids) == 2
    # Every step event carries its step's id, tool events included.
    by_id = {i: [e for e in steps if e.data.get("step_id") == i] for i in ids}
    assert sum(len(v) for v in by_id.values()) == len(steps)
    for evs in by_id.values():
        assert [e.type for e in evs] == [
            EventType.AGENT_START,
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_END,
            EventType.AGENT_END,
        ]
        tool = evs[1].data["tool_name"]
        assert evs[3].data["content"] == tool.removeprefix("tool_")
