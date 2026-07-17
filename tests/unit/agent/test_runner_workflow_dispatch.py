"""Regression tests: runner.run() must dispatch workflow agents to execute().

A workflow agent (SequentialAgent, ReflectionAgent, PlannerAgent, ...) carries
its orchestration in ``execute()``. Before the fix, ``runner.run()`` pushed
every agent through the plain conversation loop, silently flattening a
workflow into one bare, tool-less LLM call — e.g. a planner nested inside a
ReflectionAgent never delegated to its team (reflection.py drives its inner
agent via ``runner.run()``).

Covers:
- runner.run(workflow_agent) dispatches to execute() and returns its response
- the plain conversation loop (execute_loop) is NOT entered for workflow agents
- message-list input is collapsed to the last user text
- caller-supplied context is passed through; ids are validated
- circuit-breaker-open short-circuits with an error response
- ReflectionAgent wrapping a workflow inner agent runs the inner execute()
- plain agents still go through the conversation loop
- run_stream() runs workflow agents via run() and surfaces the result as
  stream events (graceful fallback) instead of flattening
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from continuum.agent.base import BaseAgent
from continuum.agent.types import AgentResponse, ResponseStatus
from continuum.agent.utils.context_utils import create_run_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class StubWorkflowAgent(BaseAgent):
    """Minimal workflow agent: orchestration lives in execute()."""

    execute_calls: list = field(default_factory=list)

    async def execute(self, input_text, runner, context, llm_client=None):
        self.execute_calls.append({"input": input_text, "runner": runner, "context": context})
        return AgentResponse(
            content=f"ORCHESTRATED:{input_text}",
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
        )


def _make_runner(llm=None):
    from continuum.agent.runner import AgentRunner

    llm = llm or MagicMock()
    llm.is_enabled = True

    container = MagicMock()
    container.llm_client = llm
    container.memory_client = None
    container.session_client = None
    container.tool_executor = None

    with patch("continuum.agent.runner.get_container", return_value=container):
        runner = AgentRunner(
            llm_client=llm,
            memory_client=None,
            session_client=None,
            tool_executor=None,
        )
    return runner, llm


# ---------------------------------------------------------------------------
# runner.run() dispatch
# ---------------------------------------------------------------------------


class TestRunDispatchesWorkflowAgents:
    async def test_execute_is_dispatched_and_response_returned(self):
        runner, llm = _make_runner()
        llm.chat = AsyncMock()
        wf = StubWorkflowAgent(name="wf", instructions="orchestrate")

        response = await runner.run(wf, "hello world")

        assert len(wf.execute_calls) == 1
        call = wf.execute_calls[0]
        assert call["input"] == "hello world"
        assert call["runner"] is runner
        assert call["context"] is not None
        assert response.content == "ORCHESTRATED:hello world"
        # The plain conversation loop must never fire for a workflow agent.
        llm.chat.assert_not_awaited()

    async def test_message_list_input_collapsed_to_last_user_text(self):
        runner, _ = _make_runner()
        wf = StubWorkflowAgent(name="wf", instructions="orchestrate")

        await runner.run(
            wf,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
        )

        assert wf.execute_calls[0]["input"] == "second"

    async def test_caller_context_is_passed_through(self):
        runner, _ = _make_runner()
        wf = StubWorkflowAgent(name="wf", instructions="orchestrate")
        ctx = create_run_context(session_id="sess-1", user_id="user-1")

        await runner.run(wf, "hi", context=ctx)

        assert wf.execute_calls[0]["context"] is ctx

    async def test_scope_ids_are_validated(self):
        runner, _ = _make_runner()
        wf = StubWorkflowAgent(name="wf", instructions="orchestrate")

        # Colons are rejected: they are Redis key delimiters (see sanitization.py).
        response = await runner.run(wf, "hi", user_id="user:1")

        assert response.status == ResponseStatus.ERROR
        assert not wf.execute_calls

    async def test_circuit_breaker_open_short_circuits(self):
        from continuum.agent.utils.circuit_breaker import CircuitBreakerOpen

        runner, _ = _make_runner()
        runner._circuit_breaker.check = MagicMock(side_effect=CircuitBreakerOpen(5.0))
        wf = StubWorkflowAgent(name="wf", instructions="orchestrate")

        response = await runner.run(wf, "hi")

        assert response.status == ResponseStatus.ERROR
        assert not wf.execute_calls
        # The remaining cooldown is exposed structurally so callers can implement
        # cooldown-aware retry instead of parsing it out of the error prose.
        assert response.run_artifacts["retry_after_s"] == 5.0

    async def test_plain_agent_still_uses_conversation_loop(self):
        runner, _ = _make_runner()
        executor = MagicMock()
        executor.execute_loop = AsyncMock(
            return_value=AgentResponse(
                content="plain", agent_name="plain-agent", status=ResponseStatus.SUCCESS
            )
        )
        runner._executor = executor
        runner._finalizer = MagicMock()
        runner._finalizer.finalize = AsyncMock()
        runner._finalizer.handle_error = AsyncMock()

        from continuum.agent.types import PrepareRunResult, RunState

        rs = RunState(run_id="run-1")
        rs.push_agent("plain-agent")
        rs.messages = []
        runner._prepare_run = AsyncMock(
            return_value=PrepareRunResult(
                success=True,
                context=create_run_context(),
                run_state=rs,
                user_message_index=0,
                tool_context_state=None,
            )
        )

        agent = BaseAgent(name="plain-agent", instructions="test")
        response = await runner.run(agent, "hi")

        executor.execute_loop.assert_awaited_once()
        assert response.content == "plain"


# ---------------------------------------------------------------------------
# The tester's scenario: workflow nested inside ReflectionAgent
# ---------------------------------------------------------------------------


class TestNestedWorkflowNotFlattened:
    async def test_reflection_agent_runs_inner_workflow_execute(self):
        from continuum.agent.workflow.reflection import ReflectionAgent
        from continuum.llm.types import LLMResponse

        llm = MagicMock()
        llm.is_enabled = True
        llm.chat = AsyncMock(return_value=LLMResponse(content="PASS", model="test-model"))
        runner, _ = _make_runner(llm=llm)

        container = MagicMock()
        container.llm_client = llm

        inner = StubWorkflowAgent(name="planner", instructions="coordinate the team")
        reflector = ReflectionAgent(name="reflector", agent=inner)

        with patch("continuum.core.container.get_container", return_value=container):
            result = await runner.run(reflector, "deep research")

        # Before the fix, the inner workflow was flattened into one bare LLM
        # call and execute() never ran.
        assert len(inner.execute_calls) == 1
        assert inner.execute_calls[0]["input"] == "deep research"
        assert result.content == "ORCHESTRATED:deep research"


# ---------------------------------------------------------------------------
# run_stream() gracefully falls back to run() for workflow agents
# ---------------------------------------------------------------------------


class TestRunStreamWorkflowFallback:
    async def test_run_stream_runs_workflow_and_emits_events(self):
        from continuum.agent.types import EventType

        runner, _ = _make_runner()
        wf = StubWorkflowAgent(name="wf", instructions="orchestrate")

        events = [e async for e in runner.run_stream(wf, "hi")]

        # The workflow's execute() actually ran (not flattened, not an error).
        assert len(wf.execute_calls) == 1
        assert wf.execute_calls[0]["input"] == "hi"

        types = [e.type for e in events]
        # Full streaming event contract is honoured.
        assert types[0] == EventType.RUN_START
        assert EventType.AGENT_START in types
        assert EventType.CONTENT_COMPLETE in types
        assert types[-1] == EventType.RUN_END
        assert EventType.RUN_ERROR not in types

        # The correct orchestrated content is surfaced through the stream.
        complete = next(e for e in events if e.type == EventType.CONTENT_COMPLETE)
        assert complete.data["content"] == "ORCHESTRATED:hi"
        end = events[-1]
        assert end.data["content"] == "ORCHESTRATED:hi"
