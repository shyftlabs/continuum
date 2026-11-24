"""Failed agent steps are retried; on_step_error decides whether the run goes on.

Before: run_agent_activity caught every error and returned status="error", so
Temporal never retried it. Now ``run_agent_step`` sets ``raise_on_error`` and
the step's retry policy applies; when the retries run out the failure comes
back as an error result, and ``on_step_error`` ("continue" by default, the old
behaviour, or "fail") decides what the workflow does with it.

Covers, with the real activity on Temporal's time-skipping test server:
- the activity: raise_on_error off returns an error result as before; on, it
  raises, retryable or not by the error, and retries an open circuit breaker
  after its cooldown
- a step that fails twice and then works is retried and the workflow completes
- "continue": a step that keeps failing is recorded and the next step runs
- "fail": the run stops as "failed", names the step, later steps do not run
- retries 0 runs once (Temporal reads 0 as unlimited); single-attempt options
- cancelling a sequential or loop run ends it CANCELED
- AgentWorkflow and ParallelAgentWorkflow parallel rules, structured keys kept
- histories recorded before the change replay with workflow.patched()
- workflows make replay-safe ids (workflow.uuid4, not uuid.uuid4)
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("temporalio")

from temporalio.client import (  # noqa: E402
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHistory,
)
from temporalio.exceptions import ApplicationError, CancelledError  # noqa: E402
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402

from continuum.agent.base import BaseAgent  # noqa: E402
from continuum.agent.exceptions import (  # noqa: E402
    AgentExecutionError,
    MaxTurnsExceededError,
    ModelAccessDeniedError,
)
from continuum.agent.types import AgentResponse, ResponseStatus  # noqa: E402
from continuum.exceptions import ConfigurationError  # noqa: E402
from continuum.temporal.activities import (  # noqa: E402
    run_agent_activity,
    send_notification_activity,
)
from continuum.temporal.registry import get_agent_registry, reset_agent_registry  # noqa: E402
from continuum.temporal.types import AgentActivityParams, WorkflowInput  # noqa: E402
from continuum.temporal.workflows import steps  # noqa: E402
from continuum.temporal.workflows.agent_workflow import AgentWorkflow  # noqa: E402
from continuum.temporal.workflows.loop_workflow import (  # noqa: E402
    LoopAgentWorkflow,
    LoopWorkflowInput,
)
from continuum.temporal.workflows.parallel_workflow import (  # noqa: E402
    ParallelAgentWorkflow,
    ParallelWorkflowInput,
)
from continuum.temporal.workflows.sequential_workflow import (  # noqa: E402
    SequentialAgentWorkflow,
    SequentialWorkflowInput,
)

WORKFLOWS = [AgentWorkflow, SequentialAgentWorkflow, ParallelAgentWorkflow, LoopAgentWorkflow]
HISTORIES = Path(__file__).parent / "histories"


class FakeRunner:
    """Answers "<input>><agent>"; an agent in ``fail_times`` fails that many times first.

    ``raises`` sets what a failure raises (a RuntimeError by default);
    ``breaker`` makes failures an open-circuit-breaker response instead;
    ``hang`` makes the agent wait until cancelled.
    """

    def __init__(self, fail_times: dict[str, int]):
        self.fail_times = dict(fail_times)
        self.calls: Counter[str] = Counter()
        self.raises: Exception | None = None
        self.breaker = False
        self.hang = False
        self.started = asyncio.Event()

    async def run(self, agent, input, **kw):
        self.calls[agent.name] += 1
        self.started.set()
        if self.hang:
            await asyncio.sleep(3600)
        if self.calls[agent.name] <= self.fail_times.get(agent.name, 0):
            if self.breaker:
                return AgentResponse(
                    content="Service temporarily unavailable",
                    agent_name=agent.name,
                    status=ResponseStatus.ERROR,
                    error="circuit open",
                    run_artifacts={"retry_after_s": 30.0},
                )
            raise self.raises or RuntimeError(f"{agent.name} model timeout")
        content = "COMPLETE" if agent.name == "looper" else f"{input}>{agent.name}"
        return AgentResponse(content=content, agent_name=agent.name, status=ResponseStatus.SUCCESS)


@pytest.fixture
def runner():
    def make(**fail_times):
        reset_agent_registry()
        registry = get_agent_registry()
        for name in ("research", "write", "a", "b", "looper"):
            registry.register(BaseAgent(name=name))
        fake = FakeRunner(fail_times)
        registry.set_runner_factory(lambda: fake)
        return fake

    yield make
    reset_agent_registry()


async def _run(workflow_run, arg):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="step-retries",
            workflows=WORKFLOWS,
            activities=[run_agent_activity, send_notification_activity],
        ):
            return await env.client.execute_workflow(
                workflow_run,
                arg,
                id=f"step-retries-{uuid.uuid4().hex[:8]}",
                task_queue="step-retries",
            )


def _seq(*names, **kw):
    return SequentialWorkflowInput(agent_names=list(names), initial_input="q", **kw)


# ---------------------------------------------------------------------------
# The activity
# ---------------------------------------------------------------------------


async def _activity(raise_on_error: bool, agent: str = "research"):
    params = AgentActivityParams(agent_name=agent, input="q", raise_on_error=raise_on_error)
    return await ActivityEnvironment().run(run_agent_activity, params)


def test_defaults_keep_old_behaviour():
    assert AgentActivityParams(agent_name="a", input="q").raise_on_error is False
    assert WorkflowInput(steps=[], initial_input="q").on_step_error == "continue"
    assert SequentialWorkflowInput(agent_names=[], initial_input="q").on_step_error == "continue"
    assert LoopWorkflowInput(agent_name="a", initial_input="q").on_step_error == "continue"
    assert ParallelWorkflowInput(agent_names=[], initial_input="q").on_step_error == "continue"


async def test_activity_returns_error_result_without_raise_on_error(runner):
    runner(research=99)

    result = await _activity(raise_on_error=False)

    assert result.status == "error"
    assert "research model timeout" in result.error


async def test_activity_raises_retryable_error_with_raise_on_error(runner):
    runner(research=99)

    with pytest.raises(ApplicationError) as exc:
        await _activity(raise_on_error=True)

    assert not exc.value.non_retryable
    assert exc.value.type == "RuntimeError"


@pytest.mark.parametrize(
    "error",
    [
        MaxTurnsExceededError(max_turns=5, current_turn=5),
        AgentExecutionError("wrapped", original_error=MaxTurnsExceededError(max_turns=5)),
        ModelAccessDeniedError(model="gpt-x", policy_name="phi"),
        ConfigurationError("missing key"),
    ],
    ids=["max-turns", "wrapped-max-turns", "policy-denied", "config"],
)
async def test_activity_errors_a_retry_cannot_fix_are_not_retryable(runner, error):
    runner(research=99).raises = error

    with pytest.raises(ApplicationError) as exc:
        await _activity(raise_on_error=True)

    assert exc.value.non_retryable


async def test_activity_unregistered_agent_is_not_retryable(runner):
    runner()

    with pytest.raises(ApplicationError) as exc:
        await _activity(raise_on_error=True, agent="missing")
    assert exc.value.non_retryable

    result = await _activity(raise_on_error=False, agent="missing")
    assert result.status == "error"


async def test_activity_open_circuit_breaker_is_retried_after_cooldown(runner):
    runner(research=99).breaker = True

    with pytest.raises(ApplicationError) as exc:
        await _activity(raise_on_error=True)
    assert not exc.value.non_retryable
    assert exc.value.next_retry_delay == timedelta(seconds=30)

    result = await _activity(raise_on_error=False)
    assert result.status == "error"


async def test_open_circuit_breaker_step_is_retried(runner):
    fake = runner(research=1)
    fake.breaker = True

    result = await _run(SequentialAgentWorkflow.run, _seq("research", "write"))

    assert result.status == "completed"
    assert result.content == "q>research>write"
    assert fake.calls["research"] == 2


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


async def test_failing_step_is_retried_then_completes(runner):
    fake = runner(research=2)

    result = await _run(SequentialAgentWorkflow.run, _seq("research", "write"))

    assert result.status == "completed"
    assert result.content == "q>research>write"
    assert fake.calls["research"] == 3


async def test_continue_runs_the_next_step_after_a_step_keeps_failing(runner):
    fake = runner(research=99)

    result = await _run(SequentialAgentWorkflow.run, _seq("research", "write"))

    assert result.status == "completed"
    assert result.content == "q>write"
    assert fake.calls["research"] == 3
    assert fake.calls["write"] == 1
    assert [r.status for r in result.step_results] == ["error", "success"]
    assert "research model timeout" in result.step_results[0].error


async def test_fail_stops_the_workflow(runner):
    fake = runner(research=99)

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", on_step_error="fail")
    )

    assert result.status == "failed"
    assert "Step 1 (research) failed" in result.error
    assert "research model timeout" in result.error
    assert fake.calls["research"] == 3
    assert fake.calls["write"] == 0


async def test_fail_with_unregistered_agent(runner):
    runner()

    result = await _run(SequentialAgentWorkflow.run, _seq("missing", on_step_error="fail"))

    assert result.status == "failed"
    assert "not registered" in result.error


async def test_sequential_single_attempt_agents_run_once(runner):
    fake = runner(research=99)

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", single_attempt_agents=["research"])
    )

    assert result.status == "completed"
    assert fake.calls["research"] == 1


async def _cancel_mid_step(workflow_run, arg):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="step-retries",
            workflows=WORKFLOWS,
            activities=[run_agent_activity, send_notification_activity],
        ):
            handle = await env.client.start_workflow(
                workflow_run,
                arg,
                id=f"step-retries-{uuid.uuid4().hex[:8]}",
                task_queue="step-retries",
            )
            fake = get_agent_registry().get_runner()
            await asyncio.wait_for(fake.started.wait(), 30)
            await handle.cancel()
            with pytest.raises(WorkflowFailureError) as exc:
                await handle.result()
            assert isinstance(exc.value.cause, CancelledError)
            return (await handle.describe()).status


@pytest.mark.parametrize("on_step_error", ["continue", "fail"])
async def test_cancelling_sequential_run_ends_canceled(runner, on_step_error):
    runner().hang = True

    status = await _cancel_mid_step(
        SequentialAgentWorkflow.run, _seq("research", on_step_error=on_step_error)
    )

    assert status == WorkflowExecutionStatus.CANCELED


async def test_cancelling_loop_run_ends_canceled(runner):
    runner().hang = True

    status = await _cancel_mid_step(
        LoopAgentWorkflow.run,
        LoopWorkflowInput(agent_name="looper", initial_input="q", on_step_error="fail"),
    )

    assert status == WorkflowExecutionStatus.CANCELED


# ---------------------------------------------------------------------------
# AgentWorkflow
# ---------------------------------------------------------------------------


def _steps(**kw):
    return [
        {"type": "agent", "agent_name": "research", "retries": 2, **kw},
        {"type": "agent", "agent_name": "write"},
    ]


async def test_agent_workflow_continue_runs_the_next_step(runner):
    fake = runner(research=99)

    result = await _run(AgentWorkflow.run, WorkflowInput(steps=_steps(), initial_input="q"))

    assert result.status == "completed"
    assert result.content == "q>write"
    assert fake.calls["research"] == 2


async def test_agent_workflow_fail_stops_the_run(runner):
    fake = runner(research=99)

    result = await _run(
        AgentWorkflow.run,
        WorkflowInput(steps=_steps(), initial_input="q", on_step_error="fail"),
    )

    assert result.status == "failed"
    assert "research" in result.error and "model timeout" in result.error
    assert fake.calls["research"] == 2
    assert fake.calls["write"] == 0


@pytest.mark.parametrize("retries", [0, 1])
async def test_agent_step_retries_0_and_1_run_once(runner, retries):
    fake = runner(research=99)

    result = await _run(
        AgentWorkflow.run,
        WorkflowInput(steps=_steps(retries=retries), initial_input="q", on_step_error="fail"),
    )

    assert result.status == "failed"
    assert fake.calls["research"] == 1


async def test_agent_workflow_condition_agent_failure(runner):
    runner(a=99)
    step = {
        "type": "conditional",
        "condition_agent": "a",
        "retries": 1,
        "if_false": [{"type": "agent", "agent_name": "b"}],
    }

    cont = await _run(AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q"))
    assert cont.status == "completed"
    assert cont.content == "q>b"

    failed = await _run(
        AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q", on_step_error="fail")
    )
    assert failed.status == "failed"
    assert "Condition agent 'a' failed" in failed.error


async def test_agent_workflow_parallel_step(runner):
    runner(a=99)
    step = {
        "type": "parallel",
        "merge_strategy": "structured",
        "agents": [{"agent_name": "a", "retries": 1}, {"agent_name": "b"}],
    }

    cont = await _run(AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q"))
    assert cont.status == "completed"
    assert '"a": ""' in cont.content and '"q>b"' in cont.content  # the failed key stays

    failed = await _run(
        AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q", on_step_error="fail")
    )
    assert failed.status == "failed"
    assert "Parallel step failed: a:" in failed.error


# ---------------------------------------------------------------------------
# ParallelAgentWorkflow and LoopAgentWorkflow
# ---------------------------------------------------------------------------


async def test_parallel_workflow_continue_and_fail(runner):
    fake = runner(a=99)
    cont = await _run(
        ParallelAgentWorkflow.run,
        ParallelWorkflowInput(
            agent_names=["a", "b"],
            initial_input="q",
            merge_strategy="structured",
            single_attempt_agents=["a"],
        ),
    )
    assert cont.status == "completed"
    assert cont.content == str({"a": "", "b": "q>b"})  # the failed agent's key stays
    assert cont.error is None
    assert fake.calls["a"] == 1

    fake = runner(a=99)
    failed = await _run(
        ParallelAgentWorkflow.run,
        ParallelWorkflowInput(agent_names=["a", "b"], initial_input="q", on_step_error="fail"),
    )
    assert failed.status == "failed"
    assert failed.content == "q>b"  # the agents that worked still merge
    assert failed.error.startswith("a:")
    assert fake.calls["a"] == 3


async def test_loop_continue_and_fail(runner):
    fake = runner(looper=1)
    cont = await _run(
        LoopAgentWorkflow.run,
        LoopWorkflowInput(agent_name="looper", initial_input="q", single_attempt=True),
    )
    assert cont.status == "completed"
    assert [r.status for r in cont.step_results] == ["error", "success"]
    assert fake.calls["looper"] == 2

    runner(looper=99)
    failed = await _run(
        LoopAgentWorkflow.run,
        LoopWorkflowInput(agent_name="looper", initial_input="q", on_step_error="fail"),
    )
    assert failed.status == "failed"
    assert "Iteration 1 (looper) failed" in failed.error


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(HISTORIES.glob("*.json")), ids=lambda p: p.stem)
async def test_histories_from_before_the_change_replay(path):
    """Recorded on dev before this change: a failed step, then parked at an
    approval or a timer (or completed). Without workflow.patched() the new code
    completes the run where the history scheduled the next activity (TMPRL1100).
    """
    history = WorkflowHistory.from_json(path.stem, path.read_text())

    await Replayer(workflows=WORKFLOWS).replay_workflow(history)


async def test_new_histories_replay(runner):
    runner(research=99)
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="step-retries",
            workflows=WORKFLOWS,
            activities=[run_agent_activity, send_notification_activity],
        ):
            handle = await env.client.start_workflow(
                SequentialAgentWorkflow.run,
                _seq("research", "write"),
                id=f"step-retries-{uuid.uuid4().hex[:8]}",
                task_queue="step-retries",
            )
            await handle.result()
            history = await handle.fetch_history()

    assert any(e.marker_recorded_event_attributes.marker_name for e in history.events)
    await Replayer(workflows=WORKFLOWS).replay_workflow(history)


def test_workflows_make_replay_safe_ids():
    from continuum.temporal.workflows import loop_workflow, sequential_workflow

    for mod in (loop_workflow, sequential_workflow, steps):
        source = inspect.getsource(mod)
        assert "uuid.uuid4" not in source
    assert "workflow.uuid4()" in inspect.getsource(sequential_workflow)
    assert "workflow.uuid4()" in inspect.getsource(loop_workflow)
