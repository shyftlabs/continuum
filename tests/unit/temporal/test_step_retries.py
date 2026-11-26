"""Failed agent steps can be retried; on_step_error decides whether the run goes on.

Before: run_agent_activity caught every error and returned status="error", so
Temporal never retried it. That is still the default. A run that sets
``retry_agent_errors=True`` goes through ``run_agent_step``, which sets
``raise_on_error`` so the step's retry policy applies; when the retries run out
the failure comes back as an error result, and ``on_step_error`` ("continue" by
default, the old behaviour, or "fail") decides what the workflow does with it.
Only such opted-in runs (or "fail" runs) record the ``continuum-step-retries``
marker; a default run records none, so a 2.1 worker can replay it.

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
import json
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
from continuum.agent.utils.circuit_breaker import CircuitBreaker, CircuitState  # noqa: E402
from continuum.exceptions import ConfigurationError, InputBlockedError  # noqa: E402
from continuum.llm.exceptions import (  # noqa: E402
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMInvalidRequestError,
)
from continuum.session.exceptions import (  # noqa: E402
    SessionConnectionError,
    SessionNotFoundError,
)
from continuum.temporal.activities import (  # noqa: E402
    run_agent_activity,
    send_notification_activity,
)
from continuum.temporal.registry import get_agent_registry, reset_agent_registry  # noqa: E402
from continuum.temporal.types import (  # noqa: E402
    AGENT_ERROR_MARKER,
    AgentActivityParams,
    WorkflowInput,
)
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
    ``hang`` makes the agent wait until cancelled; ``hang_after_fail`` makes it
    outlast a 2 s step timeout once its failures are used up. ``cb`` is a real circuit breaker that
    counts failures as AgentRunner's does. ``artifacts`` sets ``run_artifacts``
    on a successful response.
    """

    def __init__(self, fail_times: dict[str, int]):
        self.fail_times = dict(fail_times)
        self.calls: Counter[str] = Counter()
        self.raises: Exception | None = None
        self.breaker = False
        self.hang = False
        self.hang_after_fail = False
        self.artifacts = None
        self.cb = CircuitBreaker(threshold=3, cooldown=60)
        self.started = asyncio.Event()

    async def run(self, agent, input, **kw):
        self.calls[agent.name] += 1
        self.started.set()
        if self.hang:
            await asyncio.sleep(3600)
        failing = self.calls[agent.name] <= self.fail_times.get(agent.name, 0)
        if self.hang_after_fail and not failing:
            # Longer than the step's 2 s timeout, short enough for the worker to shut down.
            await asyncio.sleep(5)
        if failing and not self.breaker:
            self.cb.record_failure()
        if failing:
            if self.breaker:
                return AgentResponse(
                    content="Service temporarily unavailable",
                    agent_name=agent.name,
                    status=ResponseStatus.ERROR,
                    error="circuit open",
                    run_artifacts={"retry_after_s": 30.0},
                )
            raise self.raises or RuntimeError(f"{agent.name} model timeout")
        self.cb.record_success()
        content = "COMPLETE" if agent.name == "looper" else f"{input}>{agent.name}"
        response = AgentResponse(
            content=content, agent_name=agent.name, status=ResponseStatus.SUCCESS
        )
        if self.artifacts is not None:
            response.run_artifacts = self.artifacts
        return response


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
    assert WorkflowInput(steps=[], initial_input="q").retry_agent_errors is False
    assert SequentialWorkflowInput(agent_names=[], initial_input="q").retry_agent_errors is False
    assert LoopWorkflowInput(agent_name="a", initial_input="q").retry_agent_errors is False
    assert ParallelWorkflowInput(agent_names=[], initial_input="q").retry_agent_errors is False


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


def _wrapped_twice(error: Exception) -> Exception:
    """``error`` as the ``__cause__`` of a RuntimeError the runner wrapped again."""
    middle = RuntimeError("provider call failed")
    middle.__cause__ = error
    return AgentExecutionError("wrapped", original_error=middle)


@pytest.mark.parametrize(
    "error",
    [
        MaxTurnsExceededError(max_turns=5, current_turn=5),
        AgentExecutionError("wrapped", original_error=MaxTurnsExceededError(max_turns=5)),
        ModelAccessDeniedError(model="gpt-x", policy_name="phi"),
        ConfigurationError("missing key"),
        AgentExecutionError("wrapped", original_error=LLMAuthenticationError("bad key")),
        AgentExecutionError("wrapped", original_error=LLMContextLengthError("too long")),
        AgentExecutionError("wrapped", original_error=LLMInvalidRequestError("bad request")),
        AgentExecutionError("wrapped", original_error=LLMContentFilterError("filtered")),
        InputBlockedError("blocked", scanner="injection"),
        AgentExecutionError("wrapped", original_error=SessionNotFoundError("no session")),
        _wrapped_twice(LLMAuthenticationError("bad key")),
    ],
    ids=[
        "max-turns",
        "wrapped-max-turns",
        "policy-denied",
        "config",
        "wrapped-auth",
        "wrapped-context-length",
        "wrapped-invalid-request",
        "wrapped-content-filter",
        "input-blocked",
        "wrapped-session",
        "auth-two-levels-deep",
    ],
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

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
    )

    assert result.status == "completed"
    assert result.content == "q>research>write"
    assert fake.calls["research"] == 2


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


async def test_failing_step_is_retried_then_completes(runner):
    fake = runner(research=2)

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
    )

    assert result.status == "completed"
    assert result.content == "q>research>write"
    assert fake.calls["research"] == 3


async def test_continue_runs_the_next_step_after_a_step_keeps_failing(runner):
    fake = runner(research=99)

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
    )

    assert result.status == "completed"
    assert result.content == "q>write"
    assert fake.calls["research"] == 3
    assert fake.calls["write"] == 1
    assert [r.status for r in result.step_results] == ["error", "success"]
    assert "research model timeout" in result.step_results[0].error


async def test_fail_stops_the_workflow(runner):
    fake = runner(research=99)

    result = await _run(
        SequentialAgentWorkflow.run,
        _seq("research", "write", on_step_error="fail", retry_agent_errors=True),
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
        SequentialAgentWorkflow.run,
        _seq("research", "write", single_attempt_agents=["research"], retry_agent_errors=True),
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

    result = await _run(
        AgentWorkflow.run, WorkflowInput(steps=_steps(), initial_input="q", retry_agent_errors=True)
    )

    assert result.status == "completed"
    assert result.content == "q>write"
    assert fake.calls["research"] == 2


async def test_agent_workflow_fail_stops_the_run(runner):
    fake = runner(research=99)

    result = await _run(
        AgentWorkflow.run,
        WorkflowInput(
            steps=_steps(), initial_input="q", on_step_error="fail", retry_agent_errors=True
        ),
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
        WorkflowInput(
            steps=_steps(retries=retries),
            initial_input="q",
            on_step_error="fail",
            retry_agent_errors=True,
        ),
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

    cont = await _run(
        AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q", retry_agent_errors=True)
    )
    assert cont.status == "completed"
    assert cont.content == "q>b"

    failed = await _run(
        AgentWorkflow.run,
        WorkflowInput(
            steps=[step], initial_input="q", on_step_error="fail", retry_agent_errors=True
        ),
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

    cont = await _run(
        AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q", retry_agent_errors=True)
    )
    assert cont.status == "completed"
    assert '"a": ""' in cont.content and '"q>b"' in cont.content  # the failed key stays

    failed = await _run(
        AgentWorkflow.run,
        WorkflowInput(
            steps=[step], initial_input="q", on_step_error="fail", retry_agent_errors=True
        ),
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
            retry_agent_errors=True,
        ),
    )
    assert cont.status == "completed"
    assert cont.content == str({"a": "", "b": "q>b"})  # the failed agent's key stays
    assert cont.error is None
    assert fake.calls["a"] == 1

    fake = runner(a=99)
    failed = await _run(
        ParallelAgentWorkflow.run,
        ParallelWorkflowInput(
            agent_names=["a", "b"], initial_input="q", on_step_error="fail", retry_agent_errors=True
        ),
    )
    assert failed.status == "failed"
    assert failed.content == "q>b"  # the agents that worked still merge
    assert failed.error.startswith("a:")
    assert fake.calls["a"] == 3


async def test_loop_continue_and_fail(runner):
    fake = runner(looper=1)
    cont = await _run(
        LoopAgentWorkflow.run,
        LoopWorkflowInput(
            agent_name="looper", initial_input="q", single_attempt=True, retry_agent_errors=True
        ),
    )
    assert cont.status == "completed"
    assert [r.status for r in cont.step_results] == ["error", "success"]
    assert fake.calls["looper"] == 2

    runner(looper=99)
    failed = await _run(
        LoopAgentWorkflow.run,
        LoopWorkflowInput(
            agent_name="looper", initial_input="q", on_step_error="fail", retry_agent_errors=True
        ),
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
                _seq("research", "write", retry_agent_errors=True),
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


# ---------------------------------------------------------------------------
# Round 2: error text, which errors become error steps, the breaker, inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("boom"),
        AgentExecutionError("top-level message", original_error=RuntimeError("inner")),
    ],
    ids=["plain", "wrapped"],
)
async def test_step_error_text_matches_the_old_path(runner, error):
    """step_results[].error is the text raise_on_error=False gives (the dev text)."""
    runner(research=99).raises = error
    expected = (await _activity(raise_on_error=False)).error

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
    )

    assert result.step_results[0].error == expected
    assert not expected.startswith(type(error).__name__)


async def test_worker_errors_fail_the_run_instead_of_becoming_error_steps(runner):
    """A worker without run_agent_activity fails the run, as on dev."""
    runner()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="step-retries",
            workflows=WORKFLOWS,
            activities=[send_notification_activity],
        ):
            with pytest.raises(WorkflowFailureError):
                await env.client.execute_workflow(
                    SequentialAgentWorkflow.run,
                    _seq("research", "write"),
                    id=f"step-retries-{uuid.uuid4().hex[:8]}",
                    task_queue="step-retries",
                )


def _timing_out_steps(**kw):
    # A short timeout: the test server does not skip time while an activity runs.
    return WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "research", "timeout": 2, "retries": 3},
            {"type": "agent", "agent_name": "write"},
        ],
        initial_input="q",
        **kw,
    )


@pytest.mark.parametrize("on_step_error", ["continue", "fail"])
async def test_agent_error_then_timeouts_is_an_error_step(runner, on_step_error):
    """Attempt 1 raises, attempts 2 and 3 time out: the agent error is the step's
    result (dev returned it on attempt 1), so "continue" completes the run."""
    fake = runner(research=1)
    fake.hang_after_fail = True

    result = await _run(
        AgentWorkflow.run, _timing_out_steps(on_step_error=on_step_error, retry_agent_errors=True)
    )

    assert fake.calls["research"] == 3
    assert result.step_results[0].status == "error"
    assert result.step_results[0].error == "research model timeout"
    if on_step_error == "continue":
        assert result.status == "completed"
        assert result.content == "q>write"
    else:
        assert result.status == "failed"
        assert "research model timeout" in result.error


async def test_every_attempt_timing_out_still_fails_a_continue_run(runner):
    """Timeouts alone keep 2.1's behaviour: the step raises, so the run fails
    (AgentWorkflow returns status="failed") and the next step does not run."""
    fake = runner()
    fake.hang_after_fail = True

    result = await _run(AgentWorkflow.run, _timing_out_steps())

    assert result.status == "failed"
    assert fake.calls["research"] == 3
    assert fake.calls["write"] == 0


async def test_retries_count_once_toward_the_circuit_breaker(runner):
    """A step that fails all 3 attempts counts one failure, not 3 (threshold 3)."""
    fake = runner(research=99)

    await _run(SequentialAgentWorkflow.run, _seq("research", retry_agent_errors=True))

    assert fake.calls["research"] == 3
    assert fake.cb.failure_count == 1
    assert fake.cb.state == CircuitState.CLOSED


async def test_a_permanent_error_counts_toward_the_circuit_breaker(runner):
    fake = runner(research=99)
    fake.raises = ConfigurationError("missing key")

    await _run(SequentialAgentWorkflow.run, _seq("research", retry_agent_errors=True))

    assert fake.calls["research"] == 1
    assert fake.cb.failure_count == 1


async def test_without_raise_on_error_every_failure_counts(runner):
    fake = runner(research=99)

    await _activity(raise_on_error=False)

    assert fake.cb.failure_count == 1


async def test_session_connection_error_is_retryable(runner):
    runner(research=99).raises = AgentExecutionError(
        "wrapped", original_error=SessionConnectionError("redis down")
    )

    with pytest.raises(ApplicationError) as exc:
        await _activity(raise_on_error=True)

    assert not exc.value.non_retryable
    assert AGENT_ERROR_MARKER in exc.value.details


async def test_failure_log_line_unchanged_without_raise_on_error(runner, caplog):
    runner(research=99)

    with caplog.at_level("ERROR"):
        await _activity(raise_on_error=False)

    lines = [r.getMessage() for r in caplog.records if "failed" in r.getMessage()]
    assert lines and lines[0].startswith("Agent 'research' failed: research model timeout")


@pytest.mark.parametrize("artifacts", [["not", "a", "dict"], "text"])
async def test_non_dict_run_artifacts_do_not_break_a_successful_run(runner, artifacts):
    runner().artifacts = artifacts

    result = await _activity(raise_on_error=True)

    assert result.status == "success"


@pytest.mark.parametrize(
    "workflow_run, arg",
    [
        (SequentialAgentWorkflow.run, _seq("research", on_step_error="Fail")),
        (
            LoopAgentWorkflow.run,
            LoopWorkflowInput(agent_name="looper", initial_input="q", on_step_error="Fail"),
        ),
        (
            ParallelAgentWorkflow.run,
            ParallelWorkflowInput(agent_names=["a"], initial_input="q", on_step_error="Fail"),
        ),
        (
            AgentWorkflow.run,
            WorkflowInput(steps=_steps(), initial_input="q", on_step_error="Fail"),
        ),
    ],
    ids=["sequential", "loop", "parallel", "agent"],
)
async def test_on_step_error_typo_fails_the_run(runner, workflow_run, arg):
    """Before: the worker could not decode the Literal and the run stayed RUNNING."""
    fake = runner()

    with pytest.raises(WorkflowFailureError) as exc:
        await asyncio.wait_for(_run(workflow_run, arg), 20)

    assert "on_step_error" in str(exc.value.cause)
    assert sum(fake.calls.values()) == 0


async def test_agent_workflow_on_step_error_typo_from_a_raw_dict_fails_the_run(runner):
    """A non-Python client sends a dict; as a Literal the typo could not be decoded."""
    fake = runner()
    raw = {"steps": _steps(), "initial_input": "q", "on_step_error": "contnue"}

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="step-retries",
            workflows=WORKFLOWS,
            activities=[run_agent_activity, send_notification_activity],
        ):
            with pytest.raises(WorkflowFailureError) as exc:
                await asyncio.wait_for(
                    env.client.execute_workflow(
                        "AgentWorkflow",
                        raw,
                        id=f"step-retries-{uuid.uuid4().hex[:8]}",
                        task_queue="step-retries",
                    ),
                    20,
                )

    assert "on_step_error" in str(exc.value.cause)
    assert sum(fake.calls.values()) == 0


# ---------------------------------------------------------------------------
# Round 3: no raw cause chain in history or logs, capped text, ApplicationError
# ---------------------------------------------------------------------------


def _secret() -> str:
    # Built at run time, in shapes the redaction patterns know.
    return "sk-" + uuid.uuid4().hex + uuid.uuid4().hex[:8]


def _as_the_runner_raises(inner: Exception) -> AgentExecutionError:
    """``raise AgentExecutionError(str(inner), original_error=inner) from inner``."""
    try:
        try:
            raise inner
        except Exception as caught:
            raise AgentExecutionError(str(caught), original_error=caught) from caught
    except AgentExecutionError as wrapped:
        return wrapped


def _retryable_with(secret: str) -> AgentExecutionError:
    return _as_the_runner_raises(RuntimeError(f"401 Unauthorized: Bearer {secret} key {secret}"))


def _permanent_with(secret: str) -> AgentExecutionError:
    try:
        try:
            raise RuntimeError(f"Error code: 401 - invalid api key {secret}")
        except RuntimeError as raw:
            raise LLMAuthenticationError(f"auth failed: {secret}") from raw
    except LLMAuthenticationError as auth:
        return _as_the_runner_raises(auth)


async def _run_with_history(workflow_run, arg):
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
            result = await handle.result()
            history = await handle.fetch_history()
            return result, history.to_json()


@pytest.mark.parametrize(
    "make_error, attempts",
    [(_retryable_with, 3), (_permanent_with, 1)],
    ids=["retryable", "permanent"],
)
async def test_raw_secret_reaches_neither_history_nor_worker_log(
    runner, caplog, make_error, attempts
):
    secret = _secret()
    fake = runner(research=99)
    fake.raises = make_error(secret)
    assert secret not in str(fake.raises)  # the runner's text is redacted

    with caplog.at_level("DEBUG"):
        result, history = await _run_with_history(
            SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
        )

    assert fake.calls["research"] == attempts
    assert result.step_results[0].status == "error"
    assert "[REDACTED]" in result.step_results[0].error
    assert secret not in result.step_results[0].error
    assert secret not in history
    assert "Completing activity as failed" in caplog.text  # the SDK's traceback line
    assert secret not in caplog.text
    assert result.content == "q>write"


async def test_cyclic_cause_chain_is_an_error_step(runner):
    """A cyclic __cause__ made the failure converter recurse and fail the run."""
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__cause__, second.__cause__ = second, first
    fake = runner(research=99)
    fake.raises = AgentExecutionError("cyclic", original_error=first)
    fake.raises.__cause__ = first

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
    )

    assert result.step_results[0].status == "error"
    assert "cyclic" in result.step_results[0].error
    assert result.content == "q>write"


async def test_long_error_text_is_capped_in_the_failure_and_heartbeat(runner):
    runner(research=99).raises = RuntimeError("x" * 1_500_000)
    heartbeats: list[tuple] = []
    env = ActivityEnvironment()
    env.on_heartbeat = lambda *details: heartbeats.append(details)
    params = AgentActivityParams(agent_name="research", input="q", raise_on_error=True)

    with pytest.raises(ApplicationError) as exc:
        await env.run(run_agent_activity, params)

    assert exc.value.__cause__ is None
    message = exc.value.message
    assert len(message.encode()) <= 256 * 1024
    assert message.endswith("[truncated]")
    assert heartbeats[-1][1:] == (AGENT_ERROR_MARKER, message)


async def test_long_error_text_still_completes_the_step(runner):
    """At 1.5 MB the completion went over gRPC's 4 MB limit and timed out."""
    runner(research=99).raises = RuntimeError("x" * 1_500_000)

    result = await asyncio.wait_for(
        _run(SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)), 60
    )

    assert result.step_results[0].status == "error"
    assert result.step_results[0].error.endswith("[truncated]")
    assert result.content == "q>write"


async def test_application_error_from_a_custom_runner_keeps_the_dev_text(runner):
    """Only the breaker's marked ApplicationError is passed on as is; any other
    one from a runner is an agent error, so the step records it and the run goes on."""
    fake = runner(research=99)
    fake.raises = ApplicationError("quota gone", type="MyAppError")
    expected = (await _activity(raise_on_error=False)).error

    result = await _run(
        SequentialAgentWorkflow.run, _seq("research", "write", retry_agent_errors=True)
    )

    assert result.status == "completed"
    assert result.step_results[0].status == "error"
    assert result.step_results[0].error == expected
    assert result.content == "q>write"
    assert fake.calls["write"] == 1


# ---------------------------------------------------------------------------
# Round 4: agent-error retries are opt-in; the marker only on opted-in runs
# ---------------------------------------------------------------------------


def _history(history_json: str) -> WorkflowHistory:
    return WorkflowHistory.from_json("run", history_json)


def _markers(history: WorkflowHistory) -> list[str]:
    """The patch ids of the ``workflow.patched`` markers in ``history``."""
    return [
        json.loads(payload.data)["id"]
        for e in history.events
        if e.HasField("marker_recorded_event_attributes")
        for payload in e.marker_recorded_event_attributes.details["patch-data"].payloads
    ]


def _commands(history: WorkflowHistory) -> list[tuple[str, ...]]:
    """The commands in ``history``, in order: what a replaying worker checks its code against."""
    commands: list[tuple[str, ...]] = []
    for e in history.events:
        if e.HasField("activity_task_scheduled_event_attributes"):
            commands.append(
                ("activity", e.activity_task_scheduled_event_attributes.activity_type.name)
            )
        elif e.HasField("timer_started_event_attributes"):
            commands.append(("timer",))
        elif e.HasField("marker_recorded_event_attributes"):
            commands.append(("marker", e.marker_recorded_event_attributes.marker_name))
        elif e.HasField("workflow_execution_completed_event_attributes"):
            commands.append(("completed",))
    return commands


_DEFAULT_RUNS = [
    (SequentialAgentWorkflow.run, _seq("research", "write"), "research", 1),
    (
        LoopAgentWorkflow.run,
        LoopWorkflowInput(agent_name="looper", initial_input="q"),
        "looper",
        2,  # the failed iteration, then the one that says COMPLETE
    ),
    (
        ParallelAgentWorkflow.run,
        ParallelWorkflowInput(agent_names=["research", "b"], initial_input="q"),
        "research",
        1,
    ),
    (AgentWorkflow.run, WorkflowInput(steps=_steps(), initial_input="q"), "research", 1),
]


@pytest.mark.parametrize(
    "workflow_run, arg, agent, calls",
    _DEFAULT_RUNS,
    ids=["sequential", "loop", "parallel", "agent"],
)
async def test_default_run_does_not_retry_agent_errors_and_records_no_marker(
    runner, workflow_run, arg, agent, calls
):
    """As on dev: the agent error is a status="error" step on the first attempt,
    the run goes on, and the history has no marker a 2.1 worker would not know."""
    fake = runner(research=99, looper=1)

    result, history_json = await _run_with_history(workflow_run, arg)

    assert fake.calls[agent] == calls
    assert result.status == "completed"
    assert result.step_results[0].status == "error"
    assert "model timeout" in result.step_results[0].error
    assert _markers(_history(history_json)) == []
    assert _commands(_history(history_json))[-1] == ("completed",)
    await Replayer(workflows=WORKFLOWS).replay_workflow(_history(history_json))


@pytest.mark.parametrize(
    "opt_in, calls",
    [({"retry_agent_errors": True}, 3), ({"on_step_error": "fail"}, 1)],
    ids=["retry_agent_errors", "fail"],
)
async def test_opted_in_run_records_the_marker(runner, opt_in, calls):
    """retry_agent_errors retries the agent; "fail" alone stops the run without retrying it."""
    fake = runner(research=99)

    result, history_json = await _run_with_history(
        SequentialAgentWorkflow.run, _seq("research", "write", **opt_in)
    )

    assert fake.calls["research"] == calls
    assert result.status == ("failed" if "on_step_error" in opt_in else "completed")
    assert _markers(_history(history_json)) == [steps.STEP_RETRIES_PATCH]
    await Replayer(workflows=WORKFLOWS).replay_workflow(_history(history_json))


async def test_default_run_issues_the_same_commands_as_dev(runner):
    """A default parallel run with one failing agent, against the history dev's
    workflow code recorded for the same run: the same commands in the same
    order, so a 2.1 worker replays it."""
    runner(research=99)
    recorded_on_dev = WorkflowHistory.from_json(
        "dev", (HISTORIES / "parallel-one-failed-completed.json").read_text()
    )

    _, history_json = await _run_with_history(
        ParallelAgentWorkflow.run,
        ParallelWorkflowInput(agent_names=["research", "b"], initial_input="q"),
    )

    assert _commands(_history(history_json)) == _commands(recorded_on_dev)
    assert _commands(recorded_on_dev) == [
        ("activity", "run_agent_activity"),
        ("activity", "run_agent_activity"),
        ("completed",),
    ]
