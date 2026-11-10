"""Failed agent steps are retried, and a step that keeps failing fails the workflow.

Before: run_agent_activity caught every error and returned status="error", so
Temporal never retried it, and the sequential, loop and agent workflows carried
on past the failed step and finished "completed".

Covers, with the real activity on Temporal's time-skipping test server:
- a step that fails twice and then works is retried and the workflow completes
- a step that always fails is tried 3 times, then the workflow is "failed",
  names the step and the error, and later steps do not run
- an unregistered agent is not retried
- AgentWorkflow: a failing agent step fails the run; one failed parallel agent
  is left out of the merge; all failed fails the run
- ParallelAgentWorkflow: same partial / all-failed rules
- LoopAgentWorkflow stops as "failed"
- workflows make replay-safe ids (workflow.uuid4, not uuid.uuid4)
"""

from __future__ import annotations

import inspect
import uuid
from collections import Counter

import pytest

pytest.importorskip("temporalio")

from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from continuum.agent.base import BaseAgent  # noqa: E402
from continuum.agent.types import AgentResponse, ResponseStatus  # noqa: E402
from continuum.temporal.activities import (  # noqa: E402
    run_agent_activity,
    send_notification_activity,
)
from continuum.temporal.registry import get_agent_registry, reset_agent_registry  # noqa: E402
from continuum.temporal.types import WorkflowInput  # noqa: E402
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


class FakeRunner:
    """Answers "<input>><agent>"; an agent in ``fail_times`` raises that many times first."""

    def __init__(self, fail_times: dict[str, int]):
        self.fail_times = dict(fail_times)
        self.calls: Counter[str] = Counter()

    async def run(self, agent, input, **kw):
        self.calls[agent.name] += 1
        if self.calls[agent.name] <= self.fail_times.get(agent.name, 0):
            raise RuntimeError(f"{agent.name} model timeout")
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


def _seq(*names):
    return SequentialWorkflowInput(agent_names=list(names), initial_input="q")


async def test_failing_step_is_retried_then_completes(runner):
    fake = runner(research=2)

    result = await _run(SequentialAgentWorkflow.run, _seq("research", "write"))

    assert result.status == "completed"
    assert result.content == "q>research>write"
    assert fake.calls["research"] == 3


async def test_step_that_keeps_failing_fails_the_workflow(runner):
    fake = runner(research=99)

    result = await _run(SequentialAgentWorkflow.run, _seq("research", "write"))

    assert result.status == "failed"
    assert "Step 1 (research) failed" in result.error
    assert "research model timeout" in result.error
    assert fake.calls["research"] == 3
    assert fake.calls["write"] == 0
    assert result.step_results[-1].status == "error"


async def test_unregistered_agent_is_not_retried(runner):
    runner()

    result = await _run(SequentialAgentWorkflow.run, _seq("missing"))

    assert result.status == "failed"
    assert "not registered" in result.error


async def test_agent_workflow_step_failure_fails_run(runner):
    fake = runner(research=99)
    wf = WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "research", "retries": 2},
            {"type": "agent", "agent_name": "write"},
        ],
        initial_input="q",
    )

    result = await _run(AgentWorkflow.run, wf)

    assert result.status == "failed"
    assert "research" in result.error and "model timeout" in result.error
    assert fake.calls["research"] == 2
    assert fake.calls["write"] == 0


async def test_agent_workflow_parallel_partial_and_total_failure(runner):
    runner(a=99)
    step = {"type": "parallel", "agents": [{"agent_name": "a"}, {"agent_name": "b"}]}

    partial = await _run(AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q"))
    assert partial.status == "completed"
    assert partial.content == "q>b"

    runner(a=99, b=99)
    total = await _run(AgentWorkflow.run, WorkflowInput(steps=[step], initial_input="q"))
    assert total.status == "failed"
    assert "All parallel agents failed" in total.error


async def test_parallel_workflow_partial_and_total_failure(runner):
    runner(a=99)
    partial = await _run(
        ParallelAgentWorkflow.run, ParallelWorkflowInput(agent_names=["a", "b"], initial_input="q")
    )
    assert partial.status == "completed"
    assert partial.content == "q>b"
    assert partial.error and partial.error.startswith("a:")

    runner(a=99, b=99)
    total = await _run(
        ParallelAgentWorkflow.run, ParallelWorkflowInput(agent_names=["a", "b"], initial_input="q")
    )
    assert total.status == "failed"


async def test_loop_workflow_fails_on_failed_iteration(runner):
    runner(looper=99)

    result = await _run(
        LoopAgentWorkflow.run, LoopWorkflowInput(agent_name="looper", initial_input="q")
    )

    assert result.status == "failed"
    assert "Iteration 1 (looper) failed" in result.error


def test_workflows_make_replay_safe_ids():
    from continuum.temporal.workflows import loop_workflow, sequential_workflow

    for mod in (loop_workflow, sequential_workflow, steps):
        source = inspect.getsource(mod)
        assert "uuid.uuid4" not in source
    assert "workflow.uuid4()" in inspect.getsource(sequential_workflow)
    assert "workflow.uuid4()" in inspect.getsource(loop_workflow)
