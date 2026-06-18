"""
MEDIUM #2 fix — gate workflow orchestrators' own coordination LLM calls.

Workflow agents (scatter/sequential/planner/...) run via `execute()`, not through
AgentRunner.run, so the run-level ambient policy publisher doesn't wrap them.
Their branches are gated (each runs via runner.run), but the orchestrator's own
coordination calls (split / merge / synthesize / critique / route) run inside
execute() with no ambient -> they bypass the model-routing + telemetry policy.

Fix: a @publish_active_policy decorator on each workflow execute() publishes the
agent's policy context for the duration of the call. These tests cover the
decorator's behavior and assert every workflow execute() carries it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from continuum.agent.utils.context_utils import create_run_context


class TestPublishActivePolicyDecorator:
    async def test_publishes_ambient_with_positional_context(self):
        from continuum.agent.utils.context_utils import publish_active_policy
        from continuum.security.policy_context import get_active_policy

        captured: dict = {}

        class FakeWorkflow:
            name = "wf-agent"
            policy_store = MagicMock()

            @publish_active_policy
            async def execute(self, input, runner, context):  # noqa: A002
                captured["ap"] = get_active_policy()
                return "ok"

        ctx = create_run_context(data_labels={"pii"})
        result = await FakeWorkflow().execute("in", None, ctx)  # context positional

        assert result == "ok"
        ap = captured["ap"]
        assert ap is not None
        assert ap.subject == "wf-agent"
        assert ap.data_labels == {"pii"}
        assert get_active_policy() is None  # reset after

    async def test_publishes_ambient_with_keyword_context(self):
        from continuum.agent.utils.context_utils import publish_active_policy
        from continuum.security.policy_context import get_active_policy

        captured: dict = {}

        class FakeWorkflow:
            name = "wf2"
            policy_store = MagicMock()

            @publish_active_policy
            async def execute(self, input, runner, context):  # noqa: A002
                captured["ap"] = get_active_policy()
                return "ok"

        ctx = create_run_context(data_labels={"phi"})
        await FakeWorkflow().execute("in", None, context=ctx)  # context keyword

        assert captured["ap"] is not None
        assert captured["ap"].data_labels == {"phi"}

    async def test_no_context_is_safe_noop(self):
        from continuum.agent.utils.context_utils import publish_active_policy
        from continuum.security.policy_context import get_active_policy

        class FakeWorkflow:
            name = "wf3"
            policy_store = None

            @publish_active_policy
            async def execute(self, input):  # noqa: A002 - no context arg
                return get_active_policy()

        # No RunContext passed → decorator must not crash; just runs the method.
        assert await FakeWorkflow().execute("in") is None


class TestAllWorkflowExecutesDecorated:
    def test_every_workflow_execute_publishes_policy(self):
        from continuum.agent.workflow.dag import DAGAgent
        from continuum.agent.workflow.debate import DebateAgent
        from continuum.agent.workflow.loop import LoopAgent
        from continuum.agent.workflow.parallel import ParallelAgent
        from continuum.agent.workflow.planner import PlannerAgent
        from continuum.agent.workflow.reflection import ReflectionAgent
        from continuum.agent.workflow.router import RouterAgent
        from continuum.agent.workflow.scatter import ScatterAgent
        from continuum.agent.workflow.sequential import SequentialAgent
        from continuum.agent.workflow.supervised import SupervisedSequentialAgent

        agents = [
            DAGAgent,
            DebateAgent,
            LoopAgent,
            ParallelAgent,
            PlannerAgent,
            ReflectionAgent,
            RouterAgent,
            ScatterAgent,
            SequentialAgent,
            SupervisedSequentialAgent,
        ]
        for cls in agents:
            assert getattr(cls.execute, "__publishes_active_policy__", False), (
                f"{cls.__name__}.execute() is not wrapped with @publish_active_policy — "
                f"its coordination LLM calls would bypass the data-label gate"
            )
