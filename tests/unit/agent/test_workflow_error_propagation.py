"""Regression tests: workflow drivers must not launder a returned ERROR into SUCCESS.

``runner.run()`` returns (does not raise) an ``AgentResponse`` with
``status=ERROR`` when it short-circuits — most notably when the circuit breaker
is open, where ``content`` holds error prose like
"Service temporarily unavailable: Circuit breaker is open. Retry after 22.3s".

Before the fix, every workflow ``_drive()`` ignored ``response.status``: it
marked the failed step ``success=True``, chained the error prose into the next
step's input, assembled a final ``status=SUCCESS`` response carrying that prose,
and persisted it to memory via ``save_turn``. These tests pin the fixed
behavior — a returned ERROR is treated as a failed step, never relabeled
SUCCESS, and never written to memory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.exceptions import (
    LoopWorkflowError,
    PlannerWorkflowError,
    SequentialWorkflowError,
)
from continuum.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    TokenUsage,
)

# Error prose exactly as the runner would emit it when the breaker is open.
_BREAKER_PROSE = "Service temporarily unavailable: Circuit breaker is open. Retry after 22.3s"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_response() -> AgentResponse:
    """A real (not mocked) ERROR response, mirroring the runner breaker path."""
    return AgentResponse(
        content=_BREAKER_PROSE,
        agent_name="sub",
        status=ResponseStatus.ERROR,
        error="Circuit breaker is open. Retry after 22.3s",
        usage=TokenUsage(),
        turn_count=0,
        run_artifacts={"retry_after_s": 22.3},
    )


def _make_error_runner():
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_error_response())
    runner.save_turn = AsyncMock()
    runner.ensure_recorder = MagicMock(return_value=False)
    runner.persist_decision_trace = AsyncMock()
    return runner


def _make_base_agent(name: str):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _make_context(session_id: str = "sess-1") -> RunContext:
    ctx = RunContext(run_id="test-run")
    ctx.session_id = session_id
    return ctx


def _patch_span():
    mock_span = MagicMock()
    mock_span.set_output = MagicMock()
    mock_span.set_error = MagicMock()
    mock_span.add_metadata = MagicMock()
    mock_span.__aenter__ = AsyncMock(return_value=mock_span)
    mock_span.__aexit__ = AsyncMock(return_value=False)
    return patch("continuum.observability.trace_context.SpanScope", return_value=mock_span)


def _patch_planner(planner, sub_agent_name: str):
    plan = [{"step_id": "1", "instruction": "do it", "agent_name": sub_agent_name}]

    async def _fake_generate_plan(self, goal, llm_client):
        return plan, TokenUsage()

    mock_container = MagicMock()
    mock_container.llm_client = MagicMock()
    return (
        patch.object(type(planner), "_generate_plan", _fake_generate_plan),
        patch("continuum.core.container.get_container", return_value=mock_container),
    )


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class TestPlannerErrorPropagation:
    @pytest.mark.asyncio
    async def test_returned_error_raises_and_is_not_saved(self):
        from continuum.agent.config import PlanningConfig
        from continuum.agent.types import FailStrategy
        from continuum.agent.workflow.planner import PlannerAgent

        sub = _make_base_agent("sub")
        planner = PlannerAgent(
            name="plan",
            agent=sub,
            # Deterministic: no replan LLM call, fail-fast on the ERROR step.
            planning_config=PlanningConfig(
                replan_on_failure=False, fail_strategy=FailStrategy.FAIL_FAST
            ),
        )
        p1, p2 = _patch_planner(planner, sub.name)
        runner = _make_error_runner()

        with p1, p2, _patch_span(), pytest.raises(PlannerWorkflowError):
            await planner.execute("goal", runner, _make_context())

        # The error prose must never reach long-term memory.
        runner.save_turn.assert_not_called()


# ---------------------------------------------------------------------------
# SupervisedSequentialAgent
# ---------------------------------------------------------------------------


class TestSupervisedErrorPropagation:
    @pytest.mark.asyncio
    async def test_returned_error_raises_and_is_not_saved(self):
        from continuum.agent.types import FailStrategy
        from continuum.agent.workflow.supervised import (
            SupervisedConfig,
            SupervisedSequentialAgent,
        )

        sup = SupervisedSequentialAgent(
            name="sup",
            agents=[_make_base_agent("a")],
            supervised_config=SupervisedConfig(max_retries=0, fail_strategy=FailStrategy.FAIL_FAST),
        )
        # _get_llm patched so no real container/LLM is needed; scoring is never
        # reached because the ERROR short-circuits before it.
        score = AsyncMock(return_value=(1.0, "ok", TokenUsage()))
        runner = _make_error_runner()

        with (
            patch.object(type(sup), "_get_llm", return_value=MagicMock()),
            patch.object(type(sup), "_score_output", score),
            _patch_span(),
            pytest.raises(SequentialWorkflowError),
        ):
            await sup.execute("input", runner, _make_context())

        score.assert_not_called()  # error prose was never scored as an answer
        runner.save_turn.assert_not_called()


# ---------------------------------------------------------------------------
# LoopAgent
# ---------------------------------------------------------------------------


class TestLoopErrorPropagation:
    @pytest.mark.asyncio
    async def test_returned_error_raises_and_is_not_saved(self):
        from continuum.agent.types import TerminationConfig, TerminationType
        from continuum.agent.workflow.loop import LoopAgent

        loop = LoopAgent(
            name="loop",
            agent=_make_base_agent("a"),
            termination=TerminationConfig(
                type=TerminationType.OUTPUT_MATCH, pattern="ok", max_iterations=3
            ),
        )
        runner = _make_error_runner()

        with _patch_span(), pytest.raises(LoopWorkflowError):
            await loop.execute("input", runner, _make_context(), llm_client=MagicMock())

        runner.save_turn.assert_not_called()


# ---------------------------------------------------------------------------
# ReflectionAgent  (no failure branch in its loop — must propagate ERROR status)
# ---------------------------------------------------------------------------


class TestReflectionErrorPropagation:
    @pytest.mark.asyncio
    async def test_returned_error_propagates_status_and_is_not_saved(self):
        from continuum.agent.workflow.reflection import ReflectionAgent, ReflectionConfig

        ref = ReflectionAgent(
            name="ref",
            agent=_make_base_agent("a"),
            reflection_config=ReflectionConfig(max_reflections=1),
        )
        runner = _make_error_runner()

        with _patch_span():
            result = await ref.execute("input", runner, _make_context(), llm_client=MagicMock())

        # Not laundered into SUCCESS, and the error prose is not persisted.
        assert result.status == ResponseStatus.ERROR
        runner.save_turn.assert_not_called()
