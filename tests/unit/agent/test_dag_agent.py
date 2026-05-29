"""
Tests for DAGAgent: parallel execution, dependency gating,
cycle detection, fail-fast abort, and predecessor output passing.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.types import AgentResponse, FailStrategy, MergeStrategy, ResponseStatus
from orchestrator.agent.workflow.dag import DAGAgent, DAGCycleError, DAGStageError, create_dag_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(content: str) -> AgentResponse:
    resp = MagicMock(spec=AgentResponse)
    resp.content = content
    resp.status = ResponseStatus.SUCCESS
    resp.usage = MagicMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)
    resp.turn_count = 1
    resp.messages = []
    resp.structured_output = None
    return resp


def _make_agent(name: str):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _make_runner(responses: dict[str, str]):
    """Return a mock runner whose run() maps agent name → response content."""
    runner = MagicMock()

    async def _run(agent, input, context, **kwargs):
        return _mock_response(responses[agent.name])

    runner.run = AsyncMock(side_effect=_run)
    runner.save_turn = AsyncMock()
    return runner


def _patch_span():
    mock_span = MagicMock()
    mock_span.set_output = MagicMock()
    mock_span.set_error = MagicMock()
    mock_span.__aenter__ = AsyncMock(return_value=mock_span)
    mock_span.__aexit__ = AsyncMock(return_value=False)
    return patch("orchestrator.observability.trace_context.SpanScope", return_value=mock_span)


def _make_context():
    from orchestrator.agent.types import RunContext

    return RunContext(run_id="test-run")


# ---------------------------------------------------------------------------
# Construction and cycle detection
# ---------------------------------------------------------------------------


class TestDAGConstruction:
    def test_add_stage_chainable(self):
        dag = DAGAgent(name="dag")
        result = dag.add_stage("a", _make_agent("a"))
        assert result is dag

    def test_duplicate_stage_id_raises(self):
        dag = DAGAgent(name="dag")
        dag.add_stage("a", _make_agent("a"))
        with pytest.raises(ValueError, match="already registered"):
            dag.add_stage("a", _make_agent("a2"))

    def test_unknown_dependency_raises(self):
        dag = DAGAgent(name="dag")
        with pytest.raises(ValueError, match="not been registered"):
            dag.add_stage("b", _make_agent("b"), depends_on=["a"])

    def test_cycle_detection_simple(self):
        dag = DAGAgent(name="dag")
        dag.add_stage("a", _make_agent("a"))
        dag.add_stage("b", _make_agent("b"), depends_on=["a"])
        # Manually inject a back-edge to simulate a cycle (bypass add_stage check)
        dag._stages["a"].depends_on = ["b"]
        with pytest.raises(DAGCycleError):
            dag._validate_no_cycles()

    def test_no_stages_raises_on_execute(self):
        from orchestrator.agent.exceptions import AgentConfigurationError

        dag = DAGAgent(name="dag")
        runner = MagicMock()
        runner.save_turn = AsyncMock()
        ctx = _make_context()
        with _patch_span():
            with pytest.raises(AgentConfigurationError):
                asyncio.get_event_loop().run_until_complete(dag.execute("input", runner, ctx))

    def test_create_dag_agent_factory(self):
        dag = create_dag_agent(
            name="pipeline",
            stages=[
                ("a", _make_agent("a"), []),
                ("b", _make_agent("b"), ["a"]),
            ],
        )
        assert isinstance(dag, DAGAgent)
        assert len(dag._stages) == 2


# ---------------------------------------------------------------------------
# Execution — sequential chain
# ---------------------------------------------------------------------------


class TestDAGSequential:
    @pytest.mark.asyncio
    async def test_single_stage(self):
        dag = DAGAgent(name="dag")
        dag.add_stage("only", _make_agent("only"))
        runner = _make_runner({"only": "result"})
        ctx = _make_context()
        with _patch_span():
            resp = await dag.execute("input", runner, ctx)
        assert resp.status == ResponseStatus.SUCCESS
        assert resp.content == "result"

    @pytest.mark.asyncio
    async def test_chain_passes_output(self):
        dag = DAGAgent(name="dag")
        dag.add_stage("a", _make_agent("a"))
        dag.add_stage("b", _make_agent("b"), depends_on=["a"])

        received_inputs = {}

        async def _run(agent, input, context, **kwargs):
            received_inputs[agent.name] = input
            return _mock_response(f"{agent.name}_out")

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            resp = await dag.execute("original", runner, ctx)

        assert received_inputs["a"] == "original"
        assert "a_out" in received_inputs["b"]  # b receives a's output
        assert resp.content == "b_out"

    @pytest.mark.asyncio
    async def test_terminal_is_last_stage(self):
        dag = create_dag_agent(
            name="chain",
            stages=[
                ("a", _make_agent("a"), []),
                ("b", _make_agent("b"), ["a"]),
                ("c", _make_agent("c"), ["b"]),
            ],
        )
        runner = _make_runner({"a": "A", "b": "B", "c": "C"})
        ctx = _make_context()
        with _patch_span():
            resp = await dag.execute("in", runner, ctx)
        assert resp.content == "C"


# ---------------------------------------------------------------------------
# Execution — parallel branches
# ---------------------------------------------------------------------------


class TestDAGParallel:
    @pytest.mark.asyncio
    async def test_independent_stages_run_concurrently(self):
        start_times = {}
        end_times = {}

        async def _run(agent, input, context, **kwargs):
            start_times[agent.name] = time.monotonic()
            await asyncio.sleep(0.05)
            end_times[agent.name] = time.monotonic()
            return _mock_response(f"{agent.name}_out")

        dag = DAGAgent(name="dag")
        dag.add_stage("a", _make_agent("a"))
        dag.add_stage("b", _make_agent("b"))

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            await dag.execute("input", runner, ctx)

        # Both started before either finished
        assert start_times["a"] < end_times["b"]
        assert start_times["b"] < end_times["a"]

    @pytest.mark.asyncio
    async def test_fan_in_waits_for_all_predecessors(self):
        completed_order = []

        async def _run(agent, input, context, **kwargs):
            if agent.name in ("a", "b"):
                await asyncio.sleep(0.02)
            completed_order.append(agent.name)
            return _mock_response(f"{agent.name}_out")

        dag = create_dag_agent(
            name="fan_in",
            stages=[
                ("a", _make_agent("a"), []),
                ("b", _make_agent("b"), []),
                ("merge", _make_agent("merge"), ["a", "b"]),
            ],
        )

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            await dag.execute("input", runner, ctx)

        assert completed_order.index("merge") > completed_order.index("a")
        assert completed_order.index("merge") > completed_order.index("b")

    @pytest.mark.asyncio
    async def test_merge_strategy_concatenate(self):
        dag = create_dag_agent(
            name="merge",
            stages=[
                ("a", _make_agent("a"), []),
                ("b", _make_agent("b"), []),
                ("c", _make_agent("c"), ["a", "b"]),
            ],
            merge_strategy=MergeStrategy.CONCATENATE,
        )

        received = {}

        async def _run(agent, input, context, **kwargs):
            received[agent.name] = input
            return _mock_response(f"{agent.name}_out")

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            await dag.execute("original", runner, ctx)

        assert "a_out" in received["c"]
        assert "b_out" in received["c"]

    @pytest.mark.asyncio
    async def test_merge_strategy_structured(self):
        import json

        dag = create_dag_agent(
            name="merge",
            stages=[
                ("x", _make_agent("x"), []),
                ("y", _make_agent("y"), []),
                ("z", _make_agent("z"), ["x", "y"]),
            ],
            merge_strategy=MergeStrategy.STRUCTURED,
        )

        received = {}

        async def _run(agent, input, context, **kwargs):
            received[agent.name] = input
            return _mock_response(f"{agent.name}_out")

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            await dag.execute("original", runner, ctx)

        data = json.loads(received["z"])
        assert data["x"] == "x_out"
        assert data["y"] == "y_out"


# ---------------------------------------------------------------------------
# Fail strategy
# ---------------------------------------------------------------------------


class TestDAGFailStrategy:
    @pytest.mark.asyncio
    async def test_fail_fast_raises_on_stage_error(self):
        async def _run(agent, input, context, **kwargs):
            if agent.name == "bad":
                raise RuntimeError("stage failed")
            return _mock_response("ok")

        dag = DAGAgent(name="dag", fail_strategy=FailStrategy.FAIL_FAST)
        dag.add_stage("bad", _make_agent("bad"))
        dag.add_stage("good", _make_agent("good"))

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            with pytest.raises(DAGStageError) as exc_info:
                await dag.execute("input", runner, ctx)
        assert exc_info.value.stage_id == "bad"

    @pytest.mark.asyncio
    async def test_continue_on_error_returns_partial(self):
        async def _run(agent, input, context, **kwargs):
            if agent.name == "bad":
                raise RuntimeError("stage failed")
            return _mock_response(f"{agent.name}_ok")

        dag = DAGAgent(name="dag", fail_strategy=FailStrategy.CONTINUE_ON_ERROR)
        dag.add_stage("bad", _make_agent("bad"))
        dag.add_stage("good", _make_agent("good"))

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_run)
        runner.save_turn = AsyncMock()
        ctx = _make_context()

        with _patch_span():
            resp = await dag.execute("input", runner, ctx)

        assert resp.status == ResponseStatus.SUCCESS
        assert "good_ok" in resp.content
