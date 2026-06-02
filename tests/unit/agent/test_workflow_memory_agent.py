"""
Tests that every workflow agent passes `memory_agent` (not a hardcoded None)
into `runner.save_turn()`, and that each factory function wires the parameter
through to the resulting instance.

Covers: SequentialAgent, ParallelAgent, ScatterAgent,
        PlannerAgent, SupervisedSequentialAgent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.types import (
    AgentResponse,
    FailStrategy,
    MergeStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_response(content: str = "ok") -> AgentResponse:
    resp = MagicMock(spec=AgentResponse)
    resp.content = content
    resp.status = ResponseStatus.SUCCESS
    resp.usage = MagicMock(total_tokens=5, prompt_tokens=3, completion_tokens=2)
    resp.turn_count = 1
    resp.messages = []
    resp.structured_output = None
    return resp


def _make_base_agent(name: str):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _make_runner():
    runner = MagicMock()
    runner.run = AsyncMock(return_value=_mock_response())
    runner.save_turn = AsyncMock()
    return runner


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
    return patch("orchestrator.observability.trace_context.SpanScope", return_value=mock_span)


# ---------------------------------------------------------------------------
# SequentialAgent
# ---------------------------------------------------------------------------


class TestSequentialMemoryAgent:
    @pytest.mark.asyncio
    async def test_save_turn_agent_is_none_by_default(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        seq = SequentialAgent(name="seq", agents=[_make_base_agent("a")])
        runner = _make_runner()
        with _patch_span():
            await seq.execute("input", runner, _make_context())
        runner.save_turn.assert_called_once()
        assert runner.save_turn.call_args.kwargs["agent"] is None

    @pytest.mark.asyncio
    async def test_save_turn_uses_memory_agent_when_set(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        mem = _make_base_agent("mem")
        seq = SequentialAgent(name="seq", agents=[_make_base_agent("a")], memory_agent=mem)
        runner = _make_runner()
        with _patch_span():
            await seq.execute("input", runner, _make_context())
        assert runner.save_turn.call_args.kwargs["agent"] is mem

    def test_create_sequential_agent_passes_memory_agent(self):
        from orchestrator.agent.workflow.sequential import create_sequential_agent

        mem = _make_base_agent("mem")
        seq = create_sequential_agent("seq", agents=[_make_base_agent("a")], memory_agent=mem)
        assert seq.memory_agent is mem


# ---------------------------------------------------------------------------
# ParallelAgent  (CONCATENATE avoids an LLM merge call)
# ---------------------------------------------------------------------------


class TestParallelMemoryAgent:
    @pytest.mark.asyncio
    async def test_save_turn_agent_is_none_by_default(self):
        from orchestrator.agent.config import ParallelConfig
        from orchestrator.agent.workflow.parallel import ParallelAgent

        pa = ParallelAgent(
            name="par",
            agents=[_make_base_agent("a")],
            parallel_config=ParallelConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )
        runner = _make_runner()
        with _patch_span():
            await pa.execute("input", runner, _make_context())
        runner.save_turn.assert_called_once()
        assert runner.save_turn.call_args.kwargs["agent"] is None

    @pytest.mark.asyncio
    async def test_save_turn_uses_memory_agent_when_set(self):
        from orchestrator.agent.config import ParallelConfig
        from orchestrator.agent.workflow.parallel import ParallelAgent

        mem = _make_base_agent("mem")
        pa = ParallelAgent(
            name="par",
            agents=[_make_base_agent("a")],
            parallel_config=ParallelConfig(merge_strategy=MergeStrategy.CONCATENATE),
            memory_agent=mem,
        )
        runner = _make_runner()
        with _patch_span():
            await pa.execute("input", runner, _make_context())
        assert runner.save_turn.call_args.kwargs["agent"] is mem

    def test_create_parallel_agent_passes_memory_agent(self):
        from orchestrator.agent.workflow.parallel import create_parallel_agent

        mem = _make_base_agent("mem")
        pa = create_parallel_agent(
            "par",
            agents=[_make_base_agent("a")],
            merge_strategy=MergeStrategy.CONCATENATE,
            memory_agent=mem,
        )
        assert pa.memory_agent is mem


# ---------------------------------------------------------------------------
# ScatterAgent  (input_slices bypasses LLM splitting; CONCATENATE avoids merge LLM)
# ---------------------------------------------------------------------------


class TestScatterMemoryAgent:
    @pytest.mark.asyncio
    async def test_save_turn_agent_is_none_by_default(self):
        from orchestrator.agent.workflow.scatter import ScatterAgent, ScatterConfig

        sc = ScatterAgent(
            name="scat",
            agents=[_make_base_agent("a")],
            input_slices=["slice-a"],
            scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )
        runner = _make_runner()
        with _patch_span():
            await sc.execute("input", runner, _make_context())
        runner.save_turn.assert_called_once()
        assert runner.save_turn.call_args.kwargs["agent"] is None

    @pytest.mark.asyncio
    async def test_save_turn_uses_memory_agent_when_set(self):
        from orchestrator.agent.workflow.scatter import ScatterAgent, ScatterConfig

        mem = _make_base_agent("mem")
        sc = ScatterAgent(
            name="scat",
            agents=[_make_base_agent("a")],
            input_slices=["slice-a"],
            scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
            memory_agent=mem,
        )
        runner = _make_runner()
        with _patch_span():
            await sc.execute("input", runner, _make_context())
        assert runner.save_turn.call_args.kwargs["agent"] is mem

    def test_create_scatter_agent_passes_memory_agent(self):
        from orchestrator.agent.workflow.scatter import create_scatter_agent

        mem = _make_base_agent("mem")
        sc = create_scatter_agent(
            "scat",
            agents=[_make_base_agent("a")],
            merge_strategy=MergeStrategy.CONCATENATE,
            memory_agent=mem,
        )
        assert sc.memory_agent is mem


# ---------------------------------------------------------------------------
# PlannerAgent  (_generate_plan patched to return a fixed single-step plan)
# ---------------------------------------------------------------------------


def _patch_planner(planner, sub_agent_name: str):
    """Patch _generate_plan so no LLM call is made."""
    plan = [{"step_id": "1", "instruction": "do it", "agent_name": sub_agent_name}]

    async def _fake_generate_plan(self, goal, llm_client):
        return plan, TokenUsage()

    mock_container = MagicMock()
    mock_container.llm_client = MagicMock()

    return (
        patch.object(type(planner), "_generate_plan", _fake_generate_plan),
        patch("orchestrator.core.container.get_container", return_value=mock_container),
    )


class TestPlannerMemoryAgent:
    @pytest.mark.asyncio
    async def test_save_turn_agent_is_none_by_default(self):
        from orchestrator.agent.workflow.planner import PlannerAgent

        sub = _make_base_agent("sub")
        planner = PlannerAgent(name="plan", agent=sub)
        p1, p2 = _patch_planner(planner, sub.name)
        runner = _make_runner()
        with p1, p2, _patch_span():
            await planner.execute("goal", runner, _make_context())
        runner.save_turn.assert_called_once()
        assert runner.save_turn.call_args.kwargs["agent"] is None

    @pytest.mark.asyncio
    async def test_save_turn_uses_memory_agent_when_set(self):
        from orchestrator.agent.workflow.planner import PlannerAgent

        sub = _make_base_agent("sub")
        mem = _make_base_agent("mem")
        planner = PlannerAgent(name="plan", agent=sub, memory_agent=mem)
        p1, p2 = _patch_planner(planner, sub.name)
        runner = _make_runner()
        with p1, p2, _patch_span():
            await planner.execute("goal", runner, _make_context())
        assert runner.save_turn.call_args.kwargs["agent"] is mem

    def test_create_planner_agent_passes_memory_agent(self):
        from orchestrator.agent.workflow.planner import create_planner_agent

        mem = _make_base_agent("mem")
        planner = create_planner_agent(
            "plan",
            agent=_make_base_agent("sub"),
            memory_agent=mem,
        )
        assert planner.memory_agent is mem


# ---------------------------------------------------------------------------
# SupervisedSequentialAgent  (_score_output and _get_llm patched)
# ---------------------------------------------------------------------------


def _patch_supervised(agent):
    """Patch LLM-dependent methods so supervised agent completes without a real LLM."""
    mock_llm = MagicMock()

    async def _fake_score(self, step_num, agent_name, original_input, output, llm_client):
        return 1.0, "perfect", TokenUsage()

    return (
        patch.object(type(agent), "_score_output", _fake_score),
        patch.object(type(agent), "_get_llm", return_value=mock_llm),
    )


class TestSupervisedMemoryAgent:
    @pytest.mark.asyncio
    async def test_save_turn_agent_is_none_by_default(self):
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent

        sup = SupervisedSequentialAgent(name="sup", agents=[_make_base_agent("a")])
        p1, p2 = _patch_supervised(sup)
        runner = _make_runner()
        with p1, p2, _patch_span():
            await sup.execute("input", runner, _make_context())
        runner.save_turn.assert_called_once()
        assert runner.save_turn.call_args.kwargs["agent"] is None

    @pytest.mark.asyncio
    async def test_save_turn_uses_memory_agent_when_set(self):
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent

        mem = _make_base_agent("mem")
        sup = SupervisedSequentialAgent(
            name="sup", agents=[_make_base_agent("a")], memory_agent=mem
        )
        p1, p2 = _patch_supervised(sup)
        runner = _make_runner()
        with p1, p2, _patch_span():
            await sup.execute("input", runner, _make_context())
        assert runner.save_turn.call_args.kwargs["agent"] is mem

    def test_create_supervised_agent_passes_memory_agent(self):
        from orchestrator.agent.workflow.supervised import create_supervised_agent

        mem = _make_base_agent("mem")
        sup = create_supervised_agent(
            "sup",
            agents=[_make_base_agent("a")],
            memory_agent=mem,
        )
        assert sup.memory_agent is mem
