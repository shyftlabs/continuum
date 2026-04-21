"""
Tests for the workflow transparency pattern used by all 6 workflow agents:
- Sub-agent log_to_session is disabled during execution, restored in finally
- runner.save_turn() is called exactly once at the end with the final output
- Pipeline context injected into context.metadata for sequential steps
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agent.types import AgentResponse, ResponseStatus
from orchestrator.agent.utils.context_utils import create_run_context


def _make_agent(name="sub", log_to_session=True):
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="test",
        config=AgentConfig(log_to_session=log_to_session),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _mock_response(content="output"):
    resp = MagicMock(spec=AgentResponse)
    resp.content = content
    resp.status = ResponseStatus.SUCCESS
    resp.usage = MagicMock(total_tokens=10, prompt_tokens=5, completion_tokens=5)
    resp.turn_count = 1
    resp.messages = []
    resp.structured_output = None
    return resp


def _patch_span_scope():
    """Return a context manager patch for SpanScope that no-ops."""
    mock_span = MagicMock()
    mock_span.set_output = MagicMock()
    mock_span.set_error = MagicMock()
    mock_span.__aenter__ = AsyncMock(return_value=mock_span)
    mock_span.__aexit__ = AsyncMock(return_value=False)

    return patch(
        "orchestrator.observability.trace_context.SpanScope",
        return_value=mock_span,
    )


# ---------------------------------------------------------------------------
# SequentialAgent
# ---------------------------------------------------------------------------

class TestSequentialTransparency:
    async def test_log_to_session_disabled_during_execution(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        log_values_during_run: list[bool] = []
        agent_a = _make_agent("a", log_to_session=True)
        agent_b = _make_agent("b", log_to_session=True)

        runner = MagicMock()
        runner.save_turn = AsyncMock()

        async def capturing_run(agent, input, context=None):
            log_values_during_run.append(agent.config.log_to_session)
            return _mock_response(f"output-{agent.name}")

        runner.run = capturing_run

        pipeline = SequentialAgent(name="seq", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("user input", runner, ctx)

        # During execution both sub-agents must have had log_to_session=False
        assert all(v is False for v in log_values_during_run), (
            f"Expected all False during execution, got: {log_values_during_run}"
        )

    async def test_log_to_session_restored_after_execution(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        agent_a = _make_agent("a", log_to_session=True)
        agent_b = _make_agent("b", log_to_session=True)

        runner = MagicMock()
        runner.run = AsyncMock(return_value=_mock_response())
        runner.save_turn = AsyncMock()

        pipeline = SequentialAgent(name="seq", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("input", runner, ctx)

        assert agent_a.config.log_to_session is True
        assert agent_b.config.log_to_session is True

    async def test_log_to_session_restored_even_on_exception(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent
        from orchestrator.agent.config import SequentialConfig
        from orchestrator.agent.types import FailStrategy

        agent_a = _make_agent("a", log_to_session=True)

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=RuntimeError("step failed"))
        runner.save_turn = AsyncMock()

        pipeline = SequentialAgent(
            name="seq",
            agents=[agent_a],
            sequential_config=SequentialConfig(fail_strategy=FailStrategy.FAIL_FAST),
        )
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            with pytest.raises(Exception):
                await pipeline.execute("input", runner, ctx)

        # Must be restored even after exception
        assert agent_a.config.log_to_session is True

    async def test_save_turn_called_once_with_final_output(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        agent_a = _make_agent("a")
        agent_b = _make_agent("b")

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=[
            _mock_response("step1 output"),
            _mock_response("step2 final"),
        ])
        runner.save_turn = AsyncMock()

        pipeline = SequentialAgent(name="seq", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("user query", runner, ctx)

        runner.save_turn.assert_called_once()
        call_kwargs = runner.save_turn.call_args
        assert call_kwargs.kwargs["session_id"] == "sess-1"
        assert call_kwargs.kwargs["user_message"] == "user query"
        assert call_kwargs.kwargs["assistant_message"] == "step2 final"

    async def test_save_turn_not_called_without_session_id(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        agent_a = _make_agent("a")
        runner = MagicMock()
        runner.run = AsyncMock(return_value=_mock_response("output"))
        runner.save_turn = AsyncMock()

        pipeline = SequentialAgent(name="seq", agents=[agent_a])
        ctx = create_run_context()  # no session_id

        with _patch_span_scope():
            await pipeline.execute("input", runner, ctx)

        runner.save_turn.assert_not_called()

    async def test_pipeline_context_injected_for_step_2(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        agent_a = _make_agent("a")
        agent_b = _make_agent("b")
        captured_contexts: list = []

        runner = MagicMock()
        runner.save_turn = AsyncMock()

        async def capturing_run(agent, input, context=None):
            if context:
                captured_contexts.append(dict(context.metadata))
            return _mock_response(f"output-{agent.name}")

        runner.run = capturing_run

        pipeline = SequentialAgent(name="seq", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("original input", runner, ctx)

        # Step 1 (agent_a) should NOT see pipeline context
        assert "pipeline_context" not in (captured_contexts[0] if captured_contexts else {})
        # Step 2 (agent_b) SHOULD see pipeline context with step 1 output
        assert len(captured_contexts) >= 2
        assert "pipeline_context" in captured_contexts[1]
        assert "output-a" in captured_contexts[1]["pipeline_context"]

    async def test_pipeline_context_cleaned_from_metadata_after_run(self):
        from orchestrator.agent.workflow.sequential import SequentialAgent

        agent_a = _make_agent("a")
        runner = MagicMock()
        runner.run = AsyncMock(return_value=_mock_response("done"))
        runner.save_turn = AsyncMock()

        pipeline = SequentialAgent(name="seq", agents=[agent_a])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("input", runner, ctx)

        assert "pipeline_context" not in ctx.metadata


# ---------------------------------------------------------------------------
# ParallelAgent
# ---------------------------------------------------------------------------

class TestParallelTransparency:
    async def test_log_to_session_disabled_during_execution(self):
        from orchestrator.agent.workflow.parallel import ParallelAgent

        log_values: list[bool] = []
        agent_a = _make_agent("a", log_to_session=True)
        agent_b = _make_agent("b", log_to_session=True)

        runner = MagicMock()
        runner.save_turn = AsyncMock()

        async def capturing_run(agent, input, context=None):
            log_values.append(agent.config.log_to_session)
            return _mock_response(f"output-{agent.name}")

        runner.run = capturing_run

        pipeline = ParallelAgent(name="par", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("input", runner, ctx)

        assert all(v is False for v in log_values)

    async def test_log_to_session_restored_after_execution(self):
        from orchestrator.agent.workflow.parallel import ParallelAgent

        agent_a = _make_agent("a", log_to_session=True)
        agent_b = _make_agent("b", log_to_session=True)

        runner = MagicMock()
        runner.run = AsyncMock(return_value=_mock_response())
        runner.save_turn = AsyncMock()

        pipeline = ParallelAgent(name="par", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("input", runner, ctx)

        assert agent_a.config.log_to_session is True
        assert agent_b.config.log_to_session is True

    async def test_save_turn_called_once(self):
        from orchestrator.agent.workflow.parallel import ParallelAgent

        agent_a = _make_agent("a")
        agent_b = _make_agent("b")

        runner = MagicMock()
        runner.run = AsyncMock(return_value=_mock_response("parallel output"))
        runner.save_turn = AsyncMock()

        pipeline = ParallelAgent(name="par", agents=[agent_a, agent_b])
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await pipeline.execute("user query", runner, ctx)

        runner.save_turn.assert_called_once()
        assert runner.save_turn.call_args.kwargs["user_message"] == "user query"


# ---------------------------------------------------------------------------
# LoopAgent
# ---------------------------------------------------------------------------

class TestLoopTransparency:
    async def test_log_to_session_disabled_during_loop(self):
        from orchestrator.agent.workflow.loop import LoopAgent
        from orchestrator.agent.types import TerminationConfig, TerminationType

        log_values: list[bool] = []
        sub = _make_agent("looper", log_to_session=True)

        runner = MagicMock()
        runner.save_turn = AsyncMock()

        call_count = 0

        async def capturing_run(agent, input, context=None):
            nonlocal call_count
            log_values.append(agent.config.log_to_session)
            call_count += 1
            resp = _mock_response("done")
            resp.content = "COMPLETE: final output" if call_count >= 2 else "still going"
            return resp

        runner.run = capturing_run

        loop = LoopAgent(
            name="loop-agent",
            agent=sub,
            termination=TerminationConfig(
                type=TerminationType.OUTPUT_MATCH,
                pattern="COMPLETE",
                max_iterations=5,
            ),
        )
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await loop.execute("input", runner, ctx)

        assert all(v is False for v in log_values)

    async def test_log_to_session_restored_after_loop(self):
        from orchestrator.agent.workflow.loop import LoopAgent
        from orchestrator.agent.types import TerminationConfig, TerminationType

        sub = _make_agent("looper", log_to_session=True)
        runner = MagicMock()
        runner.save_turn = AsyncMock()

        call_count = 0

        async def mock_run(agent, input, context=None):
            nonlocal call_count
            call_count += 1
            resp = _mock_response()
            resp.content = "COMPLETE: done" if call_count >= 1 else "still going"
            return resp

        runner.run = mock_run

        loop = LoopAgent(
            name="loop-agent",
            agent=sub,
            termination=TerminationConfig(
                type=TerminationType.OUTPUT_MATCH,
                pattern="COMPLETE",
                max_iterations=5,
            ),
        )
        ctx = create_run_context(session_id="sess-1")

        with _patch_span_scope():
            await loop.execute("input", runner, ctx)

        assert sub.config.log_to_session is True
