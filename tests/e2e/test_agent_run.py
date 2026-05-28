"""
End-to-end tests — full agent execution with real LLM, Redis, and Qdrant.

Tests the complete pipeline: agent creation, runner execution, session persistence,
and response generation through real services.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


from tests.e2e.conftest import skip_if_no_api_key as _skip_if_no_api_key
from tests.e2e.conftest import skip_on_api_error as _skip_on_api_error


class TestFullAgentRun:
    """Full end-to-end agent run through real infrastructure."""

    @_skip_on_api_error
    async def test_simple_agent_returns_response(self):
        """Create an agent, run it with a real LLM, get a response."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import RunContext

        agent = BaseAgent(
            name="test-simple-agent",
            instructions="You are a helpful assistant. Be extremely concise.",
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()
        context = RunContext(run_id="e2e-simple-run")

        response = await runner.run(
            agent, "What is 2+2? Reply with just the number.", context=context
        )

        assert response is not None
        assert response.content is not None
        assert "4" in response.content
        assert response.status.value in ("success", "completed")

    @_skip_on_api_error
    async def test_agent_with_system_instructions(self):
        """Agent with custom instructions produces appropriate response."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import RunContext

        agent = BaseAgent(
            name="test-pirate-agent",
            instructions=(
                "You are a pirate. Always respond in pirate speak. Keep responses under 20 words."
            ),
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()
        context = RunContext(run_id="e2e-pirate-run")

        response = await runner.run(agent, "Hello, how are you?", context=context)

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    @_skip_on_api_error
    async def test_agent_run_tracks_usage(self):
        """Agent run should track token usage."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import RunContext

        agent = BaseAgent(
            name="test-usage-agent",
            instructions="Be concise.",
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()
        context = RunContext(run_id="e2e-usage-run")

        response = await runner.run(agent, "Say hi", context=context)

        assert response.usage is not None
        assert response.usage.total_tokens > 0

    @_skip_on_api_error
    async def test_agent_clone_runs_independently(self):
        """Cloned agent should run independently without affecting original."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import RunContext

        original = BaseAgent(
            name="original-agent",
            instructions="You always respond with 'ORIGINAL'.",
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False),
        )

        cloned = original.clone(
            name="cloned-agent",
            instructions="You always respond with 'CLONED'.",
        )

        runner = AgentRunner()

        resp_original = await runner.run(
            original, "Identify yourself", context=RunContext(run_id="e2e-orig")
        )
        resp_cloned = await runner.run(
            cloned, "Identify yourself", context=RunContext(run_id="e2e-clone")
        )

        assert resp_original.agent_name == "original-agent"
        assert resp_cloned.agent_name == "cloned-agent"
        # Both should have valid content
        assert resp_original.content
        assert resp_cloned.content


class TestEvaluatorE2E:
    """End-to-end evaluation with real LLM."""

    @_skip_on_api_error
    async def test_evaluator_agent_scores_response(self):
        """Run a real evaluator agent against an answer."""
        _skip_if_no_api_key()

        from orchestrator.evaluation.evaluator_agent import EvaluatorAgent
        from orchestrator.evaluation.types import EvalCase

        evaluator = EvaluatorAgent(
            name="e2e-judge",
            criteria=["correctness"],
            pass_threshold=0.5,
        )

        case = EvalCase(
            input_text="What is the capital of France?",
            expected_output="Paris",
        )

        result = await evaluator.evaluate(case, "The capital of France is Paris.")

        assert result is not None
        assert len(result.scores) == 1
        assert result.scores[0].criterion == "correctness"
        assert result.scores[0].score > 0  # Should score well
        assert result.overall_score is not None

    @_skip_on_api_error
    async def test_evaluator_detects_wrong_answer(self):
        """Evaluator should give low score for incorrect answer."""
        _skip_if_no_api_key()

        from orchestrator.evaluation.evaluator_agent import EvaluatorAgent
        from orchestrator.evaluation.types import EvalCase

        evaluator = EvaluatorAgent(
            name="e2e-judge-wrong",
            criteria=["correctness"],
            pass_threshold=0.7,
        )

        case = EvalCase(
            input_text="What is the capital of France?",
            expected_output="Paris",
        )

        result = await evaluator.evaluate(case, "The capital of France is Berlin.")

        assert result is not None
        assert result.scores[0].score < 0.7  # Should score poorly
