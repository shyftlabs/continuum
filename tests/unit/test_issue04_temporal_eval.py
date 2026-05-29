"""
Unit tests for Issue 04 — Temporal Workflows & Evaluation fixes.

Tests step validation, retry policy, WaitStep validation, AgentActivityResult null safety,
evaluator JSON parse error handling, RAGAS threshold, DeepEval name-based mapping,
EvalCase context coercion, secrets redaction, registry thread safety.
"""

from __future__ import annotations

import json
import threading

import pytest


# ---------------------------------------------------------------------------
# 04-#3: Retry policy has explicit backoff
# ---------------------------------------------------------------------------


class TestRetryPolicyExplicit:
    """Verify all workflows use explicit retry backoff in their source."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "orchestrator.temporal.workflows.agent_workflow",
            "orchestrator.temporal.workflows.sequential_workflow",
            "orchestrator.temporal.workflows.parallel_workflow",
            "orchestrator.temporal.workflows.loop_workflow",
        ],
    )
    def test_retry_policy_has_backoff(self, module_path):
        import importlib
        import inspect

        mod = importlib.import_module(module_path)
        source = inspect.getsource(mod)
        assert "initial_interval" in source, f"{module_path} missing initial_interval"
        assert "backoff_coefficient" in source, f"{module_path} missing backoff_coefficient"
        assert "maximum_interval" in source, f"{module_path} missing maximum_interval"


# ---------------------------------------------------------------------------
# 04-#6: Upfront step validation
# ---------------------------------------------------------------------------


class TestStepValidation:
    def test_parse_step_valid_agent(self):
        from orchestrator.temporal.types import AgentStep, parse_step

        step = parse_step({"type": "agent", "agent_name": "test-agent"})
        assert isinstance(step, AgentStep)
        assert step.agent_name == "test-agent"

    def test_parse_step_unknown_type_raises(self):
        from orchestrator.temporal.types import parse_step

        with pytest.raises(ValueError, match="Unknown step type"):
            parse_step({"type": "invalid_type"})

    def test_parse_step_missing_type_raises(self):
        from orchestrator.temporal.types import parse_step

        with pytest.raises(ValueError, match="Unknown step type: None"):
            parse_step({"agent_name": "test"})


# ---------------------------------------------------------------------------
# 04-#7: ConditionalStep has configurable timeout/retries
# ---------------------------------------------------------------------------


class TestConditionalStepConfigurable:
    def test_default_values(self):
        from orchestrator.temporal.types import ConditionalStep

        step = ConditionalStep(condition_agent="cond-agent")
        assert step.timeout == 300
        assert step.retries == 3

    def test_custom_values(self):
        from orchestrator.temporal.types import ConditionalStep

        step = ConditionalStep(
            condition_agent="cond-agent",
            timeout=600,
            retries=5,
            metadata={"key": "val"},
        )
        assert step.timeout == 600
        assert step.retries == 5
        assert step.metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# 04-#8: Evaluator JSON parse failure raises _ParseError
# ---------------------------------------------------------------------------


class TestEvaluatorJsonParsing:
    def test_valid_json_parses(self):
        from orchestrator.evaluation.evaluator_agent import _parse_json_response

        result = _parse_json_response('{"score": 0.9, "passed": true, "reasoning": "Good"}')
        assert result["score"] == 0.9

    def test_json_in_markdown_block(self):
        from orchestrator.evaluation.evaluator_agent import _parse_json_response

        text = 'Here is my evaluation:\n{"score": 0.5, "passed": false, "reasoning": "Partial"}\n'
        result = _parse_json_response(text)
        assert result["score"] == 0.5

    def test_invalid_json_raises_parse_error(self):
        from orchestrator.evaluation.evaluator_agent import _ParseError, _parse_json_response

        with pytest.raises(_ParseError, match="Could not extract valid JSON"):
            _parse_json_response("This is not JSON at all")

    def test_empty_string_raises_parse_error(self):
        from orchestrator.evaluation.evaluator_agent import _ParseError, _parse_json_response

        with pytest.raises(_ParseError):
            _parse_json_response("")


# ---------------------------------------------------------------------------
# 04-#9: RAGAS threshold configurable
# ---------------------------------------------------------------------------


class TestRagasThresholdConfigurable:
    def test_default_threshold(self):
        """RagasEvaluator should accept pass_threshold."""
        # Can't instantiate without ragas installed, test the dataclass fields
        import dataclasses

        from orchestrator.evaluation.ragas_eval import RagasEvaluator

        fields = {f.name for f in dataclasses.fields(RagasEvaluator)}
        assert "pass_threshold" in fields
        assert "metric_thresholds" in fields


# ---------------------------------------------------------------------------
# 04-#10: Secrets redaction — circular ref + max depth
# ---------------------------------------------------------------------------


class TestSecretsRedaction:
    def test_circular_reference_handled(self):
        from orchestrator.utils.secrets import redact_dict

        d: dict = {"name": "test", "api_key": "sk-secret123"}
        d["loop"] = d
        result = redact_dict(d)
        assert result["loop"] == {"_redacted": "[CIRCULAR REFERENCE]"}
        assert "****" in result["api_key"]

    def test_max_depth_returns_redacted_placeholder(self):
        from orchestrator.utils.secrets import redact_dict

        deep = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"api_key": "secret"}}}}}}}
        result = redact_dict(deep, max_depth=3)
        # At depth 4, should return redacted placeholder
        inner = result["l1"]["l2"]["l3"]["l4"]
        assert "_redacted" in inner

    def test_normal_redaction_works(self):
        from orchestrator.utils.secrets import redact_dict

        data = {
            "username": "admin",
            "password": "super-secret-pass",
            "config": {"token": "tok-abc123"},
        }
        result = redact_dict(data)
        assert result["username"] == "admin"
        assert "super-secret" not in result["password"]
        assert "tok-abc123" not in str(result["config"]["token"])


# ---------------------------------------------------------------------------
# 04-#11: Registry thread-safe reads
# ---------------------------------------------------------------------------


class TestRegistryThreadSafety:
    def test_concurrent_register_and_get(self):
        from orchestrator.agent.base import BaseAgent
        from orchestrator.temporal.registry import AgentRegistry

        registry = AgentRegistry()
        errors = []

        def register_agents():
            try:
                for i in range(50):
                    agent = BaseAgent(name=f"agent-{i}")
                    registry.register(agent)
            except Exception as e:
                errors.append(e)

        def read_agents():
            try:
                for _ in range(50):
                    registry.list_agents()
                    try:
                        registry.get("agent-0")
                    except Exception:
                        pass  # May not be registered yet
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=register_agents)
        t2 = threading.Thread(target=read_agents)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors


# ---------------------------------------------------------------------------
# 04-#14: EvalCase.context type coercion
# ---------------------------------------------------------------------------


class TestEvalCaseContextValidation:
    def test_string_context_coerced_to_list(self):
        from orchestrator.evaluation.types import EvalCase

        case = EvalCase(input_text="test", context="single string")
        assert case.context == ["single string"]

    def test_list_context_accepted(self):
        from orchestrator.evaluation.types import EvalCase

        case = EvalCase(input_text="test", context=["a", "b"])
        assert case.context == ["a", "b"]

    def test_invalid_context_raises(self):
        from orchestrator.evaluation.types import EvalCase

        with pytest.raises(TypeError, match="must be a list"):
            EvalCase(input_text="test", context=123)


# ---------------------------------------------------------------------------
# 04-#17: AgentActivityResult.from_agent_response null safety
# ---------------------------------------------------------------------------


class TestAgentActivityResultNullSafety:
    def test_from_response_with_none_usage(self):
        from orchestrator.temporal.types import AgentActivityResult

        class FakeResponse:
            content = "hello"
            status = "success"
            structured_output = None
            usage = None
            agents_used = []
            error = None

        result = AgentActivityResult.from_agent_response(FakeResponse())
        assert result.usage == {}
        assert result.content == "hello"

    def test_from_response_with_partial_usage(self):
        from orchestrator.temporal.types import AgentActivityResult

        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = None
            total_tokens = None

        class FakeResponse:
            content = "world"
            status = "success"
            structured_output = None
            usage = FakeUsage()
            agents_used = ["agent-1"]
            error = None

        result = AgentActivityResult.from_agent_response(FakeResponse())
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 0
        assert result.usage["total_tokens"] == 0


# ---------------------------------------------------------------------------
# 04-#18: WaitStep duration validation
# ---------------------------------------------------------------------------


class TestWaitStepValidation:
    def test_valid_duration(self):
        from orchestrator.temporal.types import WaitStep

        step = WaitStep(duration_seconds=60)
        assert step.duration_seconds == 60

    def test_zero_duration_rejected(self):
        from pydantic import ValidationError

        from orchestrator.temporal.types import WaitStep

        with pytest.raises(ValidationError):
            WaitStep(duration_seconds=0)

    def test_negative_duration_rejected(self):
        from pydantic import ValidationError

        from orchestrator.temporal.types import WaitStep

        with pytest.raises(ValidationError):
            WaitStep(duration_seconds=-5)

    def test_max_duration_accepted(self):
        from orchestrator.temporal.types import WaitStep

        step = WaitStep(duration_seconds=86400 * 7)  # 7 days
        assert step.duration_seconds == 86400 * 7

    def test_over_max_duration_rejected(self):
        from pydantic import ValidationError

        from orchestrator.temporal.types import WaitStep

        with pytest.raises(ValidationError):
            WaitStep(duration_seconds=86400 * 7 + 1)


# ---------------------------------------------------------------------------
# 04-#24: WorkflowInput payload size
# ---------------------------------------------------------------------------


class TestWorkflowInputPayloadSize:
    def test_normal_input_accepted(self):
        from orchestrator.temporal.types import WorkflowInput

        wi = WorkflowInput(steps=[], initial_input="Hello world")
        assert wi.initial_input == "Hello world"

    def test_oversized_input_rejected(self):
        from pydantic import ValidationError

        from orchestrator.temporal.types import WorkflowInput

        huge = "x" * 2_000_001
        with pytest.raises(ValidationError):
            WorkflowInput(steps=[], initial_input=huge)
