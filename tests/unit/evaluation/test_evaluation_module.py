"""
Unit tests for the evaluation module.

Covers:
- EvalCase / CriterionScore / EvalResult types
- EvalResult.compute_overall() logic
- EvaluatorAgent._judge_criterion() with mocked LLM
- EvaluatorAgent.evaluate() full flow
- EvaluatorAgent.execute() (workflow entry point)
- _parse_json_response() edge cases
- LangfuseDatasetClient (all methods, with mocked Langfuse client)
- Import safety: DeepEvalEvaluator / RagasEvaluator raise ImportError cleanly when deps absent
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.evaluation.types import (
    CriterionScore,
    EvalCase,
    EvalResult,
    EvalStatus,
)
from orchestrator.evaluation.evaluator_agent import (
    EvaluatorAgent,
    _parse_json_response,
    create_evaluator_agent,
)
from orchestrator.evaluation.langfuse_datasets import LangfuseDatasetClient


# ===========================================================================
# EvalCase
# ===========================================================================


class TestEvalCase:
    def test_auto_case_id(self):
        case = EvalCase(input_text="hello")
        assert case.case_id.startswith("case_")

    def test_explicit_case_id(self):
        case = EvalCase(input_text="hello", case_id="my-id")
        assert case.case_id == "my-id"

    def test_default_context_empty(self):
        case = EvalCase(input_text="hello")
        assert case.context == []

    def test_to_dict_round_trip(self):
        case = EvalCase(
            input_text="What is AI?",
            expected_output="AI is...",
            context=["chunk1", "chunk2"],
            metadata={"tag": "test"},
            case_id="c-001",
        )
        d = case.to_dict()
        restored = EvalCase.from_dict(d)
        assert restored.input_text == case.input_text
        assert restored.expected_output == case.expected_output
        assert restored.context == case.context
        assert restored.metadata == case.metadata
        assert restored.case_id == case.case_id

    def test_from_dict_missing_fields_use_defaults(self):
        case = EvalCase.from_dict({"input_text": "hi"})
        assert case.input_text == "hi"
        assert case.expected_output is None
        assert case.context == []


# ===========================================================================
# CriterionScore
# ===========================================================================


class TestCriterionScore:
    def test_to_dict(self):
        cs = CriterionScore(criterion="correctness", score=0.9, passed=True, reasoning="Good")
        d = cs.to_dict()
        assert d["criterion"] == "correctness"
        assert d["score"] == 0.9
        assert d["passed"] is True
        assert d["reasoning"] == "Good"


# ===========================================================================
# EvalResult.compute_overall
# ===========================================================================


class TestEvalResultComputeOverall:
    def test_empty_scores_gives_skipped(self):
        result = EvalResult()
        result.compute_overall()
        assert result.status == EvalStatus.SKIPPED
        assert result.overall_score is None
        assert result.overall_passed is False

    def test_all_pass(self):
        result = EvalResult(
            scores=[
                CriterionScore("correctness", 0.9, passed=True),
                CriterionScore("conciseness", 0.8, passed=True),
            ]
        )
        result.compute_overall(pass_threshold=0.7)
        assert result.overall_passed is True
        assert result.status == EvalStatus.PASSED
        assert abs(result.overall_score - 0.85) < 0.01

    def test_one_fail_marks_overall_failed(self):
        result = EvalResult(
            scores=[
                CriterionScore("correctness", 0.9, passed=True),
                CriterionScore("safety", 0.3, passed=False),
            ]
        )
        result.compute_overall(pass_threshold=0.7)
        assert result.overall_passed is False
        assert result.status == EvalStatus.FAILED

    def test_threshold_applied_to_existing_scores(self):
        result = EvalResult(
            scores=[
                CriterionScore("correctness", 0.65, passed=True),  # will be flipped to False
            ]
        )
        result.compute_overall(pass_threshold=0.7)
        assert result.scores[0].passed is False
        assert result.overall_passed is False

    def test_to_dict_excludes_private_metadata(self):
        result = EvalResult(
            scores=[CriterionScore("c", 1.0, passed=True)],
            metadata={"_usage": "should_be_hidden", "visible": "yes"},
        )
        result.compute_overall()
        d = result.to_dict()
        assert "_usage" not in d["metadata"]
        assert d["metadata"]["visible"] == "yes"

    def test_round_trip(self):
        result = EvalResult(
            scores=[CriterionScore("correctness", 0.9, passed=True, reasoning="Good")],
            evaluator_name="test-judge",
            case_id="c-001",
            agent_response="Paris.",
        )
        result.compute_overall()
        d = result.to_dict()
        restored = EvalResult.from_dict(d)
        assert restored.evaluator_name == "test-judge"
        assert restored.scores[0].criterion == "correctness"
        assert restored.scores[0].score == 0.9


# ===========================================================================
# _parse_json_response
# ===========================================================================


class TestParseJsonResponse:
    def test_clean_json(self):
        raw = '{"score": 0.9, "passed": true, "reasoning": "Good"}'
        result = _parse_json_response(raw)
        assert result["score"] == 0.9
        assert result["passed"] is True

    def test_json_embedded_in_text(self):
        raw = 'Here is my evaluation:\n{"score": 0.7, "passed": true, "reasoning": "OK"}\nDone.'
        result = _parse_json_response(raw)
        assert result["score"] == 0.7

    def test_invalid_json_returns_empty_dict(self):
        result = _parse_json_response("not json at all")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        assert _parse_json_response("") == {}


# ===========================================================================
# EvaluatorAgent construction
# ===========================================================================


class TestEvaluatorAgentConstruction:
    def test_create_with_defaults(self):
        agent = EvaluatorAgent(name="judge")
        assert agent.criteria == ["correctness", "helpfulness"]
        assert agent.pass_threshold == 0.7
        assert agent.judge_model is None

    def test_create_factory(self):
        agent = create_evaluator_agent(
            "my-judge",
            ["correctness", "tone"],
            pass_threshold=0.8,
        )
        assert agent.name == "my-judge"
        assert agent.criteria == ["correctness", "tone"]
        assert agent.pass_threshold == 0.8

    def test_no_memory_enforced(self):
        agent = EvaluatorAgent(name="judge")
        assert agent.memory_config.search_memories is False
        assert agent.memory_config.store_memories is False

    def test_no_session_enforced(self):
        agent = EvaluatorAgent(name="judge")
        assert agent.config.log_to_session is False

    def test_default_instructions_set(self):
        agent = EvaluatorAgent(name="judge")
        assert len(agent.instructions) > 0

    def test_name_required(self):
        from orchestrator.agent.exceptions import AgentConfigurationError
        with pytest.raises(AgentConfigurationError):
            EvaluatorAgent(name="")

    def test_to_dict_includes_eval_fields(self):
        agent = create_evaluator_agent("j", ["correctness"], rubrics={"correctness": "Be right."})
        d = agent.to_dict()
        assert d["workflow_type"] == "evaluator"
        assert d["criteria"] == ["correctness"]
        assert d["rubrics"]["correctness"] == "Be right."


# ===========================================================================
# EvaluatorAgent._judge_criterion (mocked LLM)
# ===========================================================================


class TestEvaluatorAgentJudgeCriterion:
    def _make_llm_response(self, score: float, passed: bool, reasoning: str) -> MagicMock:
        r = MagicMock()
        r.content = json.dumps({"score": score, "passed": passed, "reasoning": reasoning})
        r.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return r

    @pytest.mark.asyncio
    async def test_returns_correct_score(self):
        agent = EvaluatorAgent(name="judge", criteria=["correctness"])
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=self._make_llm_response(0.9, True, "Correct"))

        case = EvalCase(input_text="Q", expected_output="A")
        score, usage = await agent._judge_criterion("correctness", case, "answer", llm)

        assert score.score == 0.9
        assert score.passed is True
        assert score.reasoning == "Correct"
        assert usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_score_clamped_to_0_1(self):
        agent = EvaluatorAgent(name="judge", criteria=["c"])
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=self._make_llm_response(1.5, True, ""))

        case = EvalCase(input_text="Q")
        score, _ = await agent._judge_criterion("c", case, "ans", llm)
        assert score.score == 1.0

    @pytest.mark.asyncio
    async def test_llm_error_returns_zero_score(self):
        agent = EvaluatorAgent(name="judge", criteria=["c"])
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

        case = EvalCase(input_text="Q")
        score, usage = await agent._judge_criterion("c", case, "ans", llm)
        assert score.score == 0.0
        assert score.passed is False
        assert "LLM down" in score.reasoning
        assert usage.total_tokens == 0

    @pytest.mark.asyncio
    async def test_json_fallback_from_free_text(self):
        """LLM ignores json_mode and wraps JSON in text — should still parse."""
        agent = EvaluatorAgent(name="judge", criteria=["c"])
        r = MagicMock()
        r.content = 'Sure! Here: {"score": 0.75, "passed": true, "reasoning": "OK"}'
        r.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value=r)

        case = EvalCase(input_text="Q")
        score, _ = await agent._judge_criterion("c", case, "ans", llm)
        assert score.score == 0.75

    @pytest.mark.asyncio
    async def test_rubric_included_in_prompt(self):
        agent = EvaluatorAgent(
            name="judge",
            criteria=["tone"],
            rubrics={"tone": "Must be polite."},
        )
        captured_messages = []

        async def mock_chat(messages, config, auto_session):
            captured_messages.extend(messages)
            r = MagicMock()
            r.content = '{"score": 0.8, "passed": true, "reasoning": "Polite"}'
            r.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            return r

        llm = MagicMock()
        llm.chat = mock_chat

        case = EvalCase(input_text="Hello there")
        await agent._judge_criterion("tone", case, "Hi!", llm)
        full_text = " ".join(m["content"] for m in captured_messages)
        assert "Must be polite." in full_text

    @pytest.mark.asyncio
    async def test_context_included_in_prompt_when_present(self):
        agent = EvaluatorAgent(name="judge", criteria=["faithfulness"])
        captured = []

        async def mock_chat(messages, config, auto_session):
            captured.extend(messages)
            r = MagicMock()
            r.content = '{"score": 0.9, "passed": true, "reasoning": "faithful"}'
            r.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
            return r

        llm = MagicMock()
        llm.chat = mock_chat

        case = EvalCase(input_text="Q", context=["Chunk A", "Chunk B"])
        await agent._judge_criterion("faithfulness", case, "ans", llm)
        full_text = " ".join(m["content"] for m in captured)
        assert "Chunk A" in full_text
        assert "Chunk B" in full_text


# ===========================================================================
# EvaluatorAgent.evaluate() full flow
# ===========================================================================


class TestEvaluatorAgentEvaluate:
    def _mock_container(self, llm_responses: list[dict]) -> MagicMock:
        """Build a mock container whose llm_client returns sequential responses."""
        responses = iter(llm_responses)

        async def mock_chat(messages, config, auto_session):
            resp_data = next(responses)
            r = MagicMock()
            r.content = json.dumps(resp_data)
            r.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
            return r

        llm = MagicMock()
        llm.chat = mock_chat
        container = MagicMock()
        container.llm_client = llm
        return container

    @pytest.mark.asyncio
    async def test_all_criteria_scored(self):
        agent = EvaluatorAgent(name="judge", criteria=["correctness", "tone"])
        responses = [
            {"score": 0.9, "passed": True, "reasoning": "Correct"},
            {"score": 0.8, "passed": True, "reasoning": "Polite"},
        ]
        container = self._mock_container(responses)

        with patch("orchestrator.core.container.get_container", return_value=container):
            result = await agent.evaluate(
                EvalCase(input_text="Q"), agent_response_text="A"
            )

        assert len(result.scores) == 2
        assert result.scores[0].criterion == "correctness"
        assert result.scores[1].criterion == "tone"
        assert result.overall_passed is True

    @pytest.mark.asyncio
    async def test_overall_fails_when_one_criterion_fails(self):
        agent = EvaluatorAgent(name="judge", criteria=["correctness", "safety"], pass_threshold=0.7)
        responses = [
            {"score": 0.9, "passed": True, "reasoning": "Good"},
            {"score": 0.4, "passed": False, "reasoning": "Unsafe"},
        ]
        container = self._mock_container(responses)

        with patch("orchestrator.core.container.get_container", return_value=container):
            result = await agent.evaluate(EvalCase(input_text="Q"), "A")

        assert result.overall_passed is False
        assert result.status == EvalStatus.FAILED

    @pytest.mark.asyncio
    async def test_result_has_correct_case_id(self):
        agent = EvaluatorAgent(name="judge", criteria=["c"])
        container = self._mock_container([{"score": 1.0, "passed": True, "reasoning": ""}])

        case = EvalCase(input_text="Q", case_id="my-case-id")
        with patch("orchestrator.core.container.get_container", return_value=container):
            result = await agent.evaluate(case, "A")

        assert result.case_id == "my-case-id"


# ===========================================================================
# EvaluatorAgent.execute() — workflow entry point
# ===========================================================================


class TestEvaluatorAgentExecute:
    def _mock_container(self) -> MagicMock:
        async def mock_chat(messages, config, auto_session):
            r = MagicMock()
            r.content = '{"score": 0.85, "passed": true, "reasoning": "Good"}'
            r.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
            return r

        llm = MagicMock()
        llm.chat = mock_chat
        container = MagicMock()
        container.llm_client = llm
        return container

    @pytest.mark.asyncio
    async def test_execute_parses_json_payload(self):
        agent = EvaluatorAgent(name="judge", criteria=["correctness"])
        case = EvalCase(input_text="Q", case_id="c-001")
        payload = json.dumps({"case": case.to_dict(), "agent_response": "Paris."})

        container = self._mock_container()
        with patch("orchestrator.core.container.get_container", return_value=container):
            response = await agent.execute(payload, runner=MagicMock(), context=MagicMock())

        assert response.agent_name == "judge"
        result_dict = json.loads(response.content)
        assert result_dict["case_id"] == "c-001"
        assert result_dict["scores"][0]["criterion"] == "correctness"

    @pytest.mark.asyncio
    async def test_execute_fallback_for_non_json_input(self):
        agent = EvaluatorAgent(name="judge", criteria=["correctness"])
        container = self._mock_container()
        with patch("orchestrator.core.container.get_container", return_value=container):
            response = await agent.execute("plain text input", runner=MagicMock(), context=MagicMock())

        assert response.agent_name == "judge"

    @pytest.mark.asyncio
    async def test_execute_metadata_contains_eval_result(self):
        agent = EvaluatorAgent(name="judge", criteria=["c"])
        case = EvalCase(input_text="Q")
        payload = json.dumps({"case": case.to_dict(), "agent_response": "A"})

        container = self._mock_container()
        with patch("orchestrator.core.container.get_container", return_value=container):
            response = await agent.execute(payload, runner=MagicMock(), context=MagicMock())

        assert response.run_artifacts is not None
        assert "eval_result" in response.run_artifacts


# ===========================================================================
# LangfuseDatasetClient
# ===========================================================================


def _make_lf_client(enabled: bool = True) -> MagicMock:
    """Build a mock LangfuseClient."""
    client = MagicMock()
    client.is_enabled = enabled
    if not enabled:
        client.create_dataset.return_value = None
        client.get_dataset.return_value = None
        client.create_dataset_item.return_value = None
        client.score.return_value = None
    return client


class TestLangfuseDatasetClient:
    def test_ensure_dataset_creates_when_absent(self):
        lf = _make_lf_client()
        lf.get_dataset.return_value = None  # doesn't exist yet
        mock_dataset = MagicMock()
        lf.create_dataset.return_value = mock_dataset

        ds = LangfuseDatasetClient("my-dataset", langfuse_client=lf)
        result = ds.ensure_dataset_exists(description="test")

        lf.create_dataset.assert_called_once_with(
            name="my-dataset", description="test", metadata=None
        )
        assert result is mock_dataset

    def test_ensure_dataset_skips_if_exists(self):
        lf = _make_lf_client()
        existing = MagicMock()
        lf.get_dataset.return_value = existing

        ds = LangfuseDatasetClient("my-dataset", langfuse_client=lf)
        result = ds.ensure_dataset_exists()

        lf.create_dataset.assert_not_called()
        assert result is existing

    def test_upload_case_returns_item_id(self):
        lf = _make_lf_client()
        mock_item = MagicMock()
        mock_item.id = "item-abc"
        lf.create_dataset_item.return_value = mock_item

        ds = LangfuseDatasetClient("my-dataset", langfuse_client=lf)
        case = EvalCase(input_text="What is AI?", case_id="c-001")
        item_id = ds.upload_case(case)

        assert item_id == "item-abc"
        lf.create_dataset_item.assert_called_once()
        call_kwargs = lf.create_dataset_item.call_args.kwargs
        assert call_kwargs["dataset_name"] == "my-dataset"
        assert call_kwargs["input"]["input_text"] == "What is AI?"

    def test_upload_case_returns_none_when_item_creation_fails(self):
        lf = _make_lf_client()
        lf.create_dataset_item.return_value = None

        ds = LangfuseDatasetClient("my-dataset", langfuse_client=lf)
        result = ds.upload_case(EvalCase(input_text="Q"))
        assert result is None

    def test_upload_bulk_cases_returns_mapping(self):
        lf = _make_lf_client()
        counter = [0]

        def make_item(*args, **kwargs):
            counter[0] += 1
            m = MagicMock()
            m.id = f"item-{counter[0]}"
            return m

        lf.create_dataset_item.side_effect = make_item

        cases = [EvalCase(input_text=f"Q{i}", case_id=f"c-{i}") for i in range(3)]
        ds = LangfuseDatasetClient("ds", langfuse_client=lf)
        mapping = ds.upload_bulk_cases(cases)

        assert len(mapping) == 3
        assert mapping["c-0"] == "item-1"
        assert mapping["c-2"] == "item-3"

    def test_upload_scores_calls_score_per_criterion(self):
        lf = _make_lf_client()
        ds = LangfuseDatasetClient("ds", langfuse_client=lf)

        result = EvalResult(
            scores=[
                CriterionScore("correctness", 0.9, passed=True, reasoning="Good"),
                CriterionScore("tone", 0.8, passed=True, reasoning="Polite"),
            ],
            evaluator_name="judge",
            overall_score=0.85,
        )
        ds.upload_scores(result, trace_id="trace-123")

        # 2 criterion scores + 1 overall = 3 calls
        assert lf.score.call_count == 3
        names = [call.kwargs["name"] for call in lf.score.call_args_list]
        assert "judge/correctness" in names
        assert "judge/tone" in names
        assert "judge/overall" in names

    def test_upload_scores_noop_when_client_none(self):
        ds = LangfuseDatasetClient("ds", langfuse_client=None)
        # Should not raise
        result = EvalResult(
            scores=[CriterionScore("c", 0.9, passed=True)],
            evaluator_name="judge",
            overall_score=0.9,
        )
        ds.upload_scores(result, trace_id="t-001")

    def test_fetch_cases_returns_eval_cases(self):
        lf = _make_lf_client()

        item1 = MagicMock()
        item1.id = "i-1"
        item1.input = {"input_text": "What is Python?", "context": ["Python is..."]}
        item1.expected_output = "A language"
        item1.metadata = {}

        item2 = MagicMock()
        item2.id = "i-2"
        item2.input = {"input_text": "What is Rust?", "context": []}
        item2.expected_output = None
        item2.metadata = {}

        mock_dataset = MagicMock()
        mock_dataset.items = [item1, item2]
        lf.get_dataset.return_value = mock_dataset

        ds = LangfuseDatasetClient("ds", langfuse_client=lf)
        cases = ds.fetch_cases()

        assert len(cases) == 2
        assert cases[0].input_text == "What is Python?"
        assert cases[0].context == ["Python is..."]
        assert cases[0].expected_output == "A language"
        assert cases[0].metadata["langfuse_item_id"] == "i-1"
        assert cases[1].expected_output is None

    def test_fetch_cases_respects_limit(self):
        lf = _make_lf_client()
        items = []
        for i in range(10):
            m = MagicMock()
            m.id = f"i-{i}"
            m.input = {"input_text": f"Q{i}", "context": []}
            m.expected_output = None
            m.metadata = {}
            items.append(m)

        mock_dataset = MagicMock()
        mock_dataset.items = items
        lf.get_dataset.return_value = mock_dataset

        ds = LangfuseDatasetClient("ds", langfuse_client=lf)
        cases = ds.fetch_cases(limit=3)
        assert len(cases) == 3

    def test_fetch_cases_returns_empty_on_missing_dataset(self):
        lf = _make_lf_client()
        lf.get_dataset.return_value = None

        ds = LangfuseDatasetClient("ds", langfuse_client=lf)
        cases = ds.fetch_cases()
        assert cases == []


# ===========================================================================
# Optional dependency import safety
# ===========================================================================


class TestOptionalDependencyImports:
    def test_deepeval_evaluator_raises_import_error_when_not_installed(self):
        import sys
        # Temporarily hide deepeval if it happens to be installed
        deepeval_mod = sys.modules.pop("deepeval", None)
        try:
            from orchestrator.evaluation.deepeval_eval import (
                DeepEvalEvaluator,
                _require_deepeval,
            )
            with pytest.raises(ImportError, match="deepeval"):
                _require_deepeval()
        finally:
            if deepeval_mod is not None:
                sys.modules["deepeval"] = deepeval_mod

    def test_ragas_evaluator_raises_import_error_when_not_installed(self):
        import sys
        ragas_mod = sys.modules.pop("ragas", None)
        try:
            from orchestrator.evaluation.ragas_eval import _require_ragas
            with pytest.raises(ImportError, match="ragas"):
                _require_ragas()
        finally:
            if ragas_mod is not None:
                sys.modules["ragas"] = ragas_mod

    def test_evaluation_module_imports_without_optional_deps(self):
        """The top-level evaluation module must import cleanly even without deepeval/ragas."""
        import orchestrator.evaluation as ev
        assert hasattr(ev, "EvalCase")
        assert hasattr(ev, "EvaluatorAgent")
        assert hasattr(ev, "LangfuseDatasetClient")
