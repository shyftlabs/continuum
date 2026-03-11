"""
DeepEval integration for the evaluation module.

Wraps deepeval metrics behind the project's EvalCase / EvalResult types.

Requires:  pip install deepeval

Supported metrics (pass metric instances directly):
    AnswerRelevancyMetric   — does the response answer the question?
    FaithfulnessMetric      — does the response stick to retrieved context?
    ContextualPrecisionMetric  — was retrieved context actually useful?
    ContextualRecallMetric     — did retrieval find everything needed?
    ContextualRelevancyMetric  — are retrieved chunks relevant to the question?
    HallucinationMetric     — does the response introduce facts not in context?
    ToxicityMetric          — is the response harmful / toxic?
    GEval                   — any custom criterion in plain English (LLM-as-judge)

Example::

    from deepeval.metrics import AnswerRelevancyMetric, GEval
    from deepeval.test_case import LLMTestCaseParams
    from orchestrator.evaluation import DeepEvalEvaluator, EvalCase

    evaluator = DeepEvalEvaluator(
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            GEval(
                name="Conciseness",
                criteria="The response should be concise and avoid unnecessary filler.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            ),
        ]
    )

    result = await evaluator.evaluate(
        EvalCase(input_text="What is Python?"),
        agent_response_text="Python is a high-level programming language.",
    )
    print(result.overall_score)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from orchestrator.evaluation.types import CriterionScore, EvalCase, EvalResult, EvalStatus
from orchestrator.logging import get_logger

logger = get_logger(__name__)


def _require_deepeval() -> None:
    try:
        import deepeval  # noqa: F401
    except ImportError:
        raise ImportError(
            "deepeval is not installed.\n"
            "Install it with:  pip install deepeval"
        )


@dataclass
class DeepEvalEvaluator:
    """
    Evaluator backed by deepeval metrics.

    Wraps deepeval's synchronous evaluate() call in an executor so it does
    not block the async event loop.

    Args:
        metrics: List of deepeval BaseMetric instances to run.
        name:    Identifier used in EvalResult.evaluator_name.
    """

    metrics: list[Any] = field(default_factory=list)
    name: str = "deepeval-evaluator"

    def __post_init__(self) -> None:
        _require_deepeval()

    async def evaluate(
        self,
        case: EvalCase,
        agent_response_text: str,
    ) -> EvalResult:
        """
        Run all deepeval metrics against the case and response.

        Args:
            case:                EvalCase with input, optional expected output and context.
            agent_response_text: Produced agent text to evaluate.

        Returns:
            EvalResult with one CriterionScore per metric.
        """
        import time
        from deepeval import evaluate as _deepeval_evaluate
        from deepeval.test_case import LLMTestCase

        t0 = time.monotonic()

        test_case = LLMTestCase(
            input=case.input_text,
            actual_output=agent_response_text,
            expected_output=case.expected_output,
            retrieval_context=case.context if case.context else None,
        )

        try:
            from deepeval.evaluate.configs import DisplayConfig
            loop = asyncio.get_event_loop()
            eval_result = await loop.run_in_executor(
                None,
                lambda: _deepeval_evaluate(
                    test_cases=[test_case],
                    metrics=self.metrics,
                    display_config=DisplayConfig(print_results=False, show_indicator=False),
                ),
            )
        except Exception as exc:
            logger.error(f"DeepEvalEvaluator: deepeval.evaluate() failed: {exc}")
            return EvalResult(
                status=EvalStatus.ERROR,
                evaluator_name=self.name,
                case_id=case.case_id,
                agent_response=agent_response_text,
                latency_ms=int((time.monotonic() - t0) * 1000),
                metadata={"error": str(exc)},
            )

        # Scores are on the test result's metrics_data in the same order as self.metrics
        raw_metrics_data = []
        if eval_result and eval_result.test_results:
            raw_metrics_data = eval_result.test_results[0].metrics_data or []

        scores: list[CriterionScore] = []
        for i, metric in enumerate(self.metrics):
            md = raw_metrics_data[i] if i < len(raw_metrics_data) else None
            if md is not None:
                score_val = float(md.score or 0.0)
                passed = bool(md.success)
                reasoning = str(md.reason or "")
                criterion = md.name
            else:
                score_val = 0.0
                passed = False
                reasoning = ""
                criterion = getattr(metric, "name", type(metric).__name__)
            threshold = getattr(metric, "threshold", None)

            scores.append(
                CriterionScore(
                    criterion=criterion,
                    score=score_val,
                    passed=passed,
                    reasoning=reasoning,
                    metadata={"threshold": threshold} if threshold is not None else {},
                )
            )

        result = EvalResult(
            scores=scores,
            evaluator_name=self.name,
            case_id=case.case_id,
            agent_response=agent_response_text,
            latency_ms=int((time.monotonic() - t0) * 1000),
            status=EvalStatus.PASSED,
        )
        result.compute_overall()
        return result
