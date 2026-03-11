"""
RAGAS integration for the evaluation module.

Wraps RAGAS retrieval-augmented evaluation metrics behind the project's
EvalCase / EvalResult types.

Requires:  pip install ragas datasets

IMPORTANT — EvalCase.context must be non-empty.
RAGAS metrics measure the relationship between a question, retrieved chunks,
and the generated answer. Without retrieved context, all RAGAS scores are
meaningless. Use EvaluatorAgent or DeepEvalEvaluator for generation-only
(no retrieval) evaluation.

Supported metric names (passed as strings):
    "faithfulness"          — does the answer stay faithful to the context?
    "answer_relevancy"      — does the answer address the question?
    "context_precision"     — were the retrieved chunks actually useful?
    "context_recall"        — did retrieval surface everything needed?

Example::

    from orchestrator.evaluation import RagasEvaluator, EvalCase

    evaluator = RagasEvaluator(
        metric_names=["faithfulness", "answer_relevancy"],
    )

    result = await evaluator.evaluate(
        EvalCase(
            input_text="What is the capital of France?",
            expected_output="Paris",
            context=["France is a country in Europe. Its capital is Paris."],
        ),
        agent_response_text="The capital of France is Paris.",
    )
    print(result.scores)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from orchestrator.evaluation.types import CriterionScore, EvalCase, EvalResult, EvalStatus
from orchestrator.logging import get_logger

logger = get_logger(__name__)

_ALL_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _require_ragas() -> None:
    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        raise ImportError(
            "ragas and/or datasets are not installed.\n"
            "Install them with:  pip install ragas datasets"
        )


def _build_metrics(metric_names: list[str]) -> list[Any]:
    """Resolve metric name strings to RAGAS metric objects (deferred import)."""
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    registry: dict[str, Any] = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    resolved = []
    for name in metric_names:
        if name not in registry:
            logger.warning(f"RagasEvaluator: unknown metric '{name}', skipping")
            continue
        resolved.append(registry[name])
    return resolved


@dataclass
class RagasEvaluator:
    """
    Evaluator backed by RAGAS metrics.

    Only useful when EvalCase.context is populated with retrieved documents.
    RAGAS evaluate() is synchronous and is run in an executor to avoid
    blocking the async event loop.

    Args:
        metric_names:      RAGAS metric name strings to evaluate.
                           Defaults to all four built-in metrics.
        name:              Identifier used in EvalResult.evaluator_name.
        ragas_llm:         Optional LangchainLLM wrapper for RAGAS
                           (if None, RAGAS uses its default LLM).
        ragas_embeddings:  Optional embeddings wrapper for RAGAS.
    """

    metric_names: list[str] = field(default_factory=lambda: list(_ALL_METRIC_NAMES))
    name: str = "ragas-evaluator"
    ragas_llm: Any | None = None
    ragas_embeddings: Any | None = None

    def __post_init__(self) -> None:
        _require_ragas()

    async def evaluate(
        self,
        case: EvalCase,
        agent_response_text: str,
    ) -> EvalResult:
        """
        Run RAGAS metrics against the case and response.

        Args:
            case:                EvalCase — context should be non-empty.
            agent_response_text: Produced agent text to evaluate.

        Returns:
            EvalResult with one CriterionScore per RAGAS metric.
        """
        import time
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate

        t0 = time.monotonic()

        if not case.context:
            logger.warning(
                "RagasEvaluator: EvalCase.context is empty — "
                "RAGAS metrics will be unreliable without retrieved documents."
            )

        data = {
            "question": [case.input_text],
            "answer": [agent_response_text],
            "contexts": [case.context],
            "ground_truth": [case.expected_output or ""],
        }
        dataset = Dataset.from_dict(data)
        metrics = _build_metrics(self.metric_names)

        if not metrics:
            return EvalResult(
                status=EvalStatus.ERROR,
                evaluator_name=self.name,
                case_id=case.case_id,
                agent_response=agent_response_text,
                metadata={"error": "No valid metrics to evaluate"},
            )

        try:
            kwargs: dict[str, Any] = {}
            if self.ragas_llm:
                kwargs["llm"] = self.ragas_llm
            if self.ragas_embeddings:
                kwargs["embeddings"] = self.ragas_embeddings

            loop = asyncio.get_event_loop()
            ragas_result = await loop.run_in_executor(
                None,
                lambda: ragas_evaluate(dataset=dataset, metrics=metrics, **kwargs),
            )

            result_row = ragas_result.to_pandas().iloc[0].to_dict()

        except Exception as exc:
            logger.error(f"RagasEvaluator: ragas.evaluate() failed: {exc}")
            return EvalResult(
                status=EvalStatus.ERROR,
                evaluator_name=self.name,
                case_id=case.case_id,
                agent_response=agent_response_text,
                latency_ms=int((time.monotonic() - t0) * 1000),
                metadata={"error": str(exc)},
            )

        import math

        scores: list[CriterionScore] = []
        for metric in metrics:
            metric_name = getattr(metric, "name", type(metric).__name__)
            raw = result_row.get(metric_name, 0.0)
            raw_score = 0.0 if (raw is None or (isinstance(raw, float) and math.isnan(raw))) else float(raw)
            scores.append(
                CriterionScore(
                    criterion=metric_name,
                    score=raw_score,
                    passed=raw_score >= 0.7,
                    reasoning="",
                    metadata={"ragas_metric": metric_name},
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
