"""
RAGAS integration for the evaluation module.

Wraps RAGAS retrieval-augmented evaluation metrics behind the project's
EvalCase / EvalResult types.

Requires:  pip install ragas datasets langchain-openai

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
import os
import warnings
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
            "Install them with:  pip install ragas datasets langchain-openai"
        )


def _build_metrics(metric_names: list[str], openai_api_key: str) -> list[Any]:
    """Return old-style RAGAS metric singletons with LLM/embeddings patched in.

    ragas.evaluate() checks isinstance(m, ragas.metrics.base.Metric), so we
    must use the legacy singletons (ragas.metrics._xxx), not the collections
    classes that inherit from BaseMetric.  We suppress the deprecation warnings
    since this is intentional for ragas v0.4.x compatibility.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (  # noqa: F401
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    lc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    lc_emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    ragas_llm = LangchainLLMWrapper(lc_llm)
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    registry: dict[str, Any] = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    # Patch LLM + embeddings onto each singleton
    for m in registry.values():
        m.llm = ragas_llm
    for name in ("answer_relevancy", "context_precision", "context_recall"):
        registry[name].embeddings = ragas_emb

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
        ragas_llm:         Ignored (kept for API compatibility; LLM is built internally).
        ragas_embeddings:  Ignored (kept for API compatibility).
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
        import warnings
        from datasets import Dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas import evaluate as ragas_evaluate

        t0 = time.monotonic()

        if not case.context:
            logger.warning(
                "RagasEvaluator: EvalCase.context is empty — "
                "RAGAS metrics will be unreliable without retrieved documents."
            )

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            return EvalResult(
                status=EvalStatus.ERROR,
                evaluator_name=self.name,
                case_id=case.case_id,
                agent_response=agent_response_text,
                metadata={"error": "OPENAI_API_KEY not set"},
            )

        data = {
            "question": [case.input_text],
            "answer": [agent_response_text],
            "contexts": [case.context],
            "ground_truth": [case.expected_output or ""],
        }
        dataset = Dataset.from_dict(data)
        metrics = _build_metrics(self.metric_names, openai_api_key)

        if not metrics:
            return EvalResult(
                status=EvalStatus.ERROR,
                evaluator_name=self.name,
                case_id=case.case_id,
                agent_response=agent_response_text,
                metadata={"error": "No valid metrics to evaluate"},
            )

        try:
            loop = asyncio.get_event_loop()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ragas_result = await loop.run_in_executor(
                    None,
                    lambda: ragas_evaluate(dataset=dataset, metrics=metrics),
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
