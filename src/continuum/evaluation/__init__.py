"""
Evaluation module for the Orchestrator SDK.

Provides LLM-as-judge evaluation, optional deepeval and RAGAS integration,
and Langfuse Dataset management for regression testing.

Always available (no extra installs):
    EvalCase, CriterionScore, EvalResult, EvalStatus
    EvaluatorAgent, create_evaluator_agent
    LangfuseDatasetClient

Optional — requires extra install:
    DeepEvalEvaluator  →  pip install deepeval
    RagasEvaluator     →  pip install ragas datasets

Usage::

    from continuum.evaluation import (
        EvaluatorAgent,
        EvalCase,
        LangfuseDatasetClient,
    )

    evaluator = EvaluatorAgent(
        name="judge",
        criteria=["correctness", "conciseness"],
    )
    case = EvalCase(input_text="What is AI?", expected_output="...")
    result = await evaluator.evaluate(case, agent_response_text="AI is ...")
"""

from continuum.evaluation.evaluator_agent import EvaluatorAgent, create_evaluator_agent
from continuum.evaluation.langfuse_datasets import LangfuseDatasetClient
from continuum.evaluation.types import (
    CriterionScore,
    EvalCase,
    EvalResult,
    EvalStatus,
    contexts_from_messages,
)

__all__ = [
    # Types
    "EvalCase",
    "CriterionScore",
    "EvalResult",
    "EvalStatus",
    "contexts_from_messages",
    # Native LLM-as-judge evaluator
    "EvaluatorAgent",
    "create_evaluator_agent",
    # Langfuse Datasets
    "LangfuseDatasetClient",
]

# Optional: deepeval (pip install deepeval)
try:
    from continuum.evaluation.deepeval_eval import DeepEvalEvaluator

    __all__ += ["DeepEvalEvaluator"]
except ImportError:
    pass

# Optional: RAGAS (pip install ragas datasets)
try:
    from continuum.evaluation.ragas_eval import RagasEvaluator

    __all__ += ["RagasEvaluator"]
except ImportError:
    pass
