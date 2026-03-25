"""
Shared types for the evaluation module.

EvalCase     — one test case (input + optional expected output + optional retrieved context)
CriterionScore — score for one criterion (0–1 float + reasoning)
EvalResult   — aggregate result for one EvalCase (list of CriterionScores + overall)
EvalStatus   — outcome enum
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EvalStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class EvalCase:
    """
    A single evaluation test case.

    input_text:      The user question / prompt sent to the agent.
    expected_output: Optional gold-standard reference answer.
    context:         Retrieved document chunks — required for RAGAS metrics,
                     optional (empty list) for generation-only evaluation.
    metadata:        Arbitrary key-value pairs (dataset item IDs, tags, etc.).
    case_id:         Stable identifier — auto-generated UUID fragment if not provided.
    """

    input_text: str
    expected_output: str | None = None
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    case_id: str = field(default_factory=lambda: f"case_{uuid.uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        # Coerce context to list[str] if a single string is passed
        if isinstance(self.context, str):
            object.__setattr__(self, "context", [self.context])
        elif not isinstance(self.context, list):
            raise TypeError(
                f"EvalCase.context must be a list of strings, got {type(self.context).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "context": self.context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        return cls(
            input_text=data.get("input_text", ""),
            expected_output=data.get("expected_output"),
            context=data.get("context", []),
            metadata=data.get("metadata", {}),
            case_id=data.get("case_id", f"case_{uuid.uuid4().hex[:12]}"),
        )


@dataclass
class CriterionScore:
    """
    Score for a single evaluation criterion.

    criterion: Human-readable label, e.g. "correctness", "faithfulness".
    score:     Numeric value in [0.0, 1.0].
    passed:    True when score >= the configured pass_threshold.
    reasoning: Verbatim LLM explanation — key for debugging low scores.
    metadata:  Provider-specific extras (metric threshold, etc.).
    """

    criterion: str
    score: float
    passed: bool
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """
    Aggregate result for one EvalCase evaluated by one evaluator.

    scores:            Per-criterion CriterionScore list.
    overall_score:     Mean of all criterion scores (None if scores is empty).
    overall_passed:    True only when all criteria pass.
    status:            EvalStatus enum.
    evaluator_name:    Name of the agent / evaluator that produced this result.
    case_id:           Back-reference to EvalCase.case_id.
    agent_response:    The agent output that was judged.
    latency_ms:        Wall-clock time to produce the evaluation (milliseconds).
    langfuse_trace_id: Set by LangfuseDatasetClient when linked to a run.
    created_at:        UTC timestamp.
    metadata:          Arbitrary extras (raw token usage, error messages, etc.).
    """

    scores: list[CriterionScore] = field(default_factory=list)
    overall_score: float | None = None
    overall_passed: bool = False
    status: EvalStatus = EvalStatus.SKIPPED
    evaluator_name: str = ""
    case_id: str = ""
    agent_response: str = ""
    latency_ms: int = 0
    langfuse_trace_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_overall(self, pass_threshold: float = 0.7) -> None:
        """
        Recompute overall_score and overall_passed from self.scores in place.

        Called after the scores list is fully populated. Sets status to
        PASSED or FAILED based on whether every criterion passes.
        """
        if not self.scores:
            self.overall_score = None
            self.overall_passed = False
            self.status = EvalStatus.SKIPPED
            return

        self.overall_score = sum(s.score for s in self.scores) / len(self.scores)

        # Mark each criterion passed/failed against the threshold
        for s in self.scores:
            s.passed = s.score >= pass_threshold

        self.overall_passed = all(s.passed for s in self.scores)
        self.status = EvalStatus.PASSED if self.overall_passed else EvalStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "evaluator_name": self.evaluator_name,
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "status": self.status.value,
            "scores": [s.to_dict() for s in self.scores],
            "agent_response": self.agent_response,
            "latency_ms": self.latency_ms,
            "langfuse_trace_id": self.langfuse_trace_id,
            "created_at": self.created_at.isoformat(),
            "metadata": {k: v for k, v in self.metadata.items() if not k.startswith("_")},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        scores = [
            CriterionScore(
                criterion=s["criterion"],
                score=s["score"],
                passed=s["passed"],
                reasoning=s.get("reasoning", ""),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("scores", [])
        ]
        return cls(
            scores=scores,
            overall_score=data.get("overall_score"),
            overall_passed=data.get("overall_passed", False),
            status=EvalStatus(data.get("status", EvalStatus.SKIPPED)),
            evaluator_name=data.get("evaluator_name", ""),
            case_id=data.get("case_id", ""),
            agent_response=data.get("agent_response", ""),
            latency_ms=data.get("latency_ms", 0),
            langfuse_trace_id=data.get("langfuse_trace_id"),
            metadata=data.get("metadata", {}),
        )
