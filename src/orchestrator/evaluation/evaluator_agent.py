"""
EvaluatorAgent — native LLM-as-judge evaluation.

A workflow agent that scores any agent output against a set of criteria
using a single LLM call per criterion. No external dependencies.

Usage (direct library API)::

    from orchestrator.evaluation import EvaluatorAgent, EvalCase

    evaluator = EvaluatorAgent(
        name="quality-judge",
        criteria=["correctness", "conciseness", "safety"],
        pass_threshold=0.7,
    )

    case = EvalCase(
        input_text="What is the capital of France?",
        expected_output="Paris",
    )
    result = await evaluator.evaluate(case, agent_response_text="Paris.")
    print(result.overall_score)   # e.g. 0.93
    print(result.overall_passed)  # True

Usage (as a workflow step via runner.run)::

    import json
    payload = json.dumps({
        "case": case.to_dict(),
        "agent_response": "Paris.",
    })
    response = await runner.run(evaluator, payload, context=ctx)
    result = EvalResult.from_dict(json.loads(response.content))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
from orchestrator.agent.types import AgentResponse, ResponseStatus, TokenUsage
from orchestrator.config import settings
from orchestrator.evaluation.types import CriterionScore, EvalCase, EvalResult, EvalStatus
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner
    from orchestrator.agent.types import RunContext

logger = get_logger(__name__)

_NO_MEMORY = AgentMemoryConfig(search_memories=False, store_memories=False)
_NO_SESSION = AgentConfig(log_to_session=False)

_DEFAULT_JUDGE_INSTRUCTIONS = (
    "You are an objective, impartial evaluator. "
    "For the given criterion you will receive the user input, the agent response, "
    "and optionally the expected output and retrieved context.\n\n"
    "Respond ONLY with valid JSON in exactly this format:\n"
    '{"score": <float 0.0–1.0>, "passed": <true|false>, "reasoning": "<one sentence>"}\n\n'
    "Scoring guide:\n"
    "  1.0 — fully correct, complete, and satisfies the criterion\n"
    "  0.7 – 0.9 — mostly correct, minor issues\n"
    "  0.4 – 0.6 — partially correct, notable gaps\n"
    "  0.0 – 0.3 — incorrect or fails the criterion\n"
    "Do not include any text outside the JSON object."
)


@dataclass
class EvaluatorAgent(BaseAgent):
    """
    LLM-as-judge evaluator agent.

    Runs one LLM call per criterion and aggregates scores into an EvalResult.
    Follows the same BaseAgent extension pattern as ReflectionAgent and DebateAgent.

    Fields:
        criteria:       List of criterion labels to evaluate (e.g. ["correctness"]).
        pass_threshold: Score in [0, 1] above which a criterion is "passed".
        judge_model:    LLM model for judging. Defaults to settings.default_llm_model.
        rubrics:        Optional per-criterion prompt fragments that override the
                        default "evaluate whether the response satisfies '<criterion>'"
                        description.
    """

    criteria: list[str] = field(default_factory=lambda: ["correctness", "helpfulness"])
    pass_threshold: float = 0.7
    judge_model: str | None = None
    rubrics: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        # Ensure no memory / session overhead on the judge
        object.__setattr__(self, "memory_config", _NO_MEMORY)
        object.__setattr__(self, "config", _NO_SESSION)

        if not self.instructions:
            object.__setattr__(self, "instructions", _DEFAULT_JUDGE_INSTRUCTIONS)

    # ------------------------------------------------------------------
    # Workflow entry point (called by runner.run when agent has execute())
    # ------------------------------------------------------------------

    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """
        Called by AgentRunner when EvaluatorAgent is used inside a workflow.

        Expects input_text to be a JSON string:
            {"case": <EvalCase.to_dict()>, "agent_response": "<text>"}

        Returns an AgentResponse whose content is the JSON-serialised EvalResult.
        """
        try:
            payload = json.loads(input_text)
            case = EvalCase.from_dict(payload.get("case", {}))
            agent_response_text = payload.get("agent_response", "")
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat the whole input as the text to evaluate
            case = EvalCase(input_text=input_text)
            agent_response_text = input_text

        result = await self.evaluate(case, agent_response_text)
        usage = result.metadata.pop("_usage", TokenUsage())

        return AgentResponse(
            content=json.dumps(result.to_dict()),
            agent_name=self.name,
            status=(
                ResponseStatus.SUCCESS
                if result.status != EvalStatus.ERROR
                else ResponseStatus.ERROR
            ),
            usage=usage,
            run_artifacts={"eval_result": result.to_dict()},
        )

    # ------------------------------------------------------------------
    # Direct library API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        case: EvalCase,
        agent_response_text: str,
    ) -> EvalResult:
        """
        Score agent_response_text against all criteria for the given case.

        Args:
            case:                The EvalCase (input, expected output, context).
            agent_response_text: The produced agent text to evaluate.

        Returns:
            EvalResult with one CriterionScore per criterion.
        """
        from orchestrator.core.container import get_container

        t0 = time.monotonic()
        llm_client = get_container().llm_client
        total_usage = TokenUsage()
        scores: list[CriterionScore] = []

        for criterion in self.criteria:
            score, usage = await self._judge_criterion(
                criterion=criterion,
                case=case,
                agent_response_text=agent_response_text,
                llm_client=llm_client,
            )
            scores.append(score)
            total_usage = total_usage.add(usage)

        result = EvalResult(
            scores=scores,
            evaluator_name=self.name,
            case_id=case.case_id,
            agent_response=agent_response_text,
            latency_ms=int((time.monotonic() - t0) * 1000),
            status=EvalStatus.PASSED,
            metadata={"_usage": total_usage},
        )
        result.compute_overall(self.pass_threshold)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _judge_criterion(
        self,
        criterion: str,
        case: EvalCase,
        agent_response_text: str,
        llm_client: Any,
    ) -> tuple[CriterionScore, TokenUsage]:
        """Run one LLM call to score a single criterion. Returns (score, usage)."""
        from orchestrator.llm.config import LLMConfig

        model = self.judge_model or settings.default_llm_model
        rubric = self.rubrics.get(
            criterion,
            f"Evaluate whether the response satisfies the criterion: '{criterion}'.",
        )

        parts = [
            f"Criterion: {criterion}",
            f"Rubric: {rubric}",
            f"User input:\n{case.input_text}",
            f"Agent response:\n{agent_response_text}",
        ]
        if case.expected_output:
            parts.append(f"Expected output:\n{case.expected_output}")
        if case.context:
            parts.append("Retrieved context:\n" + "\n---\n".join(case.context))

        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

        try:
            llm_response = await llm_client.chat(
                messages=messages,
                config=LLMConfig(model=model, temperature=0.0, max_tokens=300),
                auto_session=False,
            )

            usage = TokenUsage()
            if llm_response.usage:
                usage = TokenUsage(
                    prompt_tokens=llm_response.usage.prompt_tokens or 0,
                    completion_tokens=llm_response.usage.completion_tokens or 0,
                    total_tokens=llm_response.usage.total_tokens or 0,
                )

            try:
                raw = _parse_json_response(llm_response.content or "")
            except _ParseError as parse_exc:
                # Distinguish parse failure from genuine zero score
                logger.warning(
                    f"EvaluatorAgent '{self.name}': criterion '{criterion}' "
                    f"JSON parse failed: {parse_exc}"
                )
                return (
                    CriterionScore(
                        criterion=criterion,
                        score=0.0,
                        passed=False,
                        reasoning=f"JSON parse error: {parse_exc}",
                        metadata={
                            "error": "json_parse_failure",
                            "raw_response": (llm_response.content or "")[:500],
                        },
                    ),
                    usage,
                )

            score_val = max(0.0, min(1.0, float(raw.get("score", 0.0))))
            passed = bool(raw.get("passed", score_val >= self.pass_threshold))

            return (
                CriterionScore(
                    criterion=criterion,
                    score=score_val,
                    passed=passed,
                    reasoning=str(raw.get("reasoning", "")),
                ),
                usage,
            )

        except Exception as exc:
            logger.warning(
                f"EvaluatorAgent '{self.name}': criterion '{criterion}' failed: {exc}",
                exc_info=True,
            )
            return (
                CriterionScore(
                    criterion=criterion,
                    score=0.0,
                    passed=False,
                    reasoning=f"Evaluation error: {exc}",
                    metadata={"error": str(exc)},
                ),
                TokenUsage(),
            )

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "criteria": self.criteria,
                "pass_threshold": self.pass_threshold,
                "judge_model": self.judge_model,
                "rubrics": self.rubrics,
                "workflow_type": "evaluator",
            }
        )
        return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """Raised when LLM response cannot be parsed as valid evaluation JSON."""

    pass


def _parse_json_response(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from an LLM response.

    Tries strict JSON first; falls back to extracting the first {...} block
    from free-text responses (for models that ignore json_mode).

    Raises _ParseError if no valid JSON can be extracted, so callers can
    distinguish "parse failure" from "genuine score of 0".
    """
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise _ParseError(
        f"Could not extract valid JSON from LLM response (length={len(text)}). "
        f"Preview: {text[:200]}"
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_evaluator_agent(
    name: str,
    criteria: list[str],
    *,
    pass_threshold: float = 0.7,
    judge_model: str | None = None,
    rubrics: dict[str, str] | None = None,
) -> EvaluatorAgent:
    """
    Factory for EvaluatorAgent.

    Args:
        name:           Agent name.
        criteria:       List of criterion labels (e.g. ["correctness", "safety"]).
        pass_threshold: Score threshold for a criterion to be "passed" (default 0.7).
        judge_model:    LLM model for judging (defaults to settings.default_llm_model).
        rubrics:        Per-criterion rubric overrides.

    Returns:
        Configured EvaluatorAgent.

    Example::

        evaluator = create_evaluator_agent(
            name="support-judge",
            criteria=["correctness", "tone", "conciseness"],
            rubrics={
                "tone": "The response should be empathetic and professional.",
            },
        )
    """
    return EvaluatorAgent(
        name=name,
        criteria=criteria,
        pass_threshold=pass_threshold,
        judge_model=judge_model,
        rubrics=rubrics or {},
    )
