"""
Supervised Sequential Agent — Sequential pipeline with LLM quality gating.

After each step the supervisor LLM scores the output (0.0–1.0).
If the score is below `quality_threshold`, the step is retried up to
`max_retries` times before moving on (or failing, depending on fail_strategy).

This closes the gap between Continuum's static SequentialAgent and
CrewAI/AutoGen supervisor patterns where every output is validated before
being passed to the next stage.

Usage::

    from orchestrator.agent.workflow import SupervisedSequentialAgent, create_supervised_agent

    pipeline = create_supervised_agent(
        name="supervised-pipeline",
        agents=[researcher, analyst, writer],
        quality_threshold=0.7,   # retry if score < 0.7
        max_retries=2,
    )

    result = await runner.run(pipeline, "Analyse the EV market")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.exceptions import SequentialWorkflowError
from orchestrator.agent.types import (
    AgentResponse,
    FailStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)
from orchestrator.config import settings
from orchestrator.logging import get_logger
from orchestrator.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner

logger = get_logger(__name__)


# =============================================================================
# Config
# =============================================================================


@dataclass
class SupervisedConfig:
    """Configuration for SupervisedSequentialAgent."""

    quality_threshold: float = 0.7      # Retry if supervisor score < this
    max_retries: int = 2                # Max retries per step before giving up
    supervisor_model: str | None = None # Model for quality scoring (default: agent model)
    pass_full_history: bool = False     # Pass full history vs just last output
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST
    pipeline_context_max_chars: int | None = 300  # None = no truncation

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_threshold": self.quality_threshold,
            "max_retries": self.max_retries,
            "supervisor_model": self.supervisor_model,
            "pass_full_history": self.pass_full_history,
            "fail_strategy": self.fail_strategy.value,
            "pipeline_context_max_chars": self.pipeline_context_max_chars,
        }


# =============================================================================
# Agent
# =============================================================================


@dataclass
class SupervisedSequentialAgent(BaseAgent):
    """
    Sequential pipeline where a supervisor LLM evaluates each step's output.

    After each agent runs, the supervisor scores its output (0.0–1.0).
    If the score falls below `quality_threshold`, the step is retried with
    feedback about what was missing.  If all retries are exhausted, the
    behaviour depends on `fail_strategy`:

    - FAIL_FAST         → raise SequentialWorkflowError
    - CONTINUE_ON_ERROR → pass the best output so far and continue

    Example::

        pipeline = SupervisedSequentialAgent(
            name="validated-pipeline",
            agents=[researcher, analyst, writer],
            supervised_config=SupervisedConfig(quality_threshold=0.7),
        )
    """

    agents: list[BaseAgent] = field(default_factory=list)
    supervised_config: SupervisedConfig = field(default_factory=SupervisedConfig)

    def __post_init__(self) -> None:
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError
            raise AgentConfigurationError("Agent name is required")
        if not self.agents:
            from orchestrator.agent.exceptions import AgentConfigurationError
            raise AgentConfigurationError("SupervisedSequentialAgent requires at least one agent")

    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """
        Execute the supervised sequential pipeline.

        Args:
            input_text: Initial input
            runner: Agent runner
            context: Run context

        Returns:
            Final AgentResponse from the last step
        """
        context.suppress_session_log = True
        try:
            llm_client = self._get_llm()
            current_input = input_text
            all_responses: list[AgentResponse] = []
            total_usage = TokenUsage()
            agents_used: list[str] = []
            pipeline_history: list[str] = []

            async with SpanScope(
                f"workflow.supervised.{self.name}",
                input={
                    "input_preview": input_text[:500],
                    "agent_count": len(self.agents),
                    "agents": [a.name for a in self.agents],
                    "quality_threshold": self.supervised_config.quality_threshold,
                },
                metadata={"workflow_type": "supervised_sequential"},
            ) as workflow_span:
                for i, agent in enumerate(self.agents):
                    step_num = i + 1
                    best_response: AgentResponse | None = None
                    best_score: float = 0.0

                    async with SpanScope(
                        f"workflow.supervised.step.{step_num}",
                        input={"step": step_num, "agent_name": agent.name},
                        metadata={"total_steps": len(self.agents)},
                    ) as step_span:
                        # Inject prior steps as system context for the LLM.
                        # Skip when pass_full_history=True — user message already contains all prior steps.
                        if pipeline_history and not self.supervised_config.pass_full_history:
                            context.metadata["pipeline_context"] = (
                                "Prior pipeline steps in this request:\n"
                                + "\n".join(pipeline_history)
                            )

                        attempt_input = current_input

                        for attempt in range(self.supervised_config.max_retries + 1):
                            logger.info(
                                f"SupervisedSequential step {step_num}/{len(self.agents)} "
                                f"'{agent.name}' — attempt {attempt + 1}"
                            )

                            try:
                                response = await runner.run(
                                    agent=agent,
                                    input=attempt_input,
                                    context=context,
                                )
                                total_usage = total_usage.add(response.usage)

                                # Score the output
                                score, feedback, score_usage = await self._score_output(
                                    step_num=step_num,
                                    agent_name=agent.name,
                                    original_input=current_input,
                                    output=response.content or "",
                                    llm_client=llm_client,
                                )
                                total_usage = total_usage.add(score_usage)

                                logger.info(
                                    f"SupervisedSequential step {step_num} '{agent.name}' "
                                    f"score={score:.2f} (threshold={self.supervised_config.quality_threshold})"
                                )

                                if score > best_score:
                                    best_score = score
                                    best_response = response

                                if score >= self.supervised_config.quality_threshold:
                                    logger.info(
                                        f"SupervisedSequential step {step_num} passed "
                                        f"(score={score:.2f})"
                                    )
                                    step_span.set_output({
                                        "success": True,
                                        "score": score,
                                        "attempts": attempt + 1,
                                    })
                                    break

                                # Score too low — prepare retry with feedback
                                if attempt < self.supervised_config.max_retries:
                                    logger.info(
                                        f"SupervisedSequential step {step_num} below threshold "
                                        f"(score={score:.2f}) — retrying with feedback"
                                    )
                                    attempt_input = (
                                        f"{current_input}\n\n"
                                        f"Previous attempt output:\n{response.content}\n\n"
                                        f"Quality feedback: {feedback}\n\n"
                                        f"Please improve your response based on this feedback."
                                    )
                                else:
                                    logger.warning(
                                        f"SupervisedSequential step {step_num} exhausted retries "
                                        f"(best score={best_score:.2f})"
                                    )
                                    step_span.set_output({
                                        "success": False,
                                        "best_score": best_score,
                                        "attempts": attempt + 1,
                                    })

                            except Exception as e:
                                logger.error(f"SupervisedSequential step {step_num} failed: {e}")
                                step_span.set_error(str(e))

                                if self.supervised_config.fail_strategy == FailStrategy.FAIL_FAST:
                                    workflow_span.set_error(f"Step {step_num} failed: {e}")
                                    raise SequentialWorkflowError(
                                        f"Step {step_num} ({agent.name}) failed: {e}",
                                        failed_agent=agent.name,
                                        step=step_num,
                                        run_id=context.run_id,
                                        original_error=e,
                                    ) from e

                                current_input = f"Previous step failed: {e}. Please handle this gracefully."
                                break

                        # Handle case where all retries were exhausted below threshold
                        if best_response is None:
                            if self.supervised_config.fail_strategy == FailStrategy.FAIL_FAST:
                                raise SequentialWorkflowError(
                                    f"Step {step_num} ({agent.name}) produced no valid output",
                                    failed_agent=agent.name,
                                    step=step_num,
                                    run_id=context.run_id,
                                )
                            current_input = f"Previous step produced no output. Please handle gracefully."
                            continue

                        all_responses.append(best_response)
                        agents_used.append(agent.name)
                        max_chars = self.supervised_config.pipeline_context_max_chars
                        content = best_response.content or ""
                        pipeline_history.append(
                            f"{agent.name}: {content[:max_chars] if max_chars is not None else content}"
                        )

                        # Prepare input for next step
                        if self.supervised_config.pass_full_history:
                            history_parts = [f"Original request: {input_text}"]
                            for j, resp in enumerate(all_responses):
                                history_parts.append(
                                    f"Step {j + 1} ({self.agents[j].name}): {resp.content}"
                                )
                            current_input = "\n\n".join(history_parts)
                        else:
                            current_input = best_response.content or ""

                final = (
                    all_responses[-1]
                    if all_responses
                    else AgentResponse(
                        content="No agents executed",
                        status=ResponseStatus.ERROR,
                    )
                )

                workflow_span.set_output({
                    "success": True,
                    "steps_executed": len(all_responses),
                    "agents_used": agents_used,
                    "total_tokens": total_usage.total_tokens,
                })

                result = AgentResponse(
                    content=final.content,
                    structured_output=final.structured_output,
                    agent_name=self.name,
                    status=ResponseStatus.SUCCESS,
                    usage=total_usage,
                    turn_count=sum(r.turn_count for r in all_responses),
                    agents_used=agents_used,
                    messages=final.messages,
                )

            if context.session_id and all_responses:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=final.content or "",
                    agent=None,
                )

            return result
        finally:
            context.metadata.pop("pipeline_context", None)

    async def _score_output(
        self,
        step_num: int,
        agent_name: str,
        original_input: str,
        output: str,
        llm_client: Any | None,
    ) -> tuple[float, str, TokenUsage]:
        """
        Ask the supervisor LLM to score the output (0.0–1.0).

        Returns:
            (score, feedback, token_usage)
        """
        if not llm_client or not output.strip():
            return 0.5, "No supervisor available — defaulting to pass", TokenUsage()

        from orchestrator.llm.config import LLMConfig

        model = self.supervised_config.supervisor_model or settings.default_llm_model
        prompt = (
            f"You are a quality supervisor evaluating step {step_num} output from agent '{agent_name}'.\n\n"
            f"Original task:\n{original_input[:400]}\n\n"
            f"Agent output:\n{output[:800]}\n\n"
            "Score the output on a scale of 0.0 to 1.0:\n"
            "  0.0–0.4 = poor (missing key content, off-topic, or incomplete)\n"
            "  0.5–0.6 = acceptable (covers basics but lacks depth)\n"
            "  0.7–0.9 = good (thorough and relevant)\n"
            "  1.0     = excellent (comprehensive, precise, well-structured)\n\n"
            "Reply in this exact format (two lines):\n"
            "SCORE: <number>\n"
            "FEEDBACK: <one sentence explaining what is missing or why it passed>"
        )

        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.1, max_tokens=200),
                auto_session=False,
            )

            usage = TokenUsage()
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                )

            content = (response.content or "").strip()
            score = 0.5
            feedback = "No feedback provided"

            for line in content.splitlines():
                if line.startswith("SCORE:"):
                    try:
                        score = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                    except ValueError:
                        pass
                elif line.startswith("FEEDBACK:"):
                    feedback = line.split(":", 1)[1].strip()

            return score, feedback, usage

        except Exception as e:
            logger.debug(f"Supervisor scoring failed: {e} — defaulting to 0.5")
            return 0.5, f"Scoring error: {e}", TokenUsage()

    def _get_llm(self) -> Any | None:
        try:
            from orchestrator.core.container import get_container
            return get_container().llm_client
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "agents": [a.name for a in self.agents],
            "supervised_config": self.supervised_config.to_dict(),
            "workflow_type": "supervised_sequential",
        })
        return base


# =============================================================================
# Factory
# =============================================================================


def create_supervised_agent(
    name: str,
    agents: list[BaseAgent],
    *,
    quality_threshold: float = 0.7,
    max_retries: int = 2,
    supervisor_model: str | None = None,
    pass_full_history: bool = False,
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST,
) -> SupervisedSequentialAgent:
    """
    Factory for SupervisedSequentialAgent.

    Args:
        name: Pipeline name
        agents: Agents to execute in order
        quality_threshold: Minimum acceptable score (0.0–1.0)
        max_retries: Max retries per step when below threshold
        supervisor_model: LLM model for quality scoring
        pass_full_history: Pass full accumulated history to each step
        fail_strategy: Behaviour when all retries are exhausted

    Returns:
        Configured SupervisedSequentialAgent

    Example::

        pipeline = create_supervised_agent(
            name="research-pipeline",
            agents=[researcher, analyst, writer],
            quality_threshold=0.75,
            max_retries=2,
        )
    """
    return SupervisedSequentialAgent(
        name=name,
        agents=agents,
        supervised_config=SupervisedConfig(
            quality_threshold=quality_threshold,
            max_retries=max_retries,
            supervisor_model=supervisor_model,
            pass_full_history=pass_full_history,
            fail_strategy=fail_strategy,
        ),
    )
