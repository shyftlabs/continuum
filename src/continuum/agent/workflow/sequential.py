"""
Sequential Agent - Pipeline execution agent.

Executes a sequence of agents in order, passing outputs
from one to the next.

NOTE: Workflow agents now include Langfuse span tracing for full observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from continuum.agent.base import BaseAgent
from continuum.agent.config import SequentialConfig
from continuum.agent.exceptions import SequentialWorkflowError
from continuum.agent.types import (
    AgentResponse,
    FailStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)
from continuum.agent.utils.context_utils import publish_active_policy
from continuum.logging import get_logger
from continuum.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from continuum.agent.runner import AgentRunner

logger = get_logger(__name__)


@dataclass
class SequentialAgent(BaseAgent):
    """
    Executes agents sequentially like a pipeline.

    Each agent in the sequence receives either:
    - The previous agent's output (default)
    - The full conversation history (if pass_full_history=True)

    Example:
        ```python
        from continuum.agent import BaseAgent
        from continuum.agent.workflow import SequentialAgent

        # Define pipeline stages
        researcher = BaseAgent(
            name="researcher",
            instructions="Research the topic thoroughly.",
        )

        analyst = BaseAgent(
            name="analyst",
            instructions="Analyze the research findings.",
        )

        writer = BaseAgent(
            name="writer",
            instructions="Write a clear summary.",
        )

        # Create pipeline
        pipeline = SequentialAgent(
            name="research-pipeline",
            agents=[researcher, analyst, writer],
        )

        # Run pipeline
        result = await runner.run(pipeline, "AI in healthcare")
        ```
    """

    # Sequence of agents to execute
    agents: list[BaseAgent] = field(default_factory=list)

    # Sequential configuration
    sequential_config: SequentialConfig = field(default_factory=SequentialConfig)

    # Agent whose memory_config governs post-sequence long-term memory writes.
    # If None (default), no memory is written after the sequence completes.
    memory_agent: BaseAgent | None = None

    def __post_init__(self) -> None:
        """Initialize sequential agent."""
        # Skip base validation as we're a composite agent
        if not self.name:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if not self.agents:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("SequentialAgent requires at least one agent")

    @publish_active_policy
    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """
        Execute the sequential pipeline.

        Args:
            input_text: Initial input
            runner: Agent runner for executing sub-agents
            context: Run context

        Returns:
            Final AgentResponse from the pipeline
        """
        # Own the one decision trace spanning all pipeline sub-agents (rooted at
        # this workflow). Sub-runs share this recorder but are suppressed, so we
        # persist it ourselves at the end.
        _trace_created = runner.ensure_recorder(context, self.name, input_text)
        context.suppress_session_log = True
        try:
            current_input = input_text
            all_responses: list[AgentResponse] = []
            total_usage = TokenUsage()
            agents_used = []
            pipeline_history: list[str] = []

            # Create a span for the entire sequential workflow
            async with SpanScope(
                f"workflow.sequential.{self.name}",
                input={
                    "input_preview": input_text[:500] if input_text else None,
                    "agent_count": len(self.agents),
                    "agents": [a.name for a in self.agents],
                },
                metadata={
                    "workflow_type": "sequential",
                    "pass_full_history": self.sequential_config.pass_full_history,
                    "fail_strategy": self.sequential_config.fail_strategy.value,
                },
            ) as workflow_span:
                for i, agent in enumerate(self.agents):
                    step_num = i + 1

                    logger.info(
                        f"Sequential step {step_num}/{len(self.agents)}: {agent.name}",
                        extra={"run_id": context.run_id, "step": step_num},
                    )

                    # Decision trace: mark the start of this pipeline stage.
                    if context.recorder is not None:
                        context.recorder.record_workflow_step(
                            self.name, stage=i, label=agent.name, agent_stack=[self.name]
                        )

                    # Create a span for each step in the pipeline
                    async with SpanScope(
                        f"workflow.sequential.step.{step_num}",
                        input={
                            "step": step_num,
                            "agent_name": agent.name,
                            "input_preview": current_input[:300] if current_input else None,
                        },
                        metadata={"total_steps": len(self.agents)},
                    ) as step_span:
                        try:
                            # Inject prior steps as system context for the LLM.
                            # Skip when pass_full_history=True — [N+1] user message already
                            # contains all prior steps in full, so this would be redundant.
                            # Exclude the immediately preceding step (pipeline_history[-1]) since
                            # it is already passed as the [user] input — injecting it here too
                            # would be pure duplication. Context only carries steps 1..N-2.
                            background = pipeline_history[:-1]
                            if (
                                background
                                and not self.sequential_config.pass_full_history
                                and context.metadata is not None
                            ):
                                context.metadata["pipeline_context"] = (
                                    "Prior pipeline steps in this request:\n"
                                    + "\n".join(background)
                                )

                            # Execute agent
                            response = await runner.run(
                                agent=agent,
                                input=current_input,
                                context=context,
                            )

                            all_responses.append(response)
                            agents_used.append(agent.name)
                            total_usage = total_usage.add(response.usage)
                            max_chars = self.sequential_config.pipeline_context_max_chars
                            content = response.content or ""
                            pipeline_history.append(
                                f"{agent.name}: {content[:max_chars] if max_chars is not None else content}"
                            )

                            # Update step span with result
                            step_span.set_output(
                                {
                                    "success": True,
                                    "response_preview": (response.content or "")[:200],
                                    "turn_count": response.turn_count,
                                }
                            )

                            # Prepare input for next agent
                            if self.sequential_config.pass_full_history:
                                # Build conversation so far
                                history_parts = [f"Original request: {input_text}"]
                                for j, resp in enumerate(all_responses):
                                    history_parts.append(
                                        f"Step {j + 1} ({self.agents[j].name}): {resp.content}"
                                    )
                                current_input = "\n\n".join(history_parts)
                            else:
                                # Just pass the output
                                current_input = response.content or ""

                        except Exception as e:
                            logger.error(f"Sequential step {step_num} failed: {e}")
                            step_span.set_error(str(e))
                            step_span.set_output({"success": False, "error": str(e)})

                            if self.sequential_config.fail_strategy == FailStrategy.FAIL_FAST:
                                workflow_span.set_error(f"Step {step_num} failed: {e}")
                                raise SequentialWorkflowError(
                                    f"Step {step_num} ({agent.name}) failed: {e}",
                                    failed_agent=agent.name,
                                    step=step_num,
                                    run_id=context.run_id,
                                    original_error=e,
                                ) from e

                            # Continue with error message
                            current_input = (
                                f"Previous step failed: {e}. Please handle this gracefully."
                            )

                # Return final response
                final_response = (
                    all_responses[-1]
                    if all_responses
                    else AgentResponse(
                        content="No agents executed",
                        status=ResponseStatus.ERROR,
                    )
                )

                # Update workflow span with final result
                workflow_span.set_output(
                    {
                        "success": True,
                        "total_steps_executed": len(all_responses),
                        "agents_used": agents_used,
                        "total_tokens": total_usage.total_tokens if total_usage else 0,
                    }
                )

                result = AgentResponse(
                    content=final_response.content,
                    structured_output=final_response.structured_output,
                    agent_name=self.name,
                    status=ResponseStatus.SUCCESS,
                    usage=total_usage,
                    turn_count=sum(r.turn_count for r in all_responses),
                    agents_used=agents_used,
                    messages=final_response.messages,
                )

            if context.session_id and all_responses:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=final_response.content or "",
                    agent=self.memory_agent,
                )

            if _trace_created:
                await runner.persist_decision_trace(context, result)

            return result
        finally:
            if context.metadata is not None:
                context.metadata.pop("pipeline_context", None)

    # ------------------------------------------------------------------ #
    # Forkable: resume the pipeline from the stage owning a step
    # ------------------------------------------------------------------ #
    def _segment_stages(self, trace: Any) -> tuple[dict[str, int], dict[int, Any]]:
        """Map each step_id → pipeline stage index and stage index → its first
        (snapshot-bearing) step. Delegates to the shared marker-based segmenter
        (``_forkable.segment_by_markers``) used by every Forkable orchestrator."""
        from continuum.agent.workflow._forkable import segment_by_markers

        return segment_by_markers(trace)

    async def resume_from(
        self,
        *,
        parent_trace: Any,
        from_step: str,
        override: dict[str, Any] | None,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Re-run the pipeline from the stage containing ``from_step``.

        Stages before the resume point are not re-run; the resumed stage's input
        is recovered from its first step's message checkpoint (or replaced via
        ``override={"replace_last_user": ...}``). The parent run is never mutated;
        the new run records its lineage.
        """
        from continuum.agent.workflow._forkable import link_lineage, resumed_input

        step_stage, stage_first = self._segment_stages(parent_trace)
        stage_idx = step_stage.get(from_step)
        if stage_idx is None:
            raise ValueError(f"resume_from: step '{from_step}' not found in trace")
        stage_idx = min(stage_idx, len(self.agents) - 1)

        # Recover the resumed stage's input (last user message in its first step's
        # checkpoint, with the override applied; falls back to the run query).
        stage_input = resumed_input(stage_first.get(stage_idx), override, parent_trace.user_query)

        # New trace rooted at this workflow, linked back to the parent.
        created = runner.ensure_recorder(context, self.name, parent_trace.user_query)
        if created:
            link_lineage(context, parent_trace, from_step, override, stage_idx)

        # Run the remaining stages as a sub-pipeline sharing this recorder.
        tail = SequentialAgent(
            name=self.name,
            agents=self.agents[stage_idx:],
            sequential_config=self.sequential_config,
        )
        result = await tail.execute(stage_input, runner, context)
        # Surface the forked run's id (the workflow AgentResponse doesn't set it)
        # so callers can load the persisted trace by run_id.
        result.run_id = context.run_id
        if created:
            await runner.persist_decision_trace(context, result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "agents": [a.name for a in self.agents],
                "sequential_config": self.sequential_config.to_dict(),
                "workflow_type": "sequential",
            }
        )
        return base


def create_sequential_agent(
    name: str,
    agents: list[BaseAgent],
    *,
    pass_full_history: bool = False,
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST,
    memory_agent: BaseAgent | None = None,
) -> SequentialAgent:
    """
    Factory function to create a sequential agent.

    Args:
        name: Pipeline name
        agents: List of agents to execute in order
        pass_full_history: Whether to pass full history to each agent
        fail_strategy: How to handle failures
        memory_agent: Agent whose memory_config governs post-sequence long-term memory writes

    Returns:
        Configured SequentialAgent
    """
    return SequentialAgent(
        name=name,
        agents=agents,
        memory_agent=memory_agent,
        sequential_config=SequentialConfig(
            pass_full_history=pass_full_history,
            fail_strategy=fail_strategy,
        ),
    )
