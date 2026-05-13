"""
Sequential Agent - Pipeline execution agent.

Executes a sequence of agents in order, passing outputs
from one to the next.

NOTE: Workflow agents now include Langfuse span tracing for full observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import SequentialConfig
from orchestrator.agent.exceptions import SequentialWorkflowError
from orchestrator.agent.types import (
    AgentResponse,
    FailStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)
from orchestrator.logging import get_logger
from orchestrator.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner

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
        from orchestrator.agent import BaseAgent
        from orchestrator.agent.workflow import SequentialAgent

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

    def __post_init__(self) -> None:
        """Initialize sequential agent."""
        # Skip base validation as we're a composite agent
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if not self.agents:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("SequentialAgent requires at least one agent")

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
                            if background and not self.sequential_config.pass_full_history and context.metadata is not None:
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
                            current_input = f"Previous step failed: {e}. Please handle this gracefully."

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
                    agent=None,
                )

            return result
        finally:
            if context.metadata is not None:
                context.metadata.pop("pipeline_context", None)

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
) -> SequentialAgent:
    """
    Factory function to create a sequential agent.

    Args:
        name: Pipeline name
        agents: List of agents to execute in order
        pass_full_history: Whether to pass full history to each agent
        fail_strategy: How to handle failures

    Returns:
        Configured SequentialAgent
    """
    return SequentialAgent(
        name=name,
        agents=agents,
        sequential_config=SequentialConfig(
            pass_full_history=pass_full_history,
            fail_strategy=fail_strategy,
        ),
    )
