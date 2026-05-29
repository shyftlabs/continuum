"""
Loop Agent - Iterative execution agent.

Repeatedly executes an agent until a termination condition is met.

NOTE: Workflow agents now include Langfuse span tracing for full observability.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import LoopConfig
from orchestrator.agent.exceptions import LoopWorkflowError
from orchestrator.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    TerminationConfig,
    TerminationType,
    TokenUsage,
)
from orchestrator.config import settings
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner
    from orchestrator.llm import LLMClient

logger = get_logger(__name__)


@dataclass
class LoopAgent(BaseAgent):
    """
    Repeatedly executes an agent until termination.

    Supports multiple termination conditions:
    - LLM_DECISION: LLM decides when task is complete
    - TOOL_CALL: Terminates when specific tool is called
    - OUTPUT_MATCH: Terminates when output matches pattern
    - CUSTOM: User-provided termination function

    Example:
        ```python
        from orchestrator.agent import BaseAgent
        from orchestrator.agent.workflow import LoopAgent
        from orchestrator.agent.types import TerminationConfig, TerminationType

        # Define worker agent
        refiner = BaseAgent(
            name="refiner",
            instructions="Improve the text quality.",
        )

        # Create loop agent with LLM-based termination
        loop = LoopAgent(
            name="iterative-refiner",
            agent=refiner,
            termination=TerminationConfig(
                type=TerminationType.LLM_DECISION,
                max_iterations=5,
            ),
        )

        # Run iterative refinement
        result = await runner.run(loop, "Draft: The quick brown fox...")
        ```
    """

    # Agent to execute repeatedly
    agent: BaseAgent | None = None

    # Termination configuration
    termination: TerminationConfig = field(default_factory=TerminationConfig)

    # Loop configuration
    loop_config: LoopConfig = field(default_factory=LoopConfig)

    def __post_init__(self) -> None:
        """Initialize loop agent."""
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if self.agent is None:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("LoopAgent requires an agent to execute")

    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
        llm_client: LLMClient | None = None,
    ) -> AgentResponse:
        """
        Execute the agent in a loop until termination.

        Args:
            input_text: Initial input
            runner: Agent runner
            context: Run context
            llm_client: LLM client for termination checks

        Returns:
            Final AgentResponse
        """
        context.suppress_session_log = True
        _orig_hist = self.agent.config.session_history_turns
        try:
            current_input = input_text
            iteration = 0
            all_responses: list[AgentResponse] = []
            total_usage = TokenUsage()
            iteration_history: list[dict[str, Any]] = []

            while iteration < self.termination.max_iterations:
                iteration += 1

                logger.info(
                    f"Loop iteration {iteration}/{self.termination.max_iterations}",
                    extra={"run_id": context.run_id, "iteration": iteration},
                )

                try:
                    # Execute agent
                    response = await runner.run(
                        agent=self.agent,
                        input=current_input,
                        context=context,
                    )

                    all_responses.append(response)
                    total_usage = total_usage.add(response.usage)

                    # History only needed on iteration 1 — it doesn't change between
                    # iterations (writes are blocked), so skip it from iteration 2 onwards.
                    if iteration == 1:
                        self.agent.config.session_history_turns = 0

                    # Track iteration
                    iteration_history.append(
                        {
                            "iteration": iteration,
                            "input": current_input[:500],
                            "output": (response.content or "")[:500],
                        }
                    )

                    # Check termination
                    should_terminate = await self._check_termination(
                        response=response,
                        iteration=iteration,
                        history=iteration_history,
                        llm_client=llm_client,
                    )

                    if should_terminate:
                        logger.info(f"Loop terminated at iteration {iteration}")
                        break

                    # Prepare next iteration input
                    current_input = self._build_next_input(
                        original_input=input_text,
                        last_output=response.content,
                        iteration=iteration,
                    )

                except Exception as e:
                    logger.error(f"Loop iteration {iteration} failed: {e}")
                    raise LoopWorkflowError(
                        f"Iteration {iteration} failed: {e}",
                        iteration=iteration,
                        run_id=context.run_id,
                        original_error=e,
                    ) from e

            else:
                # Max iterations reached without termination
                if iteration >= self.termination.max_iterations:
                    logger.warning(f"Loop reached max iterations ({self.termination.max_iterations})")

            # Return final response
            final_response = (
                all_responses[-1]
                if all_responses
                else AgentResponse(
                    content="No iterations completed",
                    status=ResponseStatus.ERROR,
                )
            )

            result = AgentResponse(
                content=final_response.content,
                structured_output=final_response.structured_output,
                agent_name=self.name,
                status=ResponseStatus.SUCCESS if all_responses else ResponseStatus.ERROR,
                usage=total_usage,
                turn_count=iteration,
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
            self.agent.config.session_history_turns = _orig_hist

    async def _check_termination(
        self,
        response: AgentResponse,
        iteration: int,
        history: list[dict[str, Any]],
        llm_client: LLMClient | None,
    ) -> bool:
        """Check if loop should terminate."""
        term_type = self.termination.type

        if term_type == TerminationType.LLM_DECISION:
            return await self._llm_termination_check(
                response=response,
                history=history,
                llm_client=llm_client,
            )

        elif term_type == TerminationType.TOOL_CALL:
            # Check if specific tool was called
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = (
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "")
                    )
                    if tool_name == self.termination.tool_name:
                        return True
            return False

        elif term_type == TerminationType.OUTPUT_MATCH:
            # Check if output matches pattern
            if self.termination.pattern:
                return bool(re.search(self.termination.pattern, response.content or ""))
            return False

        elif term_type == TerminationType.CUSTOM:
            # Use custom condition
            if self.termination.condition and callable(self.termination.condition):
                return self.termination.condition(response.content, history)
            return False

        return False

    async def _llm_termination_check(
        self,
        response: AgentResponse,
        history: list[dict[str, Any]],
        llm_client: LLMClient | None,
    ) -> bool:
        """Use LLM to decide if task is complete."""
        if llm_client is None:
            from orchestrator.core.container import get_container

            llm_client = get_container().llm_client

        # Build history summary
        history_text = "\n".join(
            f"Iteration {h['iteration']}: {h['output'][:200]}..."
            for h in history[-3:]  # Last 3 iterations
        )

        prompt = f"""{self.termination.decision_prompt}

Recent iterations:
{history_text}

Current output:
{response.content[:500]}

Is the task complete? Respond with exactly 'COMPLETE' or 'CONTINUE':"""

        try:
            from orchestrator.llm.config import LLMConfig

            llm_response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(
                    model=self.agent.model if self.agent else settings.default_llm_model,
                    temperature=0.1,
                    max_tokens=20,
                ),
            )

            result = (llm_response.content or "").strip().upper()
            return "COMPLETE" in result

        except Exception as e:
            logger.warning(f"LLM termination check failed: {e}")
            return False

    def _build_next_input(
        self,
        original_input: str,
        last_output: str,
        iteration: int,
    ) -> str:
        """Build input for next iteration."""
        return f"""Original task: {original_input}

Previous output (iteration {iteration}):
{last_output}

Please continue improving or refining the output. If the task is complete, indicate completion."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "agent": self.agent.name if self.agent else None,
                "termination": {
                    "type": self.termination.type.value,
                    "max_iterations": self.termination.max_iterations,
                },
                "loop_config": self.loop_config.to_dict(),
                "workflow_type": "loop",
            }
        )
        return base


def create_loop_agent(
    name: str,
    agent: BaseAgent,
    *,
    termination_type: TerminationType = TerminationType.LLM_DECISION,
    max_iterations: int = 10,
    termination_prompt: str | None = None,
    termination_tool: str | None = None,
    termination_pattern: str | None = None,
    termination_condition: Callable[[str, list[dict[str, Any]]], bool] | None = None,
) -> LoopAgent:
    """
    Factory function to create a loop agent.

    Args:
        name: Loop agent name
        agent: Agent to execute repeatedly
        termination_type: How to determine when to stop
        max_iterations: Maximum number of iterations
        termination_prompt: Custom prompt for LLM decision
        termination_tool: Tool name for TOOL_CALL termination
        termination_pattern: Pattern for OUTPUT_MATCH termination
        termination_condition: Function for CUSTOM termination

    Returns:
        Configured LoopAgent
    """
    termination = TerminationConfig(
        type=termination_type,
        max_iterations=max_iterations,
    )

    if termination_prompt:
        termination.decision_prompt = termination_prompt
    if termination_tool:
        termination.tool_name = termination_tool
    if termination_pattern:
        termination.pattern = termination_pattern
    if termination_condition:
        termination.condition = termination_condition

    return LoopAgent(
        name=name,
        agent=agent,
        termination=termination,
    )
