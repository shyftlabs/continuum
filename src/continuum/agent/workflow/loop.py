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

from continuum.agent.base import BaseAgent
from continuum.agent.config import LoopConfig
from continuum.agent.exceptions import LoopWorkflowError
from continuum.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    TerminationConfig,
    TerminationType,
    TokenUsage,
)
from continuum.agent.utils.context_utils import publish_active_policy
from continuum.config import settings
from continuum.logging import get_logger

if TYPE_CHECKING:
    from continuum.agent.runner import AgentRunner
    from continuum.llm import LLMClient

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
        from continuum.agent import BaseAgent
        from continuum.agent.workflow import LoopAgent
        from continuum.agent.types import TerminationConfig, TerminationType

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

    # Agent whose memory_config governs post-execution long-term memory writes.
    # Defaults to the loop agent itself (the natural producer of the final output).
    memory_agent: BaseAgent | None = None

    def __post_init__(self) -> None:
        """Initialize loop agent."""
        if not self.name:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if self.agent is None:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("LoopAgent requires an agent to execute")

    @publish_active_policy
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
        # Own the one decision trace spanning all iterations (rooted at this loop).
        created = runner.ensure_recorder(context, self.name, input_text)
        context.suppress_session_log = True
        _orig_hist = self.agent.config.session_history_turns
        try:
            result = await self._drive(
                input_text,
                runner,
                context,
                start_iteration=0,
                original_input=input_text,
                llm_client=llm_client,
            )

            if context.session_id and result.turn_count and result.status == ResponseStatus.SUCCESS:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=result.content or "",
                    agent=self.memory_agent,
                )

            if created:
                await runner.persist_decision_trace(context, result)
            return result
        finally:
            self.agent.config.session_history_turns = _orig_hist

    async def _drive(
        self,
        current_input: str,
        runner: AgentRunner,
        context: RunContext,
        *,
        start_iteration: int,
        original_input: str,
        llm_client: LLMClient | None,
    ) -> AgentResponse:
        """Run iterations ``start_iteration..max`` and assemble the result.

        ``start_iteration`` is the 0-based iteration index to begin at — 0 for a
        fresh run, ``k`` when resuming a fork from iteration ``k`` so the new
        trace's stage markers stay aligned with the parent's.
        """
        iteration = start_iteration
        all_responses: list[AgentResponse] = []
        total_usage = TokenUsage()
        iteration_history: list[dict[str, Any]] = []

        while iteration < self.termination.max_iterations:
            iteration += 1

            logger.info(
                f"Loop iteration {iteration}/{self.termination.max_iterations}",
                extra={"run_id": context.run_id, "iteration": iteration},
            )

            # Decision trace: mark the start of this iteration (0-based stage).
            if context.recorder is not None:
                context.recorder.record_workflow_step(
                    self.name, stage=iteration - 1, label=self.agent.name, agent_stack=[self.name]
                )

            try:
                response = await runner.run(
                    agent=self.agent,
                    input=current_input,
                    context=context,
                )
                # A returned ERROR (e.g. circuit breaker open) is a failed
                # iteration, not an answer — surface it instead of treating the
                # error prose as this iteration's output.
                if response.status == ResponseStatus.ERROR:
                    raise RuntimeError(response.error or response.content or "iteration failed")

                all_responses.append(response)
                total_usage = total_usage.add(response.usage)

                # History only needed on the first executed iteration — it doesn't
                # change between iterations (writes are blocked), so skip it after.
                if iteration == start_iteration + 1:
                    self.agent.config.session_history_turns = 0

                iteration_history.append(
                    {
                        "iteration": iteration,
                        "input": current_input[:500],
                        "output": (response.content or "")[:500],
                    }
                )

                should_terminate = await self._check_termination(
                    response=response,
                    iteration=iteration,
                    history=iteration_history,
                    llm_client=llm_client,
                )

                if should_terminate:
                    logger.info(f"Loop terminated at iteration {iteration}")
                    break

                current_input = self._build_next_input(
                    original_input=original_input,
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
            if iteration >= self.termination.max_iterations:
                logger.warning(f"Loop reached max iterations ({self.termination.max_iterations})")

        final_response = (
            all_responses[-1]
            if all_responses
            else AgentResponse(content="No iterations completed", status=ResponseStatus.ERROR)
        )

        result = AgentResponse(
            content=final_response.content,
            structured_output=final_response.structured_output,
            agent_name=self.name,
            status=ResponseStatus.SUCCESS if all_responses else ResponseStatus.ERROR,
            usage=total_usage,
            turn_count=iteration,
            messages=final_response.messages,
        )
        result.run_id = context.run_id
        return result

    async def resume_from(
        self,
        *,
        parent_trace: Any,
        from_step: str,
        override: dict[str, Any] | None,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Re-run the loop from the iteration containing ``from_step``.

        Iterations before it are not re-run; the resumed iteration's input is
        recovered from its first step's checkpoint (with ``override`` applied),
        then the loop runs forward with its normal termination checks.
        """
        from continuum.agent.workflow._forkable import (
            link_lineage,
            resumed_input,
            segment_by_markers,
        )

        step_stage, stage_first = segment_by_markers(parent_trace)
        stage_idx = step_stage.get(from_step)
        if stage_idx is None:
            raise ValueError(f"resume_from: step '{from_step}' not found in trace")
        stage_idx = max(0, min(stage_idx, self.termination.max_iterations - 1))

        new_input = resumed_input(stage_first.get(stage_idx), override, parent_trace.user_query)

        created = runner.ensure_recorder(context, self.name, parent_trace.user_query)
        if created:
            link_lineage(context, parent_trace, from_step, override, stage_idx)
        context.suppress_session_log = True
        _orig_hist = self.agent.config.session_history_turns
        try:
            result = await self._drive(
                new_input,
                runner,
                context,
                start_iteration=stage_idx,
                original_input=parent_trace.user_query,
                llm_client=None,
            )
            if created:
                await runner.persist_decision_trace(context, result)
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
            from continuum.core.container import get_container

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
            from continuum.llm.config import LLMConfig

            llm_response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(
                    model=self.agent.model if self.agent else settings.default_llm_model,
                    temperature=self.termination.decision_temperature,
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
    memory_agent: BaseAgent | None = None,
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
        memory_agent=memory_agent,
    )
