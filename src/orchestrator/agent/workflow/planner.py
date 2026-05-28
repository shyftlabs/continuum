"""
Planner Agent - Goal decomposition and dynamic execution.

Takes a high-level goal, generates an ordered task list via one LLM call,
then executes each step. Supports two modes:

  Single-agent mode  — one worker agent executes all steps with different instructions.
                       The LLM freely generates any steps based on the goal.

  Agent-pool mode    — a list of specialist agents; the LLM routes each step to the
                       right agent by name. All possible agents must be declared upfront.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import PlanningConfig
from orchestrator.agent.exceptions import PlannerWorkflowError
from orchestrator.agent.types import (
    AgentResponse,
    FailStrategy,
    ResponseStatus,
    TokenUsage,
)
from orchestrator.llm.config import LLMConfig
from orchestrator.logging import get_logger
from orchestrator.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner
    from orchestrator.agent.types import RunContext

logger = get_logger(__name__)


@dataclass
class PlannerAgent(BaseAgent):
    """
    Decomposes a high-level goal into steps, then executes them.

    Two modes:

    **Single-agent mode** (simpler, recommended):
        One worker agent executes every step with different instructions.
        The LLM freely creates any steps — no need to pre-define specialists.

        Example::

            worker = BaseAgent(name="worker", instructions="You are a helpful assistant.", ...)

            planner = PlannerAgent(
                name="planner",
                agent=worker,
                planning_config=PlanningConfig(enable_replanning=True),
            )

            result = await runner.run(planner, "Build a market research report on Tesla")

    **Agent-pool mode** (for specialist workflows):
        The LLM routes each step to a named agent from the pool.
        Every agent the LLM might need MUST be declared in ``agents`` upfront.

        Example::

            researcher = BaseAgent(name="researcher", description="Research a topic", ...)
            writer     = BaseAgent(name="writer",     description="Write a report", ...)

            planner = PlannerAgent(
                name="planner",
                agents=[researcher, writer],
            )

        WARNING: If an agent named in the plan is not in the ``agents`` list, that step
        is skipped. Make sure all possible agents are declared.
    """

    # Single-agent mode: one agent executes all steps
    agent: BaseAgent | None = None

    # Agent-pool mode: specialist agents the planner can route steps to
    agents: list[BaseAgent] = field(default_factory=list)

    # Planning configuration
    planning_config: PlanningConfig = field(default_factory=PlanningConfig)

    def __post_init__(self) -> None:
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if not self.agent and not self.agents:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError(
                "PlannerAgent requires either 'agent' (single-agent mode) "
                "or 'agents' (agent-pool mode)"
            )

        if self.agent and self.agents:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError(
                "PlannerAgent accepts either 'agent' or 'agents', not both"
            )

        # Warn user when using agent-pool mode
        if self.agents:
            agent_names = ", ".join(a.name for a in self.agents)
            logger.warning(
                f"PlannerAgent '{self.name}' is using agent-pool mode. "
                f"The LLM can ONLY use these agents: [{agent_names}]. "
                f"Any agent not in this list will be skipped. "
                f"Consider single-agent mode (agent=worker) for more flexibility."
            )

    @property
    def _mode(self) -> str:
        return "single" if self.agent else "pool"

    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """
        Decompose the goal into steps and execute them.

        Args:
            input_text: High-level goal
            runner: Agent runner for executing child agents
            context: Run context

        Returns:
            Final AgentResponse from the last executed step
        """
        from orchestrator.core.container import get_container

        llm_client = get_container().llm_client
        total_usage = TokenUsage()
        completed: list[dict[str, Any]] = []

        context.suppress_session_log = True
        try:
            async with SpanScope(
                f"workflow.planner.{self.name}",
                input={"goal": input_text[:500], "mode": self._mode},
                metadata={"workflow_type": "planner", "mode": self._mode},
            ) as workflow_span:
                # Step 1: Generate plan
                plan_steps, plan_usage = await self._generate_plan(input_text, llm_client)
                total_usage = total_usage.add(plan_usage)
                logger.info(
                    f"PlannerAgent '{self.name}' [{self._mode} mode]: "
                    f"generated {len(plan_steps)} steps"
                )

                # Step 2: Execute steps
                current_input = input_text
                step_outputs: list[dict[str, str]] = []  # accumulated outputs for assembly
                pipeline_history: list[str] = []
                i = 0  # index into plan_steps
                executed = 0  # actual agent executions (max_steps cap applies here only)

                while i < len(plan_steps) and executed < self.planning_config.max_steps:
                    step = plan_steps[i]
                    instruction = step.get("instruction", "")
                    step_id = step.get("step_id", str(i + 1))

                    # Resolve which agent runs this step
                    if self._mode == "single":
                        agent = self.agent
                        agent_name = agent.name  # type: ignore[union-attr]
                    else:
                        agent_name = step.get("agent_name", "")
                        agent = self._find_agent(agent_name)
                        if not agent:
                            if self.planning_config.strict_agent_pool:
                                workflow_span.set_error(
                                    f"Unknown agent '{agent_name}' at step {step_id}"
                                )
                                raise PlannerWorkflowError(
                                    f"Step {step_id} references unknown agent '{agent_name}'. "
                                    f"Declare it in the agents list or disable strict_agent_pool.",
                                    failed_agent=agent_name,
                                    step=i + 1,
                                    run_id=context.run_id,
                                )
                            logger.warning(
                                f"PlannerAgent: step {step_id} references unknown agent "
                                f"'{agent_name}' — skipping. "
                                f"Add it to the agents list to fix this."
                            )
                            i += 1
                            continue

                    # For the last step, pass full accumulated history so assembly works correctly
                    is_last_step = i == len(plan_steps) - 1
                    if is_last_step and len(step_outputs) > 1:
                        history = "\n\n".join(
                            f"[Step {s['step_id']} output]\n{s['output']}" for s in step_outputs
                        )
                        step_input = (
                            f"{instruction}\n\nAll previous step outputs:\n{history}"
                            if instruction
                            else history
                        )
                    else:
                        step_input = (
                            f"{instruction}\n\nInput:\n{current_input}"
                            if instruction
                            else current_input
                        )

                    logger.info(f"PlannerAgent step {step_id} → {agent_name}: {instruction[:80]}")

                    async with SpanScope(
                        f"workflow.planner.step.{step_id}",
                        input={"agent_name": agent_name, "instruction": instruction},
                    ) as step_span:
                        try:
                            # Inject prior steps so this agent sees what earlier agents produced.
                            # For regular steps: exclude the immediately preceding step (it is
                            # already the [user] input) — context carries only steps 1..N-2.
                            # For the last step: [user] already contains ALL outputs, so skip
                            # context entirely to avoid full redundancy.
                            if not is_last_step:
                                background = pipeline_history[:-1]
                                if background:
                                    context.metadata["pipeline_context"] = (
                                        "Prior pipeline steps in this request:\n"
                                        + "\n".join(background)
                                    )
                            else:
                                # Last step [user] already contains all outputs — clear any
                                # pipeline_context left over from the previous step iteration.
                                context.metadata.pop("pipeline_context", None)

                            response = await runner.run(
                                agent=agent,
                                input=step_input,
                                context=context,
                            )
                            total_usage = total_usage.add(response.usage)
                            completed.append(
                                {
                                    "step": step,
                                    "output": response.content or "",
                                    "success": True,
                                }
                            )
                            current_input = response.content or current_input
                            step_outputs.append({"step_id": step_id, "output": current_input})
                            pipeline_history.append(f"{agent_name}: {current_input[:300]}")

                            step_span.set_output(
                                {
                                    "success": True,
                                    "output_preview": (response.content or "")[:200],
                                }
                            )

                            # Optional: replan after successful step.
                            # Uses a separate counter so replan checks do not consume max_steps.
                            remaining = plan_steps[i + 1 :]
                            if self.planning_config.enable_replanning and remaining:
                                new_remaining, replan_usage = await self._maybe_replan(
                                    goal=input_text,
                                    completed=completed,
                                    remaining=remaining,
                                    last_output=current_input,
                                    llm_client=llm_client,
                                )
                                total_usage = total_usage.add(replan_usage)
                                if new_remaining is not None:
                                    plan_steps = plan_steps[: i + 1] + new_remaining
                                    logger.info(
                                        f"PlannerAgent: replanned after step {step_id} — "
                                        f"{len(new_remaining)} remaining steps"
                                    )

                            i += 1
                            executed += 1

                        except Exception as e:
                            logger.error(f"PlannerAgent step {step_id} ({agent_name}) failed: {e}")
                            step_span.set_error(str(e))
                            completed.append(
                                {
                                    "step": step,
                                    "output": str(e),
                                    "success": False,
                                }
                            )

                            if self.planning_config.replan_on_failure:
                                remaining = plan_steps[i + 1 :]
                                new_plan, replan_usage = await self._replan_on_failure(
                                    goal=input_text,
                                    completed=completed,
                                    failed_step=step,
                                    remaining=remaining,
                                    error=str(e),
                                    llm_client=llm_client,
                                )
                                total_usage = total_usage.add(replan_usage)
                                if new_plan is not None:
                                    plan_steps = plan_steps[:i] + new_plan
                                    logger.info(
                                        f"PlannerAgent: replanned after failure at step {step_id} — "
                                        f"{len(new_plan)} new steps"
                                    )
                                    continue

                            if self.planning_config.fail_strategy == FailStrategy.FAIL_FAST:
                                workflow_span.set_error(f"Step {step_id} failed: {e}")
                                raise PlannerWorkflowError(
                                    f"Step {step_id} ({agent_name}) failed: {e}",
                                    failed_agent=agent_name,
                                    step=i + 1,
                                    run_id=context.run_id,
                                ) from e

                            i += 1
                            executed += 1

                workflow_span.set_output(
                    {
                        "steps_executed": len(completed),
                        "total_tokens": total_usage.total_tokens,
                    }
                )

            final_content = current_input
            result = AgentResponse(
                content=final_content,
                agent_name=self.name,
                status=ResponseStatus.SUCCESS,
                usage=total_usage,
                turn_count=len(completed),
                agents_used=[
                    c["step"].get("agent_name", self.agent.name if self.agent else "")
                    for c in completed
                    if c["success"]
                ],
            )

            if context.session_id and completed:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=final_content or "",
                    agent=None,
                )

            return result
        finally:
            context.metadata.pop("pipeline_context", None)

    # -------------------------------------------------------------------------
    # Plan generation
    # -------------------------------------------------------------------------

    async def _generate_plan(
        self,
        goal: str,
        llm_client: Any,
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        """One LLM call to decompose the goal into an ordered step list."""

        if self._mode == "single":
            # Single-agent: steps only need instructions, no agent_name
            prompt = (
                f"You are a planning agent. Break down the following goal into ordered steps.\n\n"
                f"Goal: {goal}\n\n"
                f"Output a JSON object with this exact format:\n"
                f'{{\n  "steps": [\n'
                f'    {{"step_id": "1", "instruction": "<what to do in this step>"}},\n'
                f"    ...\n"
                f"  ]\n}}\n\n"
                f"Rules:\n"
                f"- Maximum {self.planning_config.max_steps} steps\n"
                f"- Each step's output becomes the next step's input\n"
                f"- If the goal produces a document or content piece across multiple steps, "
                f"the LAST step must combine all previous outputs into the final complete result\n"
                f"- Output only valid JSON, nothing else"
            )
        else:
            # Agent-pool: steps must name an agent from the list
            agent_catalog = "\n".join(
                f"- {a.name}: {a.description or (a.instructions[:100] + '...' if len(a.instructions) > 100 else a.instructions)}"
                for a in self.agents
            )
            prompt = (
                f"You are a planning agent. Break down the following goal into ordered steps.\n\n"
                f"Goal: {goal}\n\n"
                f"Available agents:\n{agent_catalog}\n\n"
                f"Output a JSON object with this exact format:\n"
                f'{{\n  "steps": [\n'
                f'    {{"step_id": "1", "agent_name": "<name from list>", "instruction": "<what this step does>"}},\n'
                f"    ...\n"
                f"  ]\n}}\n\n"
                f"Rules:\n"
                f"- Only use agents from the list above\n"
                f"- Maximum {self.planning_config.max_steps} steps\n"
                f"- Each step's output becomes the next step's input\n"
                f"- If the goal produces a document or content piece across multiple steps, "
                f"the LAST step must combine all previous outputs into the final complete result\n"
                f"- Output only valid JSON, nothing else"
            )

        model = self.planning_config.planning_model or self.model
        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.2, max_tokens=2000),
                auto_session=False,
            )
            usage = self._extract_usage(response)
            logger.debug(f"PlannerAgent raw plan response: {repr(response.content)}")
            steps = self._parse_steps(response.content or "")
            return steps, usage
        except Exception as e:
            logger.error(f"PlannerAgent plan generation failed: {e}")
            return [], TokenUsage()

    # -------------------------------------------------------------------------
    # Replanning
    # -------------------------------------------------------------------------

    async def _maybe_replan(
        self,
        goal: str,
        completed: list[dict[str, Any]],
        remaining: list[dict[str, Any]],
        last_output: str,
        llm_client: Any,
    ) -> tuple[list[dict[str, Any]] | None, TokenUsage]:
        """Lightweight check after a successful step — returns new remaining or None."""
        remaining_str = "\n".join(
            f"  {s['step_id']}: {s.get('agent_name', '')} — {s['instruction']}" for s in remaining
        )
        agent_hint = (
            f"Available agents: {', '.join(a.name for a in self.agents)}\n\n"
            if self._mode == "pool"
            else ""
        )
        step_format = (
            '{"step_id": "...", "agent_name": "...", "instruction": "..."}'
            if self._mode == "pool"
            else '{"step_id": "...", "instruction": "..."}'
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Last step output: {last_output[:400]}\n\n"
                    f"Remaining planned steps:\n{remaining_str}\n\n"
                    f"{agent_hint}"
                    f"Does the remaining plan still make sense given the last output?\n"
                    f'Reply ONLY with "CONTINUE" if valid, or a JSON object if replanning needed:\n'
                    f'  {{"steps": [{step_format}]}}'
                ),
            }
        ]

        model = self.planning_config.planning_model or self.model
        try:
            response = await llm_client.chat(
                messages=messages,
                config=LLMConfig(model=model, temperature=0.1, max_tokens=1500),
                auto_session=False,
            )
            usage = self._extract_usage(response)
            content = (response.content or "").strip()
            if content.upper().startswith("CONTINUE"):
                return None, usage
            steps = self._parse_steps(content)
            return (steps if steps else None), usage
        except Exception as e:
            logger.warning(f"PlannerAgent replan check failed: {e}")
            return None, TokenUsage()

    async def _replan_on_failure(
        self,
        goal: str,
        completed: list[dict[str, Any]],
        failed_step: dict[str, Any],
        remaining: list[dict[str, Any]],
        error: str,
        llm_client: Any,
    ) -> tuple[list[dict[str, Any]] | None, TokenUsage]:
        """Replan after a step failure."""
        agent_hint = (
            "Available agents:\n"
            + "\n".join(f"- {a.name}: {a.description or a.instructions[:80]}" for a in self.agents)
            + "\n\n"
            if self._mode == "pool"
            else ""
        )
        step_format = (
            '{"step_id": "...", "agent_name": "...", "instruction": "..."}'
            if self._mode == "pool"
            else '{"step_id": "...", "instruction": "..."}'
        )
        completed_str = "\n".join(
            f"  ✅ {c['step'].get('agent_name', '')} — {str(c['output'])[:100]}"
            for c in completed
            if c["success"]
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Completed steps:\n{completed_str or '  (none)'}\n\n"
                    f"Failed step: {failed_step.get('agent_name', '')} — {failed_step['instruction']}\n"
                    f"Error: {error}\n\n"
                    f"{agent_hint}"
                    f"Generate a new plan for the remaining work (replacing the failed step).\n"
                    f'Output JSON only: {{"steps": [{step_format}]}}'
                ),
            }
        ]

        model = self.planning_config.planning_model or self.model
        try:
            response = await llm_client.chat(
                messages=messages,
                config=LLMConfig(model=model, temperature=0.2, max_tokens=1500),
                auto_session=False,
            )
            usage = self._extract_usage(response)
            steps = self._parse_steps(response.content or "")
            return (steps if steps else None), usage
        except Exception as e:
            logger.warning(f"PlannerAgent replan-on-failure failed: {e}")
            return None, TokenUsage()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _find_agent(self, name: str) -> BaseAgent | None:
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def _parse_steps(self, content: str) -> list[dict[str, Any]]:
        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
            else:
                brace, start = 0, -1
                for idx, ch in enumerate(content):
                    if ch == "{":
                        if start == -1:
                            start = idx
                        brace += 1
                    elif ch == "}":
                        brace -= 1
                        if brace == 0 and start != -1:
                            content = content[start : idx + 1]
                            break
            data = json.loads(content)
            return data.get("steps", [])
        except Exception as e:
            logger.warning(f"PlannerAgent failed to parse steps: {e}")
            return []

    def _extract_usage(self, response: Any) -> TokenUsage:
        if response.usage:
            return TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return TokenUsage()

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "mode": self._mode,
                "agent": self.agent.name if self.agent else None,
                "agents": [a.name for a in self.agents],
                "planning_config": self.planning_config.to_dict(),
                "workflow_type": "planner",
            }
        )
        return base


def create_planner_agent(
    name: str,
    *,
    agent: BaseAgent | None = None,
    agents: list[BaseAgent] | None = None,
    instructions: str = "You are a planning agent that decomposes goals into steps.",
    max_steps: int = 10,
    enable_replanning: bool = False,
    replan_on_failure: bool = True,
    planning_model: str | None = None,
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST,
    strict_agent_pool: bool = False,
) -> PlannerAgent:
    """
    Factory function to create a planner agent.

    Choose one mode:

    **Single-agent mode** (recommended):
        One worker agent executes all steps with different instructions.
        The LLM freely generates any steps — no pre-defined specialist list needed.

        Example::

            planner = create_planner_agent(name="planner", agent=worker)

    **Agent-pool mode**:
        Specialist agents; the LLM routes each step to the right agent by name.
        Every agent the LLM might need MUST be in the list.
        By default, steps referencing unknown agents are skipped with a warning.
        Set strict_agent_pool=True to raise an error instead.

        Example::

            planner = create_planner_agent(
                name="planner",
                agents=[researcher, writer, editor],
                strict_agent_pool=True,
            )

    Args:
        name: Planner agent name
        agent: Single worker agent (single-agent mode)
        agents: Pool of specialist agents (agent-pool mode)
        instructions: System instructions for the planner
        max_steps: Maximum number of agent executions allowed (replan checks do not count)
        enable_replanning: Check after each step whether to replan
        replan_on_failure: Replan when a step fails
        planning_model: Model for plan generation (defaults to agent model)
        fail_strategy: How to handle unrecoverable step failures
        strict_agent_pool: Raise an error if the plan names an agent not in the pool
    """
    if not agent and not agents:
        raise ValueError(
            "create_planner_agent requires either 'agent' (single-agent mode) "
            "or 'agents' (agent-pool mode)"
        )
    if agent and agents:
        raise ValueError("create_planner_agent accepts either 'agent' or 'agents', not both")

    return PlannerAgent(
        name=name,
        instructions=instructions,
        agent=agent,
        agents=agents or [],
        planning_config=PlanningConfig(
            max_steps=max_steps,
            enable_replanning=enable_replanning,
            replan_on_failure=replan_on_failure,
            planning_model=planning_model,
            fail_strategy=fail_strategy,
            strict_agent_pool=strict_agent_pool,
        ),
    )
