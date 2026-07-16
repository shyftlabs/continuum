"""
Reflection Agent - Self-critiquing workflow agent.

Runs the inner agent, critiques its output via an LLM call, and retries
up to ``max_reflections`` times if the critique says "NEEDS IMPROVEMENT".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from continuum.agent.base import BaseAgent
from continuum.agent.config import ReflectionConfig
from continuum.agent.types import AgentResponse, ResponseStatus, TokenUsage
from continuum.agent.utils.context_utils import publish_active_policy
from continuum.config import settings
from continuum.logging import get_logger

if TYPE_CHECKING:
    from continuum.agent.runner import AgentRunner
    from continuum.agent.types import RunContext

logger = get_logger(__name__)


@dataclass
class ReflectionAgent(BaseAgent):
    """
    Workflow agent that self-critiques and retries its inner agent.

    After each run of the inner agent, a separate LLM call is made to
    evaluate the response.  If the critique starts with "NEEDS IMPROVEMENT",
    the inner agent is re-invoked with the original input plus a note about
    the previous attempt.  This repeats up to ``reflection_config.max_reflections``
    times.

    Example::

        from continuum.agent import BaseAgent
        from continuum.agent.workflow import ReflectionAgent

        worker = BaseAgent(name="writer", instructions="Write concise summaries.")

        reflector = ReflectionAgent(
            name="reflection-agent",
            agent=worker,
        )

        result = await runner.run(reflector, "Summarise the quarterly report.")
    """

    # Inner agent to run and critique
    agent: BaseAgent | None = None

    # Reflection configuration
    reflection_config: ReflectionConfig = field(default_factory=ReflectionConfig)

    # Agent whose memory_config governs post-execution long-term memory writes.
    # Defaults to the inner agent (the natural producer of the final output).
    memory_agent: BaseAgent | None = None

    def __post_init__(self) -> None:
        if not self.name:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if self.agent is None:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("ReflectionAgent requires an inner agent to execute")

    @publish_active_policy
    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
        llm_client: Any | None = None,
    ) -> AgentResponse:
        """
        Run the inner agent, critique the output, and retry if needed.

        Args:
            input_text: User input
            runner: Agent runner used to invoke the inner agent
            context: Run context
            llm_client: Optional LLM client override for critique calls

        Returns:
            Final AgentResponse (either passing or last attempt)
        """
        # Own the one decision trace spanning all reflection attempts (rooted at
        # this workflow). Sub-runs share this recorder but are suppressed, so we
        # persist it ourselves at the end.
        created = runner.ensure_recorder(context, self.name, input_text)
        context.suppress_session_log = True
        _orig_hist = self.agent.config.session_history_turns
        try:
            result = await self._drive(
                input_text,
                runner,
                context,
                start_attempt=0,
                original_input=input_text,
                llm_client=llm_client,
            )

            if context.session_id:
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
        start_attempt: int,
        original_input: str,
        llm_client: Any | None,
    ) -> AgentResponse:
        """Run reflection attempts ``start_attempt..max`` and assemble the result.

        A "stage" is one reflection attempt (the generate + critique cycle).
        ``start_attempt`` is the 0-based attempt index to begin at — 0 for a fresh
        run, ``k`` when resuming a fork from attempt ``k`` so the new trace's stage
        markers stay aligned with the parent's. Refinement inputs are built from
        ``original_input`` (the user query) just as a fresh run would.
        """
        total_usage = TokenUsage()

        # Resolve LLM client for critique calls
        if llm_client is None:
            from continuum.core.container import get_container

            llm_client = get_container().llm_client

        response: AgentResponse | None = None

        for attempt in range(start_attempt, self.reflection_config.max_reflections + 1):
            logger.info(
                f"ReflectionAgent '{self.name}': attempt {attempt + 1} / "
                f"{self.reflection_config.max_reflections + 1}"
            )

            # Decision trace: mark the start of this reflection attempt (0-based stage).
            if context.recorder is not None:
                context.recorder.record_workflow_step(
                    self.name, stage=attempt, label=self.agent.name, agent_stack=[self.name]
                )

            response = await runner.run(
                agent=self.agent,
                input=current_input,
                context=context,
            )
            total_usage = total_usage.add(response.usage)

            # History only needed on the first executed attempt — it doesn't change
            # between retries (writes are blocked), so skip it after.
            if attempt == start_attempt:
                self.agent.config.session_history_turns = 0

            # On the last allowed attempt, skip the critique and return
            if attempt == self.reflection_config.max_reflections:
                break

            critique = await self._critique(
                response_content=response.content,
                llm_client=llm_client,
            )
            total_usage = total_usage.add(critique["usage"])

            if critique["verdict"].startswith("PASS"):
                logger.info(
                    f"ReflectionAgent '{self.name}': critique passed on attempt {attempt + 1}"
                )
                break

            logger.info(
                f"ReflectionAgent '{self.name}': critique says NEEDS IMPROVEMENT — retrying. "
                f"Reason: {critique['verdict']}"
            )
            current_input = (
                f"{original_input}\n\nPrevious attempt:\n{response.content}\n\n"
                f"Feedback: {critique['verdict']}"
            )

        assert response is not None  # at least one iteration always runs

        result = AgentResponse(
            content=response.content,
            structured_output=response.structured_output,
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
            usage=total_usage,
            turn_count=response.turn_count,
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
        """Re-run reflection from the attempt containing ``from_step``.

        Attempts before it are not re-run; the resumed attempt's input is
        recovered from its first step's checkpoint (with ``override`` applied),
        then reflection runs forward with its normal critique / termination logic.
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
        stage_idx = max(0, min(stage_idx, self.reflection_config.max_reflections))

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
                start_attempt=stage_idx,
                original_input=parent_trace.user_query,
                llm_client=None,
            )
            if created:
                await runner.persist_decision_trace(context, result)
            return result
        finally:
            self.agent.config.session_history_turns = _orig_hist

    async def _critique(
        self,
        response_content: str,
        llm_client: Any,
    ) -> dict[str, Any]:
        """
        Call the LLM to evaluate the inner agent's response.

        Returns a dict with ``verdict`` (str) and ``usage`` (TokenUsage).
        """
        from continuum.llm.config import LLMConfig

        model = self.reflection_config.reflection_model or (
            self.agent.model if self.agent else settings.default_llm_model
        )

        # Temperature: an explicit reflection_temperature wins; otherwise inherit
        # the inner agent's configured temperature (which may itself be None, in
        # which case the parameter is omitted for providers that reject it).
        temperature: float | None
        if self.reflection_config.reflection_temperature is not None:
            temperature = self.reflection_config.reflection_temperature
        elif self.agent is not None:
            temperature = self.agent.temperature
        else:
            temperature = settings.default_llm_temperature

        messages = [
            {"role": "user", "content": response_content},
            {"role": "user", "content": self.reflection_config.critique_prompt},
        ]

        logger.info(
            "===== CRITIQUE PROMPT [%s] =====\n%s\n%s\n=========================",
            self.name,
            response_content[:500],
            self.reflection_config.critique_prompt,
        )

        try:
            llm_response = await llm_client.chat(
                messages=messages,
                config=LLMConfig(model=model, temperature=temperature, max_tokens=256),
                auto_session=False,
            )

            usage = TokenUsage()
            if llm_response.usage:
                usage = TokenUsage(
                    prompt_tokens=llm_response.usage.prompt_tokens or 0,
                    completion_tokens=llm_response.usage.completion_tokens or 0,
                    total_tokens=llm_response.usage.total_tokens or 0,
                )

            verdict = (llm_response.content or "PASS").strip()
            logger.info(
                "===== CRITIQUE VERDICT [%s] =====\n%s\n=========================",
                self.name,
                verdict,
            )
            return {"verdict": verdict, "usage": usage}

        except Exception as e:
            logger.warning(f"ReflectionAgent critique call failed: {e}")
            return {"verdict": "PASS", "usage": TokenUsage()}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "agent": self.agent.name if self.agent else None,
                "reflection_config": {
                    "critique_prompt": self.reflection_config.critique_prompt,
                    "max_reflections": self.reflection_config.max_reflections,
                    "reflection_model": self.reflection_config.reflection_model,
                },
                "workflow_type": "reflection",
            }
        )
        return base


async def generate_critique_prompt(
    user_query: str,
    llm_client: Any,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Generate a critique prompt tailored to the user's query.

    Instead of a fixed critique prompt per domain, this asks an LLM to write
    a critique prompt for this specific query.  Use this when queries come from
    unpredictable domains and a single fixed critique would not cover all cases.

    Args:
        user_query: The user's original query
        llm_client: LLM client to use for generation
        model: Model to use (defaults to the configured default)
        temperature: Temperature for the generation call. None inherits the
            default LLM temperature; pass a float to override.

    Returns:
        A critique prompt string ready for use in ReflectionConfig

    Example::

        from continuum.core.container import get_container
        from continuum.agent.workflow import generate_critique_prompt, ReflectionAgent

        llm_client = get_container().llm_client
        critique = await generate_critique_prompt(user_query, llm_client)
        agent = ReflectionAgent(
            name="dynamic-reflector",
            agent=inner_agent,
            reflection_config=ReflectionConfig(critique_prompt=critique),
        )
    """
    from continuum.llm.config import LLMConfig

    _model = model or settings.default_llm_model
    config = LLMConfig(model=_model, max_tokens=300)
    if temperature is not None:
        config = config.with_overrides(temperature=temperature)

    messages = [
        {
            "role": "user",
            "content": (
                f"A user asked: '{user_query}'\n\n"
                "Write a critique prompt for evaluating a response to this query.\n"
                "Rules:\n"
                "- List 3-5 specific, concrete things a complete answer must include\n"
                "- Each requirement must be checkable (not vague like 'be thorough')\n"
                "- Start with: Reply ONLY 'PASS' if the response includes ALL of:\n"
                "- End with: Otherwise reply 'NEEDS IMPROVEMENT: <list what is missing>'\n"
                "Return only the critique prompt text, nothing else."
            ),
        }
    ]

    try:
        response = await llm_client.chat(
            messages=messages,
            config=config,
            auto_session=False,
        )
        return (response.content or "").strip()
    except Exception as e:
        logger.warning(f"generate_critique_prompt failed: {e} — using default")
        return ReflectionConfig().critique_prompt


def create_reflection_agent(
    name: str,
    agent: BaseAgent,
    *,
    critique_prompt: str | None = None,
    max_reflections: int = 2,
    reflection_model: str | None = None,
    reflection_temperature: float | None = None,
    memory_agent: BaseAgent | None = None,
) -> ReflectionAgent:
    """
    Factory function to create a reflection agent.

    Args:
        name: Name for the reflection agent
        agent: Inner agent to run and critique
        critique_prompt: Custom critique prompt (overrides default)
        max_reflections: Maximum number of reflection retries
        reflection_model: Model to use for critique calls (defaults to inner agent's model)
        reflection_temperature: Temperature for critique calls. None inherits the
            inner agent's temperature; pass a float to override.

    Returns:
        Configured ReflectionAgent
    """
    config = ReflectionConfig(
        max_reflections=max_reflections,
        reflection_model=reflection_model,
        reflection_temperature=reflection_temperature,
    )
    if critique_prompt is not None:
        config.critique_prompt = critique_prompt

    return ReflectionAgent(
        name=name,
        agent=agent,
        reflection_config=config,
        memory_agent=memory_agent,
    )
