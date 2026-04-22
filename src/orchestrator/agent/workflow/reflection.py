"""
Reflection Agent - Self-critiquing workflow agent.

Runs the inner agent, critiques its output via an LLM call, and retries
up to ``max_reflections`` times if the critique says "NEEDS IMPROVEMENT".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import ReflectionConfig
from orchestrator.agent.types import AgentResponse, ResponseStatus, TokenUsage
from orchestrator.config import settings
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner
    from orchestrator.agent.types import RunContext

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

        from orchestrator.agent import BaseAgent
        from orchestrator.agent.workflow import ReflectionAgent

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

    def __post_init__(self) -> None:
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if self.agent is None:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("ReflectionAgent requires an inner agent to execute")

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
        # Disable inner agent Redis saves; one clean pair is saved at the end.
        _orig_log = self.agent.config.log_to_session
        _orig_hist = self.agent.config.session_history_turns
        self.agent.config.log_to_session = False
        try:
            total_usage = TokenUsage()

            # Resolve LLM client for critique calls
            if llm_client is None:
                from orchestrator.core.container import get_container
                llm_client = get_container().llm_client

            current_input = input_text
            response: AgentResponse | None = None

            for attempt in range(self.reflection_config.max_reflections + 1):
                logger.info(
                    f"ReflectionAgent '{self.name}': attempt {attempt + 1} / "
                    f"{self.reflection_config.max_reflections + 1}"
                )

                response = await runner.run(
                    agent=self.agent,
                    input=current_input,
                    context=context,
                )
                total_usage = total_usage.add(response.usage)

                # History only needed on attempt 1 — it doesn't change between
                # retries (writes are blocked), so skip it from attempt 2 onwards.
                if attempt == 0:
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
                    logger.info(f"ReflectionAgent '{self.name}': critique passed on attempt {attempt + 1}")
                    break

                logger.info(
                    f"ReflectionAgent '{self.name}': critique says NEEDS IMPROVEMENT — retrying. "
                    f"Reason: {critique['verdict']}"
                )
                current_input = (
                    f"{input_text}\n\nPrevious attempt:\n{response.content}\n\n"
                    f"Feedback: {critique['verdict']}"
                )

            assert response is not None  # at least one iteration always runs

            if context.session_id:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=response.content or "",
                    agent=None,
                )

            return AgentResponse(
                content=response.content,
                structured_output=response.structured_output,
                agent_name=self.name,
                status=ResponseStatus.SUCCESS,
                usage=total_usage,
                turn_count=response.turn_count,
            )
        finally:
            self.agent.config.log_to_session = _orig_log
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
        from orchestrator.llm.config import LLMConfig

        model = (
            self.reflection_config.reflection_model
            or (self.agent.model if self.agent else settings.default_llm_model)
        )

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
                config=LLMConfig(model=model, temperature=0.1, max_tokens=256),
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
            logger.info("===== CRITIQUE VERDICT [%s] =====\n%s\n=========================", self.name, verdict)
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

    Returns:
        A critique prompt string ready for use in ReflectionConfig

    Example::

        from orchestrator.core.container import get_container
        from orchestrator.agent.workflow import generate_critique_prompt, ReflectionAgent

        llm_client = get_container().llm_client
        critique = await generate_critique_prompt(user_query, llm_client)
        agent = ReflectionAgent(
            name="dynamic-reflector",
            agent=inner_agent,
            reflection_config=ReflectionConfig(critique_prompt=critique),
        )
    """
    from orchestrator.llm.config import LLMConfig

    _model = model or settings.default_llm_model

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
            config=LLMConfig(model=_model, temperature=0.1, max_tokens=300),
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
) -> ReflectionAgent:
    """
    Factory function to create a reflection agent.

    Args:
        name: Name for the reflection agent
        agent: Inner agent to run and critique
        critique_prompt: Custom critique prompt (overrides default)
        max_reflections: Maximum number of reflection retries
        reflection_model: Model to use for critique calls (defaults to inner agent's model)

    Returns:
        Configured ReflectionAgent
    """
    config = ReflectionConfig(
        max_reflections=max_reflections,
        reflection_model=reflection_model,
    )
    if critique_prompt is not None:
        config.critique_prompt = critique_prompt

    return ReflectionAgent(
        name=name,
        agent=agent,
        reflection_config=config,
    )
