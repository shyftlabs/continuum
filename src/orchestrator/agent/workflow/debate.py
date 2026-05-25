"""
Debate Agent — Two-sided argumentation with a judge synthesiser.

Three LLM calls, no new infrastructure:
  1. pro_agent  — argues one position
  2. con_agent  — argues the opposing position
  3. judge_agent — reads both, synthesises a balanced recommendation

Pro and con run concurrently; the judge only runs after both complete.

Useful for:
  - Architecture decisions ("microservices vs monolith")
  - Code review ("is this design good?")
  - Risk analysis ("risks vs opportunities")
  - Research ("explore both sides of a topic")

Usage::

    from orchestrator.agent.workflow import DebateAgent, create_debate_agent

    debate = create_debate_agent(
        name="architecture-debate",
        topic_description="Should we use microservices or a monolith?",
        pro_stance="Argue strongly FOR microservices.",
        con_stance="Argue strongly FOR a monolith.",
    )

    result = await runner.run(debate, "Our team is 5 engineers building a SaaS product.")
    # result.content = judge's balanced synthesis
    # result.metadata["pro_argument"] = pro agent's argument
    # result.metadata["con_argument"] = con agent's argument
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
from orchestrator.agent.types import AgentResponse, ResponseStatus, RunContext, TokenUsage
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
class DebateConfig:
    """Configuration for DebateAgent."""

    # When True: before passing to the judge, each side runs a short LLM call
    # to compress its own argument into 3-5 bullet points. This preserves the
    # most important points rather than blindly truncating by character count.
    # Adds 2 extra LLM calls (one per side), run in parallel.
    summarise_arguments: bool = False

    # Model for the self-summarisation calls (defaults to the debate agent's model)
    summarise_model: str | None = None

    # Hard character limit applied AFTER summarisation (or instead of it when
    # summarise_arguments=False). Set to None to disable truncation entirely.
    truncate_chars: int | None = 2000

    def to_dict(self) -> dict[str, Any]:
        return {
            "summarise_arguments": self.summarise_arguments,
            "summarise_model": self.summarise_model,
            "truncate_chars": self.truncate_chars,
        }


# =============================================================================
# Agent
# =============================================================================


@dataclass
class DebateAgent(BaseAgent):
    """
    Orchestrates a structured debate between two agents, judged by a third.

    The pro_agent and con_agent run in parallel on the same input.
    The judge_agent then receives both arguments and synthesises a
    balanced recommendation.

    Both the individual arguments and the final synthesis are available
    in the returned AgentResponse.

    Example::

        debate = DebateAgent(
            name="build-vs-buy",
            pro_agent=BaseAgent(name="build-advocate", instructions="Argue FOR building in-house."),
            con_agent=BaseAgent(name="buy-advocate",   instructions="Argue FOR buying a vendor solution."),
            judge_agent=BaseAgent(name="cto-judge",    instructions="Synthesise both arguments into a recommendation."),
        )
        result = await runner.run(debate, "Should we build our own auth system?")
    """

    pro_agent: BaseAgent | None = None
    con_agent: BaseAgent | None = None
    judge_agent: BaseAgent | None = None
    debate_config: DebateConfig = field(default_factory=DebateConfig)

    def __post_init__(self) -> None:
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError
            raise AgentConfigurationError("Agent name is required")
        if self.pro_agent is None or self.con_agent is None or self.judge_agent is None:
            from orchestrator.agent.exceptions import AgentConfigurationError
            raise AgentConfigurationError(
                "DebateAgent requires pro_agent, con_agent, and judge_agent"
            )

    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """
        Run the debate: pro + con in parallel, then judge synthesises.

        Args:
            input_text: The question or topic being debated
            runner: Agent runner
            context: Run context

        Returns:
            AgentResponse whose content is the judge's synthesis.
            The raw arguments are stored in context.metadata under
            "debate_pro" and "debate_con" keys.
        """
        context.suppress_session_log = True
        result = await self._execute_inner(input_text, runner, context)
        if context.session_id:
            await runner.save_turn(
                session_id=context.session_id,
                user_message=input_text,
                assistant_message=result.content or "",
                agent=None,
            )
        return result

    async def _execute_inner(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        total_usage = TokenUsage()

        async with SpanScope(
            f"workflow.debate.{self.name}",
            input={"topic_preview": input_text[:400]},
            metadata={
                "workflow_type": "debate",
                "pro_agent": self.pro_agent.name,
                "con_agent": self.con_agent.name,
                "judge_agent": self.judge_agent.name,
            },
        ) as workflow_span:

            # Step 1 — run pro and con concurrently
            logger.info(
                f"DebateAgent '{self.name}': running '{self.pro_agent.name}' "
                f"and '{self.con_agent.name}' in parallel"
            )

            pro_task = asyncio.create_task(
                runner.run(agent=self.pro_agent, input=input_text, context=context)
            )
            con_task = asyncio.create_task(
                runner.run(agent=self.con_agent, input=input_text, context=context)
            )

            pro_response, con_response = await asyncio.gather(
                pro_task, con_task, return_exceptions=True
            )

            # Handle failures in either side
            if isinstance(pro_response, Exception):
                logger.error(f"DebateAgent: pro_agent failed: {pro_response}")
                pro_content = f"[Pro argument unavailable: {pro_response}]"
                pro_usage = TokenUsage()
            else:
                pro_content = pro_response.content or ""
                pro_usage = pro_response.usage
                total_usage = total_usage.add(pro_usage)

            if isinstance(con_response, Exception):
                logger.error(f"DebateAgent: con_agent failed: {con_response}")
                con_content = f"[Con argument unavailable: {con_response}]"
                con_usage = TokenUsage()
            else:
                con_content = con_response.content or ""
                con_usage = con_response.usage
                total_usage = total_usage.add(con_usage)

            logger.info(
                f"DebateAgent '{self.name}': both sides complete — "
                f"pro={len(pro_content)} chars, con={len(con_content)} chars"
            )

            workflow_span.add_metadata("pro_preview", pro_content[:200])
            workflow_span.add_metadata("con_preview", con_content[:200])

            # Step 2 — prepare excerpts for the judge
            # Option A (summarise_arguments=True): each side compresses its own
            #   argument into bullet points — preserves important content.
            # Option B (default): hard character truncation as a safety net.
            pro_excerpt, con_excerpt, summarise_usage = await self._prepare_excerpts(
                pro_content=pro_content,
                con_content=con_content,
            )
            total_usage = total_usage.add(summarise_usage)

            judge_input = (
                f"Topic / Question:\n{input_text}\n\n"
                f"--- Argument FOR ({self.pro_agent.name}) ---\n"
                f"{pro_excerpt}\n\n"
                f"--- Argument AGAINST ({self.con_agent.name}) ---\n"
                f"{con_excerpt}\n\n"
                f"Based on both arguments above, provide a balanced, actionable synthesis."
            )

            logger.info(f"DebateAgent '{self.name}': running judge '{self.judge_agent.name}'")

            try:
                judge_response = await runner.run(
                    agent=self.judge_agent,
                    input=judge_input,
                    context=context,
                )
                total_usage = total_usage.add(judge_response.usage)
                synthesis = judge_response.content or ""
            except Exception as e:
                logger.error(f"DebateAgent: judge_agent failed: {e}")
                synthesis = (
                    f"Judge synthesis unavailable ({e}).\n\n"
                    f"Pro argument:\n{pro_content}\n\n"
                    f"Con argument:\n{con_content}"
                )

            # Store individual arguments in context metadata for caller access
            context.metadata["debate_pro"] = pro_content
            context.metadata["debate_con"] = con_content

            workflow_span.set_output({
                "success": True,
                "synthesis_length": len(synthesis),
                "total_tokens": total_usage.total_tokens,
            })

            return AgentResponse(
                content=synthesis,
                agent_name=self.name,
                status=ResponseStatus.SUCCESS,
                usage=total_usage,
                turn_count=3,  # pro + con + judge
                agents_used=[
                    self.pro_agent.name,
                    self.con_agent.name,
                    self.judge_agent.name,
                ],
            )

    async def _prepare_excerpts(
        self,
        pro_content: str,
        con_content: str,
    ) -> tuple[str, str, TokenUsage]:
        """
        Return (pro_excerpt, con_excerpt, token_usage) ready for the judge.

        If summarise_arguments=True: ask each side to compress its own argument
        into 3-5 bullet points. Both summarisation calls run in parallel.

        If summarise_arguments=False: apply hard character truncation only
        (or no truncation if truncate_chars=None).
        """
        total_usage = TokenUsage()

        if self.debate_config.summarise_arguments:
            limit = self.debate_config.truncate_chars
            pro_needs_summary = limit is None or len(pro_content) > limit
            con_needs_summary = limit is None or len(con_content) > limit

            if not pro_needs_summary and not con_needs_summary:
                # Both arguments already fit — skip LLM summarisation entirely
                logger.info(
                    f"DebateAgent '{self.name}': skipping summarisation — "
                    f"both arguments fit within truncate_chars limit "
                    f"(pro={len(pro_content)}, con={len(con_content)}, limit={limit})"
                )
            else:
                llm = self._get_llm()
                if llm:
                    # Only summarise sides that exceed the limit
                    async def _maybe_summarise(content: str, stance: str, needs: bool) -> tuple[str, TokenUsage]:
                        if needs:
                            return await self._summarise_side(content, stance, llm)
                        return content, TokenUsage()

                    (pro_excerpt, pro_usage), (con_excerpt, con_usage) = await asyncio.gather(
                        _maybe_summarise(pro_content, "FOR", pro_needs_summary),
                        _maybe_summarise(con_content, "AGAINST", con_needs_summary),
                    )
                    total_usage = total_usage.add(pro_usage).add(con_usage)
                    logger.info(
                        f"DebateAgent '{self.name}': summarised arguments — "
                        f"pro {len(pro_content)}→{len(pro_excerpt)} chars, "
                        f"con {len(con_content)}→{len(con_excerpt)} chars"
                    )
                    return pro_excerpt, con_excerpt, total_usage
                else:
                    logger.warning(
                        "DebateAgent: summarise_arguments=True but no LLM client available "
                        "— falling back to truncation"
                    )

        # Truncation fallback
        limit = self.debate_config.truncate_chars
        if limit is not None:
            pro_excerpt = pro_content[:limit] + ("…" if len(pro_content) > limit else "")
            con_excerpt = con_content[:limit] + ("…" if len(con_content) > limit else "")
        else:
            pro_excerpt = pro_content
            con_excerpt = con_content

        return pro_excerpt, con_excerpt, total_usage

    async def _summarise_side(
        self,
        content: str,
        stance: str,
        llm: Any,
    ) -> tuple[str, TokenUsage]:
        """
        Ask the LLM to compress one side's argument into 3-5 bullet points,
        each 2-4 sentences with supporting evidence.

        The LLM knows the argument — it picks what matters, not a char slice.
        Returns (summary_text, token_usage).
        """
        from orchestrator.llm.config import LLMConfig

        model = self.debate_config.summarise_model or self.model
        prompt = (
            f"You are summarising an argument that argues {stance} a position.\n\n"
            f"Argument:\n{content}\n\n"
            f"Compress this into 3-5 bullet points that capture the STRONGEST and most "
            f"important points. Do not add new arguments. Do not editorialize. "
            f"Each bullet should be 2-4 sentences capturing the main point and its supporting evidence."
        )

        try:
            response = await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.1, max_tokens=1000),
                auto_session=False,
            )
            usage = TokenUsage()
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                )
            return (response.content or content), usage
        except Exception as e:
            logger.warning(f"DebateAgent: argument summarisation failed ({e}) — using original")
            return content, TokenUsage()

    def _get_llm(self) -> Any | None:
        try:
            from orchestrator.core.container import get_container
            return get_container().llm_client
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "pro_agent": self.pro_agent.name if self.pro_agent else None,
            "con_agent": self.con_agent.name if self.con_agent else None,
            "judge_agent": self.judge_agent.name if self.judge_agent else None,
            "debate_config": self.debate_config.to_dict(),
            "workflow_type": "debate",
        })
        return base


# =============================================================================
# Factory
# =============================================================================


def create_debate_agent(
    name: str,
    *,
    topic_description: str = "",
    pro_stance: str,
    con_stance: str,
    judge_instructions: str | None = None,
    model: str | None = None,
    summarise_arguments: bool = False,
    summarise_model: str | None = None,
    truncate_chars: int | None = 2000,
) -> DebateAgent:
    """
    Factory for DebateAgent — creates pro, con, and judge agents automatically.

    All three agents share the same model. For fine-grained control (different
    models, tools, configs), construct DebateAgent directly.

    Args:
        name: Debate agent name
        topic_description: Context prepended to all agent instructions
        pro_stance: Instructions for the agent arguing the FOR position
        con_stance: Instructions for the agent arguing the AGAINST position
        judge_instructions: Instructions for the judge (default: balanced synthesis)
        model: LLM model for all three agents

    Returns:
        Configured DebateAgent

    Example::

        debate = create_debate_agent(
            name="arch-debate",
            topic_description="We are a 5-engineer startup.",
            pro_stance="Argue strongly FOR microservices. Give concrete technical reasons.",
            con_stance="Argue strongly FOR a monolith. Give concrete practical reasons.",
        )
        result = await runner.run(debate, "Should we use microservices or a monolith?")
    """
    _model = model or settings.default_llm_model
    _no_memory = AgentMemoryConfig(search_memories=False, store_memories=False)
    _config = AgentConfig(log_to_session=False)

    context_prefix = f"{topic_description}\n\n" if topic_description else ""

    pro_agent = BaseAgent(
        name=f"{name}-pro",
        instructions=f"{context_prefix}{pro_stance}",
        model=_model,
        config=_config,
        memory_config=_no_memory,
    )

    con_agent = BaseAgent(
        name=f"{name}-con",
        instructions=f"{context_prefix}{con_stance}",
        model=_model,
        config=_config,
        memory_config=_no_memory,
    )

    _judge_instructions = judge_instructions or (
        "You are an impartial judge. You have received two opposing arguments. "
        "Read both carefully, identify the strongest points on each side, and "
        "provide a balanced, actionable recommendation. "
        "Acknowledge trade-offs honestly and tailor your recommendation to the specific context provided."
    )

    judge_agent = BaseAgent(
        name=f"{name}-judge",
        instructions=f"{context_prefix}{_judge_instructions}",
        model=_model,
        config=_config,
        memory_config=_no_memory,
    )

    return DebateAgent(
        name=name,
        model=_model,
        pro_agent=pro_agent,
        con_agent=con_agent,
        debate_config=DebateConfig(
            summarise_arguments=summarise_arguments,
            summarise_model=summarise_model,
            truncate_chars=truncate_chars,
        ),
        judge_agent=judge_agent,
    )
