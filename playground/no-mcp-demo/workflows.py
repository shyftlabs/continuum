"""
All 10 workflow modes — no MCP, no external tools.

Each class follows the same pattern as multi-agent-shop/workflows.py:
  initialize() → lifecycle → container → agents → runner
  chat(message, user_id, conversation_id) → str
  close()
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import DemoConfig, default_config

from agents import (
    make_analyst,
    make_editor,
    make_fact_checker,
    make_researcher,
    make_summarizer,
    make_support_agent,
    make_writer,
)
from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    Handoff,
    RunnerConfig,
    get_logger,
)
from orchestrator.agent.types import AgentResponse, MergeStrategy, ResponseStatus, Route, TerminationConfig, TerminationType
from orchestrator.agent.config import (
    ParallelConfig,
    PlanningConfig,
    ReflectionConfig,
    RouterConfig,
    SequentialConfig,
)
from orchestrator.agent.workflow.debate import DebateAgent
from orchestrator.agent.workflow.loop import LoopAgent
from orchestrator.agent.workflow.parallel import ParallelAgent
from orchestrator.agent.workflow.planner import PlannerAgent
from orchestrator.agent.workflow.reflection import ReflectionAgent
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.agent.workflow.scatter import ScatterAgent
from orchestrator.agent.workflow.sequential import SequentialAgent
from orchestrator.agent.workflow.supervised import SupervisedConfig, SupervisedSequentialAgent
from orchestrator.core.container import get_container
from orchestrator.core.lifecycle import get_lifecycle_manager

logger = get_logger(__name__)


# =============================================================================
# Base — shared init/teardown logic (no MCP step)
# =============================================================================

class _BaseWorkflow:
    # Workflow agents must be invoked via .execute() directly; runner.run() only
    # drives the single-agent LLM loop and never calls the workflow execute() method.
    # Set to False in subclasses whose _agent is a plain BaseAgent (e.g. HandoffDemo).
    _use_direct_execute: bool = True

    def __init__(self, config: DemoConfig | None = None):
        self.config = config or default_config
        self._lifecycle = None
        self._container = None
        self._runner: AgentRunner | None = None
        self._agent = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=True,
            enable_signal_handlers=False,
        )
        await self._lifecycle.initialize()
        self._container = get_container()
        self._build_workflow()
        self._runner = AgentRunner(
            container=self._container,
            config=RunnerConfig(persist_state=False, default_max_turns=self.config.max_turns),
        )
        self._initialized = True
        logger.info(f"✓ {self.__class__.__name__} ready")

    def _build_workflow(self) -> None:
        raise NotImplementedError

    async def chat(self, message: str, user_id: str, conversation_id: str) -> str:
        if not self._initialized:
            await self.initialize()

        session_id = None
        if self._container and self.config.enable_session:
            sc = self._container.session_client
            if sc and sc.is_enabled:
                try:
                    session_id = await sc.get_or_create_session(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                except Exception as e:
                    logger.warning(f"Session init failed: {e}")

        try:
            if self._use_direct_execute:
                from orchestrator.agent.utils.context_utils import create_run_context
                ctx = create_run_context(
                    session_id=session_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                response = await self._agent.execute(message, self._runner, ctx)
            else:
                response = await self._runner.run(
                    agent=self._agent,
                    input=message,
                    session_id=session_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            return response.content or ""
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {e}"

    async def close(self) -> None:
        if self._lifecycle:
            await self._lifecycle.shutdown()


# =============================================================================
# 1. Sequential — question → research → write → edit
# =============================================================================

class SequentialDemo(_BaseWorkflow):
    """
    Use case: "Explain quantum computing"
    Step 1 (researcher):  gathers facts and context
    Step 2 (writer):      turns research into a readable essay
    Step 3 (editor):      polishes the draft
    Step 4 (summarizer):  condenses everything into a short reply
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        self._agent = SequentialAgent(
            name="sequential-demo",
            agents=[
                make_researcher(m),
                make_writer(m),
                make_editor(m),
                make_summarizer(m),
            ],
            sequential_config=SequentialConfig(
                pass_full_history=True,
                pipeline_context_max_chars=None,
            ),
        )


# =============================================================================
# 2. Parallel — research from multiple angles simultaneously
# =============================================================================

@dataclass
class ParallelCoordinatorAgent(BaseAgent):
    """
    Coordinator for parallel mode:
      1. Three analysts research different angles simultaneously (stateless).
      2. Synthesiser combines results into a user-facing reply.
    """
    synthesiser: BaseAgent | None = None
    parallel: ParallelAgent | None = None

    async def execute(self, input_text: str, runner: Any, context: Any, llm_client: Any = None) -> AgentResponse:
        from orchestrator.agent.utils.context_utils import create_run_context

        context.suppress_session_log = True

        parallel_ctx = create_run_context(
            user_id=context.user_id,
            conversation_id=context.conversation_id,
        )
        parallel_result = await self.parallel.execute(input_text, runner, parallel_ctx)

        synthesis_input = (
            f"User asked: {input_text}\n\n"
            f"Multiple research angles:\n{parallel_result.content}\n\n"
            f"Write a coherent, unified answer covering all angles."
        )
        final = await runner.run(
            agent=self.synthesiser,
            input=synthesis_input,
            context=context,
        )

        if context.session_id:
            await runner.save_turn(
                session_id=context.session_id,
                user_message=input_text,
                assistant_message=final.content or "",
                agent=None,
            )

        total_usage = parallel_result.usage.add(final.usage)
        return AgentResponse(
            content=final.content,
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
            usage=total_usage,
        )


class ParallelDemo(_BaseWorkflow):
    """
    Use case: "Explain climate change"
    Three analysts cover: scientific, economic, and political angles simultaneously.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = self.config.enable_memory and memory_client is not None and memory_client.is_enabled

        science_analyst = BaseAgent(
            name="science-analyst",
            instructions=(
                "Focus ONLY on the scientific angle of the topic. "
                "Cover physical/biological/technical mechanisms and evidence. "
                "2-3 clear paragraphs."
            ),
            model=m,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False, session_history_turns=0),
        )
        economic_analyst = BaseAgent(
            name="economic-analyst",
            instructions=(
                "Focus ONLY on the economic angle of the topic. "
                "Cover costs, benefits, market forces, and financial implications. "
                "2-3 clear paragraphs."
            ),
            model=m,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False, session_history_turns=0),
        )
        social_analyst = BaseAgent(
            name="social-analyst",
            instructions=(
                "Focus ONLY on the social and human angle of the topic. "
                "Cover people, communities, ethics, and real-world impact. "
                "2-3 clear paragraphs."
            ),
            model=m,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False, session_history_turns=0),
        )

        parallel = ParallelAgent(
            name="parallel-inner",
            agents=[science_analyst, economic_analyst, social_analyst],
            parallel_config=ParallelConfig(
                merge_strategy=MergeStrategy.LLM_SUMMARIZE,
                summary_prompt=(
                    "Combine these three research angles into one coherent overview. "
                    "Label each section (Scientific / Economic / Social). "
                    "Keep each section to 2-3 sentences."
                ),
            ),
        )

        synthesiser = BaseAgent(
            name="parallel-synthesiser",
            instructions=(
                "You are a research synthesiser. "
                "Given multi-angle research and the user's original question, "
                "write a clear, unified answer that weaves all angles together."
            ),
            model=m,
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )

        self._agent = ParallelCoordinatorAgent(
            name="parallel-demo",
            synthesiser=synthesiser,
            parallel=parallel,
        )


# =============================================================================
# 3. Loop — refine answer until satisfied
# =============================================================================

class LoopDemo(_BaseWorkflow):
    """
    Use case: "Explain recursion until I fully understand"
    The researcher loops and refines until the answer starts with 'DONE:'.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        agent = BaseAgent(
            name="loop-researcher",
            instructions=(
                "Answer the user's question. "
                "After each attempt, evaluate if your answer is complete, clear, and beginner-friendly. "
                "If yes, start your final response with 'DONE:' followed by the complete answer. "
                "If not, start with 'REFINING:' and write a better version."
            ),
            model=m,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=True),
        )
        self._agent = LoopAgent(
            name="loop-demo",
            agent=agent,
            termination=TerminationConfig(
                type=TerminationType.OUTPUT_MATCH,
                pattern="DONE:",
                max_iterations=4,
            ),
        )


# =============================================================================
# 4. Scatter — split topic, analyse subtopics in parallel
# =============================================================================

class ScatterDemo(_BaseWorkflow):
    """
    Use case: "Compare Python, JavaScript, and Rust"
    Three analysts each cover one language in parallel.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = self.config.enable_memory and memory_client is not None and memory_client.is_enabled

        analysts = [make_analyst(m) for _ in range(3)]
        for i, a in enumerate(analysts, 1):
            a.name = f"analyst-{i}"

        scatter = ScatterAgent(
            name="scatter-inner",
            agents=analysts,
        )

        self._agent = scatter


# =============================================================================
# 5. Supervised — write with quality gate
# =============================================================================

class SupervisedDemo(_BaseWorkflow):
    """
    Use case: "Write an essay about the Renaissance"
    writer drafts; supervisor scores it. Retries up to 2 times if < 0.7.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        self._agent = SupervisedSequentialAgent(
            name="supervised-demo",
            agents=[make_writer(m)],
            supervised_config=SupervisedConfig(
                quality_threshold=0.7,
                max_retries=2,
                supervisor_model=m,
                pipeline_context_max_chars=None,
            ),
        )


# =============================================================================
# 6. Planner — dynamic research plan
# =============================================================================

class PlannerDemo(_BaseWorkflow):
    """
    Use case: "Help me understand machine learning from scratch"
    The planner breaks the goal into steps, routes each to the right agent.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        self._agent = PlannerAgent(
            name="planner-demo",
            agents=[
                make_researcher(m),
                make_writer(m),
                make_analyst(m),
                make_editor(m),
            ],
            planning_config=PlanningConfig(
                max_steps=6,
                enable_replanning=False,
            ),
        )


# =============================================================================
# 7. Debate — pro vs con with judge
# =============================================================================

class DebateDemo(_BaseWorkflow):
    """
    Use case: "Should AI replace human jobs?"
    pro-agent argues FOR; con-agent argues AGAINST.
    judge-agent synthesises a balanced conclusion.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        no_memory = AgentMemoryConfig(search_memories=False, store_memories=False)
        cfg = AgentConfig(log_to_session=True)

        self._agent = DebateAgent(
            name="debate-demo",
            pro_agent=BaseAgent(
                name="pro-side",
                instructions=(
                    "You are arguing FOR the proposition in the user's message. "
                    "Build the strongest possible case with evidence, logic, and examples. "
                    "Be persuasive and specific. 3-4 paragraphs."
                ),
                model=m,
                memory_config=no_memory,
                config=cfg,
            ),
            con_agent=BaseAgent(
                name="con-side",
                instructions=(
                    "You are arguing AGAINST the proposition in the user's message. "
                    "Build the strongest possible counter-argument with evidence and examples. "
                    "Be persuasive and specific. 3-4 paragraphs."
                ),
                model=m,
                memory_config=no_memory,
                config=cfg,
            ),
            judge_agent=BaseAgent(
                name="judge",
                instructions=(
                    "You are an impartial judge. "
                    "Read both arguments and give a clear, nuanced verdict. "
                    "Acknowledge the strongest points on each side, then state your conclusion."
                ),
                model=m,
                memory_config=no_memory,
                config=cfg,
            ),
        )


# =============================================================================
# 8. Reflection — write and self-critique until PASS
# =============================================================================

class ReflectionDemo(_BaseWorkflow):
    """
    Use case: "Write a cover letter for a software engineer role"
    writer drafts; critique LLM evaluates it.
    Retries with feedback until the critique says PASS.
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        self._agent = ReflectionAgent(
            name="reflection-demo",
            agent=make_writer(m),
            reflection_config=ReflectionConfig(
                max_reflections=2,
                critique_prompt=(
                    "Evaluate this piece of writing. "
                    "Check: Is it clear? Specific? Well-structured? Appropriate length (under 300 words)? "
                    "If all criteria are met, respond with 'PASS'. "
                    "Otherwise respond with 'NEEDS IMPROVEMENT: ' and the specific issue to fix."
                ),
                reflection_model=m,
            ),
        )


# =============================================================================
# 9. Router — triage to researcher, writer, or fact-checker
# =============================================================================

class RouterDemo(_BaseWorkflow):
    """
    Use case: any message — router decides which specialist handles it.
    - researcher:    factual questions, explanations, how-things-work
    - writer:        content creation, essays, emails, summaries
    - fact-checker:  verifying claims, checking accuracy
    - support-agent: general help, unclear intent
    """

    def _build_workflow(self) -> None:
        m = self.config.model
        researcher = make_researcher(m)
        writer = make_writer(m)
        fact_checker = make_fact_checker(m)
        support = make_support_agent(m)

        for specialist in (researcher, writer, fact_checker, support):
            specialist.config.session_history_turns = 0

        self._agent = RouterAgent(
            name="router-demo",
            model=m,
            routes=[
                Route(
                    agent_name="researcher",
                    description="factual questions, explanations, how things work, background information",
                ),
                Route(
                    agent_name="writer",
                    description="write content, create essays, draft emails, summarize, rewrite",
                ),
                Route(
                    agent_name="fact-checker",
                    description="verify claims, check accuracy, evaluate statements",
                ),
                Route(
                    agent_name="support-agent",
                    description="general help, greetings, unclear requests, meta questions",
                ),
            ],
            fallback_agent_name="support-agent",
            router_config=RouterConfig(routing_strategy="llm"),
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=True),
        )
        self._specialist_agents = {
            "researcher": researcher,
            "writer": writer,
            "fact-checker": fact_checker,
            "support-agent": support,
        }

    async def chat(self, message: str, user_id: str, conversation_id: str) -> str:
        if not self._initialized:
            await self.initialize()

        session_id = None
        if self._container and self.config.enable_session:
            sc = self._container.session_client
            if sc and sc.is_enabled:
                try:
                    session_id = await sc.get_or_create_session(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                except Exception as e:
                    logger.warning(f"Session init failed: {e}")

        try:
            from orchestrator.core.container import get_container as _gc
            llm = _gc().llm_client
            agent_name = await self._agent.route(message, llm_client=llm)
            if not agent_name:
                agent_name = self._agent.fallback_agent_name or "support-agent"

            target = self._specialist_agents.get(agent_name)
            if not target:
                return f"No agent found for route '{agent_name}'"

            logger.info(f"Router → {agent_name}")
            response = await self._runner.run(
                agent=target,
                input=message,
                session_id=session_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            return f"[→ {agent_name}]\n{response.content or ''}"
        except Exception as e:
            logger.error(f"Router chat error: {e}")
            return f"Error: {e}"


# =============================================================================
# 10. Handoff — orchestrator plans, hands off to researcher
# =============================================================================

class HandoffDemo(_BaseWorkflow):
    """
    Use case: any research question.
    The orchestrator (no tools) understands the question, then hands off to the
    researcher which answers it in detail. Control returns to the orchestrator
    which writes the final user-facing summary.

    Key difference from Router: the orchestrator runs AGAIN after the researcher
    finishes (return_to_parent=True) and can synthesise or follow up.
    """

    _use_direct_execute = False

    def _build_workflow(self) -> None:
        m = self.config.model
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = self.config.enable_memory and memory_client is not None and memory_client.is_enabled

        user_memory = AgentMemoryConfig(
            search_memories=memory_enabled,
            store_memories=memory_enabled,
            search_scope=AgentMemoryScope.USER,
            store_scope=AgentMemoryScope.USER,
            search_limit=5,
        )
        no_memory = AgentMemoryConfig(search_memories=False, store_memories=False)

        self._researcher_agent = BaseAgent(
            name="handoff-researcher",
            instructions=(
                "You are a deep research specialist. "
                "Given a topic or question, provide a thorough, detailed answer. "
                "Structure your response clearly. Aim for 4-6 paragraphs."
            ),
            model=m,
            memory_config=no_memory,
            config=AgentConfig(log_to_session=True),
        )

        self._agent = BaseAgent(
            name="handoff-orchestrator",
            instructions=(
                "You are a research orchestrator. "
                "Understand what the user wants to learn, then hand off to the researcher to investigate. "
                "After the researcher returns with findings, synthesise them into a clear, concise answer for the user."
            ),
            model=m,
            handoffs=[
                Handoff(
                    target_agent="handoff-researcher",
                    description=(
                        "Hand off to the researcher to investigate a topic or answer a question in depth."
                    ),
                    return_to_parent=True,
                )
            ],
            memory_config=user_memory,
            config=AgentConfig(log_to_session=True),
        )

    async def initialize(self) -> None:
        await super().initialize()
        self._runner.register_agent(self._agent)
        self._runner.register_agent(self._researcher_agent)


# =============================================================================
# Factory
# =============================================================================

MODES: dict[str, type[_BaseWorkflow]] = {
    "sequential":  SequentialDemo,
    "parallel":    ParallelDemo,
    "loop":        LoopDemo,
    "scatter":     ScatterDemo,
    "supervised":  SupervisedDemo,
    "planner":     PlannerDemo,
    "debate":      DebateDemo,
    "reflection":  ReflectionDemo,
    "router":      RouterDemo,
    "handoff":     HandoffDemo,
}


def create_workflow(mode: str, config: DemoConfig | None = None) -> _BaseWorkflow:
    cls = MODES.get(mode)
    if not cls:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
    return cls(config)
