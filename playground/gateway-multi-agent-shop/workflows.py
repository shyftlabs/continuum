"""
All 10 workflow modes for gateway-multi-agent-shop.

Identical logic to multi-agent-shop/workflows.py, with gateway_mode
passed to every agent so the Smart Gateway can route each independently.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from agents import (
    make_analyst_agent,
    make_cart_agent,
    make_recommend_agent,
    make_search_agent,
    make_summary_agent,
    make_support_agent,
    make_writer_agent,
)
from config import WorkflowShopConfig, default_config

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    Handoff,
    MCPServerStreamableHttp,
    MCPUtil,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from orchestrator.agent.config import (
    ParallelConfig,
    PlanningConfig,
    ReflectionConfig,
    RouterConfig,
    SequentialConfig,
)
from orchestrator.agent.types import (
    AgentResponse,
    MergeStrategy,
    ResponseStatus,
    Route,
    TerminationConfig,
    TerminationType,
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
from orchestrator.tools.types import ToolContextConfig, ToolContextVariable

logger = get_logger(__name__)


# =============================================================================
# Base — shared init/teardown logic for every mode
# =============================================================================


class _BaseWorkflow:
    _use_direct_execute: bool = True

    def __init__(self, config: WorkflowShopConfig | None = None):
        self.config = config or default_config
        self._lifecycle = None
        self._container = None
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None
        self._tools: list[dict[str, Any]] = []
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
        await self._connect_mcp()
        self._build_workflow()
        self._runner = AgentRunner(
            container=self._container,
            tool_executor=self._tool_executor,
            config=RunnerConfig(persist_state=False, default_max_turns=self.config.max_turns),
        )
        self._initialized = True
        logger.info(
            f"✓ {self.__class__.__name__} ready (gateway_mode={self.config.gateway_mode!r})"
        )

    async def _connect_mcp(self) -> None:
        logger.info(f"Connecting to MCP: {self.config.mcp_url}")
        context_config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="session_id",
                    inject_into=["add_to_cart", "view_cart", "checkout"],
                )
            ],
            auto_capture_common=False,
        )
        self._mcp_server = MCPServerStreamableHttp(
            params={"url": self.config.mcp_url},
            client_session_timeout_seconds=self.config.mcp_timeout,
            context_config=context_config,
        )
        await self._mcp_server.connect()
        tool_defs = await MCPUtil.get_function_tools(self._mcp_server)
        self._tools = []
        for t in tool_defs:
            if isinstance(t, dict):
                self._tools.append(t)
            elif hasattr(t, "model_dump"):
                self._tools.append(t.model_dump())
            else:
                self._tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(t, "name", str(t)),
                            "description": getattr(t, "description", ""),
                            "parameters": getattr(t, "parameters", {}),
                        },
                    }
                )
        # Strip injected parameters from schemas so the LLM never sees them
        # as required fields and doesn't ask the user for values the executor provides.
        _injected = {"session_id"}
        for tool_def in self._tools:
            fn = tool_def.get("function", {})
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            for p in _injected:
                props.pop(p, None)
            params["required"] = [r for r in params.get("required", []) if r not in _injected]
        self._tool_executor = ToolExecutor({self._mcp_server: None})
        await self._tool_executor.initialize()
        names = [t.get("function", {}).get("name", "?") for t in self._tools]
        logger.info(f"✓ {len(self._tools)} tools: {', '.join(names)}")

    def _build_workflow(self) -> None:
        raise NotImplementedError

    async def chat(self, message: str, user_id: str, conversation_id: str) -> str:
        if not self._initialized:
            await self.initialize()

        namespace = self._mcp_server.name if self._mcp_server else "gateway-multi-agent-shop"
        cart_session_id = f"{user_id}:{conversation_id}"
        if self._tool_executor:
            self._tool_executor.context_state.set(namespace, "session_id", cart_session_id)

        session_id = None
        if self._container and self.config.enable_session:
            sc = self._container.session_client
            if sc and sc.is_enabled:
                try:
                    session_id = await sc.get_or_create_session(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    if (
                        self._runner
                        and hasattr(self._runner, "_session_service")
                        and self._runner._session_service
                    ):
                        existing = await self._runner._session_service.load_tool_context_state(
                            session_id
                        )
                        existing.set(namespace, "session_id", cart_session_id)
                        await self._runner._session_service.save_tool_context_state(
                            session_id, existing
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
        if self._mcp_server:
            try:
                await self._mcp_server.cleanup()
            except Exception:
                pass
        if self._lifecycle:
            await self._lifecycle.shutdown()

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools


# =============================================================================
# 1. Sequential
# =============================================================================


class SequentialShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        self._agent = SequentialAgent(
            name="sequential-shop",
            agents=[
                make_search_agent(self._tools, self._tool_executor, m, gm),
                make_recommend_agent(m, gm),
                make_cart_agent(self._tools, self._tool_executor, m, gm),
                make_summary_agent(m, gm),
            ],
            sequential_config=SequentialConfig(
                pass_full_history=True,
                pipeline_context_max_chars=None,
            ),
        )


# =============================================================================
# 2. Parallel
# =============================================================================


@dataclass
class ParallelCoordinatorAgent(BaseAgent):
    synthesiser: BaseAgent | None = None
    parallel: ParallelAgent | None = None

    async def execute(
        self, input_text: str, runner: Any, context: Any, llm_client: Any = None
    ) -> AgentResponse:
        from orchestrator.agent.utils.context_utils import create_run_context

        context.suppress_session_log = True
        parallel_ctx = create_run_context(
            user_id=context.user_id,
            conversation_id=context.conversation_id,
        )
        parallel_result = await self.parallel.execute(input_text, runner, parallel_ctx)

        synthesis_input = (
            f"User asked: {input_text}\n\n"
            f"Search results:\n{parallel_result.content}\n\n"
            f"Write a clear, concise response grouped by animal type."
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


class ParallelShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = (
            self.config.enable_memory and memory_client is not None and memory_client.is_enabled
        )

        dog_searcher = BaseAgent(
            name="dog-search-agent",
            instructions=(
                "Search for dog products only. "
                "Use search_products with animal='dog'. "
                "Return a clear list of results with IDs and prices."
            ),
            model=m,
            gateway_mode=gm,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False, session_history_turns=0),
        )
        cat_searcher = BaseAgent(
            name="cat-search-agent",
            instructions=(
                "Search for cat products only. "
                "Use search_products with animal='cat'. "
                "Return a clear list of results with IDs and prices."
            ),
            model=m,
            gateway_mode=gm,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=False, session_history_turns=0),
        )
        parallel = ParallelAgent(
            name="parallel-inner",
            agents=[dog_searcher, cat_searcher],
            model=m,
            gateway_mode=gm,
            parallel_config=ParallelConfig(
                merge_strategy=MergeStrategy.LLM_SUMMARIZE,
                summary_prompt=(
                    "Combine the dog and cat product results into one short response. "
                    "List only real products with their IDs and prices. "
                    "Group by animal type. Maximum 6 bullet points total."
                ),
            ),
        )
        synthesiser = BaseAgent(
            name="parallel-synthesiser",
            instructions=(
                "You are a pet shop assistant. "
                "Given search results and the user's request, write a clear, concise response "
                "grouped by animal type. Use product IDs and prices."
            ),
            model=m,
            gateway_mode=gm,
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
            name="parallel-shop",
            synthesiser=synthesiser,
            parallel=parallel,
        )


# =============================================================================
# 3. Loop
# =============================================================================


class LoopShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        searcher = BaseAgent(
            name="budget-search-agent",
            instructions=(
                "Search for pet products matching the user's request. "
                "If you find a product matching all criteria (including any budget constraint), "
                "start your response with 'FOUND:' followed by the product details. "
                "If not found yet, describe what you tried and suggest a refined search."
            ),
            model=m,
            gateway_mode=gm,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=True),
        )
        self._agent = LoopAgent(
            name="loop-shop",
            agent=searcher,
            termination=TerminationConfig(
                type=TerminationType.OUTPUT_MATCH,
                pattern="FOUND:",
                max_iterations=5,
            ),
        )


# =============================================================================
# 4. Scatter
# =============================================================================


@dataclass
class ScatterCoordinatorAgent(BaseAgent):
    coordinator: BaseAgent | None = None
    scatter: ScatterAgent | None = None

    async def execute(
        self, input_text: str, runner: Any, context: Any, llm_client: Any = None
    ) -> AgentResponse:
        from orchestrator.agent.utils.context_utils import create_run_context
        from orchestrator.config import settings
        from orchestrator.llm.config import LLMConfig

        if llm_client is None:
            try:
                from orchestrator.core.container import get_container

                llm_client = get_container().llm_client
            except Exception:
                llm_client = None

        import json as _json

        n_agents = len(self.scatter.agents)
        resolved_task = input_text
        input_slices: list[str] | None = None

        if llm_client:
            try:
                history_msgs: list[dict] = []
                if (
                    context.session_id
                    and hasattr(runner, "_session_service")
                    and runner._session_service
                ):
                    history = await runner._session_service.get_conversation_history(
                        context.session_id, limit=6
                    )
                    history_msgs = history or []

                system_prompt = (
                    f"You are a task router for a pet shop analyst system with {n_agents} analyst agents.\n"
                    "Read the conversation history and the user's message.\n"
                    "Output a JSON object with two keys:\n"
                    '  "task": one sentence summarising what the user wants to compare\n'
                    f'  "slices": array of exactly {n_agents} strings, ONE PER PRODUCT — not one per analytical dimension\n'
                    "Each slice: 'Fetch and assess [product name] ([price]) for value, quality, and suitability.'\n"
                    "Resolve vague words ('those', 'them', 'the first one') to explicit product IDs using history.\n"
                    "Output ONLY valid JSON — no markdown, no extra text.\n"
                    "Example for 3 products:\n"
                    '{"task": "Compare dog food p1 ($29.99), cat food p2 ($18.99), dog toy p5 ($6.99) for value.",\n'
                    ' "slices": [\n'
                    '   "Fetch and assess dog food p1 ($29.99) for value, quality, and suitability.",\n'
                    '   "Fetch and assess cat food p2 ($18.99) for value, quality, and suitability.",\n'
                    '   "Fetch and assess dog toy p5 ($6.99) for value, quality, and suitability."\n'
                    " ]}"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    *history_msgs,
                    {"role": "user", "content": input_text},
                ]
                model = (
                    self.coordinator.model if self.coordinator else None
                ) or settings.default_llm_model
                response = await llm_client.chat(
                    messages=messages,
                    config=LLMConfig(model=model, max_tokens=800),
                    auto_session=False,
                )
                content = (response.content or "").strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                parsed = _json.loads(content)
                resolved_task = parsed.get("task", input_text)
                input_slices = parsed.get("slices")
                logger.info(f"ScatterCoordinator task: '{resolved_task}'")
                logger.info(f"ScatterCoordinator slices: {input_slices}")
            except Exception as e:
                logger.warning(f"ScatterCoordinator routing failed ({e}), using raw input")

        _orig_slices = self.scatter.input_slices
        if input_slices:
            self.scatter.input_slices = input_slices
        try:
            scatter_ctx = create_run_context(
                user_id=context.user_id,
                conversation_id=context.conversation_id,
            )
            scatter_result = await self.scatter.execute(resolved_task, runner, scatter_ctx)
        finally:
            self.scatter.input_slices = _orig_slices

        if context.session_id:
            await runner.save_turn(
                session_id=context.session_id,
                user_message=input_text,
                assistant_message=scatter_result.content or "",
                agent=None,
            )

        return AgentResponse(
            content=scatter_result.content,
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
            usage=scatter_result.usage,
        )


class ScatterShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = (
            self.config.enable_memory and memory_client is not None and memory_client.is_enabled
        )

        analysts = [make_analyst_agent(m, gm) for _ in range(3)]
        for i, a in enumerate(analysts, 1):
            a.name = f"analyst-agent-{i}"

        scatter = ScatterAgent(name="scatter-inner", agents=analysts, model=m, gateway_mode=gm)

        coordinator = BaseAgent(
            name="scatter-coordinator",
            instructions=(
                "You are a task router. "
                "Read the user's message and conversation history, then output ONE sentence "
                "that restates the request explicitly for product analysts. "
                "Rules: "
                "1. Never greet, never ask questions, never say you need more info. "
                "2. Always resolve vague words ('those', 'them', 'the first one') to explicit "
                "   product IDs using conversation history. "
                "3. If products have prices in the message, include them. "
                "4. Output exactly one sentence — nothing else."
            ),
            model=m,
            gateway_mode=gm,
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )
        self._agent = ScatterCoordinatorAgent(
            name="scatter-shop",
            coordinator=coordinator,
            scatter=scatter,
        )


# =============================================================================
# 5. Supervised
# =============================================================================


class SupervisedShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        self._agent = SupervisedSequentialAgent(
            name="supervised-shop",
            agents=[make_writer_agent(m, gm)],
            supervised_config=SupervisedConfig(
                quality_threshold=0.7,
                max_retries=2,
                supervisor_model=m,
                pipeline_context_max_chars=None,
            ),
        )


# =============================================================================
# 6. Planner
# =============================================================================


class PlannerShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        self._agent = PlannerAgent(
            name="planner-shop",
            model=m,
            gateway_mode=gm,
            agents=[
                make_search_agent(self._tools, self._tool_executor, m, gm),
                make_recommend_agent(m, gm),
                make_writer_agent(m, gm),
            ],
            planning_config=PlanningConfig(
                max_steps=6,
                enable_replanning=False,
            ),
        )


# =============================================================================
# 7. Debate
# =============================================================================


class DebateShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        no_memory = AgentMemoryConfig(search_memories=False, store_memories=False)
        cfg = AgentConfig(log_to_session=True)

        self._agent = DebateAgent(
            name="debate-shop",
            model=m,
            gateway_mode=gm,
            pro_agent=BaseAgent(
                name="pro-premium",
                instructions=(
                    "You are arguing FOR premium dog food. "
                    "Make a strong case: better nutrition, longer life, fewer vet bills. "
                    "Be persuasive and specific."
                ),
                model=m,
                gateway_mode=gm,
                memory_config=no_memory,
                config=cfg,
            ),
            con_agent=BaseAgent(
                name="pro-budget",
                instructions=(
                    "You are arguing FOR budget dog food. "
                    "Make a strong case: meets nutritional standards, costs less, dogs are happy. "
                    "Be persuasive and specific."
                ),
                model=m,
                gateway_mode=gm,
                memory_config=no_memory,
                config=cfg,
            ),
            judge_agent=BaseAgent(
                name="food-judge",
                instructions=(
                    "You are an impartial pet nutrition expert. "
                    "Read both arguments and give the owner a clear, practical recommendation "
                    "that acknowledges their budget and their dog's needs."
                ),
                model=m,
                gateway_mode=gm,
                memory_config=no_memory,
                config=cfg,
            ),
        )


# =============================================================================
# 8. Reflection
# =============================================================================


class ReflectionShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        self._agent = ReflectionAgent(
            name="reflection-shop",
            agent=make_writer_agent(m, gm),
            reflection_config=ReflectionConfig(
                max_reflections=2,
                critique_prompt=(
                    "Evaluate this pet product recommendation email. "
                    "Check: is it friendly, specific, includes a product ID, and under 150 words? "
                    "If all criteria are met respond with 'PASS'. "
                    "Otherwise respond with 'NEEDS IMPROVEMENT: ' and the specific issue."
                ),
                reflection_model=m,
            ),
        )


# =============================================================================
# 9. Router
# =============================================================================


class RouterShop(_BaseWorkflow):
    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        search = make_search_agent(self._tools, self._tool_executor, m, gm)
        cart = make_cart_agent(self._tools, self._tool_executor, m, gm)
        support = make_support_agent(m, gm)

        for specialist in (search, cart, support):
            specialist.config.session_history_turns = 0

        self._agent = RouterAgent(
            name="router-shop",
            model=m,
            gateway_mode=gm,
            routes=[
                Route(
                    agent_name="search-agent",
                    description="search products, browse catalogue, find items by name or category",
                ),
                Route(
                    agent_name="cart-agent",
                    description="add to cart, view cart, checkout, place order",
                ),
                Route(
                    agent_name="support-agent",
                    description="pet care advice, nutrition questions, general help, greetings, unclear intent",
                ),
            ],
            fallback_agent_name="support-agent",
            router_config=RouterConfig(routing_strategy="llm"),
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(log_to_session=True),
        )
        self._specialist_agents = {
            "search-agent": search,
            "cart-agent": cart,
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
                agent_name = self._agent.fallback_agent_name or "search-agent"

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
# 10. Handoff
# =============================================================================


class HandoffShop(_BaseWorkflow):
    _use_direct_execute = False

    def _build_workflow(self) -> None:
        m, gm = self.config.model, self.config.gateway_mode
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = (
            self.config.enable_memory and memory_client is not None and memory_client.is_enabled
        )
        user_memory = AgentMemoryConfig(
            search_memories=memory_enabled,
            store_memories=memory_enabled,
            search_scope=AgentMemoryScope.USER,
            store_scope=AgentMemoryScope.USER,
            search_limit=5,
        )
        no_memory = AgentMemoryConfig(search_memories=False, store_memories=False)

        self._executor_agent = BaseAgent(
            name="handoff-executor",
            instructions=(
                "You are a pet shop executor. "
                "Use the available tools to search products, manage carts, and checkout. "
                "Return a clear, complete result."
            ),
            model=m,
            gateway_mode=gm,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=no_memory,
        )

        self._agent = BaseAgent(
            name="handoff-orchestrator",
            instructions=(
                "You are a pet shop orchestrator. "
                "Understand what the user wants, then hand off to the executor to perform the action. "
                "After the executor returns, summarise the result clearly for the user."
            ),
            model=m,
            gateway_mode=gm,
            tools=[],
            tool_executor=self._tool_executor,
            handoffs=[
                Handoff(
                    target_agent="handoff-executor",
                    description=(
                        "Hand off to the executor to search products, add to cart, view cart, or checkout."
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
        self._runner.register_agent(self._executor_agent)


# =============================================================================
# Factory
# =============================================================================

MODES: dict[str, type[_BaseWorkflow]] = {
    "sequential": SequentialShop,
    "parallel": ParallelShop,
    "loop": LoopShop,
    "scatter": ScatterShop,
    "supervised": SupervisedShop,
    "planner": PlannerShop,
    "debate": DebateShop,
    "reflection": ReflectionShop,
    "router": RouterShop,
    "handoff": HandoffShop,
}


def create_workflow(mode: str, config: WorkflowShopConfig | None = None) -> _BaseWorkflow:
    cls = MODES.get(mode)
    if not cls:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
    return cls(config)
