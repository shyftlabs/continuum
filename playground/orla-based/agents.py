"""
Agent definitions for the orla-based playground.

Demonstrates:
- RouterAgent routing free vs premium users, stamping dispatch_priority
- FreeAgent / PremiumAgent with different tool access (policy-enforced)
- SummarizerAgent with low stage_priority
- DAGAgent parallel pipeline: search + recommend → synthesize → reply
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    Route,
    RouterAgent,
    RunnerConfig,
    ToolExecutor,
)
from orchestrator.agent.config import RouterConfig
from orchestrator.agent.workflow.dag import create_dag_agent
from orchestrator.core.container import get_container
from orchestrator.core.lifecycle import get_lifecycle_manager
from orchestrator.tools.util import MCPUtil

from config import AppConfig, default_config
from tools import build_tool_server


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

class OrlaPlayground:
    """Wires all agents, tools, dispatcher, and policy store together."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or default_config
        self._initialized = False
        self.runner: AgentRunner | None = None
        self.router: RouterAgent | None = None
        self.free_agent: BaseAgent | None = None
        self.premium_agent: BaseAgent | None = None
        self.summarizer: BaseAgent | None = None
        self.dag_agent = None
        self.tool_executor: ToolExecutor | None = None
        self._tools: list[dict] = []
        self.container = None

    async def initialize(self) -> None:
        if self._initialized:
            return

        lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=False,
            enable_signal_handlers=False,
        )
        await lifecycle.initialize()
        self.container = get_container()
        container = self.container

        memory_client = container.memory_client if container else None
        memory_enabled = (
            self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled
        )

        # --- In-process MCP tool server ---
        tool_server = build_tool_server()
        await tool_server.connect()

        tool_definitions = await MCPUtil.get_function_tools(tool_server)
        self._tools = []
        for tool in tool_definitions:
            if isinstance(tool, dict):
                self._tools.append(tool)
            elif hasattr(tool, "model_dump"):
                self._tools.append(tool.model_dump())
            else:
                self._tools.append({
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", str(tool)),
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "parameters", {}),
                    },
                })

        self.tool_executor = ToolExecutor({tool_server: None})
        await self.tool_executor.initialize()

        # --- Agents ---
        self.free_agent = BaseAgent(
            name="free-agent",
            instructions=(
                "You are a pet shop assistant for free-tier users. "
                "Help users search for products and manage their cart. "
                "You cannot process checkouts — when a user tries to checkout, tell them it is "
                "only available on premium tier and instruct them to type '/tier premium' to upgrade."
            ),
            model=self.config.model,
            temperature=self.config.temperature,
            tools=self._tools,
            tool_executor=self.tool_executor,
            policy_store=self.config.policy_store,
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(
                max_turns=self.config.max_turns,
                log_to_session=self.config.enable_session,
                stage_priority=5,
            ),
        )

        self.premium_agent = BaseAgent(
            name="premium-agent",
            instructions=(
                "You are a pet shop assistant for premium users. "
                "You have full access: search, cart management, and checkout. "
                "Be helpful and efficient."
            ),
            model=self.config.model,
            temperature=self.config.temperature,
            tools=self._tools,
            tool_executor=self.tool_executor,
            policy_store=self.config.policy_store,
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(
                max_turns=self.config.max_turns,
                log_to_session=self.config.enable_session,
                stage_priority=5,
            ),
        )

        self.summarizer = BaseAgent(
            name="summarizer-agent",
            instructions=(
                "Summarize the conversation so far in one sentence. "
                "Focus on what the user was looking for and any actions taken."
            ),
            model=self.config.model,
            temperature=0.3,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(
                max_turns=2,
                log_to_session=False,
                stage_priority=2,
            ),
        )

        # --- DAG debug hooks ---
        def _dag_on_start(agent, data):
            text = str(data.get("input", ""))
            print(f"\n  ┌─ [{agent.name}] INPUT {'─' * max(0, 40 - len(agent.name))}")
            print(f"  │  {text[:300].replace(chr(10), chr(10) + '  │  ')}")
            print(f"  └{'─' * 43}")

        def _dag_on_end(agent, data):
            resp = data.get("response")
            text = str(getattr(resp, "content", resp) or "")
            print(f"\n  ┌─ [{agent.name}] OUTPUT {'─' * max(0, 39 - len(agent.name))}")
            print(f"  │  {text[:300].replace(chr(10), chr(10) + '  │  ')}")
            print(f"  └{'─' * 43}")

        # --- DAG pipeline agents ---
        fetch_agent = BaseAgent(
            name="fetch-agent",
            instructions="Search for products matching the user's query. Return a list of relevant products.",
            model=self.config.model,
            tools=self._tools,
            tool_executor=self.tool_executor,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(max_turns=3, log_to_session=False, stage_priority=7),
            on_start=_dag_on_start, # comment this out to disable debug logging
            on_end=_dag_on_end, # comment this out to disable debug logging
        )

        recommend_agent = BaseAgent(
            name="recommend-agent",
            instructions="Based on the user's query, suggest complementary products they might also need.",
            model=self.config.model,
            tools=self._tools,
            tool_executor=self.tool_executor,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(max_turns=3, log_to_session=False, stage_priority=7),
            on_start=_dag_on_start, # comment this out to disable debug logging
            on_end=_dag_on_end, # comment this out to disable debug logging
        )

        synthesize_agent = BaseAgent(
            name="synthesize-agent",
            instructions=(
                "You receive search results and recommendations. "
                "Combine them into a single coherent product overview for the user."
            ),
            model=self.config.model,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(max_turns=2, log_to_session=False, stage_priority=8),
            on_start=_dag_on_start, # comment this out to disable debug logging
            on_end=_dag_on_end, # comment this out to disable debug logging
        )

        reply_agent = BaseAgent(
            name="reply-agent",
            instructions=(
                "You receive a product overview. Write a friendly, concise reply to the user "
                "highlighting the best options and next steps."
            ),
            model=self.config.model,
            memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
            config=AgentConfig(max_turns=2, log_to_session=False, stage_priority=9),
            on_start=_dag_on_start, # comment this out to disable debug logging
            on_end=_dag_on_end, # comment this out to disable debug logging
        )

        # fetch and recommend run in parallel; synthesize waits for both; reply waits for synthesize
        self.dag_agent = create_dag_agent(
            name="shop-dag",
            stages=[
                ("fetch",      fetch_agent,      []),
                ("recommend",  recommend_agent,  []),
                ("synthesize", synthesize_agent, ["fetch", "recommend"]),
                ("reply",      reply_agent,      ["synthesize"]),
            ],
        )

        # --- Router ---
        self.router = RouterAgent(
            name="triage-router",
            instructions=(
                "Route the user to the correct agent based on their tier. "
                "Use 'free-agent' for free users and 'premium-agent' for premium users."
            ),
            routes=[
                Route(
                    agent_name="free-agent",
                    description="Free tier users — basic shopping, no checkout",
                    dispatch_priority=2,
                ),
                Route(
                    agent_name="premium-agent",
                    description="Premium tier users — full access including checkout",
                    dispatch_priority=9,
                ),
            ],
            router_config=RouterConfig(routing_strategy="rule_based"),
        )

        # --- Runner ---
        self.runner = AgentRunner(
            container=container,
            tool_executor=self.tool_executor,
            config=RunnerConfig(
                persist_state=False,
                default_max_turns=self.config.max_turns,
            ),
        )
        self.runner.register_agent(self.free_agent)
        self.runner.register_agent(self.premium_agent)
        self.runner.register_agent(self.summarizer)

        self._initialized = True

    @property
    def tools(self) -> list[dict]:
        return self._tools
