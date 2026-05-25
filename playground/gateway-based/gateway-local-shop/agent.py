"""
Local Shop Agent.

Single agent using MCPServerStreamableHttp (HTTP transport) — same pattern as commerce-chat
but against a local MCP server instead of the remote one.
"""

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import ShopConfig, default_config

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    MCPServerStreamableHttp,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from orchestrator.tools.tool_attention.config import ToolAttentionConfig
from orchestrator.tools.types import ToolContextConfig, ToolContextVariable
from orchestrator.agent.types import generate_run_id
from orchestrator.core.container import Container, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager

logger = get_logger(__name__)

_CART_TOOLS = {"get_cart", "cart", "get_cart_items"}
_TOTAL_KEYS = {"total", "subtotal", "total_cents", "subtotal_cents", "taxes", "tax_cents"}


class CartDebugToolExecutor(ToolExecutor):
    """ToolExecutor subclass that adds cart-specific debug logging after each tool call."""

    def _on_tool_result(self, tool_name: str, result: str, artifact: Any) -> None:
        sc = artifact.structured_content if artifact else None

        # Log totals from structuredContent for any tool that returns them
        if sc:
            total_fields = {k: v for k, v in sc.items() if k in _TOTAL_KEYS}
            if total_fields:
                logger.info(f"💰 {tool_name} structuredContent totals: {total_fields}")

            # Log cart item count and sample price fields
            items = sc.get("items") or sc.get("cart_items")
            if isinstance(items, list) and items:
                sample = {k: v for k, v in items[0].items()
                          if "price" in k.lower() or "total" in k.lower()}
                logger.info(
                    f"🛒 {tool_name} cart: items_count={len(items)}, "
                    f"sample_price_fields={sample}, "
                    f"structuredContent_keys={list(sc.keys())}"
                )

        # Extra detail for known cart tools
        if tool_name in _CART_TOOLS:
            if sc:
                llm_totals = {k: v for k, v in sc.items()
                              if "total" in k.lower() or "subtotal" in k.lower() or "tax" in k.lower()}
                if llm_totals:
                    logger.info(f"📤 {tool_name} sending to LLM (structuredContent): totals={llm_totals}")
                else:
                    logger.warning(
                        f"⚠️ {tool_name} structuredContent has NO totals for LLM! "
                        f"Keys: {list(sc.keys())}"
                    )


class LocalShopAgent:
    def __init__(self, config: ShopConfig | None = None):
        self.config = config or default_config
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[Any] = []
        self._resource_context: str = ""
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False, 
            verify_connections=True,
            enable_signal_handlers=False
        )
        await self._lifecycle.initialize()

        self._container = get_container()

        await self._connect_mcp()
        self._create_agent()

        self._runner = AgentRunner(
            container=self._container,
            tool_executor=self._tool_executor,
            config=RunnerConfig(persist_state=False, default_max_turns=self.config.max_turns),
        )
        self._runner.register_agent(self._agent)
        self._initialized = True
        logger.info(f"✓ LocalShopAgent ready! (llm model used: {self.config.agent_model})")

    async def _connect_mcp(self) -> None:
        logger.info(f"Connecting to MCP server: {self.config.mcp_url}")

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

        self._tool_executor = CartDebugToolExecutor({self._mcp_server: None})
        await self._tool_executor.initialize()

        self._tools = self._tool_executor.get_tool_definitions()
        # Strip injected parameters from schemas so the LLM never sees them as
        # required fields and doesn't ask the user for values the executor provides.
        _injected = {"session_id"}
        for tool_def in self._tools:
            params = tool_def.function.parameters or {}
            props = params.get("properties", {})
            for p in _injected:
                props.pop(p, None)
            params["required"] = [r for r in params.get("required", []) if r not in _injected]

        names = [t.function.name for t in self._tools]
        logger.info(f"✓ Discovered {len(self._tools)} tools: {', '.join(names)}")

        await self._fetch_resources()

    async def _fetch_resources(self) -> None:
        try:
            catalogue = await self._mcp_server.read_resource("shop://catalogue")
            categories = await self._mcp_server.read_resource("shop://categories")
            self._resource_context = (
                f"Product catalogue:\n{catalogue}\n\nCategories:\n{categories}"
            )
            logger.info("✓ Loaded shop resources (catalogue + categories)")
        except Exception as e:
            logger.warning(f"Could not load shop resources: {e}")

    def _create_agent(self) -> None:
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = self.config.enable_memory and memory_client is not None and memory_client.is_enabled

        instructions = self.config.system_instructions

        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=instructions,
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            gateway_mode=self.config.gateway_mode,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=AgentMemoryConfig(
                search_memories=memory_enabled,
                store_memories=memory_enabled,
                search_scope=AgentMemoryScope.USER,
                store_scope=AgentMemoryScope.USER,
                search_limit=5,
                extraction_prompt=(
                    "You are a memory extraction system for a pet shop assistant. "
                    "Only extract long-term facts about the user's pets, animal preferences, "
                    "favourite products, or dietary needs. "
                    "Do NOT store transient actions like adding to cart, checkout requests, "
                    "or one-off searches. Do NOT store unrelated personal facts (e.g. favourite colour)."
                ),
            ),
            config=AgentConfig(
                max_turns=self.config.max_turns,
                log_to_session=self.config.enable_session,
                # Tool-attention: local-shop has ~5 tools so set min_tools=3 to trigger.
                # k=3 means top-3 semantically relevant tools promoted each turn.
                tool_attention=ToolAttentionConfig(k=3, min_tools=3),
            ),
        )

    async def chat(self, message: str, user_id: str, conversation_id: str) -> str:
        if not self._initialized:
            await self.initialize()

        namespace = self._mcp_server.name if self._mcp_server else "local-shop"
        cart_session_id = f"{user_id}:{conversation_id}"

        # Seed in-memory context (works when session service is disabled).
        if self._tool_executor:
            self._tool_executor.context_state.set(namespace, "session_id", cart_session_id)

        session_id = None
        if self._container:
            session_client = self._container.session_client
            if session_client and session_client.is_enabled:
                try:
                    session_id = await session_client.get_or_create_session(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    logger.info(f"✓ Active Session ID: {session_id}")
                    # The runner loads tool context from Redis and overwrites the
                    # in-memory context_state, so we must also persist cart_session_id
                    # to Redis before run() is called.
                    existing = await self._runner._session_service.load_tool_context_state(
                        session_id
                    )
                    existing.set(namespace, "session_id", cart_session_id)
                    await self._runner._session_service.save_tool_context_state(
                        session_id, existing
                    )
                except Exception as e:
                    logger.warning(f"Session init failed for user {user_id}: {e}")

        try:
            response = await self._runner.run(
                agent=self._agent,
                input=message,
                session_id=session_id,
                user_id=user_id,
            )
            return response.content or ""
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"

    async def close(self) -> None:
        if self._mcp_server:
            try:
                await self._mcp_server.cleanup()
            except Exception:
                pass
        if self._lifecycle:
            await self._lifecycle.shutdown()

    @property
    def tools(self) -> list[Any]:
        return self._tools


async def create_shop_agent() -> LocalShopAgent:
    agent = LocalShopAgent()
    await agent.initialize()
    return agent
