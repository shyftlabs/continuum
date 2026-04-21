"""
Local Shop Agent.

Single agent using MCPServerStreamableHttp (HTTP transport) — same pattern as commerce-chat
but against a local MCP server instead of the remote one.
"""

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
    MCPUtil,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from orchestrator.agent.types import generate_run_id
from orchestrator.core.container import Container, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager

logger = get_logger(__name__)


class LocalShopAgent:
    def __init__(self, config: ShopConfig | None = None):
        self.config = config or default_config
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[dict[str, Any]] = []
        self._initialized = False
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
        logger.info("✓ LocalShopAgent ready!")

    async def _connect_mcp(self) -> None:
        logger.info(f"Connecting to MCP server: {self.config.mcp_url}")
        self._mcp_server = MCPServerStreamableHttp(
            params={"url": self.config.mcp_url},
            client_session_timeout_seconds=self.config.mcp_timeout,
        )
        await self._mcp_server.connect()

        tool_definitions = await MCPUtil.get_function_tools(self._mcp_server)
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

        names = [t.get("function", {}).get("name", "?") for t in self._tools if isinstance(t, dict)]
        logger.info(f"✓ Discovered {len(self._tools)} tools: {', '.join(names)}")

        self._tool_executor = ToolExecutor({self._mcp_server: None})
        await self._tool_executor.initialize()

    def _create_agent(self) -> None:
        memory_client = self._container.memory_client if self._container else None
        memory_enabled = self.config.enable_memory and memory_client is not None and memory_client.is_enabled

        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=self.config.system_instructions,
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            tools=self._tools,
            tool_executor=self._tool_executor,
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
            ),
        )

    async def chat(self, message: str, user_id: str, conversation_id: str) -> str:
        if not self._initialized:
            await self.initialize()
            
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
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools


async def create_shop_agent() -> LocalShopAgent:
    agent = LocalShopAgent()
    await agent.initialize()
    return agent
