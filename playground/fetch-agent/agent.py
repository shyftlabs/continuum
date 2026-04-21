"""
Fetch Agent.

An agent that connects to the mcp-server-fetch MCP server via stdio,
allowing it to fetch and read web pages.
"""

import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import FetchAgentConfig, default_config

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    BaseAgent,
    MCPUtil,
    RunnerConfig,
    ToolExecutor,
    get_logger,
)
from orchestrator.agent.types import generate_run_id
from orchestrator.core.container import Container, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager
from orchestrator.tools.mcp import MCPServerStdio

logger = get_logger(__name__)


class FetchAgent:
    """
    Fetch Agent.

    An agent that uses the mcp-server-fetch MCP server to fetch web pages.
    """

    def __init__(self, config: FetchAgentConfig | None = None):
        self.config = config or default_config

        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None

        self._mcp_server: MCPServerStdio | None = None
        self._tool_executor: ToolExecutor | None = None

        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[dict[str, Any]] = []

        self._initialized = False
        self._current_session_id: str | None = None
        self._current_user_id: str | None = None

    async def initialize(self, user_id: str | None = None) -> None:
        if self._initialized:
            return

        logger.info("Initializing Fetch Agent...")

        self._current_user_id = user_id or f"user_{generate_run_id()[-8:]}"

        # Initialize lifecycle
        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=True,
        )
        init_result = await self._lifecycle.initialize()

        if not init_result.success:
            logger.warning(f"Lifecycle initialization had issues: {init_result.errors}")
        else:
            logger.info("✓ OrchestratorLifecycle initialized")

        # Get container
        self._container = get_container()
        logger.info("✓ Container initialized")

        # Create session
        session_client = self._container.session_client
        if session_client and session_client.is_enabled:
            try:
                self._current_session_id = await session_client.get_or_create_session(
                    user_id=self._current_user_id,
                    conversation_id=self.config.agent_name,
                )
                logger.info(f"✓ Session initialized: {self._current_session_id}")
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")

        # Connect to MCP server
        await self._connect_mcp()

        # Create agent
        self._create_agent()

        # Create runner
        self._runner = AgentRunner(
            container=self._container,
            tool_executor=self._tool_executor,
            config=RunnerConfig(
                persist_state=False,
                default_max_turns=self.config.max_turns,
            ),
        )
        self._runner.register_agent(self._agent)

        self._initialized = True
        logger.info("✓ Fetch Agent ready!")

    async def _connect_mcp(self) -> None:
        """Connect to mcp-server-fetch via stdio."""
        logger.info(f"Starting MCP server: {self.config.mcp_command} {' '.join(self.config.mcp_args)}")

        try:
            self._mcp_server = MCPServerStdio(
                {
                    "command": self.config.mcp_command,
                    "args": self.config.mcp_args,
                }
            )

            await self._mcp_server.connect()
            logger.info("✓ Connected to mcp-server-fetch")

            # Discover tools
            tool_definitions = await MCPUtil.get_function_tools(self._mcp_server)

            self._tools = []
            for tool in tool_definitions:
                if isinstance(tool, dict):
                    self._tools.append(tool)
                elif hasattr(tool, "model_dump"):
                    self._tools.append(tool.model_dump())
                elif hasattr(tool, "to_dict"):
                    self._tools.append(tool.to_dict())
                else:
                    self._tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": getattr(tool, "name", str(tool)),
                                "description": getattr(tool, "description", ""),
                                "parameters": getattr(tool, "parameters", {}),
                            },
                        }
                    )

            tool_names = [t.get("function", {}).get("name", "?") for t in self._tools if isinstance(t, dict)]
            logger.info(f"✓ Discovered {len(self._tools)} tools: {', '.join(tool_names)}")

            self._tool_executor = ToolExecutor({self._mcp_server: None})
            await self._tool_executor.initialize()

        except Exception as e:
            logger.error(f"Failed to connect to MCP: {e}")
            raise

    def _create_agent(self) -> None:
        memory_client = self._container.memory_client if self._container else None

        memory_config = AgentMemoryConfig(
            search_memories=self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled,
            store_memories=self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled,
            search_scope=AgentMemoryScope.CONVERSATION,
            store_scope=AgentMemoryScope.CONVERSATION,
        )

        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=self.config.system_instructions,
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            tools=self._tools,
            tool_executor=self._tool_executor,
            memory_config=memory_config,
            config=AgentConfig(
                max_turns=self.config.max_turns,
                log_to_session=self.config.enable_session,
            ),
        )

        logger.info(f"✓ Agent created with {len(self._tools)} tools")

    async def chat(self, message: str) -> str:
        if not self._initialized:
            await self.initialize()

        try:
            response = await self._runner.run(
                agent=self._agent,
                input=message,
                session_id=self._current_session_id,
                user_id=self._current_user_id,
            )
            return response.content
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error: {str(e)}"

    async def get_memories(self, limit: int = 20) -> list[dict]:
        memory_client = self._container.memory_client if self._container else None
        if not memory_client or not memory_client.is_enabled:
            return []
        try:
            memories = await memory_client.get_all(
                user_id=self._current_user_id,
                conversation_id=self._current_session_id,
                limit=limit,
            )
            return [m.to_dict() for m in memories]
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []

    async def close(self) -> None:
        # Clean up MCP resources in the current task before lifecycle shutdown,
        # because lifecycle.shutdown() wraps callbacks in asyncio.gather (new tasks),
        # which causes anyio's cancel scope to throw a cross-task RuntimeError.
        if self._mcp_server:
            try:
                await self._mcp_server.cleanup()
            except Exception:
                pass
        if self._lifecycle:
            await self._lifecycle.shutdown()
            logger.info("✓ Cleanup complete")

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    @property
    def session_id(self) -> str | None:
        return self._current_session_id

    @property
    def user_id(self) -> str | None:
        return self._current_user_id


async def create_fetch_agent(user_id: str | None = None) -> FetchAgent:
    agent = FetchAgent()
    await agent.initialize(user_id)
    return agent
