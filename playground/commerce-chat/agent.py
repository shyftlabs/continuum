"""
Petco Retail Agent.

A dynamic retail agent that handles all shopping interactions using MCP tools.
No specialist agents - single intelligent agent makes all decisions dynamically.
"""

import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import PetcoConfig, default_config

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentMemoryScope,
    AgentRunner,
    # Agent
    BaseAgent,
    CompressionStrategy,
    # Context Management
    ContextManagementConfig,
    # Tools - Use StreamableHttp for this MCP server
    MCPServerStreamableHttp,
    MCPUtil,
    RunnerConfig,
    ToolExecutor,
    # Logging
    get_logger,
)
from orchestrator.agent.types import generate_run_id
from orchestrator.core.container import Container, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager

logger = get_logger(__name__)


class PetcoRetailAgent:
    """
    Petco Retail Shopping Agent.

    A single intelligent agent that dynamically handles all shopping
    interactions using MCP tools. No specialist agents needed.

    Features:
    - Dynamic tool discovery from MCP server
    - Personalized recommendations via memory
    - Conversation history via sessions
    - Full Langfuse tracing
    """

    def __init__(self, config: PetcoConfig | None = None):
        """
        Initialize the Petco retail agent.

        Args:
            config: Configuration for the agent
        """
        self.config = config or default_config

        # Container and Lifecycle (for SDK services)
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None

        # MCP-specific resources (not in Container)
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._tool_executor: ToolExecutor | None = None

        # Agent components
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None
        self._tools: list[dict[str, Any]] = []

        # State
        self._initialized = False
        self._current_session_id: str | None = None
        self._current_user_id: str | None = None

    async def initialize(self, user_id: str | None = None) -> None:
        """
        Initialize all components.

        Args:
            user_id: Optional user ID for personalization
        """
        if self._initialized:
            return

        logger.info("Initializing Petco Retail Agent with Container and Lifecycle...")

        self._current_user_id = user_id or f"user_{generate_run_id()[-8:]}"

        # Initialize OrchestratorLifecycle for SDK services
        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,  # Don't fail if optional services are unavailable
            verify_connections=True,  # Verify connections at startup
        )
        init_result = await self._lifecycle.initialize()

        if not init_result.success:
            logger.warning(f"Lifecycle initialization had issues: {init_result.errors}")
        else:
            logger.info("✓ OrchestratorLifecycle initialized")
            if init_result.warnings:
                logger.info(f"Warnings: {init_result.warnings}")

        # Get Container (DI) for client management
        self._container = get_container()
        logger.info("✓ Container (DI) initialized")

        # Access clients from container
        llm_client = self._container.llm_client
        memory_client = self._container.memory_client
        session_client = self._container.session_client

        logger.info(f"✓ LLM client: {'available' if llm_client else 'not available'}")
        memory_status = (
            "available" if memory_client and memory_client.is_enabled else "not available"
        )
        logger.info(f"✓ Memory client: {memory_status}")
        session_status = (
            "available" if session_client and session_client.is_enabled else "not available"
        )
        logger.info(f"✓ Session client: {session_status}")

        # Create session if session client is available
        if session_client and session_client.is_enabled:
            try:
                self._current_session_id = await session_client.get_or_create_session(
                    user_id=self._current_user_id,
                    conversation_id=self.config.agent_name,
                )
                logger.info(f"✓ Session initialized: {self._current_session_id}")
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")

        # Connect to MCP server and discover tools
        await self._connect_mcp()

        # Create the agent
        self._create_agent()

        # Create runner using Container (DI)
        self._runner = AgentRunner(
            container=self._container,  # Use Container for client management
            tool_executor=self._tool_executor,  # MCP-specific, not in container
            config=RunnerConfig(
                persist_state=False,  # Using session for state
                default_max_turns=self.config.max_turns,
            ),
        )

        self._runner.register_agent(self._agent)
        logger.info("✓ Runner initialized with Container (DI)")

        # Tracing is handled by Container and Lifecycle
        if self._container.has_langfuse_client():
            logger.info("✓ Langfuse tracing enabled via Container")

        self._initialized = True
        logger.info("✓ Petco Retail Agent ready!")

    async def _connect_mcp(self) -> None:
        """Connect to MCP server and discover tools."""
        logger.info(f"Connecting to MCP server: {self.config.mcp_url}")

        try:
            # Create MCP StreamableHttp connection
            self._mcp_server = MCPServerStreamableHttp(
                {
                    "url": self.config.mcp_url,
                    "timeout": self.config.mcp_timeout,
                    "sse_read_timeout": self.config.mcp_sse_timeout,
                }
            )

            # Connect
            await self._mcp_server.connect()
            logger.info("✓ Connected to MCP server")

            # Discover tools - get as ToolDefinition dicts for LLM
            tool_definitions = await MCPUtil.get_function_tools(self._mcp_server)

            # Convert to dict format if needed
            self._tools = []
            for tool in tool_definitions:
                if isinstance(tool, dict):
                    self._tools.append(tool)
                elif hasattr(tool, "model_dump"):
                    # Pydantic model
                    self._tools.append(tool.model_dump())
                elif hasattr(tool, "to_dict"):
                    self._tools.append(tool.to_dict())
                else:
                    # Try to access as dict-like
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

            logger.info(f"✓ Discovered {len(self._tools)} tools from MCP")

            # Log tool names
            tool_names = []
            for t in self._tools:
                if isinstance(t, dict):
                    tool_names.append(t.get("function", {}).get("name", "?"))
                else:
                    tool_names.append(getattr(t, "name", "?"))
            logger.info(
                f"  Tools: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}"
            )

            # Create tool executor
            self._tool_executor = ToolExecutor({self._mcp_server: None})
            await self._tool_executor.initialize()

        except Exception as e:
            logger.error(f"Failed to connect to MCP: {e}")
            raise

    def _create_agent(self) -> None:
        """Create the retail agent with discovered tools."""
        # Get memory client from container
        memory_client = self._container.memory_client if self._container else None

        # Memory configuration
        # Use RUN scope for memory context - memories are retrieved in the context of the current session
        #
        # IMPORTANT: Memory isolation mode is set via MEMORY_ISOLATION environment variable.
        # Currently testing with MEMORY_ISOLATION=user mode.
        #
        # Mode behavior with RUN scope:
        # - MEMORY_ISOLATION=user + RUN scope:
        #   Uses user_id as primary identifier, NO session filter
        #   Memories are SHARED across all sessions for the same user (proper user continuity)
        #
        # - MEMORY_ISOLATION=run + RUN scope:
        #   Uses run_id as primary identifier, filters by session_id in metadata
        #   Memories are ISOLATED per session (session-level isolation)
        #
        # - MEMORY_ISOLATION=agent + RUN scope:
        #   Uses agent_id as primary identifier, NO session filter
        #   Memories are SHARED across all sessions for the same agent
        #
        # - MEMORY_ISOLATION=shared + RUN scope:
        #   Uses agent_id="shared", NO session filter
        #   Memories are SHARED across all sessions and users
        #
        # The SDK automatically adapts storage and retrieval based on the configured isolation mode.
        # RUN scope behavior depends on isolation mode: session filtering only when isolation="run".
        memory_config = AgentMemoryConfig(
            search_memories=self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled,
            store_memories=self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled,
            search_scope=AgentMemoryScope.CONVERSATION,  # Session-level isolation (per chat)
            store_scope=AgentMemoryScope.CONVERSATION,  # Session-level isolation (per chat)
            search_limit=self.config.memory_search_limit,
        )

        # Create agent
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
                # Enable context management for long conversations
                # Uses smart strategy: summarize old messages, fallback to truncation
                context_management=ContextManagementConfig(
                    enabled=True,
                    compression_threshold=0.8,  # Compress at 80% of context limit
                    compression_strategy=CompressionStrategy.SMART,  # Smart: summarize + truncate fallback
                    keep_recent_messages=10,  # Keep last 10 messages intact for continuity
                ),
            ),
        )

        logger.info(f"✓ Agent created with {len(self._tools)} tools")
        logger.info(
            f"✓ Context management enabled: {CompressionStrategy.SMART.value} strategy, "
            f"threshold=80%, keep_recent=10 messages"
        )

    async def chat(self, message: str) -> str:
        """
        Process a user message and return the response.

        Args:
            message: User's message

        Returns:
            Agent's response
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Run agent
            response = await self._runner.run(
                agent=self._agent,
                input=message,
                session_id=self._current_session_id,
                user_id=self._current_user_id,
            )

            # Log metrics
            if response.usage.total_tokens > 0:
                logger.debug(
                    f"Response generated: {response.usage.total_tokens} tokens, "
                    f"{response.latency_ms}ms, {response.turn_count} turns"
                )

            return response.content

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up Petco Retail Agent...")

        # Shutdown lifecycle (handles SDK services)
        if self._lifecycle:
            await self._lifecycle.shutdown()
            logger.info("✓ Cleanup complete")

    async def close(self) -> None:
        """Clean up resources (alias for cleanup for backward compatibility)."""
        await self.cleanup()

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get list of available tools."""
        return self._tools

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._current_session_id

    @property
    def user_id(self) -> str | None:
        """Get current user ID."""
        return self._current_user_id


async def create_petco_agent(
    user_id: str | None = None,
    config: PetcoConfig | None = None,
) -> PetcoRetailAgent:
    """
    Factory function to create and initialize a Petco retail agent.

    Args:
        user_id: Optional user ID for personalization
        config: Optional configuration

    Returns:
        Initialized PetcoRetailAgent
    """
    agent = PetcoRetailAgent(config)
    await agent.initialize(user_id)
    return agent
