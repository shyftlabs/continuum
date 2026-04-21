"""
Memory Modes Demo Agent.

Demonstrates the new provider-based memory architecture with all 4 isolation modes:
- shared: All memories accessible to all users/agents
- user: Memories isolated per user
- agent: Memories isolated per agent
- run: Memories isolated per session

Uses the new memory module features:
- MemoryScope for explicit scope management
- MemoryClient with provider abstraction
- Custom prompts for fact extraction
"""

import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config import MemoryModesConfig, default_config

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    AgentRunner,
    BaseAgent,
    LogLevel,
    get_logger,
    setup_logging,
)
from orchestrator.agent.types import MemoryScope as AgentMemoryScope
from orchestrator.agent.types import generate_run_id
from orchestrator.core.container import Container, get_container
from orchestrator.core.lifecycle import OrchestratorLifecycle, get_lifecycle_manager
from orchestrator.llm.context_management import CompressionStrategy, ContextManagementConfig
from orchestrator.memory import (
    MemoryAddResult,
    MemoryClient,
    MemorySearchResult,
)

logger = get_logger(__name__)


class MemoryModesDemoAgent:
    """
    Demo agent showcasing the new provider-based memory architecture.

    This agent demonstrates:
    - All 4 memory isolation modes (shared, user, agent, run)
    - MemoryScope for explicit scope management
    - Direct memory operations using MemoryClient
    - Custom prompts for fact extraction
    """

    def __init__(self, config: MemoryModesConfig | None = None):
        """
        Initialize the memory modes demo agent.

        Args:
            config: Configuration for the agent
        """
        self.config = config or default_config

        # Container and Lifecycle
        self._container: Container | None = None
        self._lifecycle: OrchestratorLifecycle | None = None

        # Agent components
        self._agent: BaseAgent | None = None
        self._runner: AgentRunner | None = None

        # State
        self._initialized = False
        self._current_session_id: str | None = None
        self._current_user_id: str | None = None
        self._current_agent_id: str | None = None

    @property
    def memory_client(self) -> MemoryClient | None:
        """Get the memory client from container."""
        if self._container:
            return self._container.memory_client
        return None

    async def initialize(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """
        Initialize all components.

        Args:
            user_id: User ID for personalization
            agent_id: Agent ID for agent-level isolation
        """
        if self._initialized:
            return

        logger.info("Initializing Memory Modes Demo Agent...")

        self._current_user_id = user_id or f"user-{generate_run_id()[-8:]}"
        self._current_agent_id = agent_id or self.config.agent_name

        # Setup logging
        setup_logging(level=LogLevel.INFO)

        # Initialize Lifecycle
        self._lifecycle = get_lifecycle_manager(
            fail_on_unhealthy=False,
            verify_connections=True,
        )
        init_result = await self._lifecycle.initialize()

        if not init_result.success:
            logger.warning(f"Lifecycle initialization issues: {init_result.errors}")
        else:
            logger.info("✓ Lifecycle initialized")

        # Get Container
        self._container = get_container()
        logger.info("✓ Container initialized")

        # Log client status
        memory_client = self._container.memory_client
        session_client = self._container.session_client

        if memory_client and memory_client.is_enabled:
            isolation_mode = memory_client.config.memory_isolation
            logger.info(f"✓ Memory client enabled (isolation: {isolation_mode})")
        else:
            logger.warning("⚠ Memory client not available")

        if session_client and session_client.is_enabled:
            logger.info("✓ Session client enabled")
        else:
            logger.warning("⚠ Session client not available")

        # Create session
        if session_client and session_client.is_enabled:
            try:
                self._current_session_id = await session_client.get_or_create_session(
                    user_id=self._current_user_id,
                    conversation_id=self._current_agent_id,
                )
                logger.info(f"✓ Session: {self._current_session_id[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")

        # Create agent and runner
        self._create_agent()
        self._runner = AgentRunner(container=self._container)

        self._initialized = True
        logger.info("✓ Memory Modes Demo Agent ready!")

    def _create_agent(self) -> None:
        """Create the demo agent with memory configuration."""
        memory_client = self.memory_client

        # Memory config - uses RUN scope for session-level context
        memory_config = AgentMemoryConfig(
            search_memories=self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled,
            store_memories=self.config.enable_memory
            and memory_client is not None
            and memory_client.is_enabled,
            search_scope=AgentMemoryScope.CONVERSATION,
            store_scope=AgentMemoryScope.CONVERSATION,
            search_limit=self.config.memory_search_limit,
            extraction_prompt=(
                "Extract ONLY facts that the USER explicitly stated about themselves: "
                "preferences, personal details, goals, or opinions they expressed. "
                "Do NOT extract anything the assistant said, and do NOT extract meta-questions "
                "about how the system works. Return an empty list if there are no user facts."
            ),
        )

        # Create agent
        self._agent = BaseAgent(
            name=self.config.agent_name,
            instructions=self.config.system_instructions,
            model=self.config.agent_model,
            temperature=self.config.agent_temperature,
            memory_config=memory_config,
            config=AgentConfig(
                max_turns=self.config.max_turns,
                log_to_session=self.config.enable_session,
                context_management=ContextManagementConfig(
                    enabled=True,
                    compression_threshold=0.8,
                    compression_strategy=CompressionStrategy.SMART,
                    keep_recent_messages=10,
                ),
            ),
        )

        logger.info("✓ Agent created with memory support")

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
            response = await self._runner.run(
                agent=self._agent,
                input=message,
                session_id=self._current_session_id,
                user_id=self._current_user_id,
                conversation_id=self._current_agent_id,
            )
            return response.content or ""
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"

    # =========================================================================
    # Direct Memory Operations (demonstrating the new API)
    # =========================================================================

    async def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        custom_prompt: str | None = None,
    ) -> MemoryAddResult | None:
        """
        Add a memory directly using the new MemoryClient API.

        Args:
            content: Memory content to add
            metadata: Optional metadata
            custom_prompt: Custom fact extraction prompt

        Returns:
            MemoryAddResult or None if failed
        """
        if not self.memory_client or not self.memory_client.is_enabled:
            logger.warning("Memory client not available")
            return None

        try:
            result = await self.memory_client.add(
                content,
                user_id=self._current_user_id,
                agent_id=self._current_agent_id,
                conversation_id=self._current_agent_id,
                metadata=metadata,
                custom_prompt=custom_prompt,
            )
            logger.info(f"Memory added: {result.message}")
            return result
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
    ) -> MemorySearchResult | None:
        """
        Search memories directly using the new MemoryClient API.

        Args:
            query: Search query
            limit: Max results

        Returns:
            MemorySearchResult or None if failed
        """
        if not self.memory_client or not self.memory_client.is_enabled:
            logger.warning("Memory client not available")
            return None

        try:
            result = await self.memory_client.search(
                query,
                user_id=self._current_user_id,
                agent_id=self._current_agent_id,
                conversation_id=self._current_agent_id,
                limit=limit,
            )
            logger.info(f"Search found {result.total_results} memories")
            return result
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return None

    async def get_all_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get all memories for current scope.

        Args:
            limit: Max memories to return

        Returns:
            List of memory dicts
        """
        if not self.memory_client or not self.memory_client.is_enabled:
            return []

        try:
            memories = await self.memory_client.get_all(
                user_id=self._current_user_id,
                agent_id=self._current_agent_id,
                conversation_id=self._current_agent_id,
                limit=limit,
            )
            return [m.to_dict() for m in memories]
        except Exception as e:
            logger.error(f"Failed to get memories: {e}")
            return []

    async def delete_all_memories(self) -> bool:
        """Delete all memories for current scope."""
        if not self.memory_client or not self.memory_client.is_enabled:
            return False

        try:
            return await self.memory_client.delete_all(
                user_id=self._current_user_id,
                agent_id=self._current_agent_id,
                conversation_id=self._current_agent_id,
            )
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")
            return False

    async def get_session_history(self) -> list[dict]:
        session_client = self._container.session_client if self._container else None
        if not session_client or not session_client.is_enabled or not self._current_session_id:
            return []
        try:
            messages = await session_client.get_conversation_history(self._current_session_id)
            return [{"role": m.role, "content": m.content} for m in messages]
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []

    # =========================================================================
    # User/Agent/Session Management
    # =========================================================================

    async def switch_user(self, user_id: str) -> None:
        """Switch to a different user."""
        if not self._initialized:
            await self.initialize()

        old_user = self._current_user_id
        self._current_user_id = user_id

        # Create new session for new user
        if (
            self._container
            and self._container.session_client
            and self._container.session_client.is_enabled
        ):
            try:
                self._current_session_id = (
                    await self._container.session_client.get_or_create_session(
                        user_id=self._current_user_id,
                        conversation_id=self._current_agent_id,
                    )
                )
                logger.info(f"Switched user: {old_user} -> {self._current_user_id}")
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")
        else:
            logger.info(f"Switched user: {old_user} -> {self._current_user_id}")

    async def switch_agent(self, agent_id: str) -> None:
        """Switch to a different agent."""
        if not self._initialized:
            await self.initialize()

        old_agent = self._current_agent_id
        self._current_agent_id = agent_id

        # Create new session for new agent
        if (
            self._container
            and self._container.session_client
            and self._container.session_client.is_enabled
        ):
            try:
                self._current_session_id = (
                    await self._container.session_client.get_or_create_session(
                        user_id=self._current_user_id,
                        conversation_id=self._current_agent_id,
                    )
                )
                logger.info(f"Switched agent: {old_agent} -> {self._current_agent_id}")
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")
        else:
            logger.info(f"Switched agent: {old_agent} -> {self._current_agent_id}")

    async def new_session(self) -> None:
        """Start a new chat session."""
        if not self._initialized:
            await self.initialize()

        old_session = self._current_session_id

        if (
            self._container
            and self._container.session_client
            and self._container.session_client.is_enabled
        ):
            try:
                self._current_session_id = (
                    await self._container.session_client.get_or_create_session(
                        user_id=self._current_user_id,
                        conversation_id=self._current_agent_id,
                    )
                )
                logger.info(
                    f"New session: {old_session[:8] if old_session else 'none'}... -> {self._current_session_id[:8]}..."
                )
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")
        else:
            self._current_session_id = f"session-{generate_run_id()}"
            logger.info(f"New session: {self._current_session_id[:8]}...")

    # =========================================================================
    # Info and Cleanup
    # =========================================================================

    async def get_memory_info(self) -> dict[str, Any]:
        """Get information about current memory configuration."""
        if not self.memory_client:
            return {"error": "Memory client not available"}

        config = self.memory_client.config

        return {
            "isolation_mode": config.memory_isolation,
            "is_enabled": self.memory_client.is_enabled,
            "provider": config.provider,
            "vector_store_provider": config.vector_store_provider,
            "embedder_provider": config.embedder_provider,
            "embedder_model": config.embedder_model,
            "embedding_dims": config.embedding_dims,
            "user_id": self._current_user_id,
            "agent_id": self._current_agent_id,
            "session_id": self._current_session_id,
            "search_limit": config.search_limit,
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._lifecycle:
            await self._lifecycle.shutdown()
            logger.info("✓ Cleanup complete")
