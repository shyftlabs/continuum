"""
Memory Client - Unified interface for long-term memory.

Provides a high-level client that delegates to memory providers (mem0, etc.)
for actual memory operations.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from orchestrator.logging import get_logger
from orchestrator.memory.base import BaseMemoryProvider
from orchestrator.memory.config import MemoryConfig
from orchestrator.memory.exceptions import (
    MemoryIdentifierError,
    MemoryNotEnabledError,
)
from orchestrator.memory.providers import create_provider, list_providers
from orchestrator.memory.scopes import MemoryScope
from orchestrator.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemoryMetadata,
    MemorySearchResult,
)

T = TypeVar("T")

logger = get_logger(__name__)

# Global client state
_global_lock = threading.Lock()
_global_memory_client: "MemoryClient | None" = None
_initialized = False


class MemoryClient:
    """
    Unified memory client.

    This client provides a high-level interface for memory operations,
    delegating to a provider (e.g., Mem0Provider) for actual implementation.

    Features:
        - Multi-level memory scoping (user, agent, run, shared)
        - Both async and sync interfaces
        - Custom prompts for fact extraction and updates
        - Provider abstraction for future extensibility
        - Graceful error handling

    Example:
        ```python
        from orchestrator.memory import MemoryClient

        # Initialize with default configuration
        client = MemoryClient()

        # Add memories (async)
        await client.add(
            "User prefers dark mode",
            user_id="user-123",
            metadata={"category": "preferences"}
        )

        # Search memories (async)
        results = await client.search(
            "What are the user's preferences?",
            user_id="user-123",
            limit=5
        )

        # Sync versions available
        results = client.search_sync("query", user_id="user-123")
        ```
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        provider: BaseMemoryProvider | None = None,
        auto_initialize: bool = True,
    ):
        """
        Initialize the memory client.

        Args:
            config: Memory configuration. Uses defaults from environment if not provided.
            provider: Memory provider. Created automatically if not provided.
            auto_initialize: Whether to initialize the provider immediately.
        """
        self._config = config or MemoryConfig()
        self._provider = provider
        self._initialized = False

        if auto_initialize and self._config.enabled:
            self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Initialize the memory provider using the registry."""
        if self._provider is not None:
            self._initialized = self._provider.is_initialized
            return

        if not self._config.enabled:
            logger.info("Memory is disabled. Set MEMORY_ENABLED=true to enable.")
            return

        try:
            provider_name = self._config.provider
            available = list_providers()

            if not available:
                logger.error(
                    "No memory providers available. Install a provider package "
                    "(e.g., pip install mem0ai for mem0 provider)"
                )
                return

            if provider_name not in available:
                logger.warning(
                    f"Provider '{provider_name}' not available. "
                    f"Available providers: {available}. Falling back to '{available[0]}'"
                )
                provider_name = available[0]

            self._provider = create_provider(provider_name, self._config)
            self._initialized = self._provider.is_initialized

            logger.info(f"Memory provider initialized: {provider_name}")

        except ImportError as e:
            logger.error(
                f"Failed to import memory provider '{provider_name}': {e}. "
                "Install the required package (e.g., pip install mem0ai).",
                exc_info=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize memory provider: {e}", exc_info=True)

    @property
    def config(self) -> MemoryConfig:
        """Get the current configuration."""
        return self._config

    @property
    def provider(self) -> BaseMemoryProvider | None:
        """Get the current provider."""
        return self._provider

    @property
    def is_enabled(self) -> bool:
        """Check if memory is enabled and initialized."""
        return self._config.enabled and self._initialized and self._provider is not None

    def _ensure_enabled(self) -> None:
        """Raise error if memory is not enabled."""
        if not self.is_enabled:
            raise MemoryNotEnabledError(
                "Memory operations require memory to be enabled. "
                "Set MEMORY_ENABLED=true in your environment."
            )

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Run an async coroutine synchronously.

        Handles the case where we're already in an event loop
        (e.g., Jupyter notebook) vs. no event loop.

        Uses a dedicated thread with its own event loop to avoid deadlocks
        when called from within an existing async context.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    def _build_scope(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> MemoryScope:
        """
        Build a MemoryScope from identifiers based on isolation mode.

        Args:
            user_id: User identifier
            agent_id: Agent identifier
            conversation_id: Conversation identifier

        Returns:
            MemoryScope configured for the current isolation mode.
        """
        mode = self._config.memory_isolation

        try:
            return MemoryScope.from_isolation_mode(
                mode=mode,
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
            )
        except ValueError as e:
            raise MemoryIdentifierError(
                str(e),
                isolation_level=mode,
                required_identifier=mode if mode != "shared" else None,
            ) from e

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def add(
        self,
        messages: str | list[dict[str, Any]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        metadata: MemoryMetadata | dict[str, Any] | None = None,
        custom_prompt: str | None = None,
        infer: bool = True,
    ) -> MemoryAddResult:
        """
        Add memories from messages or text.

        Args:
            messages: Message(s) to extract memories from
            user_id: User identifier for scoping
            agent_id: Agent identifier for scoping
            conversation_id: Conversation identifier for scoping
            metadata: Additional metadata for the memories
            custom_prompt: Custom prompt for fact extraction

        Returns:
            MemoryAddResult with status and extracted memories.
        """
        self._ensure_enabled()

        # Build scope from identifiers
        scope = self._build_scope(user_id, agent_id, conversation_id)
        identifiers = scope.to_identifiers()

        # Convert metadata if needed
        if isinstance(metadata, MemoryMetadata):
            metadata_dict = metadata.to_dict()
        else:
            metadata_dict = metadata

        return await self._provider.add(
            messages,
            **identifiers,
            metadata=metadata_dict,
            custom_prompt=custom_prompt,
            infer=infer,
        )

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> MemorySearchResult:
        """
        Search memories using semantic similarity.

        Args:
            query: Search query text
            user_id: User identifier for scoping
            agent_id: Agent identifier for scoping
            conversation_id: Conversation identifier for scoping
            limit: Maximum results to return
            filters: Additional metadata filters (provider-specific)

        Returns:
            MemorySearchResult with matching memories.
        """
        self._ensure_enabled()

        scope = self._build_scope(user_id, agent_id, conversation_id)
        identifiers = scope.to_identifiers()
        search_limit = limit or self._config.search_limit

        # Log search parameters
        logger.info(
            f"🔍 MEMORY CLIENT SEARCH: query='{query[:100]}...', "
            f"isolation={self._config.memory_isolation}, "
            f"scope={scope}, identifiers={identifiers}, "
            f"limit={search_limit}, filters={filters}"
        )

        result = await self._provider.search(
            query,
            **identifiers,
            limit=search_limit,
            filters=filters,
        )

        # Log search results
        logger.info(
            f"✅ MEMORY CLIENT SEARCH RESULT: found {len(result.results)} memories "
            f"(total_results={result.total_results if hasattr(result, 'total_results') else 'N/A'})"
        )

        return result

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            MemoryEntry if found, None otherwise.
        """
        self._ensure_enabled()
        return await self._provider.get(memory_id)

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """
        Get all memories for the specified scope.

        Args:
            user_id: User identifier for scoping
            agent_id: Agent identifier for scoping
            conversation_id: Conversation identifier for scoping
            limit: Maximum memories to return

        Returns:
            List of MemoryEntry objects.
        """
        self._ensure_enabled()

        scope = self._build_scope(user_id, agent_id, conversation_id)
        identifiers = scope.to_identifiers()

        return await self._provider.get_all(**identifiers, limit=limit)

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if deleted successfully.
        """
        self._ensure_enabled()
        return await self._provider.delete(memory_id)

    async def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """
        Delete all memories for the specified scope.

        Args:
            user_id: User identifier for scoping
            agent_id: Agent identifier for scoping
            conversation_id: Conversation identifier for scoping

        Returns:
            True if deleted successfully.
        """
        self._ensure_enabled()

        scope = self._build_scope(user_id, agent_id, conversation_id)
        identifiers = scope.to_identifiers()

        return await self._provider.delete_all(**identifiers)

    async def update(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
    ) -> MemoryEntry:
        """
        Update a specific memory.

        Args:
            memory_id: The ID of the memory to update
            data: New data for the memory
            custom_prompt: Custom prompt for memory update

        Returns:
            Updated MemoryEntry.
        """
        self._ensure_enabled()
        return await self._provider.update(memory_id, data, custom_prompt=custom_prompt)

    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        """
        Get the history of a memory (all versions).

        Args:
            memory_id: The ID of the memory

        Returns:
            List of memory history entries.
        """
        self._ensure_enabled()
        return await self._provider.history(memory_id)

    async def reset(self) -> bool:
        """
        Reset the entire memory store (USE WITH CAUTION).

        Returns:
            True if reset successfully.
        """
        self._ensure_enabled()
        return await self._provider.reset()

    async def close(self) -> None:
        """Close the memory client and release resources."""
        if self._provider:
            await self._provider.close()
        self._initialized = False
        logger.debug("Memory client closed")

    async def __aenter__(self) -> "MemoryClient":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.close()

    # =========================================================================
    # Sync Methods - Delegate to async methods via _run_sync helper
    # =========================================================================

    def add_sync(
        self,
        messages: str | list[dict[str, Any]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        metadata: MemoryMetadata | dict[str, Any] | None = None,
        custom_prompt: str | None = None,
        infer: bool = True,
    ) -> MemoryAddResult:
        """Synchronous version of add()."""
        return self._run_sync(
            self.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                metadata=metadata,
                custom_prompt=custom_prompt,
                infer=infer,
            )
        )

    def search_sync(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> MemorySearchResult:
        """Synchronous version of search()."""
        return self._run_sync(
            self.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                limit=limit,
                filters=filters,
            )
        )

    def get_sync(self, memory_id: str) -> MemoryEntry | None:
        """Synchronous version of get()."""
        return self._run_sync(self.get(memory_id))

    def get_all_sync(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Synchronous version of get_all()."""
        return self._run_sync(
            self.get_all(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                limit=limit,
            )
        )

    def delete_sync(self, memory_id: str) -> bool:
        """Synchronous version of delete()."""
        return self._run_sync(self.delete(memory_id))

    def delete_all_sync(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """Synchronous version of delete_all()."""
        return self._run_sync(
            self.delete_all(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
            )
        )

    def update_sync(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
    ) -> MemoryEntry:
        """Synchronous version of update()."""
        return self._run_sync(self.update(memory_id, data, custom_prompt=custom_prompt))

    def history_sync(self, memory_id: str) -> list[dict[str, Any]]:
        """Synchronous version of history()."""
        return self._run_sync(self.history(memory_id))

    def reset_sync(self) -> bool:
        """Synchronous version of reset()."""
        return self._run_sync(self.reset())


# =============================================================================
# Global Memory Client Functions
# =============================================================================


def initialize_global_memory(config: MemoryConfig | None = None) -> bool:
    """
    Initialize the global Memory client.

    This should be called once at application startup.

    Args:
        config: Optional configuration. Uses environment variables if not provided.

    Returns:
        True if initialization was successful.
    """
    global _global_memory_client, _initialized

    with _global_lock:
        if _initialized:
            return _global_memory_client is not None and _global_memory_client.is_enabled

        _global_memory_client = MemoryClient(config=config, auto_initialize=True)
        _initialized = True

        return _global_memory_client.is_enabled


def get_global_memory_client() -> MemoryClient:
    """
    Get the global Memory client.

    Auto-initializes if not already initialized.

    Returns:
        The global MemoryClient instance.
    """
    global _global_memory_client, _initialized

    if not _initialized:
        initialize_global_memory()

    if _global_memory_client is None:
        with _global_lock:
            if _global_memory_client is None:
                _global_memory_client = MemoryClient(auto_initialize=True)
                _initialized = True

    return _global_memory_client


def reset_global_memory() -> None:
    """Reset the global memory client. Useful for testing."""
    global _global_memory_client, _initialized

    with _global_lock:
        _global_memory_client = None
        _initialized = False
