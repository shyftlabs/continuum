"""
Base Memory Provider - Abstract base class for memory providers.

Defines the interface that all memory providers must implement.
This enables easy addition of new providers (Pinecone, Weaviate, etc.)
while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Any

from orchestrator.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemorySearchResult,
)


class BaseMemoryProvider(ABC):
    """
    Abstract base class for memory providers.

    All memory providers (Mem0, Pinecone, Weaviate, etc.) must implement
    this interface to be compatible with the MemoryClient.

    The provider handles both async and sync operations. Async is preferred
    for production use, while sync is available for simpler use cases.

    Example implementation:
        ```python
        class CustomProvider(BaseMemoryProvider):
            async def add(self, messages, user_id=None, ...):
                # Implementation
                pass
        ```
    """

    # =========================================================================
    # Async Methods (Primary Interface)
    # =========================================================================

    @abstractmethod
    async def add(
        self,
        messages: str | list[dict[str, Any]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        custom_prompt: str | None = None,
        infer: bool = True,
    ) -> MemoryAddResult:
        """
        Add memories from messages or text.

        Args:
            messages: Message(s) to extract memories from. Can be:
                - String: Single message text
                - List[str]: Multiple message texts
                - List[dict]: Chat messages with 'role' and 'content'
            user_id: User identifier for scoping
            agent_id: Agent identifier for scoping
            conversation_id: Conversation identifier for scoping
            metadata: Additional metadata for the memories
            custom_prompt: Custom prompt for fact extraction

        Returns:
            MemoryAddResult with status and extracted memories.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> MemorySearchResult:
        """
        Search memories using semantic similarity.

        Args:
            query: Search query text
            user_id: User identifier for scoping
            agent_id: Agent identifier for scoping
            conversation_id: Conversation identifier for scoping
            limit: Maximum number of results to return
            filters: Additional metadata filters (provider-specific)

        Returns:
            MemorySearchResult with matching memories.
        """
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            MemoryEntry if found, None otherwise.
        """
        ...

    @abstractmethod
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
            limit: Maximum number of memories to return

        Returns:
            List of MemoryEntry objects.
        """
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if deleted successfully.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        """
        Get the history of a memory (all versions).

        Args:
            memory_id: The ID of the memory

        Returns:
            List of memory history entries.
        """
        ...

    @abstractmethod
    async def reset(self) -> bool:
        """
        Reset the entire memory store (USE WITH CAUTION).

        This deletes ALL memories in the system.

        Returns:
            True if reset successfully.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the provider and release resources."""
        ...

    # =========================================================================
    # Sync Methods (Alternative Interface)
    # =========================================================================

    @abstractmethod
    def add_sync(
        self,
        messages: str | list[dict[str, Any]] | list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        custom_prompt: str | None = None,
        infer: bool = True,
    ) -> MemoryAddResult:
        """Synchronous version of add()."""
        ...

    @abstractmethod
    def search_sync(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> MemorySearchResult:
        """Synchronous version of search()."""
        ...

    @abstractmethod
    def get_sync(self, memory_id: str) -> MemoryEntry | None:
        """Synchronous version of get()."""
        ...

    @abstractmethod
    def get_all_sync(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Synchronous version of get_all()."""
        ...

    @abstractmethod
    def delete_sync(self, memory_id: str) -> bool:
        """Synchronous version of delete()."""
        ...

    @abstractmethod
    def delete_all_sync(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """Synchronous version of delete_all()."""
        ...

    @abstractmethod
    def update_sync(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
    ) -> MemoryEntry:
        """Synchronous version of update()."""
        ...

    @abstractmethod
    def history_sync(self, memory_id: str) -> list[dict[str, Any]]:
        """Synchronous version of history()."""
        ...

    @abstractmethod
    def reset_sync(self) -> bool:
        """Synchronous version of reset()."""
        ...

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> "BaseMemoryProvider":
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
    # Provider Info
    # =========================================================================

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'mem0', 'pinecone')."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the provider is initialized and ready."""
        ...
