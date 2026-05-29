"""
Mem0 Provider - Memory provider implementation using mem0.

Uses mem0's Memory for both sync and async operations (async via asyncio.to_thread).
Leverages mem0's native functionality for Qdrant integration, metadata filtering,
and custom prompts.

mem0 Operations Used:
    - add: Add memories from messages with fact extraction
    - search: Semantic search with native Qdrant filtering
    - get: Get a specific memory by ID
    - get_all: Get all memories for a scope
    - delete: Delete a specific memory
    - delete_all: Delete all memories for a scope
    - update: Update a memory with optional custom prompt
    - history: Get version history of a memory
    - reset: Reset entire memory store

See: https://docs.mem0.ai/open-source/python-quickstart
"""

import asyncio
from typing import Any

# mem0 imports - optional dependency
try:
    from mem0 import Memory

    MEM0_AVAILABLE = True
except ImportError:
    Memory = None  # type: ignore
    MEM0_AVAILABLE = False

from orchestrator.logging import get_logger
from orchestrator.memory.base import BaseMemoryProvider
from orchestrator.memory.config import MemoryConfig
from orchestrator.memory.exceptions import (
    MemoryConfigurationError,
    MemoryUpdateError,
)
from orchestrator.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemorySearchResult,
)
from orchestrator.observability.decorators import observe
from orchestrator.observability.error_reporter import report_error

logger = get_logger(__name__)


class Mem0Provider(BaseMemoryProvider):
    """
    Memory provider using mem0.

    This provider leverages mem0's native functionality:
    - AsyncMemory for non-blocking async operations
    - Memory for synchronous operations
    - Native Qdrant metadata filtering
    - Custom fact extraction prompts
    - Custom memory update prompts

    All operations directly call mem0's API methods without additional wrapping,
    ensuring we use mem0's built-in functionality for:
    - Automatic fact extraction from messages
    - Vector embedding and storage
    - Semantic similarity search
    - Memory consolidation and deduplication

    Example:
        ```python
        from orchestrator.memory.config import MemoryConfig
        from orchestrator.memory.providers.mem0 import Mem0Provider

        config = MemoryConfig()
        provider = Mem0Provider(config)

        # Async usage
        result = await provider.add("User likes pizza", user_id="user-123")

        # Sync usage
        result = provider.add_sync("User likes pizza", user_id="user-123")
        ```
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the Mem0 provider.

        Args:
            config: Memory configuration

        Raises:
            ImportError: If mem0ai package is not installed
        """
        if not MEM0_AVAILABLE:
            raise ImportError("mem0ai package not installed. Run: pip install mem0ai")

        self._config = config
        self._sync_memory: Memory | None = None
        self._mem0_config: dict | None = None
        self._initialized = False

        if config.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize mem0 Memory client (sync version only, async uses this internally)."""
        if self._initialized:
            return

        if not self._config.enabled:
            logger.info("Memory is disabled")
            return

        if not self._config.is_configured():
            logger.warning(
                "Memory not properly configured. Check required settings: "
                "QDRANT_HOST, MEMORY_LLM_MODEL, EMBEDDER_MODEL, EMBEDDING_DIMS"
            )
            return

        try:
            # Build mem0 config from our MemoryConfig
            self._mem0_config = self._config.to_mem0_config()

            logger.debug(f"Initializing mem0 with config: {self._mem0_config}")

            # Initialize sync client - mem0's Memory.from_config() is synchronous
            self._sync_memory = Memory.from_config(self._mem0_config)

            self._initialized = True
            self._patch_milvus_strong_consistency()
            logger.info(
                "Mem0Provider initialized successfully",
                extra={
                    "vector_store": self._config.vector_store_provider,
                    "qdrant_host": self._config.qdrant_host,
                    "embedder_provider": self._config.embedder_provider,
                    "embedder_model": self._config.embedder_model,
                    "isolation": self._config.memory_isolation,
                },
            )

        except Exception as e:
            logger.error(f"Failed to initialize Mem0Provider: {e}")
            report_error(e, context="mem0_provider_init")
            raise MemoryConfigurationError(
                f"Failed to initialize mem0: {e}",
                config_key="mem0",
            ) from e

    def _patch_milvus_strong_consistency(self) -> None:
        """Patch MilvusDB.list() to use Strong consistency so filter queries see all writes.

        Milvus inserts land in growing (unsealed) segments. JSON-field filter queries
        (used by mem0's list/get_all) cannot see growing segments without an explicit
        consistency level. Patching list() to pass consistency_level="Strong" tells
        Milvus to wait until the latest write is visible before executing the query —
        the correct production approach vs. forcing an expensive flush after every write.
        """
        if self._config.vector_store_provider != "milvus":
            return
        try:
            vs = getattr(self._sync_memory, "vector_store", None)
            if vs is None or not hasattr(vs, "client") or not hasattr(vs, "collection_name"):
                return

            def _list_strong(self_vs, filters=None, limit=100):
                from mem0.vector_stores.milvus import OutputData

                query_filter = self_vs._create_filter(filters) if filters else None
                result = self_vs.client.query(
                    collection_name=self_vs.collection_name,
                    filter=query_filter,
                    limit=limit,
                    consistency_level="Strong",
                )
                memories = [
                    OutputData(id=d.get("id"), score=None, payload=d.get("metadata"))
                    for d in result
                ]
                return [memories]

            import types

            vs.list = types.MethodType(_list_strong, vs)
            logger.debug("Patched MilvusDB.list() with consistency_level=Strong")
        except Exception as e:
            logger.debug(f"Milvus consistency patch skipped: {e}")

    def _flush_milvus(self) -> None:
        """Flush Milvus before delete_all so mem0's list() sees all growing segments.

        Used only in delete_all (a rare, expensive operation where correctness beats
        throughput). Do NOT call after add() — use the Strong consistency patch instead.
        """
        if self._config.vector_store_provider != "milvus":
            return
        try:
            vs = getattr(self._sync_memory, "vector_store", None)
            if vs is not None and hasattr(vs, "client") and hasattr(vs, "collection_name"):
                vs.client.flush(vs.collection_name)
        except Exception as e:
            logger.debug(f"Milvus flush skipped: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure provider is initialized and ready."""
        if not self._initialized:
            raise MemoryConfigurationError(
                "Mem0Provider not initialized. Check configuration and ensure memory is enabled."
            )

    def _build_identifiers(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, str]:
        """Build identifier dict for mem0. Maps conversation_id → run_id (mem0 has no native conversation_id)."""
        identifiers: dict[str, str] = {}
        if user_id:
            identifiers["user_id"] = user_id
        if agent_id:
            identifiers["agent_id"] = agent_id
        if conversation_id:
            identifiers["run_id"] = (
                conversation_id  # mem0 uses run_id for conversation-level scoping
            )
        return identifiers

    # =========================================================================
    # Provider Info
    # =========================================================================

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "mem0"

    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized

    # =========================================================================
    # Async Methods - Uses asyncio.to_thread() with sync Memory client
    # =========================================================================

    @observe(name="memory_add", capture_output=True)
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
        Add memories using mem0's Memory.add() via asyncio.to_thread().

        mem0 handles:
        - Automatic fact extraction from messages
        - Vector embedding generation
        - Storage in Qdrant
        - Memory consolidation/deduplication

        See: https://docs.mem0.ai/open-source/python-quickstart
        """
        self._ensure_initialized()

        # Build kwargs for mem0.add()
        kwargs: dict[str, Any] = {
            "messages": messages,
            **self._build_identifiers(user_id, agent_id, conversation_id),
        }

        if metadata:
            kwargs["metadata"] = metadata

        if not infer:
            kwargs["infer"] = False

        # Custom fact extraction prompt
        # See: https://docs.mem0.ai/open-source/features/custom-fact-extraction-prompt
        if custom_prompt:
            kwargs["prompt"] = custom_prompt

        try:
            logger.debug(
                f"mem0.add() with: user_id={user_id}, agent_id={agent_id}, conversation_id={conversation_id}"
            )

            # Run sync memory.add() in thread pool
            response = await asyncio.to_thread(self._sync_memory.add, **kwargs)

            result = MemoryAddResult.from_mem0_response(response)
            logger.debug(f"mem0.add() result: {result.message}, {len(result.results)} memories")
            return result

        except Exception as e:
            logger.error(f"mem0.add() failed: {e}")
            report_error(e, context="memory_add")
            return MemoryAddResult(message="Memory operation failed", results=[])

    @observe(name="memory_search", capture_output=True)
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
        Search memories using mem0's Memory.search() via asyncio.to_thread().

        mem0 handles:
        - Query embedding
        - Vector similarity search in Qdrant
        - Native Qdrant metadata filtering

        See: https://docs.mem0.ai/open-source/features/metadata-filtering#qdrant
        """
        self._ensure_initialized()

        identifiers = self._build_identifiers(user_id, agent_id, conversation_id)

        kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
            **identifiers,
        }
        if filters:
            kwargs["filters"] = filters

        try:
            logger.debug(f"mem0.search() query='{query[:50]}...', limit={limit}")

            # Run sync memory.search() in thread pool
            response = await asyncio.to_thread(self._sync_memory.search, **kwargs)

            result = MemorySearchResult.from_mem0_response(response, query, limit)
            logger.debug(f"mem0.search() found {result.total_results} results")
            return result

        except Exception as e:
            logger.error(f"mem0.search() failed: {e}")
            report_error(e, context="memory_search")
            return MemorySearchResult(results=[], query=query, limit=limit, total_results=0)

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Get a memory by ID using mem0's Memory.get() via asyncio.to_thread().
        """
        self._ensure_initialized()

        try:
            logger.debug(f"mem0.get() memory_id={memory_id}")

            # Run sync memory.get() in thread pool
            response = await asyncio.to_thread(self._sync_memory.get, memory_id=memory_id)

            if response:
                return MemoryEntry.from_mem0_result(response)
            return None

        except Exception as e:
            logger.error(f"mem0.get() failed for {memory_id}: {e}")
            report_error(e, context="memory_get")
            return None

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """
        Get all memories for a scope using mem0's Memory.get_all() via asyncio.to_thread().
        """
        self._ensure_initialized()

        # Build kwargs for mem0.get_all()
        kwargs: dict[str, Any] = self._build_identifiers(user_id, agent_id, conversation_id)
        if limit:
            kwargs["limit"] = limit

        try:
            logger.debug(f"mem0.get_all() with: {kwargs}")

            # Run sync memory.get_all() in thread pool
            response = await asyncio.to_thread(self._sync_memory.get_all, **kwargs)

            memories = [MemoryEntry.from_mem0_result(m) for m in response.get("results", [])]
            logger.debug(f"mem0.get_all() returned {len(memories)} memories")
            return memories

        except Exception as e:
            logger.error(f"mem0.get_all() failed: {e}")
            report_error(e, context="memory_get_all")
            return []

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID using mem0's Memory.delete() via asyncio.to_thread().
        """
        self._ensure_initialized()

        try:
            logger.debug(f"mem0.delete() memory_id={memory_id}")

            # Run sync memory.delete() in thread pool
            await asyncio.to_thread(self._sync_memory.delete, memory_id=memory_id)

            logger.info(f"Memory deleted: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"mem0.delete() failed for {memory_id}: {e}")
            report_error(e, context="memory_delete")
            return False

    async def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """
        Delete all memories for a scope using mem0's Memory.delete_all() via asyncio.to_thread().
        """
        self._ensure_initialized()

        kwargs = self._build_identifiers(user_id, agent_id, conversation_id)

        try:
            logger.debug(f"mem0.delete_all() with: {kwargs}")

            # Flush so mem0's internal list() sees all growing segments before deleting
            await asyncio.to_thread(self._flush_milvus)
            # Run sync memory.delete_all() in thread pool
            await asyncio.to_thread(self._sync_memory.delete_all, **kwargs)

            logger.info(f"All memories deleted for: {kwargs}")
            return True

        except Exception as e:
            logger.error(f"mem0.delete_all() failed: {e}")
            report_error(e, context="memory_delete_all")
            return False

    async def update(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
    ) -> MemoryEntry:
        """
        Update a memory using mem0's Memory.update() via asyncio.to_thread().

        See: https://docs.mem0.ai/open-source/features/custom-update-memory-prompt
        """
        self._ensure_initialized()

        kwargs: dict[str, Any] = {
            "memory_id": memory_id,
            "data": data,
        }

        # Custom update prompt
        if custom_prompt:
            kwargs["prompt"] = custom_prompt

        try:
            logger.debug(f"mem0.update() memory_id={memory_id}")

            # Run sync memory.update() in thread pool
            response = await asyncio.to_thread(self._sync_memory.update, **kwargs)

            if response is None:
                raise MemoryUpdateError(
                    "mem0.update() returned None",
                    memory_id=memory_id,
                )

            logger.info(f"Memory updated: {memory_id}")
            return MemoryEntry.from_mem0_result(response)

        except MemoryUpdateError:
            raise
        except Exception as e:
            logger.error(f"mem0.update() failed for {memory_id}: {e}")
            report_error(e, context="memory_update")
            raise MemoryUpdateError(
                f"Failed to update memory: {e}",
                memory_id=memory_id,
                original_error=e,
            ) from e

    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        """
        Get memory history using mem0's Memory.history() via asyncio.to_thread().

        Returns all versions of a memory.
        """
        self._ensure_initialized()

        try:
            logger.debug(f"mem0.history() memory_id={memory_id}")

            # Run sync memory.history() in thread pool
            history = await asyncio.to_thread(self._sync_memory.history, memory_id=memory_id)

            logger.debug(f"mem0.history() returned {len(history)} versions")
            return history

        except Exception as e:
            logger.error(f"mem0.history() failed for {memory_id}: {e}")
            report_error(e, context="memory_history")
            return []

    async def reset(self) -> bool:
        """
        Reset entire memory store using mem0's Memory.reset() via asyncio.to_thread().

        WARNING: This deletes ALL memories in the system.
        """
        self._ensure_initialized()

        try:
            logger.warning("mem0.reset() - Resetting entire memory store")

            # Run sync memory.reset() in thread pool
            await asyncio.to_thread(self._sync_memory.reset)

            logger.info("Memory store reset successfully")
            return True

        except Exception as e:
            logger.error(f"mem0.reset() failed: {e}")
            report_error(e, context="memory_reset")
            return False

    async def close(self) -> None:
        """Close the provider and release resources."""
        self._initialized = False
        self._sync_memory = None
        self._mem0_config = None
        logger.debug("Mem0Provider closed")

    # =========================================================================
    # Sync Methods - Direct mem0 Memory API calls
    # =========================================================================

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
        """Add memories using mem0's Memory.add()."""
        self._ensure_initialized()

        kwargs: dict[str, Any] = {
            "messages": messages,
            **self._build_identifiers(user_id, agent_id, conversation_id),
        }

        if metadata:
            kwargs["metadata"] = metadata
        if custom_prompt:
            kwargs["prompt"] = custom_prompt
        if not infer:
            kwargs["infer"] = False

        try:
            response = self._sync_memory.add(**kwargs)
            return MemoryAddResult.from_mem0_response(response)
        except Exception as e:
            logger.error(f"mem0.add() sync failed: {e}")
            return MemoryAddResult(message="Memory operation failed", results=[])

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
        """Search memories using mem0's Memory.search()."""
        self._ensure_initialized()

        identifiers = self._build_identifiers(user_id, agent_id, conversation_id)

        kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
            **identifiers,
        }
        if filters:
            kwargs["filters"] = filters

        try:
            response = self._sync_memory.search(**kwargs)
            return MemorySearchResult.from_mem0_response(response, query, limit)
        except Exception as e:
            logger.error(f"mem0.search() sync failed: {e}")
            return MemorySearchResult(results=[], query=query, limit=limit, total_results=0)

    def get_sync(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID using mem0's Memory.get()."""
        self._ensure_initialized()

        try:
            response = self._sync_memory.get(memory_id=memory_id)
            if response:
                return MemoryEntry.from_mem0_result(response)
            return None
        except Exception as e:
            logger.error(f"mem0.get() sync failed for {memory_id}: {e}")
            return None

    def get_all_sync(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Get all memories using mem0's Memory.get_all()."""
        self._ensure_initialized()

        kwargs: dict[str, Any] = self._build_identifiers(user_id, agent_id, conversation_id)
        if limit:
            kwargs["limit"] = limit

        try:
            response = self._sync_memory.get_all(**kwargs)
            return [MemoryEntry.from_mem0_result(m) for m in response.get("results", [])]
        except Exception as e:
            logger.error(f"mem0.get_all() sync failed: {e}")
            return []

    def delete_sync(self, memory_id: str) -> bool:
        """Delete a memory by ID using mem0's Memory.delete()."""
        self._ensure_initialized()

        try:
            self._sync_memory.delete(memory_id=memory_id)
            return True
        except Exception as e:
            logger.error(f"mem0.delete() sync failed for {memory_id}: {e}")
            return False

    def delete_all_sync(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        """Delete all memories using mem0's Memory.delete_all()."""
        self._ensure_initialized()

        kwargs = self._build_identifiers(user_id, agent_id, conversation_id)

        try:
            self._sync_memory.delete_all(**kwargs)
            return True
        except Exception as e:
            logger.error(f"mem0.delete_all() sync failed: {e}")
            return False

    def update_sync(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
    ) -> MemoryEntry:
        """Update a memory using mem0's Memory.update()."""
        self._ensure_initialized()

        kwargs: dict[str, Any] = {
            "memory_id": memory_id,
            "data": data,
        }

        if custom_prompt:
            kwargs["prompt"] = custom_prompt

        try:
            response = self._sync_memory.update(**kwargs)
            if response is None:
                raise MemoryUpdateError(
                    "mem0.update() returned None",
                    memory_id=memory_id,
                )
            return MemoryEntry.from_mem0_result(response)
        except MemoryUpdateError:
            raise
        except Exception as e:
            logger.error(f"mem0.update() sync failed for {memory_id}: {e}")
            raise MemoryUpdateError(
                f"Failed to update memory: {e}",
                memory_id=memory_id,
                original_error=e,
            ) from e

    def history_sync(self, memory_id: str) -> list[dict[str, Any]]:
        """Get memory history using mem0's Memory.history()."""
        self._ensure_initialized()

        try:
            return self._sync_memory.history(memory_id=memory_id)
        except Exception as e:
            logger.error(f"mem0.history() sync failed for {memory_id}: {e}")
            return []

    def reset_sync(self) -> bool:
        """Reset entire memory store using mem0's Memory.reset()."""
        self._ensure_initialized()

        try:
            logger.warning("mem0.reset() sync - Resetting entire memory store")
            self._sync_memory.reset()
            return True
        except Exception as e:
            logger.error(f"mem0.reset() sync failed: {e}")
            return False
