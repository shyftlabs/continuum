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

from continuum.logging import get_logger, log_content, log_id
from continuum.memory.base import BaseMemoryProvider
from continuum.memory.config import MemoryConfig
from continuum.memory.exceptions import (
    MemoryConfigurationError,
    MemoryUpdateError,
)
from continuum.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemorySearchResult,
)
from continuum.observability.decorators import observe
from continuum.observability.error_reporter import report_error
from continuum.utils.secrets import redact_dict

logger = get_logger(__name__)


def _loggable_config(config: dict[str, Any]) -> dict[str, Any]:
    """The mem0 config with its credentials masked.

    Deliberately not routed through ``log_content()``/``LOG_PROMPT_CONTENT``: a
    provider key is not a debugging convenience an operator should be able to
    switch back on, and the rest of the config is exactly what they need to see.
    ``redact_dict`` recurses, which matters here -- the keys sit two levels down
    under ``llm.config`` and ``embedder.config``.
    """
    try:
        return redact_dict(config)
    except Exception:  # a config shape redact_dict cannot walk must not leak
        return {"_redacted": "config could not be masked"}


class Mem0Provider(BaseMemoryProvider):
    """
    Memory provider using mem0.

    This provider leverages mem0's native functionality:
    - Memory (the SYNCHRONOUS class) run through asyncio.to_thread. Not
      AsyncMemory: the pre_store_filter gate is mixed into the sync class only,
      so switching would silently remove it (see filtered_memory).
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
        from continuum.memory.config import MemoryConfig
        from continuum.memory.providers.mem0 import Mem0Provider

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

            # redact_dict, not log_content: an operator wants to see which
            # provider, which model, which host -- only the credentials must go,
            # and they must go whatever LOG_PROMPT_CONTENT says. to_mem0_config()
            # embeds the embedder's and the fact-extraction LLM's live API keys,
            # and this line printed them in full at DEBUG.
            logger.debug("Initializing mem0 with config: %s", _loggable_config(self._mem0_config))

            # Initialize sync client - mem0's Memory.from_config() is synchronous
            # A Memory subclass with the pre_store_filter gate mixed in. Gating
            # inside mem0 is what makes the filter a veto instead of a delete
            # after the fact; see filtered_memory for why that distinction is not
            # cosmetic.
            from continuum.memory.providers.filtered_memory import build_filtered_memory_class

            self._sync_memory = build_filtered_memory_class().from_config(self._mem0_config)

            self._initialized = True
            self._patch_milvus_strong_consistency()
            logger.info(
                "Mem0Provider initialized successfully",
                extra={
                    "vector_store": self._config.vector_store_provider,
                    "qdrant_host": self._config.qdrant_host,
                    "embedder_provider": self._config.embedder_provider,
                    "embedder_model": self._config.embedder_model,
                    "embedder_api_base": self._config.embedder_api_base,
                    "isolation": self._config.memory_isolation,
                },
            )

        except Exception as e:
            logger.error("Failed to initialize Mem0Provider: %s", e)
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
            logger.debug("Milvus consistency patch skipped: %s", e)

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
            logger.debug("Milvus flush skipped: %s", e)

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
        pre_store_filter: Any | None = None,
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
                "mem0.add() with: user_id=%s, agent_id=%s, conversation_id=%s",
                log_id(user_id),
                log_id(agent_id),
                log_id(conversation_id),
            )

            # Run sync memory.add() in thread pool.
            #
            # The filter is scoped onto the Memory instance rather than passed
            # in: mem0 has no parameter for one, and the gate lives inside
            # _create_memory several frames down -- past a ThreadPoolExecutor of
            # mem0's own, which a ContextVar does not survive. The instance does
            # cross that boundary; see filtered_memory for the measurement.
            from continuum.memory.providers.filtered_memory import use_pre_store_filter

            def _add_with_gate() -> tuple[Any, list[str]]:
                with use_pre_store_filter(self._sync_memory, pre_store_filter) as sup:
                    return self._sync_memory.add(**kwargs), list(sup)

            response, suppressed = await asyncio.to_thread(_add_with_gate)

            result = MemoryAddResult.from_mem0_response(response)
            result.suppressed = list(suppressed)
            if suppressed:
                logger.info(
                    "🚫 pre_store_filter suppressed %d fact(s) before the write", len(suppressed)
                )
            logger.debug("mem0.add() result: %s, %s memories", result.message, len(result.results))
            return result

        except Exception as e:
            logger.error("mem0.add() failed: %s", e, exc_info=True)
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
            logger.debug("mem0.search() query='%s', limit=%s", log_content(query), limit)

            # Run sync memory.search() in thread pool
            response = await asyncio.to_thread(self._sync_memory.search, **kwargs)

            result = MemorySearchResult.from_mem0_response(response, query, limit)
            logger.debug("mem0.search() found %s results", result.total_results)
            return result

        except Exception as e:
            logger.error("mem0.search() failed: %s", e, exc_info=True)
            report_error(e, context="memory_search")
            return MemorySearchResult(results=[], query=query, limit=limit, total_results=0)

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Get a memory by ID using mem0's Memory.get() via asyncio.to_thread().
        """
        self._ensure_initialized()

        try:
            logger.debug("mem0.get() memory_id=%s", log_id(memory_id))

            # Run sync memory.get() in thread pool
            response = await asyncio.to_thread(self._sync_memory.get, memory_id=memory_id)

            if response:
                return MemoryEntry.from_mem0_result(response)
            return None

        except Exception as e:
            logger.error("mem0.get() failed for %s: %s", log_id(memory_id), e, exc_info=True)
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
            logger.debug("mem0.get_all() with: %s", log_id(kwargs))

            # Run sync memory.get_all() in thread pool
            response = await asyncio.to_thread(self._sync_memory.get_all, **kwargs)

            memories = [MemoryEntry.from_mem0_result(m) for m in response.get("results", [])]
            logger.debug("mem0.get_all() returned %s memories", len(memories))
            return memories

        except Exception as e:
            logger.error("mem0.get_all() failed: %s", e, exc_info=True)
            report_error(e, context="memory_get_all")
            return []

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID using mem0's Memory.delete() via asyncio.to_thread().
        """
        self._ensure_initialized()

        try:
            logger.debug("mem0.delete() memory_id=%s", log_id(memory_id))

            # Run sync memory.delete() in thread pool
            await asyncio.to_thread(self._sync_memory.delete, memory_id=memory_id)

            logger.info("Memory deleted: %s", log_id(memory_id))
            return True

        except Exception as e:
            logger.error("mem0.delete() failed for %s: %s", log_id(memory_id), e, exc_info=True)
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
            logger.debug("mem0.delete_all() with: %s", log_id(kwargs))

            # Flush so mem0's internal list() sees all growing segments before deleting
            await asyncio.to_thread(self._flush_milvus)
            # Run sync memory.delete_all() in thread pool
            await asyncio.to_thread(self._sync_memory.delete_all, **kwargs)

            logger.info("All memories deleted for: %s", log_id(kwargs))
            return True

        except Exception as e:
            logger.error("mem0.delete_all() failed: %s", e, exc_info=True)
            report_error(e, context="memory_delete_all")
            return False

    async def update(
        self,
        memory_id: str,
        data: str,
        *,
        custom_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Update a memory using mem0's Memory.update() via asyncio.to_thread().

        ``metadata`` REPLACES the row's payload rather than merging into it.
        mem0's docstring says the opposite -- "existing metadata fields not
        specified here will be preserved" -- but ``_update_memory`` rebuilds the
        payload from what it is given and re-preserves only user_id, agent_id,
        run_id, actor_id and role. Verified against a live Milvus. Callers should
        read the row and pass its full metadata back with their edit applied.

        See: https://docs.mem0.ai/open-source/features/custom-update-memory-prompt
        """
        self._ensure_initialized()

        kwargs: dict[str, Any] = {
            "memory_id": memory_id,
            "data": data,
        }

        if metadata is not None:
            kwargs["metadata"] = metadata

        # Custom update prompt
        if custom_prompt:
            kwargs["prompt"] = custom_prompt

        try:
            logger.debug("mem0.update() memory_id=%s", log_id(memory_id))

            # Run sync memory.update() in thread pool
            response = await asyncio.to_thread(self._sync_memory.update, **kwargs)

            if response is None:
                raise MemoryUpdateError(
                    "mem0.update() returned None",
                    memory_id=memory_id,
                )

            logger.info("Memory updated: %s", log_id(memory_id))
            return MemoryEntry.from_mem0_result(response)

        except MemoryUpdateError:
            raise
        except Exception as e:
            logger.error("mem0.update() failed for %s: %s", log_id(memory_id), e, exc_info=True)
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
            logger.debug("mem0.history() memory_id=%s", log_id(memory_id))

            # Run sync memory.history() in thread pool
            history = await asyncio.to_thread(self._sync_memory.history, memory_id=memory_id)

            logger.debug("mem0.history() returned %s versions", len(history))
            return history

        except Exception as e:
            logger.error("mem0.history() failed for %s: %s", log_id(memory_id), e, exc_info=True)
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
            logger.error("mem0.reset() failed: %s", e, exc_info=True)
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
            logger.error("mem0.add() sync failed: %s", e, exc_info=True)
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
            logger.error("mem0.search() sync failed: %s", e, exc_info=True)
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
            logger.error("mem0.get() sync failed for %s: %s", log_id(memory_id), e, exc_info=True)
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
            logger.error("mem0.get_all() sync failed: %s", e, exc_info=True)
            return []

    def delete_sync(self, memory_id: str) -> bool:
        """Delete a memory by ID using mem0's Memory.delete()."""
        self._ensure_initialized()

        try:
            self._sync_memory.delete(memory_id=memory_id)
            return True
        except Exception as e:
            logger.error(
                "mem0.delete() sync failed for %s: %s", log_id(memory_id), e, exc_info=True
            )
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
            logger.error("mem0.delete_all() sync failed: %s", e, exc_info=True)
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
            logger.error(
                "mem0.update() sync failed for %s: %s", log_id(memory_id), e, exc_info=True
            )
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
            logger.error(
                "mem0.history() sync failed for %s: %s", log_id(memory_id), e, exc_info=True
            )
            return []

    def reset_sync(self) -> bool:
        """Reset entire memory store using mem0's Memory.reset()."""
        self._ensure_initialized()

        try:
            logger.warning("mem0.reset() sync - Resetting entire memory store")
            self._sync_memory.reset()
            return True
        except Exception as e:
            logger.error("mem0.reset() sync failed: %s", e, exc_info=True)
            return False
