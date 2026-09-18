"""
Memory Client - Unified interface for long-term memory.

Provides a high-level client that delegates to memory providers (mem0, etc.)
for actual memory operations.
"""

import asyncio
import threading
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from continuum.llm.untrusted_content import strip_hidden_chars
from continuum.logging import get_logger
from continuum.memory.base import BaseMemoryProvider
from continuum.memory.config import MemoryConfig
from continuum.memory.exceptions import (
    MemoryIdentifierError,
    MemoryNotEnabledError,
)
from continuum.memory.providers import create_provider, list_providers
from continuum.memory.scopes import MemoryScope
from continuum.memory.types import (
    PROVENANCE_LABELS_KEY,
    REVIEWED_KEY,
    MemoryAddResult,
    MemoryEntry,
    MemoryMetadata,
    MemorySearchResult,
)
from continuum.security.policy_context import resolve_active_policy

if TYPE_CHECKING:
    from continuum.security.policy import PolicyStore

T = TypeVar("T")


def _strip_hidden_from_messages(
    messages: str | list[dict[str, Any]] | list[str],
) -> str | list[dict[str, Any]] | list[str]:
    """Remove invisible codepoints from anything on its way into long-term memory.

    A zero-width or bidi-override sequence carries instructions the model's
    tokenizer reads while a human reviewer, a query over the collection, and a
    text classifier do not. ``_clean_tool`` already closes that channel for tool
    descriptions on first contact; memory is the same channel with a far longer
    half-life, because a stored payload is replayed into every future session.

    Done on write rather than on read on purpose. This is the only point where
    the payload can be destroyed rather than labelled, and it protects readers
    that never come through this SDK -- a dashboard, an export, another service
    querying the same collection.

    Copy-not-mutate: the session save loop reuses its message dicts, and editing
    in place would change what the short-term store persists.
    """
    if isinstance(messages, str):
        return strip_hidden_chars(messages)
    if not isinstance(messages, list):
        return messages

    cleaned: list[Any] = []
    for item in messages:
        if isinstance(item, str):
            cleaned.append(strip_hidden_chars(item))
        elif isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                cleaned.append({**item, "content": strip_hidden_chars(content)})
            else:
                cleaned.append(item)  # structured content: left as-is
        else:
            cleaned.append(item)
    return cleaned  # type: ignore[return-value]


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
        from continuum.memory import MemoryClient

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
        self._warned_shared_write = False

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

    def _enforce_memory_policy(
        self,
        operation: str,
        scope_label: str,
        policy_store: "PolicyStore | None",
        subject: str | None,
        data_labels: set[str] | None,
    ) -> set[str]:
        """Gate one memory operation on the run's data labels.

        Returns the effective labels, so a caller that also needs them -- the
        write path stamps them onto the row as provenance -- does not resolve the
        ambient policy a second time and risk disagreeing with the gate.

        One implementation for reads and writes. There used to be two: ``add``
        resolved the ambient run policy while ``search`` used its raw arguments,
        and since automatic retrieval passes no policy arguments the read gate
        never ran at all -- a run the policy said must not touch memory could
        still read every row out of it. Two copies of one rule is how one copy
        ends up wrong, and ``resolve_active_policy`` warns about exactly this:
        threading policy args through every call site is "fragile, and silently
        bypassed by any call site that forgets".

        Resources checked, in order:

        - ``memory:<operation>:<scope>`` -- the precise form, so a deployment can
          say "never persist this, but recalling is fine". With one shared
          resource string that was inexpressible, and a rule written to stop
          persistence would silently start denying retrieval.
        - ``memory:<scope>`` -- the legacy form. ``memory:*`` covers both new
          shapes by fnmatch, but an exact ``memory:u1`` covers neither, and such
          policies are already shipped. Checked so they do not quietly lapse.

        Labels ride as additional subjects, the same convention the tool and
        session gates use.
        """
        eff_store, eff_subject, eff_labels = resolve_active_policy(
            policy_store, subject, data_labels
        )
        labels = set(eff_labels or ())
        if eff_store is None or eff_subject is None:
            return labels

        from continuum.agent.exceptions import MemoryAccessDeniedError

        subjects = [eff_subject, *sorted(eff_labels)] if eff_labels else eff_subject
        for resource in (f"memory:{operation}:{scope_label}", f"memory:{scope_label}"):
            decision = eff_store.check(subjects, resource)
            if not decision.allowed:
                raise MemoryAccessDeniedError(
                    operation=operation,
                    scope=scope_label,
                    policy_name=decision.policy_name,
                )
        return labels

    def _provider_supports_gate(self) -> bool:
        """Does the provider accept ``pre_store_filter``?

        Checked by signature rather than by catching TypeError: a provider whose
        own body raises TypeError for an unrelated reason would otherwise look
        like an old provider and be silently retried without the gate.
        """
        import inspect

        try:
            params = inspect.signature(self._provider.add).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            return False
        # An explicit parameter only. A provider with **kwargs would ACCEPT the
        # argument and silently drop it -- no error, no gate, and no warning
        # either, which is worse than the TypeError this check exists to avoid.
        # Naming the parameter is how a provider says it will act on one.
        return "pre_store_filter" in params

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
        policy_store: "PolicyStore | None" = None,
        subject: str | None = None,
        data_labels: set[str] | None = None,
        pre_store_filter: Any | None = None,
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
            policy_store: Optional access control policy store. When provided,
                a "memory:write" resource check is performed before writing.
            subject: Caller identity for policy evaluation (typically agent name).

        Returns:
            MemoryAddResult with status and extracted memories.
        """
        self._ensure_enabled()

        # Access control. Explicit policy args win; otherwise the ambient run
        # policy is used — the session-save write path doesn't thread RunContext,
        # so that is how a tainted run's labels reach the gate.
        eff_labels = self._enforce_memory_policy(
            "write", agent_id or user_id or "unknown", policy_store, subject, data_labels
        )

        # Build scope from identifiers
        scope = self._build_scope(user_id, agent_id, conversation_id)
        identifiers = scope.to_identifiers()

        # A shared-scope write is global knowledge: one user's poisoned memory
        # becomes every user's retrieved fact, with no per-user scoping between
        # them. Permitted, and a legitimate deployment choice -- but not one to
        # arrive at by leaving a config field at a value set months ago, so say
        # it once on the path that actually does it.
        if self._config.memory_isolation == "shared" and not self._warned_shared_write:
            self._warned_shared_write = True
            logger.warning(
                "memory_isolation='shared': this write goes to a single global scope "
                "visible to every user and agent, so anything stored here -- including a "
                "fact extracted from attacker-influenced content -- is recalled for "
                "everyone. Set MEMORY_ISOLATION=user (the default) unless a shared "
                "knowledge base is intended, and deny 'memory:shared' in a PolicyStore "
                "for runs whose data must not become global."
            )

        messages = _strip_hidden_from_messages(messages)

        # Convert metadata if needed
        if isinstance(metadata, MemoryMetadata):
            metadata_dict = metadata.to_dict()
        else:
            metadata_dict = metadata

        # Provenance stamp (security finding F6).
        #
        # Record how tainted the run that produced this memory was, so a later
        # run that recalls the row can inherit the taint and be gated on it. The
        # labels come from the same resolution the write gate above uses, so the
        # session-save path -- which never threads RunContext -- is covered by
        # the ambient publish rather than needing a new parameter.
        #
        # Copy-not-mutate: the caller's dict is reused across messages in a save
        # loop, so stamping in place would leak one message's labels onto the
        # next. Absent labels write no key at all: a clean row must stay clean so
        # the read side can tell "never labelled" from "labelled with nothing".
        if eff_labels:
            metadata_dict = {
                **(metadata_dict or {}),
                PROVENANCE_LABELS_KEY: sorted(eff_labels),
            }

        # BaseMemoryProvider is a public interface, and a provider written
        # before the gate existed does not accept this parameter -- passing it
        # would raise TypeError and break memory entirely for an integration
        # that was working. Degrade instead, and say so: without the gate the
        # filter reverts to delete-after-write, which is weaker and racy, so an
        # operator who configured a filter needs to know which one they have.
        provider_kwargs: dict[str, Any] = {}
        if pre_store_filter is not None:
            if self._provider_supports_gate():
                provider_kwargs["pre_store_filter"] = pre_store_filter
            elif not getattr(self, "_warned_no_gate", False):
                self._warned_no_gate = True
                logger.warning(
                    "%s does not accept pre_store_filter, so rejected facts are deleted "
                    "AFTER the write rather than stopped before it. That delete can lose a "
                    "race with the store's write visibility and leave the fact searchable. "
                    "Use a provider that supports the gate, or infer=False for content that "
                    "must never be written.",
                    type(self._provider).__name__,
                )

        return await self._provider.add(
            messages,
            **provider_kwargs,
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
        policy_store: "PolicyStore | None" = None,
        subject: str | None = None,
        data_labels: set[str] | None = None,
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
            policy_store: Optional access control policy store. When provided,
                a "memory:read" resource check is performed before reading.
            subject: Caller identity for policy evaluation (typically agent name).

        Returns:
            MemorySearchResult with matching memories.
        """
        self._ensure_enabled()

        # Bound the query before it reaches the embedder. A large synthesized
        # prompt (e.g. a planner/drafter step input) can exceed the embedder's
        # input cap (~8191 tokens), which either hard-fails or silently returns
        # no results depending on the call path. Truncating a *search* query only
        # marginally affects recall, so this is a safe guard. Since a token spans
        # at least one character, capping characters caps tokens.
        max_query_chars = self._config.max_query_chars
        if max_query_chars is not None and len(query) > max_query_chars:
            logger.warning(
                f"Memory search query of {len(query)} chars exceeds max_query_chars="
                f"{max_query_chars}; truncating before embedding. Raise "
                f"MemoryConfig.max_query_chars (or set it to None) to change this."
            )
            query = query[:max_query_chars]

        # Access control. Same gate as the write path: automatic retrieval
        # passes no policy arguments, so resolving the ambient run policy here is
        # what makes this reachable at all.
        self._enforce_memory_policy(
            "read", agent_id or user_id or "unknown", policy_store, subject, data_labels
        )

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
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Update a specific memory.

        Args:
            memory_id: The ID of the memory to update
            data: New data for the memory
            custom_prompt: Custom prompt for memory update
            metadata: Replacement metadata for the row. REPLACES rather than
                merges: mem0's own docstring claims unspecified fields are
                preserved, and they are not -- it rebuilds the payload from what
                you pass, re-preserving only a fixed set (user_id, agent_id,
                run_id, actor_id, role) and dropping everything else, including
                Continuum's own ``_user_id``/``session_id``. So read the row and
                pass its whole metadata back with your edit applied. For the
                common case of clearing provenance after review, use
                :meth:`mark_reviewed`, which does that for you.

        Returns:
            Updated MemoryEntry.
        """
        self._ensure_enabled()
        kwargs: dict[str, Any] = {"custom_prompt": custom_prompt}
        # Only forward when supplied: passing metadata=None would still be a
        # replacement, wiping the row's payload for every existing caller.
        if metadata is not None:
            kwargs["metadata"] = metadata
        return await self._provider.update(memory_id, data, **kwargs)

    async def mark_reviewed(self, memory_id: str, *, reviewer: str) -> MemoryEntry:
        """Record that a person reviewed a tainted row, clearing its provenance.

        Provenance labelling is deliberately coarse: every row written by a
        tainted run is stamped, so a useful fact picked up from a fetched page
        carries the same label as a planted instruction. The SDK cannot tell them
        apart and does not try. This is the operation that lets a person who can.

        The label is replaced by a review record rather than deleted, so a
        blessed row stays distinguishable from one nobody ever looked at, and the
        decision keeps an owner. Once cleared, the row renders in the plain
        profile block and no longer taints runs that recall it -- which also
        means it stops denying whatever actions that label was gating. That is
        the reviewer's call to make, and worth surfacing to them before they
        make it.

        Args:
            memory_id: Row to mark.
            reviewer: Who reviewed it. Recorded verbatim for audit; in a real
                deployment this should be a staff identity, not the end user
                whose session may itself have been poisoned.

        Returns:
            The updated MemoryEntry.

        Raises:
            MemoryNotFoundError: If no such row exists -- telling a reviewer
                "approved" about a row that is not there would report a decision
                that was never recorded.
        """
        self._ensure_enabled()

        entry = await self._provider.get(memory_id)
        if entry is None:
            from continuum.memory.exceptions import MemoryNotFoundError

            raise MemoryNotFoundError(f"No memory with id {memory_id!r} to review")

        existing = entry.metadata if isinstance(entry.metadata, dict) else {}
        metadata = dict(existing)
        cleared = metadata.pop(PROVENANCE_LABELS_KEY, None)
        metadata[REVIEWED_KEY] = {
            "by": reviewer,
            "at": datetime.now(UTC).isoformat(),
            "cleared": sorted(cleared) if isinstance(cleared, list | tuple) else [],
        }

        return await self.update(memory_id, entry.memory, metadata=metadata)

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
        policy_store: "PolicyStore | None" = None,
        subject: str | None = None,
        data_labels: set[str] | None = None,
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
                policy_store=policy_store,
                subject=subject,
                data_labels=data_labels,
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
