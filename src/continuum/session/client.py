"""
Session Client - Unified interface for short-term and long-term memory.

Integrates session providers (short-term) with mem0 (long-term) using standardized IDs.
Provides a high-level API for managing conversations and memory.

Tracing is handled automatically via the @observe decorator.
"""

import asyncio
import threading
from typing import Any

from continuum.core.background_tasks import BackgroundTaskRegistry
from continuum.logging import get_logger
from continuum.memory import MemoryClient
from continuum.observability.decorators import observe
from continuum.observability.error_reporter import report_error
from continuum.session.base import BaseSessionProvider
from continuum.session.config import SessionConfig
from continuum.session.exceptions import (
    SessionConnectionError,
    SessionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
from continuum.session.providers import create_provider, list_providers
from continuum.session.providers.memory import MemorySessionProvider
from continuum.session.types import ChatMessage, SessionMetadata

logger = get_logger(__name__)


def _in_temporal_activity() -> bool:
    """Return True if executing inside a Temporal activity right now.

    Used to force synchronous memory writes under Temporal regardless of the
    configured mode: a fire-and-forget task would escape the activity's durable,
    retriable boundary and could be lost when a worker is recycled. Detection is
    per-call (not a static config flag) so a process serving both HTTP requests
    and Temporal activities routes each correctly. Import-safe when the optional
    ``temporalio`` extra is not installed.
    """
    try:
        from temporalio import activity

        return activity.in_activity()
    except Exception:
        return False


# Global client state
_global_lock = threading.Lock()
_global_session_client: "SessionClient | None" = None
_initialized = False


class SessionClient:
    """
    Unified session client integrating session providers (short-term) and mem0 (long-term) memory.

    This client provides a high-level interface that:
        - Manages conversation history via session providers (short-term)
        - Integrates with mem0 for long-term memory
        - Uses standardized IDs (session_id maps to run_id in mem0)
        - Automatic tracing via @observe decorator
        - Supports multiple providers (Redis, DynamoDB, etc.)

    Example:
        ```python
        from continuum.session import SessionClient

        client = SessionClient()

        # Create or get session
        session_id = await client.get_or_create_session(
            user_id="user-123",
            conversation_id="conv-456"
        )

        # Add user message
        await client.add_message(
            session_id=session_id,
            message=ChatMessage(role="user", content="What's the weather like?")
        )

        # Get conversation history (short-term from provider)
        messages = await client.get_conversation_history(session_id)

        # Get relevant long-term memories (from mem0)
        memories = await client.get_relevant_memories(
            session_id=session_id,
            query="What does the user prefer?"
        )
        ```
    """

    def __init__(
        self,
        session_config: SessionConfig | None = None,
        memory_client: MemoryClient | None = None,
        provider: BaseSessionProvider | None = None,
        auto_initialize: bool = True,
        background_tasks: BackgroundTaskRegistry | None = None,
    ):
        """
        Initialize the Session Client.

        Args:
            session_config: Optional session configuration. Uses global settings if not provided.
            memory_client: Optional memory client. Uses global client if not provided.
            provider: Optional session provider. Created from registry if not provided.
            auto_initialize: Whether to initialize clients immediately.
            background_tasks: Optional registry for fire-and-forget memory writes
                (used when ``memory_write_mode='background'``). Falls back to the
                container's registry when not provided; if neither is available,
                the client transparently degrades to synchronous memory writes.
        """
        self._session_config = session_config or SessionConfig()
        self._provider: BaseSessionProvider | None = provider
        self._memory_client: MemoryClient | None = memory_client
        self._background_tasks = background_tasks
        self._initialized = False
        self._lock = threading.Lock()
        # An explicitly supplied provider is trusted as-is — never probed or
        # swapped for the in-memory fallback. A provider chosen lazily by the
        # client (the default path) is resolved once on first async use.
        self._explicit_provider = provider is not None
        self._provider_resolved = provider is not None
        self._resolve_lock = asyncio.Lock()

        if auto_initialize:
            self.initialize()

    @property
    def provider(self) -> BaseSessionProvider:
        """Get the session provider."""
        if not self._provider:
            self._initialize_provider()
        if self._provider is None:
            raise SessionNotEnabledError(
                "Session provider is not available. "
                "Check SESSION_ENABLED=true and provider configuration (SESSION_REDIS_HOST, SESSION_REDIS_PORT)."
            )
        return self._provider

    @property
    def config(self) -> SessionConfig:
        """Get the current configuration."""
        return self._session_config

    @property
    def memory_client(self) -> MemoryClient:
        """Get the memory client from Container."""
        if not self._memory_client:
            from continuum.core.container import get_container

            client = get_container().memory_client
            if client is None:
                raise RuntimeError(
                    "MemoryClient is not available. Ensure memory is enabled "
                    "and properly configured (MEMORY_ENABLED=true)."
                )
            self._memory_client = client
        return self._memory_client

    @property
    def background_tasks(self) -> BackgroundTaskRegistry | None:
        """Get the background task registry (from constructor or container).

        Returns None if no registry is available, in which case callers should
        fall back to synchronous execution.
        """
        if self._background_tasks is None:
            try:
                from continuum.core.container import get_container

                self._background_tasks = get_container().background_tasks
            except Exception:
                # Container not available (e.g. standalone/tests) — degrade gracefully.
                return None
        return self._background_tasks

    @property
    def is_enabled(self) -> bool:
        """Check if sessions are enabled."""
        return self._session_config.enabled

    def set_provider(self, provider: BaseSessionProvider) -> None:
        """Set the session provider explicitly (trusted; never probed/swapped)."""
        self._provider = provider
        self._explicit_provider = True
        self._provider_resolved = True

    async def _aprovider(self) -> BaseSessionProvider:
        """Resolve the active session provider, lazily and exactly once.

        On first use, chooses between Redis and the in-memory fallback:
        connectivity is probed once (not per request), and an unconfigured or
        unreachable Redis degrades to a non-durable in-memory provider with a
        single warning. An explicitly injected provider is returned untouched.
        """
        if self._provider is not None and self._provider_resolved:
            return self._provider

        async with self._resolve_lock:
            if self._provider is not None and self._provider_resolved:
                return self._provider
            provider = await self._resolve_provider()
            self._provider = provider
            self._provider_resolved = True
            return provider

    async def _resolve_provider(self) -> BaseSessionProvider:
        """Pick the concrete provider based on configuration and reachability."""
        cfg = self._session_config

        # Explicitly requested in-memory provider — chosen, not a degradation.
        if cfg.provider == "memory":
            logger.info("Session provider initialized: memory (in-process)")
            return MemorySessionProvider(cfg)

        # Redis host/credentials not configured — degrade quietly to in-memory.
        if not cfg.is_configured():
            return self._make_memory_fallback("Redis is not configured")

        # Configured: build the Redis provider lazily and probe it once.
        try:
            redis_provider = create_provider(cfg.provider, cfg)
        except Exception as e:  # provider class missing (e.g. redis not installed)
            return self._make_memory_fallback(f"Redis provider unavailable ({e})")

        aping = getattr(redis_provider, "aping", None)
        reachable = await aping() if aping is not None else True
        if reachable:
            logger.info(
                "Session provider initialized: %s",
                getattr(redis_provider, "provider_name", cfg.provider),
            )
            return redis_provider

        return self._make_memory_fallback("Redis is unreachable")

    def _make_memory_fallback(self, reason: str) -> BaseSessionProvider:
        """Build the in-memory provider and emit a single degradation warning."""
        logger.warning(
            "Session persistence falling back to non-durable in-memory store: %s. "
            "Sessions will not survive a restart and are not shared across workers. "
            "Set SESSION_ENABLED=false to silence this, or configure Redis "
            "(SESSION_REDIS_HOST / SESSION_REDIS_PORT) for durable sessions.",
            reason,
        )
        return MemorySessionProvider(self._session_config)

    def _degrade_to_memory(self, reason: str) -> None:
        """Swap the active provider to in-memory after a mid-session connection loss.

        Idempotent: once degraded, further connection errors don't re-warn. The
        warning is emitted exactly once, so a Redis that dies mid-session costs
        one log line, not an error per request.
        """
        if isinstance(self._provider, MemorySessionProvider):
            return
        self._provider = self._make_memory_fallback(reason)
        self._provider_resolved = True

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a provider method, degrading to in-memory on a connection loss.

        Resolves the provider (lazy, once), runs the method, and — if the
        resolved Redis provider raises a connection/timeout error mid-session —
        switches to the in-memory provider and retries the call there. Logical
        errors (SessionNotFoundError / SessionMessageLimitError) and an
        explicitly injected provider are never degraded.
        """
        provider = await self._aprovider()
        try:
            return await getattr(provider, method)(*args, **kwargs)
        except SessionConnectionError as e:
            # Already in-memory, or a caller-supplied provider — nothing to fall
            # back to; let the error surface.
            if self._explicit_provider or isinstance(provider, MemorySessionProvider):
                raise
            self._degrade_to_memory(f"Redis became unreachable mid-session ({e})")
            return await getattr(self._provider, method)(*args, **kwargs)

    def _initialize_provider(self) -> None:
        """Initialize the session provider using the registry."""
        if self._provider is not None:
            return

        if not self._session_config.enabled:
            logger.info("Sessions are disabled. Set SESSION_ENABLED=true to enable.")
            return

        try:
            provider_name = self._session_config.provider
            available = list_providers()

            if not available:
                logger.error(
                    "No session providers available. Install a provider package "
                    "(e.g., pip install redis for Redis provider)"
                )
                return

            if provider_name not in available:
                logger.warning(
                    f"Provider '{provider_name}' not available. "
                    f"Available providers: {available}. Falling back to '{available[0]}'"
                )
                provider_name = available[0]

            self._provider = create_provider(provider_name, self._session_config)
            logger.info(f"Session provider initialized: {provider_name}")

        except ImportError as e:
            logger.error(f"Failed to import session provider: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize session provider: {e}")

    def initialize(self) -> bool:
        """
        Initialize the session and memory clients.

        Thread-safe initialization that only runs once.

        Returns:
            True if initialization was successful, False otherwise.
        """
        with self._lock:
            if self._initialized:
                return True

            # NOTE: the session provider is intentionally NOT created here.
            # Connecting is deferred to first use (_aprovider) so that merely
            # constructing the client never opens a connection, and a disabled
            # or unreachable Redis costs no eager connection attempt.

            # Initialize memory client from Container (if not provided)
            if not self._memory_client:
                from continuum.core.container import get_container

                self._memory_client = get_container().memory_client

            self._initialized = True
            return True

    def _ensure_enabled(self) -> None:
        """Raise error if sessions are not enabled."""
        if not self.is_enabled:
            raise SessionNotEnabledError(
                "Session operations require sessions to be enabled. "
                "Set SESSION_ENABLED=true in your environment."
            )

    @observe(name="session_get_or_create", capture_output=True)
    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """
        Get existing session or create a new one.

        Creates or retrieves a session using deterministic session IDs.

        Args:
            session_id: Optional session ID. If not provided, generates a new UUID.
            user_id: Optional user identifier.
            conversation_id: Optional conversation identifier (e.g. chat window ID from caller).
                             When provided, scopes the session per conversation so history
                             is not shared across different chat windows.

        Returns:
            Session ID (existing or newly created).

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis connection fails.
        """
        self._ensure_enabled()

        try:
            session_id = await self._call(
                "get_or_create_session",
                session_id=session_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )

            logger.info(
                f"Session ready: {session_id}",
                extra={"user_id": user_id, "conversation_id": conversation_id},
            )

            return session_id

        except Exception as e:
            logger.error(f"Failed to get or create session: {e}")
            report_error(
                e,
                context="session_get_or_create",
                user_id=user_id,
                metadata={"session_id": session_id},
            )
            raise

    @observe(name="session_add_message", capture_output=True)
    async def add_message(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        store_in_memory: bool = True,
        extraction_prompt: str | None = None,
        pre_store_filter: Any | None = None,
        on_stored: Any | None = None,
    ) -> None:
        """
        Add a message to the session.

        Optionally stores the message in long-term memory (mem0) for fact extraction.

        Args:
            session_id: Session ID.
            message: Chat message to add.
            metadata: Additional metadata for the message.
            store_in_memory: Whether to also store in long-term memory (mem0).

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """
        self._ensure_enabled()

        try:
            # Add to short-term memory (via provider)
            await self._call(
                "add_message",
                session_id=session_id,
                message=message,
                metadata=metadata,
            )

            # Optionally add to long-term memory (mem0).
            # Memory storage is best-effort — failures must never break session ops.
            # In 'background' mode the (potentially slow) mem0 fact-extraction is
            # scheduled as a fire-and-forget task so it does not add latency to the
            # response; the short-term Redis write above is always synchronous.
            if store_in_memory and self._memory_client and self._memory_client.is_enabled:
                registry = self.background_tasks
                # Resolve the EFFECTIVE write mode at call time. Inside a Temporal
                # activity we always force 'sync' so the mem0 write completes within
                # the durable/retriable activity boundary (a fire-and-forget task
                # would escape Temporal's tracking and could be lost on worker
                # recycle). Detection is per-call, so a process that serves both
                # HTTP requests and Temporal activities routes each correctly.
                configured_mode = self._session_config.memory_write_mode
                in_temporal = _in_temporal_activity()
                effective_mode = "sync" if in_temporal else configured_mode
                sid = session_id[:8] if session_id else "none"
                if effective_mode == "background" and registry is not None:
                    logger.info(
                        "🧠 Memory write mode=background — scheduling mem0 write off the "
                        "response path (session=%s)",
                        sid,
                    )
                    scheduled = registry.spawn(
                        self._store_in_memory(
                            session_id=session_id,
                            message=message,
                            agent_id=agent_id,
                            metadata=metadata,
                            extraction_prompt=extraction_prompt,
                            pre_store_filter=pre_store_filter,
                            on_stored=on_stored,
                        ),
                        label=f"mem-write:{session_id[:8] if session_id else 'none'}",
                    )
                    # spawn() returns None if there's no running loop to schedule
                    # onto — fall back to inline so the write is never dropped.
                    if scheduled is None:
                        await self._store_in_memory(
                            session_id=session_id,
                            message=message,
                            agent_id=agent_id,
                            metadata=metadata,
                            extraction_prompt=extraction_prompt,
                            pre_store_filter=pre_store_filter,
                            on_stored=on_stored,
                        )
                else:
                    reason = (
                        " (forced: Temporal activity)"
                        if in_temporal and configured_mode == "background"
                        else ""
                    )
                    logger.info(
                        "🧠 Memory write mode=sync%s — awaiting mem0 write inline (session=%s)",
                        reason,
                        sid,
                    )
                    await self._store_in_memory(
                        session_id=session_id,
                        message=message,
                        agent_id=agent_id,
                        metadata=metadata,
                        extraction_prompt=extraction_prompt,
                        pre_store_filter=pre_store_filter,
                        on_stored=on_stored,
                    )

            logger.debug(
                f"Added message to session: {session_id}",
                extra={"role": message.role, "store_in_memory": store_in_memory},
            )

        except (SessionNotFoundError, SessionMessageLimitError):
            raise
        except Exception as e:
            logger.error(f"Failed to add message to session: {e}")
            report_error(
                e,
                context="session_add_message",
                metadata={"session_id": session_id, "message_role": message.role},
            )
            raise SessionError(
                f"Failed to add message to session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def _store_in_memory(
        self,
        *,
        session_id: str,
        message: ChatMessage,
        agent_id: str | None,
        metadata: dict[str, Any] | None,
        extraction_prompt: str | None,
        pre_store_filter: Any | None,
        on_stored: Any | None,
    ) -> None:
        """Extract and store long-term memory (mem0) for a single message.

        Best-effort: any failure is logged/reported but never raised, so this is
        safe to run either inline (sync mode) or as a fire-and-forget background
        task (background mode). The short-term session write is handled separately
        by the caller and is always synchronous.
        """
        try:
            # Get session metadata to extract user_id and agent_id
            session_metadata = await self._call("get_session_metadata", session_id)

            if session_metadata:
                # Build memory metadata for observability
                memory_metadata = dict(metadata) if metadata else {}
                if session_id:
                    memory_metadata["session_id"] = session_id
                if session_metadata.user_id:
                    memory_metadata["_user_id"] = session_metadata.user_id
                if agent_id:
                    memory_metadata["_agent_id"] = agent_id

                logger.debug(
                    f"🧠 Extracting memory: role={message.role} "
                    f"session={session_id[:8] if session_id else 'none'} "
                    f"user={session_metadata.user_id[:8] if session_metadata.user_id else 'none'}"
                )

                result = await self.memory_client.add(
                    messages=[message.to_dict()],
                    user_id=session_metadata.user_id,
                    agent_id=agent_id,
                    conversation_id=session_metadata.conversation_id,
                    metadata=memory_metadata,
                    custom_prompt=extraction_prompt,
                )

                # Build list of (fact_text, fact_id) for stored facts
                stored_pairs: list[tuple[str, str | None]] = []
                for fact in result.results:
                    if isinstance(fact, dict):
                        fact_text = fact.get("memory") or fact.get("text") or str(fact)
                        fact_id = fact.get("id")
                    else:
                        fact_text = (
                            getattr(fact, "memory", None)
                            or getattr(fact, "text", None)
                            or str(fact)
                        )
                        fact_id = getattr(fact, "id", None)
                    if fact_text:
                        stored_pairs.append((fact_text, fact_id))

                # Apply pre_store_filter: delete facts that don't pass (best-effort)
                if pre_store_filter and stored_pairs:
                    fact_texts = [t for t, _ in stored_pairs]
                    try:
                        allowed = set(pre_store_filter(fact_texts))
                    except Exception as fe:
                        logger.warning(f"pre_store_filter failed: {fe}")
                        allowed = set(fact_texts)
                    filtered_out = [(t, i) for t, i in stored_pairs if t not in allowed]
                    if filtered_out:
                        logger.info(
                            f"🚫 PII filter blocked {len(filtered_out)} fact(s): {[t for t, _ in filtered_out]}"
                        )
                    for _fact_text, fact_id in filtered_out:
                        if fact_id:
                            try:
                                await self.memory_client.delete(fact_id)
                            except Exception as de:
                                logger.warning(f"Failed to delete filtered fact {fact_id}: {de}")
                    stored_pairs = [(t, i) for t, i in stored_pairs if t in allowed]

                if stored_pairs:
                    facts_preview = "; ".join(t[:60] for t, _ in stored_pairs[:3])
                    logger.info(f"✅ Memory: {len(stored_pairs)} fact(s) stored — {facts_preview}")
                else:
                    logger.debug("🧠 Memory: no new facts extracted")

                # Fire on_stored callback with final stored fact texts
                final_facts = [t for t, _ in stored_pairs]
                if on_stored and final_facts:
                    try:
                        on_stored(final_facts)
                    except Exception as ce:
                        logger.warning(f"on_stored callback failed: {ce}")
            else:
                logger.warning(
                    f"⚠️ Cannot store memory: Session metadata not found for session_id={session_id[:8] if session_id else 'none'}"
                )
        except Exception as mem_error:
            # Memory storage failures should not break session operations.
            from continuum.agent.exceptions import MemoryAccessDeniedError

            if isinstance(mem_error, MemoryAccessDeniedError):
                # Expected: a data-label policy blocked the long-term write. This
                # is the gate working as designed — not a failure. Log quietly
                # (no traceback) and don't escalate to error reporting.
                logger.info(
                    "🛡️ Long-term memory write blocked by policy '%s' "
                    "(run carried restricted data labels)",
                    mem_error.context.get("policy_name"),
                )
            else:
                logger.error(
                    f"❌ Memory storage failed: {mem_error}",
                    exc_info=True,
                    extra={"session_id": session_id, "role": message.role},
                )
                report_error(
                    mem_error,
                    context="session_memory_storage",
                    metadata={"session_id": session_id, "message_role": message.role},
                )

    @observe(name="session_get_history", capture_output=True)
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """
        Get conversation history from short-term memory (Redis).

        Args:
            session_id: Session ID.
            limit: Number of complete turns (request+response pairs) to retrieve.

        Returns:
            List of ChatMessage objects in chronological order.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """
        self._ensure_enabled()

        try:
            messages: list[ChatMessage] = await self._call(
                "get_messages",
                session_id=session_id,
                limit=limit,
            )
            return messages

        except (SessionNotFoundError, SessionMessageLimitError):
            raise
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            report_error(
                e,
                context="session_get_history",
                metadata={"session_id": session_id},
            )
            raise SessionError(
                f"Failed to get conversation history: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_get_memories", capture_output=True)
    async def get_relevant_memories(
        self,
        session_id: str,
        query: str,
        *,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        """
        Get relevant long-term memories from mem0.

        Args:
            session_id: Session ID.
            query: Search query for semantic search.
            limit: Maximum number of memories to retrieve.

        Returns:
            List of MemoryEntry objects.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """
        self._ensure_enabled()

        if not self._memory_client or not self._memory_client.is_enabled:
            logger.warning("Memory client not enabled, returning empty list")
            return []

        try:
            # Get session metadata to extract user_id and agent_id
            session_metadata = await self._call("get_session_metadata", session_id)

            if not session_metadata:
                logger.warning(f"Session metadata not found: {session_id}")
                return []

            search_result = await self.memory_client.search(
                query=query,
                user_id=session_metadata.user_id,
                agent_id=agent_id,
                conversation_id=session_metadata.conversation_id,
                limit=limit,
            )

            return search_result.results

        except Exception as e:
            logger.error(f"Failed to get relevant memories: {e}")
            report_error(
                e,
                context="session_get_memories",
                metadata={"session_id": session_id, "query": query[:100]},
            )
            raise SessionError(
                f"Failed to get relevant memories: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_clear", capture_output=True)
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session (but keep metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if cleared successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """
        self._ensure_enabled()

        try:
            result: bool = await self._call("clear_session", session_id=session_id)
            return result

        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            report_error(
                e,
                context="session_clear",
                metadata={"session_id": session_id},
            )
            raise SessionError(
                f"Failed to clear session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_delete", capture_output=True)
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely (messages and metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if deleted successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """
        self._ensure_enabled()

        try:
            result: bool = await self._call("delete_session", session_id=session_id)
            return result

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            report_error(
                e,
                context="session_delete",
                metadata={"session_id": session_id},
            )
            raise SessionError(
                f"Failed to delete session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_get_metadata", capture_output=True)
    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        """
        Get session metadata.

        Args:
            session_id: Session ID.

        Returns:
            SessionMetadata if found, None otherwise.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """
        self._ensure_enabled()

        try:
            metadata_result: SessionMetadata | None = await self._call(
                "get_session_metadata", session_id=session_id
            )
            return metadata_result

        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            report_error(
                e,
                context="session_get_metadata",
                metadata={"session_id": session_id},
            )
            raise SessionError(
                f"Failed to get session metadata: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_update_metadata", capture_output=True)
    async def update_session_metadata(
        self,
        session_id: str,
        metadata: SessionMetadata,
    ) -> bool:
        """
        Update session metadata.

        Args:
            session_id: Session ID.
            metadata: Updated session metadata.

        Returns:
            True if updated successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionError: If operation fails.
        """

        self._ensure_enabled()

        try:
            result: bool = await self._call("update_session_metadata", session_id, metadata)
            if not result:
                logger.warning(f"Session not found when updating metadata: {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")
            report_error(
                e,
                context="session_update_metadata",
                metadata={"session_id": session_id},
            )
            raise SessionError(
                f"Failed to update session metadata: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e


# =============================================================================
# Global Session Client Functions
# =============================================================================


def initialize_global_session_client(
    session_config: SessionConfig | None = None,
    memory_client: MemoryClient | None = None,
) -> bool:
    """
    Initialize the global Session Client.

    This should be called once at application startup. Subsequent calls
    will return the existing initialization status.

    Args:
        session_config: Optional session configuration. Uses global settings if not provided.
        memory_client: Optional memory client. Uses global client if not provided.

    Returns:
        True if initialization was successful.

    Example:
        ```python
        from continuum.session import initialize_global_session_client

        # At application startup
        if initialize_global_session_client():
            print("Session Client ready")
        else:
            print("Session Client not configured or disabled")
        ```
    """
    global _global_session_client, _initialized

    with _global_lock:
        if _initialized:
            return _global_session_client is not None and _global_session_client.is_enabled

        _global_session_client = SessionClient(
            session_config=session_config,
            memory_client=memory_client,
            auto_initialize=True,
        )
        _initialized = True

        return _global_session_client.is_enabled


def get_global_session_client() -> SessionClient:
    """
    Get the global Session Client.

    Auto-initializes if not already initialized.

    Returns:
        The global SessionClient instance.

    Example:
        ```python
        client = get_global_session_client()
        if client.is_enabled:
            session_id = await client.get_or_create_session(user_id="user-123")
        ```
    """
    global _global_session_client, _initialized

    if not _initialized:
        initialize_global_session_client()

    if _global_session_client is None:
        with _global_lock:
            if _global_session_client is None:
                _global_session_client = SessionClient(auto_initialize=True)
                _initialized = True

    return _global_session_client


def reset_global_session() -> None:
    """Reset the global session client. Useful for testing."""
    global _global_session_client, _initialized

    with _global_lock:
        _global_session_client = None
        _initialized = False
