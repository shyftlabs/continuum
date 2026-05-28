"""
Session Client - Unified interface for short-term and long-term memory.

Integrates session providers (short-term) with mem0 (long-term) using standardized IDs.
Provides a high-level API for managing conversations and memory.

Tracing is handled automatically via the @observe decorator.
"""

import threading
from typing import Any

from orchestrator.logging import get_logger
from orchestrator.memory import MemoryClient
from orchestrator.observability.decorators import observe
from orchestrator.observability.error_reporter import report_error
from orchestrator.session.base import BaseSessionProvider
from orchestrator.session.config import SessionConfig
from orchestrator.session.exceptions import (
    SessionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
from orchestrator.session.providers import create_provider, list_providers
from orchestrator.session.types import ChatMessage, SessionMetadata

logger = get_logger(__name__)

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
        from orchestrator.session import SessionClient

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
    ):
        """
        Initialize the Session Client.

        Args:
            session_config: Optional session configuration. Uses global settings if not provided.
            memory_client: Optional memory client. Uses global client if not provided.
            provider: Optional session provider. Created from registry if not provided.
            auto_initialize: Whether to initialize clients immediately.
        """
        self._session_config = session_config or SessionConfig()
        self._provider: BaseSessionProvider | None = provider
        self._memory_client: MemoryClient | None = memory_client
        self._initialized = False
        self._lock = threading.Lock()

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
            from orchestrator.core.container import get_container

            client = get_container().memory_client
            if client is None:
                raise RuntimeError(
                    "MemoryClient is not available. Ensure memory is enabled "
                    "and properly configured (MEMORY_ENABLED=true)."
                )
            self._memory_client = client
        return self._memory_client

    @property
    def is_enabled(self) -> bool:
        """Check if sessions are enabled."""
        return self._session_config.enabled

    def set_provider(self, provider: BaseSessionProvider) -> None:
        """Set the session provider."""
        self._provider = provider

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

            # Initialize session provider (if not already provided via constructor)
            if self._session_config.enabled and not self._provider:
                self._initialize_provider()

            # Initialize memory client from Container (if not provided)
            if not self._memory_client:
                from orchestrator.core.container import get_container

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
            session_id = await self.provider.get_or_create_session(
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
            await self.provider.add_message(
                session_id=session_id,
                message=message,
                metadata=metadata,
            )

            # Optionally add to long-term memory (mem0)
            # Memory storage is best-effort - failures should not break session operations
            if store_in_memory and self._memory_client and self._memory_client.is_enabled:
                try:
                    # Get session metadata to extract user_id and agent_id
                    session_metadata = await self.provider.get_session_metadata(session_id)

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
                                        logger.warning(
                                            f"Failed to delete filtered fact {fact_id}: {de}"
                                        )
                            stored_pairs = [(t, i) for t, i in stored_pairs if t in allowed]

                        if stored_pairs:
                            facts_preview = "; ".join(t[:60] for t, _ in stored_pairs[:3])
                            logger.info(
                                f"✅ Memory: {len(stored_pairs)} fact(s) stored — {facts_preview}"
                            )
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
                    # Memory storage failures should not break session operations
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
            messages = await self.provider.get_messages(
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
            session_metadata = await self.provider.get_session_metadata(session_id)

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
            result = await self.provider.clear_session(session_id=session_id)
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
            result = await self.provider.delete_session(session_id=session_id)
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
            return await self.provider.get_session_metadata(session_id=session_id)

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
            result = await self.provider.update_session_metadata(session_id, metadata)
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
        from orchestrator.session import initialize_global_session_client

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
