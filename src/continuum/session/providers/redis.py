"""
Redis Session Provider - Session provider implementation using Redis.

Uses Redis for efficient session storage with:
- Redis Lists for message storage
- JSON for metadata storage
- TTL support for automatic expiration
- Sliding window for message limit management
"""

import json
import threading
from datetime import UTC, datetime
from typing import Any

from continuum.connectors.redis import RedisConnector
from continuum.logging import get_logger
from continuum.observability.decorators import observe
from continuum.observability.error_reporter import report_error
from continuum.session.base import BaseSessionProvider
from continuum.session.config import SessionConfig
from continuum.session.exceptions import (
    SessionConnectionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
from continuum.session.types import (
    ChatMessage,
    SessionMessage,
    SessionMetadata,
    generate_session_id,
)
from continuum.utils.sanitization import validate_conversation_id, validate_user_id

logger = get_logger(__name__)


class RedisSessionProvider(BaseSessionProvider):
    """
    Session provider using Redis.

    This provider leverages Redis for efficient session storage:
    - Redis Lists for message storage (chronological order)
    - JSON for metadata storage
    - TTL support for automatic expiration
    - Sliding window strategy for message limits

    Example:
        ```python
        from continuum.session.config import SessionConfig
        from continuum.session.providers.redis import RedisSessionProvider

        config = SessionConfig()
        provider = RedisSessionProvider(config)

        # Create session
        session_id = await provider.get_or_create_session(user_id="user-123")

        # Add message
        await provider.add_message(session_id, ChatMessage(role="user", content="Hello"))
        ```
    """

    def __init__(
        self,
        config: SessionConfig,
        auto_initialize: bool = True,
    ):
        """
        Initialize the Redis session provider.

        Args:
            config: Session configuration
            auto_initialize: Whether to initialize the Redis connection immediately.

        Raises:
            ImportError: If redis package is not installed
        """
        self._config = config
        self._redis: Any = None  # redis.Redis instance
        self._pool: Any = None  # redis.ConnectionPool instance
        # The connector is the single place the Redis client/pool is built.
        self._connector = RedisConnector(config)
        self._initialized = False
        self._lock = threading.Lock()

        if auto_initialize:
            self.initialize()

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "redis"

    @property
    def config(self) -> SessionConfig:
        """Get the current configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized and ready."""
        return self._config.enabled and self._initialized and self._redis is not None

    def initialize(self) -> bool:
        """
        Initialize the Redis connection.

        Thread-safe initialization that only runs once.

        Returns:
            True if initialization was successful, False otherwise.
        """
        with self._lock:
            if self._initialized:
                return self._redis is not None

            if not self._config.enabled:
                logger.info("Sessions are disabled. Set SESSION_ENABLED=true to enable.")
                self._initialized = True
                return False

            if not self._config.is_configured():
                logger.warning(
                    "Session not properly configured. Check required settings: "
                    "SESSION_REDIS_HOST, SESSION_REDIS_PORT"
                )
                self._initialized = True
                return False

            try:
                # The RedisConnector owns the client/pool construction (single
                # source of truth for connection params, mode, and auth). It uses
                # a BlockingConnectionPool so bursts above max_connections queue
                # for a free connection rather than dropping the request
                # (prevents silent write loss under concurrent load, S-001).
                self._redis = self._connector.build_client()
                self._pool = self._connector.pool

                self._initialized = True
                logger.info(
                    "Redis Session Provider initialized successfully",
                    extra={
                        "redis_host": self._config.redis_host,
                        "redis_port": self._config.redis_port,
                        "ttl_seconds": self._config.ttl_seconds,
                        "max_messages": self._config.max_messages,
                        "max_connections": self._config.redis_max_connections,
                    },
                )
                return True

            except ImportError:
                logger.error("redis package not installed. Run: pip install redis")
                self._initialized = True
                return False
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to initialize Redis Session Provider: {error_msg}")
                self._initialized = True
                return False

    async def aping(self) -> bool:
        """Best-effort connectivity probe — returns False on any failure, never raises.

        Ensures the (lazy) connection pool exists, then issues a single PING.
        The session client uses this once, on first use, to decide between Redis
        and the in-memory fallback — so an unreachable Redis costs one quiet probe
        instead of an error logged on every request.
        """
        try:
            if not self._initialized:
                self.initialize()
            if self._redis is None:
                return False
            await self._redis.ping()
            return True
        except Exception:
            return False

    def _ensure_enabled(self) -> None:
        """Raise error if sessions are not enabled."""
        if not self.is_initialized:
            raise SessionNotEnabledError(
                "Session operations require sessions to be enabled. "
                "Set SESSION_ENABLED=true in your environment."
            )

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session messages."""
        return f"{self._config.key_prefix}:{session_id}:messages"

    def _get_metadata_key(self, session_id: str) -> str:
        """Get Redis key for session metadata."""
        return f"{self._config.key_prefix}:{session_id}:metadata"

    def _compute_session_id(
        self,
        session_id: str | None,
        user_id: str | None,
        conversation_id: str | None,
    ) -> str:
        """Compute deterministic session ID.

        Priority:
        - explicit session_id → use as-is (internal handoff calls)
        - conversation_id + user_id → "c:{conversation_id}:u:{user_id}"
        - user_id only             → "u:{user_id}"
        - fallback                 → generate UUID

        Namespace prefixes ("c:" / "u:") prevent collision between a bare
        user_id like "foo:bar" and a conversation_id="foo" + user_id="bar"
        pair, which would otherwise both produce the same key "foo:bar".

        Raises:
            InvalidIdentifierError: if user_id/conversation_id contain characters
                unsafe for a Redis key fragment. An explicit session_id is trusted
                as-is (internal handoff calls supply framework-generated ids).
        """
        if session_id:
            return session_id
        user_id = validate_user_id(user_id)
        conversation_id = validate_conversation_id(conversation_id)
        if conversation_id and user_id:
            return f"c:{conversation_id}:u:{user_id}"
        if user_id:
            return f"u:{user_id}"
        return generate_session_id()

    @observe(name="session_provider_get_or_create", capture_output=True)
    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """
        Get existing session or create a new one.

        Session ID is deterministic:
        - explicit session_id       → used as-is (for internal handoff calls)
        - conversation_id + user_id → "c:{conversation_id}:u:{user_id}"
        - user_id only              → "u:{user_id}"
        - neither                   → generate UUID

        Args:
            session_id: Optional explicit session ID (overrides computed key).
            user_id: User identifier.
            conversation_id: Conversation identifier from caller (e.g. chat window ID).

        Returns:
            Session ID (existing or newly created).

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis connection fails.
        """
        self._ensure_enabled()

        resolved_session_id = self._compute_session_id(session_id, user_id, conversation_id)

        try:
            metadata_key = self._get_metadata_key(resolved_session_id)
            metadata_json = await self._redis.get(metadata_key)

            if metadata_json:
                # Session exists — refresh TTLs and return
                metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                metadata.last_accessed_at = datetime.now(UTC)
                messages_key = self._get_session_key(resolved_session_id)
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.set(
                        metadata_key, json.dumps(metadata.to_dict()), ex=self._config.ttl_seconds
                    )
                    pipe.expire(messages_key, self._config.ttl_seconds)
                    await pipe.execute()
                logger.debug(f"Retrieved existing session: {resolved_session_id}")
                return resolved_session_id

            # Session doesn't exist — create it atomically with SET NX so that
            # concurrent requests with the same deterministic key don't both
            # write and have the second silently overwrite the first.
            metadata = SessionMetadata(
                session_id=resolved_session_id,
                user_id=user_id,
                conversation_id=conversation_id,
                created_at=datetime.now(UTC),
                last_accessed_at=datetime.now(UTC),
                message_count=0,
            )
            nx_ok = await self._redis.set(
                metadata_key,
                json.dumps(metadata.to_dict()),
                ex=self._config.ttl_seconds,
                nx=True,
            )
            if not nx_ok:
                # Lost the race — another request created the session first.
                # Refresh TTLs on the winner's keys and return.
                messages_key = self._get_session_key(resolved_session_id)
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.expire(metadata_key, self._config.ttl_seconds)
                    pipe.expire(messages_key, self._config.ttl_seconds)
                    await pipe.execute()
                logger.debug(
                    f"Race resolved: session already created by concurrent request: {resolved_session_id}"
                )
            else:
                logger.info(
                    f"Created new session: {resolved_session_id}",
                    extra={"user_id": user_id, "conversation_id": conversation_id},
                )
            return resolved_session_id

        except (SessionNotEnabledError, SessionConnectionError):
            raise
        except Exception as e:
            logger.error(f"Failed to get or create session: {e}")
            report_error(
                e,
                context="session_provider_get_or_create",
                metadata={"session_id": resolved_session_id, "user_id": user_id},
            )
            raise SessionConnectionError(
                f"Failed to get or create session: {str(e)}",
                session_id=resolved_session_id,
                original_error=e,
            ) from e

    @observe(name="session_provider_add_message", capture_output=True)
    async def add_message(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a message to the session.

        Args:
            session_id: Session ID.
            message: Chat message to add.
            metadata: Additional metadata for the message.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionNotFoundError: If session doesn't exist.
            SessionMessageLimitError: If message limit is exceeded (when strategy='error').
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            # Check if session exists
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if not metadata_json:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}",
                    session_id=session_id,
                )

            # Update session metadata
            try:
                session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
            except (json.JSONDecodeError, TypeError, KeyError) as parse_err:
                logger.error(
                    f"Corrupt session metadata for {session_id}: {parse_err}. "
                    f"Raw preview: {str(metadata_json)[:200]}"
                )
                raise SessionConnectionError(
                    f"Corrupt session metadata for session: {session_id}",
                    session_id=session_id,
                    original_error=parse_err,
                ) from parse_err
            session_metadata.last_accessed_at = datetime.now(UTC)

            messages_key = self._get_session_key(session_id)

            # Use the actual list length (atomic source of truth) for the limit
            # check rather than the stored counter, which can drift under
            # concurrent writes (read-modify-write race on message_count).
            current_count = await self._redis.llen(messages_key)
            session_metadata.message_count = current_count

            # Check message limit and apply strategy
            if session_metadata.message_count >= self._config.max_messages:
                if self._config.message_limit_strategy == "error":
                    raise SessionMessageLimitError(
                        f"Session message limit exceeded: {session_metadata.message_count} >= {self._config.max_messages}",
                        session_id=session_id,
                        current_count=session_metadata.message_count,
                        max_messages=self._config.max_messages,
                    )
                else:
                    # Sliding window strategy: remove oldest messages
                    trim_count = self._config.sliding_window_trim_count

                    # Use LTRIM to remove oldest messages (keep from trim_count to end)
                    await self._redis.ltrim(messages_key, trim_count, -1)

                    # Update message count after trim
                    actual_count = await self._redis.llen(messages_key)
                    session_metadata.message_count = actual_count

                    logger.info(
                        f"Sliding window triggered: trimmed {trim_count} oldest messages",
                        extra={
                            "session_id": session_id,
                            "trimmed_count": trim_count,
                            "new_count": actual_count,
                        },
                    )

            # Create session message with metadata (no trace_id/span_id - handled by @observe)
            session_message = SessionMessage(
                message=message,
                timestamp=datetime.now(UTC),
                metadata=metadata or {},
            )

            # Atomically write the message and refresh its TTL. RPUSH returns the
            # new list length atomically, which we use as the authoritative
            # message_count (no read-modify-write race under concurrency).
            message_json = json.dumps(session_message.to_dict())
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(messages_key, message_json)
                pipe.expire(messages_key, self._config.ttl_seconds)
                results = await pipe.execute()

            # results[0] is the RPUSH return value = new list length
            session_metadata.message_count = results[0]
            metadata_json_updated = json.dumps(session_metadata.to_dict())
            await self._redis.set(metadata_key, metadata_json_updated, ex=self._config.ttl_seconds)

            logger.debug(
                f"Added message to session: {session_id}",
                extra={"message_count": session_metadata.message_count, "role": message.role},
            )

        except (SessionNotFoundError, SessionMessageLimitError):
            raise
        except Exception as e:
            logger.error(f"Failed to add message to session: {e}")
            report_error(
                e,
                context="session_provider_add_message",
                metadata={"session_id": session_id, "message_role": message.role},
            )
            raise SessionConnectionError(
                f"Failed to add message to session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_provider_get_messages", capture_output=True)
    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID.
            limit: Number of complete turns (request+response pairs) to retrieve.
                   Fetches limit*2 raw messages then trims to a clean turn boundary.

        Returns:
            List of ChatMessage objects in chronological order.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionNotFoundError: If session doesn't exist.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            # Check if session exists
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if not metadata_json:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}",
                    session_id=session_id,
                )

            # Get messages from Redis List
            messages_key = self._get_session_key(session_id)
            message_jsons = await self._redis.lrange(messages_key, 0, -1)

            # Parse messages (skip malformed JSON entries gracefully)
            session_messages = []
            for msg_json in message_jsons:
                try:
                    session_messages.append(SessionMessage.from_dict(json.loads(msg_json)))
                except (json.JSONDecodeError, TypeError, KeyError, ValueError) as parse_err:
                    logger.warning(
                        f"Skipping malformed session message in {session_id}: {parse_err}. "
                        f"Raw preview: {str(msg_json)[:200]}"
                    )

            # Convert to ChatMessage list
            messages = [sm.message for sm in session_messages]

            # Apply limit if specified. limit is measured in complete turns
            # (request+response pairs), so fetch limit*2 raw messages.
            # Then trim to the first user message as a safety net against
            # any orphaned assistant message from a prior partial save.
            if limit and limit > 0:
                sliced = messages[-(limit * 2) :]
                first_user = next(
                    (i for i, m in enumerate(sliced) if m.role == "user"),
                    len(sliced),
                )
                messages = sliced[first_user:]

            # Update last_accessed_at and refresh TTL on both keys.
            # Always done — even when the messages list is empty — so that a
            # read-heavy session doesn't let the messages list expire while the
            # metadata key stays alive.
            session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
            session_metadata.last_accessed_at = datetime.now(UTC)
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.set(
                    metadata_key,
                    json.dumps(session_metadata.to_dict()),
                    ex=self._config.ttl_seconds,
                )
                pipe.expire(messages_key, self._config.ttl_seconds)
                await pipe.execute()

            logger.debug(
                f"Retrieved {len(messages)} messages from session: {session_id}",
                extra={"limit": limit},
            )

            return messages

        except SessionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get messages from session: {e}")
            report_error(
                e,
                context="session_provider_get_messages",
                metadata={"session_id": session_id},
            )
            raise SessionConnectionError(
                f"Failed to get messages from session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_provider_get_metadata", capture_output=True)
    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        """
        Get session metadata.

        Args:
            session_id: Session ID.

        Returns:
            SessionMetadata if found, None otherwise.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if not metadata_json:
                return None

            metadata = SessionMetadata.from_dict(json.loads(metadata_json))
            # message_count is derived from the actual list length (the single
            # source of truth). The stored counter can drift under concurrent
            # writes (read-modify-write race), so we always reconcile from LLEN.
            metadata.message_count = await self._redis.llen(self._get_session_key(session_id))
            return metadata

        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            report_error(
                e,
                context="session_provider_get_metadata",
                metadata={"session_id": session_id},
            )
            raise SessionConnectionError(
                f"Failed to get session metadata: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_provider_update_metadata", capture_output=True)
    async def update_session_metadata(self, session_id: str, metadata: SessionMetadata) -> bool:
        """
        Persist updated session metadata and refresh TTLs on both keys.

        Args:
            session_id: Session ID.
            metadata: Updated SessionMetadata to persist.

        Returns:
            True if updated successfully, False if the session does not exist.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            metadata_key = self._get_metadata_key(session_id)
            messages_key = self._get_session_key(session_id)

            # Only update if the session actually exists
            if not await self._redis.exists(metadata_key):
                return False

            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.set(metadata_key, json.dumps(metadata.to_dict()), ex=self._config.ttl_seconds)
                pipe.expire(messages_key, self._config.ttl_seconds)
                await pipe.execute()

            logger.debug(f"Updated session metadata: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")
            report_error(
                e,
                context="session_provider_update_metadata",
                metadata={"session_id": session_id},
            )
            raise SessionConnectionError(
                f"Failed to update session metadata: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_provider_clear", capture_output=True)
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session (but keep metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if cleared successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            messages_key = self._get_session_key(session_id)
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            if metadata_json:
                session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                session_metadata.message_count = 0
                session_metadata.last_accessed_at = datetime.now(UTC)
                # Delete messages and update metadata atomically so a concurrent
                # add_message cannot slip in and leave the count permanently wrong.
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.delete(messages_key)
                    pipe.set(
                        metadata_key,
                        json.dumps(session_metadata.to_dict()),
                        ex=self._config.ttl_seconds,
                    )
                    await pipe.execute()
            else:
                await self._redis.delete(messages_key)

            logger.info(f"Cleared session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            report_error(
                e,
                context="session_provider_clear",
                metadata={"session_id": session_id},
            )
            raise SessionConnectionError(
                f"Failed to clear session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    @observe(name="session_provider_delete", capture_output=True)
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely (messages and metadata).

        Also cleans up the user+agent to session_id mapping if it exists.

        Args:
            session_id: Session ID.

        Returns:
            True if deleted successfully.

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis operation fails.
        """
        self._ensure_enabled()

        try:
            messages_key = self._get_session_key(session_id)
            metadata_key = self._get_metadata_key(session_id)

            # Delete session data
            await self._redis.delete(messages_key, metadata_key)

            logger.info(f"Deleted session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            report_error(
                e,
                context="session_provider_delete",
                metadata={"session_id": session_id},
            )
            raise SessionConnectionError(
                f"Failed to delete session: {str(e)}",
                session_id=session_id,
                original_error=e,
            ) from e

    async def close(self) -> None:
        """
        Close Redis connections and cleanup resources.

        Respects shared_services_enabled setting - if True, doesn't close
        Redis connections (they persist as a shared service).
        """
        from continuum.config import settings

        if not self._initialized or self._redis is None:
            return

        # If Redis is a shared service, don't close connections
        if settings.shared_services_enabled:
            logger.debug("Redis is a shared service, skipping connection close")
            return

        with self._lock:
            try:
                # Close Redis connection pool
                if self._pool is not None:
                    await self._pool.aclose()
                    logger.debug("Redis connection pool closed")

                # Close Redis client
                if self._redis is not None:
                    await self._redis.aclose()
                    logger.debug("Redis client closed")

                self._redis = None
                self._initialized = False
                logger.info("Redis Session Provider closed")

            except Exception as e:
                logger.warning(f"Error closing Redis Session Provider: {e}")
                self._initialized = False
