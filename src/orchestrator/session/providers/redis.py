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
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe
from orchestrator.observability.error_reporter import report_error
from orchestrator.session.base import BaseSessionProvider
from orchestrator.session.config import SessionConfig
from orchestrator.session.exceptions import (
    SessionConnectionError,
    SessionMessageLimitError,
    SessionNotEnabledError,
    SessionNotFoundError,
)
from orchestrator.session.types import (
    ChatMessage,
    SessionMessage,
    SessionMetadata,
    generate_session_id,
)

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
        from orchestrator.session.config import SessionConfig
        from orchestrator.session.providers.redis import RedisSessionProvider

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
                # Create connection pool for better performance
                self._pool = redis.ConnectionPool(
                    host=self._config.redis_host,
                    port=self._config.redis_port,
                    password=self._config.redis_password,
                    db=self._config.redis_db,
                    max_connections=self._config.redis_max_connections,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )

                # Create async Redis connection using pool
                self._redis = redis.Redis(
                    connection_pool=self._pool,
                    decode_responses=True,
                )

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

    def _get_user_agent_session_key(self, user_id: str, agent_id: str) -> str:
        """Get Redis key for user+agent to session_id mapping."""
        # Normalize user_id and agent_id to avoid issues with special characters
        # Use a simple format that's safe for Redis keys
        return f"{self._config.key_prefix}:user:{user_id}:agent:{agent_id}:session"

    @observe(name="session_provider_get_or_create", capture_output=True)
    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        """
        Get existing session or create a new one.

        If session_id is not provided but user_id and agent_id are, this will:
        1. Look up existing session for the user_id + agent_id combination
        2. If found, return that session_id
        3. If not found, create a new session and store the mapping

        Args:
            session_id: Optional session ID. If not provided, will look up by user_id+agent_id or generate new.
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.

        Returns:
            Session ID (existing or newly created).

        Raises:
            SessionNotEnabledError: If sessions are disabled.
            SessionConnectionError: If Redis connection fails.
        """
        self._ensure_enabled()

        try:
            # If session_id is provided, check if it exists
            if session_id:
                metadata_key = self._get_metadata_key(session_id)
                metadata_json = await self._redis.get(metadata_key)

                if metadata_json:
                    # Session exists, update last_accessed_at
                    metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                    metadata.last_accessed_at = datetime.now()
                    await self._redis.setex(
                        metadata_key,
                        self._config.ttl_seconds,
                        json.dumps(metadata.to_dict()),
                    )
                    logger.debug(f"Retrieved existing session: {session_id}")
                    return session_id
                else:
                    # Session ID provided but doesn't exist - create it
                    metadata = SessionMetadata(
                        session_id=session_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        created_at=datetime.now(),
                        last_accessed_at=datetime.now(),
                        message_count=0,
                    )
                    await self._redis.setex(
                        metadata_key,
                        self._config.ttl_seconds,
                        json.dumps(metadata.to_dict()),
                    )

                    # Store mapping if user_id and agent_id are provided
                    if user_id and agent_id:
                        mapping_key = self._get_user_agent_session_key(user_id, agent_id)
                        await self._redis.setex(
                            mapping_key,
                            self._config.ttl_seconds,
                            session_id,
                        )

                    logger.info(
                        f"Created new session: {session_id}",
                        extra={"user_id": user_id, "agent_id": agent_id},
                    )
                    return session_id

            # No session_id provided - look up by user_id + agent_id if available
            if user_id and agent_id:
                mapping_key = self._get_user_agent_session_key(user_id, agent_id)
                existing_session_id = await self._redis.get(mapping_key)

                if existing_session_id:
                    # Found existing session - verify it still exists and update it
                    metadata_key = self._get_metadata_key(existing_session_id)
                    metadata_json = await self._redis.get(metadata_key)

                    if metadata_json:
                        # Session exists, update last_accessed_at
                        metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                        metadata.last_accessed_at = datetime.now()
                        await self._redis.setex(
                            metadata_key,
                            self._config.ttl_seconds,
                            json.dumps(metadata.to_dict()),
                        )
                        # Refresh mapping TTL
                        await self._redis.setex(
                            mapping_key,
                            self._config.ttl_seconds,
                            existing_session_id,
                        )
                        logger.debug(
                            f"Retrieved existing session by user+agent: {existing_session_id}",
                            extra={"user_id": user_id, "agent_id": agent_id},
                        )
                        return existing_session_id
                    else:
                        # Mapping exists but session doesn't - clean up stale mapping
                        await self._redis.delete(mapping_key)
                        logger.debug(
                            f"Cleaned up stale session mapping for user_id={user_id}, agent_id={agent_id}"
                        )

            # No existing session found - create a new one
            session_id = generate_session_id()
            metadata = SessionMetadata(
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
                message_count=0,
            )
            metadata_key = self._get_metadata_key(session_id)
            await self._redis.setex(
                metadata_key,
                self._config.ttl_seconds,
                json.dumps(metadata.to_dict()),
            )

            # Store mapping if user_id and agent_id are provided
            if user_id and agent_id:
                mapping_key = self._get_user_agent_session_key(user_id, agent_id)
                await self._redis.setex(
                    mapping_key,
                    self._config.ttl_seconds,
                    session_id,
                )

            logger.info(
                f"Created new session: {session_id}",
                extra={"user_id": user_id, "agent_id": agent_id},
            )
            return session_id

        except Exception as e:
            logger.error(f"Failed to get or create session: {e}")
            report_error(
                e,
                context="session_provider_get_or_create",
                metadata={"session_id": session_id, "user_id": user_id, "agent_id": agent_id},
            )
            raise SessionConnectionError(
                f"Failed to get or create session: {str(e)}",
                session_id=session_id,
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
            session_metadata.last_accessed_at = datetime.now()

            messages_key = self._get_session_key(session_id)

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

            # Increment count for the new message
            session_metadata.message_count += 1

            # Create session message with metadata (no trace_id/span_id - handled by @observe)
            session_message = SessionMessage(
                message=message,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Store message in Redis List (append to end)
            message_json = json.dumps(session_message.to_dict())
            await self._redis.rpush(messages_key, message_json)

            # Update session metadata
            await self._redis.setex(
                metadata_key,
                self._config.ttl_seconds,
                json.dumps(session_metadata.to_dict()),
            )

            # Set TTL on messages list (same as session TTL)
            await self._redis.expire(messages_key, self._config.ttl_seconds)

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
            limit: Optional limit on number of messages to retrieve.

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

            if not message_jsons:
                return []

            # Parse messages (skip malformed JSON entries gracefully)
            session_messages = []
            for msg_json in message_jsons:
                try:
                    session_messages.append(SessionMessage.from_dict(json.loads(msg_json)))
                except (json.JSONDecodeError, TypeError, KeyError) as parse_err:
                    logger.warning(
                        f"Skipping malformed session message in {session_id}: {parse_err}. "
                        f"Raw preview: {str(msg_json)[:200]}"
                    )

            # Convert to ChatMessage list
            messages = [sm.message for sm in session_messages]

            # Apply limit if specified
            if limit and limit > 0:
                messages = messages[-limit:]  # Get last N messages

            # Update last_accessed_at
            session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
            session_metadata.last_accessed_at = datetime.now()
            await self._redis.setex(
                metadata_key,
                self._config.ttl_seconds,
                json.dumps(session_metadata.to_dict()),
            )

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

            return SessionMetadata.from_dict(json.loads(metadata_json))

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
            await self._redis.delete(messages_key)

            # Reset message count in metadata
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)
            if metadata_json:
                session_metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                session_metadata.message_count = 0
                session_metadata.last_accessed_at = datetime.now()
                await self._redis.setex(
                    metadata_key,
                    self._config.ttl_seconds,
                    json.dumps(session_metadata.to_dict()),
                )

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
            # Get session metadata to find user_id and agent_id for mapping cleanup
            metadata_key = self._get_metadata_key(session_id)
            metadata_json = await self._redis.get(metadata_key)

            messages_key = self._get_session_key(session_id)

            # Delete session data
            await self._redis.delete(messages_key, metadata_key)

            # Clean up user+agent mapping if metadata exists
            if metadata_json:
                try:
                    metadata = SessionMetadata.from_dict(json.loads(metadata_json))
                    if metadata.user_id and metadata.agent_id:
                        mapping_key = self._get_user_agent_session_key(
                            metadata.user_id, metadata.agent_id
                        )
                        # Only delete if it points to this session_id
                        existing_session_id = await self._redis.get(mapping_key)
                        if existing_session_id == session_id:
                            await self._redis.delete(mapping_key)
                            logger.debug(f"Cleaned up user+agent mapping for session: {session_id}")
                except Exception as e:
                    # Don't fail if mapping cleanup fails
                    logger.debug(f"Could not clean up mapping for session {session_id}: {e}")

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
        from orchestrator.config import settings

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
