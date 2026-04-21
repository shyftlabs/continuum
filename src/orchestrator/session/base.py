"""
Base Session Provider - Abstract base class for session providers.

Defines the interface that all session providers must implement.
This enables easy addition of new providers (DynamoDB, PostgreSQL, MongoDB, etc.)
while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Any

from orchestrator.session.types import ChatMessage, SessionMetadata


class BaseSessionProvider(ABC):
    """
    Abstract base class for session providers.

    All session providers (Redis, DynamoDB, PostgreSQL, etc.) must implement
    this interface to be compatible with the SessionClient.

    The provider handles async operations for session management including:
    - Session creation and retrieval
    - Message storage and retrieval
    - Session metadata management
    - Session cleanup

    Example implementation:
        ```python
        class CustomProvider(BaseSessionProvider):
            async def get_or_create_session(self, session_id=None, ...):
                # Implementation
                pass
        ```
    """

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional session ID. If not provided, generates a new UUID.
            user_id: Optional user identifier.
            conversation_id: Optional conversation identifier (chat window ID from caller).

        Returns:
            Session ID (existing or newly created).
        """
        ...

    @abstractmethod
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
        """
        ...

    @abstractmethod
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
        """
        ...

    @abstractmethod
    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        """
        Get session metadata.

        Args:
            session_id: Session ID.

        Returns:
            SessionMetadata if found, None otherwise.
        """
        ...

    @abstractmethod
    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session (but keep metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if cleared successfully.
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely (messages and metadata).

        Args:
            session_id: Session ID.

        Returns:
            True if deleted successfully.
        """
        ...

    @abstractmethod
    async def update_session_metadata(self, session_id: str, metadata: SessionMetadata) -> bool:
        """
        Persist updated session metadata.

        Implementations must refresh the TTL on both the metadata key and the
        messages list key so they stay in sync.

        Args:
            session_id: Session ID.
            metadata: Updated SessionMetadata to persist.

        Returns:
            True if updated successfully, False if the session does not exist.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the provider and release resources."""
        ...

    # =========================================================================
    # Provider Info
    # =========================================================================

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'redis', 'dynamodb')."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the provider is initialized and ready."""
        ...

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> "BaseSessionProvider":
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
