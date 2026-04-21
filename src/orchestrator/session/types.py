"""
Type definitions for the Session module.

Provides Pydantic models for session management and conversation history.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.llm.types import ChatMessage


class SessionMetadata(BaseModel):
    """Metadata for a session."""

    session_id: str  # UUID string
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    created_at: datetime
    last_accessed_at: datetime
    message_count: int = 0
    custom: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "message_count": self.message_count,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        """Create SessionMetadata from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            conversation_id=data.get("conversation_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            message_count=data.get("message_count", 0),
            custom=data.get("custom", {}),
        )


class Session(BaseModel):
    """Represents a session with its metadata and messages."""

    session_id: str  # UUID string
    metadata: SessionMetadata
    messages: list[ChatMessage] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "metadata": self.metadata.to_dict(),
            "messages": [msg.to_dict() for msg in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create Session from dictionary."""
        messages = [
            ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in data.get("messages", [])
        ]
        return cls(
            session_id=data["session_id"],
            metadata=SessionMetadata.from_dict(data["metadata"]),
            messages=messages,
        )


class SessionMessage(BaseModel):
    """Represents a message in a session with additional metadata.

    Note: Tracing (trace_id/span_id) is handled automatically by the @observe decorator
    via SpanScope, so these fields are not stored in the session message.
    """

    message: ChatMessage
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "message": self.message.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMessage":
        """Create SessionMessage from dictionary."""
        message_data = data["message"]
        if isinstance(message_data, dict):
            message = ChatMessage(**message_data)
        else:
            message = message_data

        timestamp_str = data.get("timestamp")
        if not isinstance(timestamp_str, str):
            raise ValueError(
                f"SessionMessage is missing a valid 'timestamp' field: got {timestamp_str!r}"
            )
        timestamp = datetime.fromisoformat(timestamp_str)

        return cls(
            message=message,
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


def generate_session_id() -> str:
    """Generate a new UUID-based session ID."""
    return str(uuid.uuid4())
