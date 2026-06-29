"""
In-Memory Session Provider — process-local fallback for short-term sessions.

Used when Redis-backed persistence is unavailable: either explicitly
unconfigured or unreachable at runtime. Stores sessions in plain dicts so the
SDK remains fully usable without any external service. State is NOT durable —
it lives only for the process lifetime and is not shared across workers — so
this is a graceful-degradation path, not a production persistence backend.

Semantics mirror RedisSessionProvider (deterministic session IDs, sliding-window
message limits, metadata bookkeeping) so callers behave identically regardless
of which provider is active. Operations never raise connection errors.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from continuum.logging import get_logger
from continuum.session.base import BaseSessionProvider
from continuum.session.config import SessionConfig
from continuum.session.exceptions import (
    SessionMessageLimitError,
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


class MemorySessionProvider(BaseSessionProvider):
    """Session provider backed by an in-process dictionary.

    Functionally equivalent to the Redis provider for a single process, minus
    durability and cross-process sharing. Intended as the automatic fallback
    when Redis is disabled, unconfigured, or unreachable.
    """

    def __init__(self, config: SessionConfig, auto_initialize: bool = True):
        self._config = config
        # session_id -> {"metadata": SessionMetadata, "messages": list[SessionMessage]}
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._initialized = config.enabled

    @property
    def provider_name(self) -> str:
        return "memory"

    @property
    def config(self) -> SessionConfig:
        return self._config

    @property
    def is_initialized(self) -> bool:
        return self._config.enabled and self._initialized

    def initialize(self) -> bool:
        """No external resource to set up; ready immediately when enabled."""
        self._initialized = self._config.enabled
        return self._initialized

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_session_id(
        self,
        session_id: str | None,
        user_id: str | None,
        conversation_id: str | None,
    ) -> str:
        """Deterministic session ID — same scheme as the Redis provider."""
        if session_id:
            return session_id
        user_id = validate_user_id(user_id)
        conversation_id = validate_conversation_id(conversation_id)
        if conversation_id and user_id:
            return f"c:{conversation_id}:u:{user_id}"
        if user_id:
            return f"u:{user_id}"
        return generate_session_id()

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        resolved = self._compute_session_id(session_id, user_id, conversation_id)
        with self._lock:
            entry = self._store.get(resolved)
            now = datetime.now(UTC)
            if entry is not None:
                entry["metadata"].last_accessed_at = now
                return resolved
            self._store[resolved] = {
                "metadata": SessionMetadata(
                    session_id=resolved,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    created_at=now,
                    last_accessed_at=now,
                    message_count=0,
                ),
                "messages": [],
            }
        return resolved

    async def add_message(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}", session_id=session_id
                )

            messages: list[SessionMessage] = entry["messages"]

            if len(messages) >= self._config.max_messages:
                if self._config.message_limit_strategy == "error":
                    raise SessionMessageLimitError(
                        f"Session message limit exceeded: "
                        f"{len(messages)} >= {self._config.max_messages}",
                        session_id=session_id,
                        current_count=len(messages),
                        max_messages=self._config.max_messages,
                    )
                # Sliding window: drop the oldest messages to make room.
                del messages[: self._config.sliding_window_trim_count]

            messages.append(
                SessionMessage(
                    message=message,
                    timestamp=datetime.now(UTC),
                    metadata=metadata or {},
                )
            )
            meta: SessionMetadata = entry["metadata"]
            meta.message_count = len(messages)
            meta.last_accessed_at = datetime.now(UTC)

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}", session_id=session_id
                )
            messages = [sm.message for sm in entry["messages"]]
            entry["metadata"].last_accessed_at = datetime.now(UTC)

        # limit is measured in complete turns (request+response pairs); mirror the
        # Redis provider's trim-to-first-user-message behavior.
        if limit and limit > 0:
            sliced = messages[-(limit * 2) :]
            first_user = next(
                (i for i, m in enumerate(sliced) if m.role == "user"),
                len(sliced),
            )
            messages = sliced[first_user:]
        return messages

    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            meta: SessionMetadata = entry["metadata"]
            meta.message_count = len(entry["messages"])
            return meta

    async def update_session_metadata(
        self, session_id: str, metadata: SessionMetadata
    ) -> bool:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return False
            entry["metadata"] = metadata
            return True

    async def clear_session(self, session_id: str) -> bool:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is not None:
                entry["messages"] = []
                entry["metadata"].message_count = 0
                entry["metadata"].last_accessed_at = datetime.now(UTC)
        return True

    async def delete_session(self, session_id: str) -> bool:
        with self._lock:
            self._store.pop(session_id, None)
        return True

    async def close(self) -> None:
        """Release in-memory state. No external connections to close."""
        with self._lock:
            self._store.clear()
            self._initialized = False
