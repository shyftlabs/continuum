"""
PostgreSQL Session Provider - durable sessions in a database the product already runs.

Two tables, named from ``postgres_table_prefix`` (default ``continuum_session``):
- ``continuum_sessions``: one row per session (metadata and expiry)
- ``continuum_session_messages``: one row per message, deleted with its session

Semantics mirror RedisSessionProvider: deterministic session ids (keyed when
``hash_session_ids`` is on), the TTL is refreshed on every touch, and
``max_messages`` either trims the oldest messages or raises. Expired rows read as
missing; ``purge_expired()`` deletes them and is meant for a periodic job.

A product that already holds an asyncpg pool can hand it over instead of a DSN,
so sessions share its connections, credentials and backups::

    provider = PostgresSessionProvider(SessionConfig(provider="postgres"), pool=app_pool)
    session_client.set_provider(provider)

The provider never closes a pool it was given.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from continuum.logging import get_logger, log_id
from continuum.session.base import BaseSessionProvider
from continuum.session.config import SessionConfig
from continuum.session.exceptions import (
    SessionConnectionError,
    SessionMessageLimitError,
    SessionNotFoundError,
)
from continuum.session.identity import compute_session_id, legacy_session_id
from continuum.session.types import ChatMessage, SessionMetadata

try:
    import asyncpg  # type: ignore[import-untyped]
except ImportError:  # optional extra: pip install 'shyftlabs-continuum[postgres]'
    asyncpg = None  # type: ignore[assignment]

logger = get_logger(__name__)

# Table names are interpolated into SQL, so the prefix must be a plain identifier.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")


def _connection_errors() -> tuple[type[BaseException], ...]:
    """Failures that mean "the database is unreachable", not "the query is wrong"."""
    errors: tuple[type[BaseException], ...] = (OSError, TimeoutError)
    if asyncpg is not None:
        errors += (
            asyncpg.InterfaceError,
            asyncpg.PostgresConnectionError,
            asyncpg.CannotConnectNowError,
        )
    return errors


class PostgresSessionProvider(BaseSessionProvider):
    """Session provider backed by PostgreSQL (asyncpg)."""

    def __init__(self, config: SessionConfig, pool: Any | None = None):
        if asyncpg is None:
            raise ImportError(
                "PostgresSessionProvider needs asyncpg: pip install 'shyftlabs-continuum[postgres]'"
            )
        prefix = config.postgres_table_prefix
        if not _IDENTIFIER.match(prefix):
            raise ValueError(
                f"postgres_table_prefix must be a lowercase SQL identifier, got {prefix!r}"
            )
        self._config = config
        self._pool = pool
        self._owns_pool = pool is None
        self._sessions = f"{prefix}s"
        self._messages = f"{prefix}_messages"
        self._ready = False
        self._init_lock = asyncio.Lock()
        self._last_probe_error: str | None = None

    @property
    def provider_name(self) -> str:
        return "postgres"

    @property
    def config(self) -> SessionConfig:
        return self._config

    @property
    def is_initialized(self) -> bool:
        return self._config.enabled and self._ready

    @property
    def last_probe_error(self) -> str | None:
        """Reason the most recent aping() failed, or None if it succeeded."""
        return self._last_probe_error

    @property
    def _ttl(self) -> timedelta:
        return timedelta(seconds=self._config.ttl_seconds)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    async def _ensure_ready(self) -> Any:
        if self._ready:
            return self._pool
        async with self._init_lock:
            if self._ready:
                return self._pool
            if self._pool is None:
                if not self._config.postgres_dsn:
                    raise SessionConnectionError("SESSION_POSTGRES_DSN is not set")
                self._pool = await asyncpg.create_pool(
                    self._config.postgres_dsn,
                    min_size=1,
                    max_size=self._config.postgres_max_connections,
                )
            if self._config.postgres_create_tables:
                await self._create_tables(self._pool)
            self._ready = True
            return self._pool

    async def _create_tables(self, pool: Any) -> None:
        s, m = self._sessions, self._messages
        async with pool.acquire() as conn, conn.transaction():
            # Several workers starting at once would race on CREATE TABLE IF NOT EXISTS.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", s)
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {s} (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    agent_id TEXT,
                    conversation_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    last_accessed_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    custom JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                CREATE TABLE IF NOT EXISTS {m} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL
                        REFERENCES {s}(session_id) ON DELETE CASCADE ON UPDATE CASCADE,
                    message JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS {m}_session_idx ON {m} (session_id, id);
                CREATE INDEX IF NOT EXISTS {s}_expires_idx ON {s} (expires_at);
                """
            )

    @asynccontextmanager
    async def _conn(self, *, transaction: bool = False) -> AsyncIterator[Any]:
        """A pooled connection; an unreachable database surfaces as SessionConnectionError,
        which lets SessionClient apply its fallback policy as it does for Redis."""
        try:
            pool = await self._ensure_ready()
            async with pool.acquire() as conn:
                if transaction:
                    async with conn.transaction():
                        yield conn
                else:
                    yield conn
        except _connection_errors() as e:
            raise SessionConnectionError(f"PostgreSQL session store unreachable: {e}") from e

    async def aping(self) -> bool:
        """Probe once; SessionClient calls this before choosing the provider."""
        try:
            async with self._conn() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as e:  # noqa: BLE001 — a probe reports, it never raises
            self._last_probe_error = f"{type(e).__name__}: {e}"
            logger.debug("PostgreSQL session probe failed: %s", self._last_probe_error)
            return False
        self._last_probe_error = None
        return True

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    async def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        resolved = compute_session_id(
            session_id,
            user_id,
            conversation_id,
            secret=self._config.active_session_id_secret,
        )
        s = self._sessions
        async with self._conn(transaction=True) as conn:
            await conn.execute(
                f"DELETE FROM {s} WHERE session_id = $1 AND expires_at <= now()", resolved
            )
            touched = await conn.fetchval(
                f"UPDATE {s} SET last_accessed_at = now(), expires_at = now() + $2::interval "
                "WHERE session_id = $1 RETURNING 1",
                resolved,
                self._ttl,
            )
            if touched:
                return resolved
            if await self._migrate_legacy_session(
                conn, resolved, session_id, user_id, conversation_id
            ):
                return resolved
            await conn.execute(
                f"INSERT INTO {s} (session_id, user_id, conversation_id, created_at, "
                "last_accessed_at, expires_at) VALUES ($1, $2, $3, now(), now(), now() + $4::interval) "
                "ON CONFLICT (session_id) DO UPDATE SET last_accessed_at = now(), "
                "expires_at = EXCLUDED.expires_at",
                resolved,
                user_id,
                conversation_id,
                self._ttl,
            )
        return resolved

    async def _migrate_legacy_session(
        self,
        conn: Any,
        resolved: str,
        session_id: str | None,
        user_id: str | None,
        conversation_id: str | None,
    ) -> bool:
        """Move a pre-hashing plaintext session under its keyed id (messages follow
        through ON UPDATE CASCADE), so the guessable id stops resolving."""
        if not self._config.active_session_id_secret:
            return False
        legacy = legacy_session_id(session_id, user_id, conversation_id)
        if legacy is None or legacy == resolved:
            return False
        moved = await conn.fetchval(
            f"UPDATE {self._sessions} SET session_id = $2, last_accessed_at = now(), "
            "expires_at = now() + $3::interval WHERE session_id = $1 AND expires_at > now() "
            "RETURNING 1",
            legacy,
            resolved,
            self._ttl,
        )
        if moved:
            logger.info("Migrated session to keyed id: %s", log_id(resolved))
        return bool(moved)

    async def add_message(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        s, m = self._sessions, self._messages
        async with self._conn(transaction=True) as conn:
            # Row lock: concurrent writers to one session take turns on the limit check.
            exists = await conn.fetchval(
                f"SELECT 1 FROM {s} WHERE session_id = $1 AND expires_at > now() FOR UPDATE",
                session_id,
            )
            if not exists:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}", session_id=session_id
                )
            count = await conn.fetchval(
                f"SELECT count(*) FROM {m} WHERE session_id = $1", session_id
            )
            if count >= self._config.max_messages:
                if self._config.message_limit_strategy == "error":
                    raise SessionMessageLimitError(
                        f"Session message limit exceeded: {count} >= {self._config.max_messages}",
                        session_id=session_id,
                        current_count=count,
                        max_messages=self._config.max_messages,
                    )
                await conn.execute(
                    f"DELETE FROM {m} WHERE id IN (SELECT id FROM {m} WHERE session_id = $1 "
                    "ORDER BY id LIMIT $2)",
                    session_id,
                    self._config.sliding_window_trim_count,
                )
            await conn.execute(
                f"INSERT INTO {m} (session_id, message, metadata) VALUES ($1, $2::jsonb, $3::jsonb)",
                session_id,
                json.dumps(message.to_dict()),
                json.dumps(metadata or {}),
            )
            await conn.execute(
                f"UPDATE {s} SET last_accessed_at = now(), expires_at = now() + $2::interval "
                "WHERE session_id = $1",
                session_id,
                self._ttl,
            )

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        s, m = self._sessions, self._messages
        async with self._conn() as conn:
            # Reading refreshes the TTL, as with Redis, so a read-heavy session stays alive.
            alive = await conn.fetchval(
                f"UPDATE {s} SET last_accessed_at = now(), expires_at = now() + $2::interval "
                "WHERE session_id = $1 AND expires_at > now() RETURNING 1",
                session_id,
                self._ttl,
            )
            if not alive:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}", session_id=session_id
                )
            if limit and limit > 0:
                rows = await conn.fetch(
                    f"SELECT message FROM (SELECT id, message FROM {m} WHERE session_id = $1 "
                    "ORDER BY id DESC LIMIT $2) recent ORDER BY id",
                    session_id,
                    limit * 2,
                )
            else:
                rows = await conn.fetch(
                    f"SELECT message FROM {m} WHERE session_id = $1 ORDER BY id", session_id
                )
        messages = [ChatMessage(**json.loads(r["message"])) for r in rows]

        # limit counts complete turns; start at a user message, as the Redis provider does.
        if limit and limit > 0:
            first_user = next(
                (i for i, msg in enumerate(messages) if msg.role == "user"), len(messages)
            )
            messages = messages[first_user:]
        return messages

    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        s, m = self._sessions, self._messages
        async with self._conn() as conn:
            row = await conn.fetchrow(
                f"SELECT s.*, (SELECT count(*) FROM {m} WHERE {m}.session_id = s.session_id) "
                f"AS message_count FROM {s} s WHERE s.session_id = $1 AND s.expires_at > now()",
                session_id,
            )
        if row is None:
            return None
        return SessionMetadata(
            session_id=row["session_id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            message_count=row["message_count"],
            custom=json.loads(row["custom"]),
        )

    async def update_session_metadata(self, session_id: str, metadata: SessionMetadata) -> bool:
        async with self._conn() as conn:
            updated = await conn.fetchval(
                f"UPDATE {self._sessions} SET user_id = $2, agent_id = $3, conversation_id = $4, "
                "custom = $5::jsonb, last_accessed_at = $6, expires_at = now() + $7::interval "
                "WHERE session_id = $1 AND expires_at > now() RETURNING 1",
                session_id,
                metadata.user_id,
                metadata.agent_id,
                metadata.conversation_id,
                json.dumps(metadata.custom),
                metadata.last_accessed_at,
                self._ttl,
            )
        return bool(updated)

    async def clear_session(self, session_id: str) -> bool:
        async with self._conn(transaction=True) as conn:
            await conn.execute(f"DELETE FROM {self._messages} WHERE session_id = $1", session_id)
            await conn.execute(
                f"UPDATE {self._sessions} SET last_accessed_at = now() WHERE session_id = $1",
                session_id,
            )
        return True

    async def delete_session(self, session_id: str) -> bool:
        async with self._conn() as conn:
            await conn.execute(f"DELETE FROM {self._sessions} WHERE session_id = $1", session_id)
        return True

    async def purge_expired(self) -> int:
        """Delete expired sessions and their messages. Returns how many sessions went."""
        async with self._conn() as conn:
            status = await conn.execute(f"DELETE FROM {self._sessions} WHERE expires_at <= now()")
        return int(status.split()[-1])

    async def close(self) -> None:
        if self._owns_pool and self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._ready = False
