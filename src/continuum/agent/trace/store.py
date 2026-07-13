"""
Trace persistence.

The recorder/runner depend only on the :class:`TraceStore` protocol, never on a
backend. Three backends ship:

* :class:`NullTraceStore` — no-op; used when persistence is disabled.
* :class:`InMemoryTraceStore` — process-local; used in unit tests.
* :class:`RedisTraceStore` — durable JSON-on-Redis, reusing the session Redis
  instance (the one Continuum already runs). One key per run, TTL'd.

We deliberately store the trace as an *exact* structured record fetched by
``run_id`` — not through mem0/Milvus, whose semantic-fact pipeline would
paraphrase and embed it (wrong for an audit record). Langfuse remains the
human-facing view via each step's ``span_id``; this store is the app-queryable
copy.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from continuum.agent.trace.types import DecisionTrace
from continuum.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class TraceStore(Protocol):
    """Minimal persistence surface for decision traces."""

    async def save(self, trace: DecisionTrace) -> None: ...
    async def get(self, run_id: str) -> DecisionTrace | None: ...
    async def delete(self, run_id: str) -> bool: ...


class NullTraceStore:
    """Persists nothing. Used when DECISION_TRACE_STORE=null/none."""

    async def save(self, trace: DecisionTrace) -> None:
        return None

    async def get(self, run_id: str) -> DecisionTrace | None:
        return None

    async def delete(self, run_id: str) -> bool:
        return False


class InMemoryTraceStore:
    """Process-local store for tests and ephemeral use."""

    def __init__(self) -> None:
        self._runs: dict[str, DecisionTrace] = {}

    async def save(self, trace: DecisionTrace) -> None:
        self._runs[trace.run_id] = trace

    async def get(self, run_id: str) -> DecisionTrace | None:
        return self._runs.get(run_id)

    async def delete(self, run_id: str) -> bool:
        return self._runs.pop(run_id, None) is not None


class RedisTraceStore:
    """Durable JSON-on-Redis trace store.

    Each run is one key ``{prefix}:{run_id}`` holding ``DecisionTrace.to_dict()``,
    expiring after ``ttl_seconds``. Defaults to the session Redis instance so no
    new infrastructure is introduced. An injected ``client`` (e.g. fakeredis) is
    used as-is, which keeps tests offline.
    """

    def __init__(
        self,
        *,
        client: object | None = None,
        prefix: str = "orchestrator:trace",
        ttl_seconds: int = 3600 * 24 * 14,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        db: int = 0,
    ) -> None:
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._redis: Any
        if client is not None:
            self._redis = client
        else:
            import redis.asyncio as redis

            from continuum.config import settings

            self._redis = redis.Redis(
                host=host or settings.session_redis_host,
                port=port or settings.session_redis_port,
                password=password if password is not None else settings.session_redis_password,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

    def _key(self, run_id: str) -> str:
        return f"{self._prefix}:{run_id}"

    async def save(self, trace: DecisionTrace) -> None:
        try:
            await self._redis.set(self._key(trace.run_id), json.dumps(trace.to_dict()), ex=self._ttl)
        except Exception as e:  # never let trace persistence break a run
            logger.warning("Failed to persist decision trace %s: %s", trace.run_id, e)

    async def get(self, run_id: str) -> DecisionTrace | None:
        try:
            blob = await self._redis.get(self._key(run_id))
        except Exception as e:
            logger.warning("Failed to read decision trace %s: %s", run_id, e)
            return None
        if blob is None:
            return None
        return DecisionTrace.from_dict(json.loads(blob))

    async def delete(self, run_id: str) -> bool:
        try:
            return bool(await self._redis.delete(self._key(run_id)))
        except Exception as e:
            logger.warning("Failed to delete decision trace %s: %s", run_id, e)
            return False
