"""
Redis-backed integration tests for the runner session preflight.

Drives a REAL AgentRunner + SessionClient over a live Redis session provider
(port 6380) — only the LLM is mocked. Closes the gap left by the unit tests
(which mock the session client) by exercising the actual load/save/preflight
paths against real Redis.

Contract verified:
  - session_id passed but never created → WARN by default (run proceeds,
    nothing persisted); RAISE SessionNotCreatedError under strict mode.
  - create-then-run (same id + user_id) → no warning, messages persisted.
  - user_id mismatch on an existing session → WARN.
  - stateless run (no session_id) → no guardrail warning at all.

Requires Redis on port 6380 (skipped otherwise).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import RunnerConfig
from continuum.agent.runner import AgentRunner
from continuum.llm.types import LLMResponse
from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionNotCreatedError, SessionNotFoundError
from continuum.session.providers.redis import RedisSessionProvider

pytestmark = [pytest.mark.integration, pytest.mark.redis]

_GUARD_LOGGERS = ("continuum.agent.runner", "continuum.agent.services.session_service")


def _make_config(strict: bool = False) -> SessionConfig:
    return SessionConfig(
        enabled=True,
        redis_host="localhost",
        redis_port=6380,
        redis_password="sdk123456789",
        ttl_seconds=300,
        max_messages=100,
        strict_sessions=strict,
    )


@pytest.fixture
async def redis_provider():
    provider = RedisSessionProvider(config=_make_config())
    if not provider.initialize():
        pytest.skip("Redis session provider failed to initialize")
    yield provider
    await provider.close()


def _session_client(provider, *, strict: bool = False) -> SessionClient:
    # Explicit provider → trusted as-is, never swapped for the in-memory
    # fallback, so persistence_degraded stays False (preflight stays active).
    sc = SessionClient(session_config=_make_config(strict=strict), provider=provider)
    assert sc.persistence_degraded is False
    return sc


def _make_llm() -> MagicMock:
    c = MagicMock()
    c.chat = AsyncMock(
        return_value=LLMResponse(model="gpt-4o-mini", content="Noted!", role="assistant")
    )
    c.chat_stream = AsyncMock()
    return c


def _make_runner(session_client: SessionClient) -> AgentRunner:
    mem = MagicMock()
    mem.is_enabled = False
    return AgentRunner(
        llm_client=_make_llm(),
        memory_client=mem,
        session_client=session_client,
        config=RunnerConfig(persist_state=False),
    )


def _make_agent() -> BaseAgent:
    a = BaseAgent(name="tea-bot", instructions="You remember things.", model="gpt-4o-mini")
    # This test is about short-term session persistence + the preflight; keep
    # long-term memory (mem0/vector store) out of the picture.
    a.memory_config.search_memories = False
    a.memory_config.store_memories = False
    return a


class _WarnCapture:
    """Capture WARNING+ records from the guardrail loggers (they set
    propagate=False, so caplog can't see them — attach handlers directly)."""

    def __init__(self):
        self._records: list[str] = []
        self._handler = logging.Handler()
        self._handler.emit = lambda rec: (  # type: ignore[method-assign]
            self._records.append(rec.getMessage()) if rec.levelno >= logging.WARNING else None
        )
        self._loggers = [logging.getLogger(n) for n in _GUARD_LOGGERS]

    def __enter__(self):
        for lg in self._loggers:
            lg.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        for lg in self._loggers:
            lg.removeHandler(self._handler)

    def has(self, substr: str) -> bool:
        return any(substr in m for m in self._records)


class TestSessionPreflightRedis:
    async def test_missing_session_warns_and_does_not_persist(self, redis_provider, test_id):
        sc = _session_client(redis_provider)
        runner = _make_runner(sc)
        sid = f"missing-{test_id}"

        with _WarnCapture() as cap:
            resp = await runner.run(
                _make_agent(), "Remember I like tea", session_id=sid, user_id="u1"
            )

        assert resp.status.value == "success"  # non-breaking
        assert cap.has("get_or_create_session")  # loud, actionable warning
        # Nothing persisted: the session was never created.
        with pytest.raises(SessionNotFoundError):
            await sc.get_conversation_history(sid)

    async def test_missing_session_raises_in_strict_mode(self, redis_provider, test_id):
        sc = _session_client(redis_provider, strict=True)
        runner = _make_runner(sc)
        with pytest.raises(SessionNotCreatedError):
            await runner.run(_make_agent(), "hi", session_id=f"strict-{test_id}", user_id="u1")

    async def test_require_session_per_call_overrides_config(self, redis_provider, test_id):
        # Config is non-strict, but the per-call flag forces a raise.
        sc = _session_client(redis_provider, strict=False)
        runner = _make_runner(sc)
        with pytest.raises(SessionNotCreatedError):
            await runner.run(
                _make_agent(),
                "hi",
                session_id=f"percall-{test_id}",
                user_id="u1",
                require_session=True,
            )

    async def test_create_then_run_persists_and_no_warning(self, redis_provider, test_id):
        sc = _session_client(redis_provider)
        runner = _make_runner(sc)
        sid = await sc.get_or_create_session(session_id=f"good-{test_id}", user_id="u1")

        with _WarnCapture() as cap:
            resp = await runner.run(
                _make_agent(), "Remember I like tea", session_id=sid, user_id="u1"
            )

        assert resp.status.value == "success"
        assert not cap.has("get_or_create_session")
        assert not cap.has("user_id mismatch")

        history = await sc.get_conversation_history(sid)
        roles = [m.role for m in history]
        assert "user" in roles and "assistant" in roles

    async def test_user_id_mismatch_warns(self, redis_provider, test_id):
        sc = _session_client(redis_provider)
        runner = _make_runner(sc)
        sid = await sc.get_or_create_session(session_id=f"mismatch-{test_id}", user_id="u1")

        with _WarnCapture() as cap:
            await runner.run(_make_agent(), "hi", session_id=sid, user_id="u2")

        assert cap.has("user_id mismatch")

    async def test_stateless_run_no_guardrail_warning(self, redis_provider):
        # No session_id → stateless. Even with strict mode on, no guardrail fires.
        sc = _session_client(redis_provider, strict=True)
        runner = _make_runner(sc)

        with _WarnCapture() as cap:
            resp = await runner.run(_make_agent(), "just answer", user_id="u1")

        assert resp.status.value == "success"
        assert not cap.has("get_or_create_session")
        assert not cap.has("user_id mismatch")
