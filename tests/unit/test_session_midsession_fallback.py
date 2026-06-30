"""
Operation-level fallback: Redis fails AFTER the session already resolved to it
(flaky/half-started mid-session). The client must degrade to in-memory on the
connection error — log once, retry on memory, serve subsequent ops from memory —
instead of logging an error on every request.

Genuine logical errors (SessionNotFoundError / SessionMessageLimitError) must
NOT trigger a degrade.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionConnectionError, SessionNotFoundError
from continuum.session.providers.memory import MemorySessionProvider
from continuum.session.providers.redis import RedisSessionProvider
from continuum.session.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def _resolve_to_redis(monkeypatch):
    """Make the client resolve to Redis (probe passes)."""
    monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=True))


def _spies(monkeypatch):
    logspy = MagicMock()
    errspy = MagicMock()
    monkeypatch.setattr("continuum.session.client.logger", logspy)
    monkeypatch.setattr("continuum.session.client.report_error", errspy)
    return logspy, errspy


class TestMidSessionDegrade:
    async def test_degrades_to_memory_warns_once_no_per_request_errors(self, monkeypatch):
        _resolve_to_redis(monkeypatch)
        # Redis op fails with a connection error on every call.
        monkeypatch.setattr(
            RedisSessionProvider,
            "get_or_create_session",
            AsyncMock(side_effect=SessionConnectionError("Timeout reading from localhost:6380")),
        )
        logspy, errspy = _spies(monkeypatch)

        sc = SessionClient(session_config=SessionConfig(enabled=True), auto_initialize=False)

        results = []
        for _ in range(5):
            results.append(await sc.get_or_create_session(user_id="shopper-1"))

        # All requests succeed (served from in-memory after degrade).
        assert all(results)
        assert isinstance(sc._provider, MemorySessionProvider)
        # Degraded once, not once-per-request.
        assert logspy.warning.call_count == 1
        # No error spam, no per-request error reports.
        assert logspy.error.call_count == 0
        assert errspy.call_count == 0

    async def test_roundtrip_works_after_degrade(self, monkeypatch):
        _resolve_to_redis(monkeypatch)
        monkeypatch.setattr(
            RedisSessionProvider,
            "get_or_create_session",
            AsyncMock(side_effect=SessionConnectionError("down")),
        )
        _spies(monkeypatch)

        sc = SessionClient(session_config=SessionConfig(enabled=True), auto_initialize=False)

        sid = await sc.get_or_create_session(user_id="shopper-2")  # triggers degrade
        await sc.add_message(sid, _msg("user", "hello"))
        history = await sc.get_conversation_history(sid)

        assert [m.content for m in history] == ["hello"]
        assert isinstance(sc._provider, MemorySessionProvider)


class TestLogicalErrorsDoNotDegrade:
    async def test_session_not_found_propagates_without_degrade(self, monkeypatch):
        _resolve_to_redis(monkeypatch)
        # A logical error (not a connection failure) must NOT trigger fallback.
        monkeypatch.setattr(
            RedisSessionProvider,
            "get_messages",
            AsyncMock(side_effect=SessionNotFoundError("no such session", session_id="x")),
        )
        _spies(monkeypatch)

        sc = SessionClient(session_config=SessionConfig(enabled=True), auto_initialize=False)

        with pytest.raises(SessionNotFoundError):
            await sc.get_conversation_history("x")

        # Still on Redis — a missing session is not a reason to drop persistence.
        assert isinstance(sc._provider, RedisSessionProvider)


class TestRealProviderDegradeIsQuiet:
    """Exercise the REAL Redis provider (dead port) so provider-level logging is
    actually executed — a method-level mock would bypass it and give false
    confidence. The whole stack must be quiet on the degrade path: no ERROR logs
    and no report_error from EITHER the provider or the client.
    """

    async def test_real_redis_failure_degrades_quietly(self, monkeypatch):
        # Resolve to the real Redis provider (probe "passes")...
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=True))
        # ...but real ops hit a closed port → real connection error inside the provider.
        # NOTE: the provider no longer imports report_error at all (quiet by
        # design), so only the client's report_error is spy-able.
        report_client = MagicMock()
        monkeypatch.setattr("continuum.session.client.report_error", report_client)

        errors: list[str] = []

        class _ErrCounter(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.levelno >= logging.ERROR:
                    errors.append(record.name)

        handler = _ErrCounter()
        watched = ("continuum.session.providers.redis", "continuum.session.client")
        for name in watched:
            logging.getLogger(name).addHandler(handler)
        try:
            cfg = SessionConfig(
                enabled=True, redis_host="127.0.0.1", redis_port=6399, fallback_mode="degrade"
            )
            sc = SessionClient(session_config=cfg, auto_initialize=False)
            sids = [await sc.get_or_create_session(user_id="a") for _ in range(3)]
        finally:
            for name in watched:
                logging.getLogger(name).removeHandler(handler)

        assert all(sids)
        assert isinstance(sc._provider, MemorySessionProvider)
        assert sc.persistence_degraded is True
        # The degrade path must be quiet end-to-end: no ERROR logs from the
        # provider OR the client, and no error reports.
        assert errors == [], f"unexpected ERROR logs on degrade path: {errors}"
        assert report_client.call_count == 0
