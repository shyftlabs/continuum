"""
Behavioral contract for Part A of "Improve Initialization of all connectors":
Redis session persistence must initialize lazily, stay silent when disabled,
and degrade cleanly to an in-memory provider (with a single warning, never a
per-request error) when Redis is unconfigured or unreachable.

These are mock-first unit tests — no live Redis required. They pin the
observable behavior (connection attempts, log volume, fallback correctness)
before the implementation exists.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionNotEnabledError
from continuum.session.providers.memory import MemorySessionProvider
from continuum.session.types import ChatMessage


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def _spy_client_logger(monkeypatch) -> MagicMock:
    """Replace the SessionClient module logger with a spy to count log calls."""
    spy = MagicMock()
    monkeypatch.setattr("continuum.session.client.logger", spy)
    return spy


def _spy_report_error(monkeypatch) -> MagicMock:
    spy = MagicMock()
    monkeypatch.setattr("continuum.session.client.report_error", spy)
    return spy


def _warning_count(spy: MagicMock) -> int:
    return spy.warning.call_count


# =============================================================================
# 5. In-memory fallback is functionally correct (round-trip)
# =============================================================================


class TestMemoryProviderRoundTrip:
    async def test_create_add_get_clear_delete(self):
        provider = MemorySessionProvider(SessionConfig(enabled=True))

        sid = await provider.get_or_create_session(user_id="alice")
        assert sid

        await provider.add_message(sid, _msg("user", "hello"))
        await provider.add_message(sid, _msg("assistant", "hi there"))

        messages = await provider.get_messages(sid)
        assert [m.content for m in messages] == ["hello", "hi there"]

        meta = await provider.get_session_metadata(sid)
        assert meta is not None
        assert meta.message_count == 2

        assert await provider.clear_session(sid) is True
        assert await provider.get_messages(sid) == []

        assert await provider.delete_session(sid) is True
        assert await provider.get_session_metadata(sid) is None

    async def test_deterministic_session_id_from_user(self):
        provider = MemorySessionProvider(SessionConfig(enabled=True))
        a = await provider.get_or_create_session(user_id="bob")
        b = await provider.get_or_create_session(user_id="bob")
        assert a == b

    async def test_is_initialized_when_enabled(self):
        assert MemorySessionProvider(SessionConfig(enabled=True)).is_initialized is True

    async def test_never_raises_connection_error(self):
        # No external service exists; operations must simply work.
        provider = MemorySessionProvider(SessionConfig(enabled=True))
        sid = await provider.get_or_create_session(user_id="carol")
        await provider.add_message(sid, _msg("user", "x"))  # must not raise


# =============================================================================
# 1. Disabled -> completely silent, no connection machinery
# =============================================================================


class TestDisabledIsSilent:
    async def test_ops_raise_not_enabled_and_emit_no_warning(self, monkeypatch):
        spy = _spy_client_logger(monkeypatch)
        client = SessionClient(
            session_config=SessionConfig(enabled=False),
            auto_initialize=False,
        )
        with pytest.raises(SessionNotEnabledError):
            await client.get_or_create_session(user_id="alice")
        assert _warning_count(spy) == 0
        assert spy.error.call_count == 0

    async def test_no_redis_provider_constructed_when_disabled(self, monkeypatch):
        init_spy = MagicMock(side_effect=AssertionError("Redis provider must not be built"))
        monkeypatch.setattr(
            "continuum.session.providers.redis.RedisSessionProvider.__init__",
            init_spy,
            raising=True,
        )
        client = SessionClient(
            session_config=SessionConfig(enabled=False),
            auto_initialize=True,  # construction must still not build Redis
        )
        # Touching is_enabled / construction must not have built a Redis provider.
        assert client.is_enabled is False
        init_spy.assert_not_called()


# =============================================================================
# 4. Lazy init — constructing the client triggers no connection
# =============================================================================


class TestLazyInit:
    async def test_construction_does_not_probe_or_build_redis(self, monkeypatch):
        aping = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "continuum.session.providers.redis.RedisSessionProvider.aping",
            aping,
            raising=False,
        )
        cfg = SessionConfig(enabled=True, redis_host="localhost", redis_port=6380,
                            redis_password="ut-strong-redis-pw-0123456789")
        SessionClient(session_config=cfg, auto_initialize=True)
        # Merely constructing the client must not probe Redis.
        aping.assert_not_called()

    async def test_probe_happens_on_first_operation(self, monkeypatch):
        # Reachable Redis -> provider resolves to the real Redis provider, and the
        # probe is performed exactly once, lazily, on first use.
        aping = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "continuum.session.providers.redis.RedisSessionProvider.aping",
            aping,
            raising=False,
        )
        # Stub the actual Redis op so we don't need a live server.
        op = AsyncMock(return_value="sess-1")
        monkeypatch.setattr(
            "continuum.session.providers.redis.RedisSessionProvider.get_or_create_session",
            op,
            raising=True,
        )
        cfg = SessionConfig(enabled=True, redis_host="localhost", redis_port=6380,
                            redis_password="ut-strong-redis-pw-0123456789")
        client = SessionClient(session_config=cfg, auto_initialize=False)

        sid = await client.get_or_create_session(user_id="alice")
        assert sid == "sess-1"
        aping.assert_awaited_once()


# =============================================================================
# 2. Enabled but unconfigured -> in-memory fallback, exactly one warning
# =============================================================================


class TestUnconfiguredFallback:
    async def test_falls_back_to_memory_with_single_warning(self, monkeypatch):
        spy = _spy_client_logger(monkeypatch)
        err = _spy_report_error(monkeypatch)
        cfg = SessionConfig(enabled=True, redis_host="", fallback_mode="degrade")  # not configured
        client = SessionClient(session_config=cfg, auto_initialize=False)

        sid = await client.get_or_create_session(user_id="alice")
        await client.add_message(sid, _msg("user", "hi"))
        history = await client.get_conversation_history(sid)

        assert [m.content for m in history] == ["hi"]
        assert isinstance(client._provider, MemorySessionProvider)
        # Exactly one warning across the whole sequence; never per-request.
        assert _warning_count(spy) == 1
        assert "in-memory" in str(spy.warning.call_args).lower()
        err.assert_not_called()


# =============================================================================
# 3. Enabled + unreachable -> warn once, then degrade (no per-request errors)
# =============================================================================


class TestUnreachableFallback:
    async def test_warns_once_then_serves_from_memory(self, monkeypatch):
        spy = _spy_client_logger(monkeypatch)
        err = _spy_report_error(monkeypatch)
        # Redis is configured but the connectivity probe fails.
        monkeypatch.setattr(
            "continuum.session.providers.redis.RedisSessionProvider.aping",
            AsyncMock(return_value=False),
            raising=False,
        )
        cfg = SessionConfig(
            enabled=True, redis_host="localhost", redis_port=6380,
            redis_password="ut-strong-redis-pw-0123456789", fallback_mode="degrade"
        )
        client = SessionClient(session_config=cfg, auto_initialize=False)

        sid = await client.get_or_create_session(user_id="alice")
        # Hammer it: many operations must not produce more warnings or any errors.
        for i in range(10):
            await client.add_message(sid, _msg("user", f"m{i}"))
        history = await client.get_conversation_history(sid)

        assert len(history) == 10
        assert isinstance(client._provider, MemorySessionProvider)
        assert _warning_count(spy) == 1
        assert err.call_count == 0
        assert spy.error.call_count == 0


# =============================================================================
# 6. Happy path unchanged — explicitly injected provider is respected, no probe
# =============================================================================


class TestHappyPathUnchanged:
    async def test_injected_provider_used_without_probe_or_fallback(self, monkeypatch):
        aping = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "continuum.session.providers.redis.RedisSessionProvider.aping",
            aping,
            raising=False,
        )
        injected = MagicMock()
        injected.get_or_create_session = AsyncMock(return_value="sess-x")
        injected.is_initialized = True

        cfg = SessionConfig(enabled=True, redis_host="localhost", redis_port=6380,
                            redis_password="ut-strong-redis-pw-0123456789")
        client = SessionClient(session_config=cfg, provider=injected, auto_initialize=False)

        sid = await client.get_or_create_session(user_id="alice")
        assert sid == "sess-x"
        injected.get_or_create_session.assert_awaited_once()
        # An explicitly injected provider must never be probed or swapped out.
        aping.assert_not_called()
        assert client._provider is injected
