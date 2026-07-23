"""
When the Redis connectivity probe fails, the SessionClient must report the
ACTUAL cause (auth error, TLS handshake, bad config, ...) captured by the
provider's ``last_probe_error`` — not a blanket "Redis is unreachable", which
mislabels every config/setup failure as a network outage.

Mock-first unit tests — no live Redis. They pin the fallback *message* while
leaving the degrade/fail decision (governed by fallback_mode) unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionConnectionError


def _spy_client_logger(monkeypatch) -> MagicMock:
    spy = MagicMock()
    monkeypatch.setattr("continuum.session.client.logger", spy)
    return spy


def _patch_provider(monkeypatch, *, reachable: bool, reason: str | None) -> None:
    """Make RedisSessionProvider.aping() report the given reachability + reason."""
    monkeypatch.setattr(
        "continuum.session.providers.redis.RedisSessionProvider.aping",
        AsyncMock(return_value=reachable),
        raising=False,
    )
    # last_probe_error is a property on the class; override with a plain attribute
    # value for the test by patching the property to return our reason.
    monkeypatch.setattr(
        "continuum.session.providers.redis.RedisSessionProvider.last_probe_error",
        property(lambda self: reason),
        raising=False,
    )


class TestFallbackMessageReportsCause:
    async def test_captured_reason_appears_in_fail_message(self, monkeypatch):
        # fallback_mode='fail' surfaces the reason as a raised error we can assert on.
        _patch_provider(
            monkeypatch, reachable=False, reason="AuthenticationError: WRONGPASS invalid password"
        )
        cfg = SessionConfig(
            enabled=True, redis_host="localhost", redis_port=6380,
            redis_password="ut-strong-redis-pw-0123456789", fallback_mode="fail"
        )
        client = SessionClient(session_config=cfg, auto_initialize=False)

        with pytest.raises(SessionConnectionError) as ei:
            await client.get_or_create_session(user_id="alice")
        msg = str(ei.value)
        assert "AuthenticationError" in msg
        assert "WRONGPASS" in msg
        assert "unreachable" not in msg.lower()

    async def test_missing_reason_falls_back_to_generic(self, monkeypatch):
        # Provider exposes no reason (None) -> keep the generic wording (back-compat).
        _patch_provider(monkeypatch, reachable=False, reason=None)
        cfg = SessionConfig(
            enabled=True, redis_host="localhost", redis_port=6380,
            redis_password="ut-strong-redis-pw-0123456789", fallback_mode="fail"
        )
        client = SessionClient(session_config=cfg, auto_initialize=False)

        with pytest.raises(SessionConnectionError) as ei:
            await client.get_or_create_session(user_id="alice")
        assert "unreachable" in str(ei.value).lower()

    async def test_reachable_returns_redis_provider(self, monkeypatch):
        # Happy path unchanged: reachable -> real Redis provider, no fallback.
        _patch_provider(monkeypatch, reachable=True, reason=None)
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
        from continuum.session.providers.redis import RedisSessionProvider

        assert isinstance(client._provider, RedisSessionProvider)
