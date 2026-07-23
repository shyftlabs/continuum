"""
Production hardening of the session fallback:

- SESSION_FALLBACK_MODE = "degrade" (default) keeps the in-memory fallback;
  "fail" raises instead of silently degrading.
- SessionClient.persistence_degraded is a health flag — True only when the
  client actually fell back from Redis (not when in-memory was chosen on purpose).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from continuum.exceptions import InsecureConfigurationError
from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.exceptions import SessionConnectionError
from continuum.session.providers.memory import MemorySessionProvider
from continuum.session.providers.redis import RedisSessionProvider


class TestDegradeMode:
    async def test_unreachable_degrades_and_sets_flag(self, monkeypatch):
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=False))
        cfg = SessionConfig(enabled=True, redis_host="localhost",
                            redis_password="ut-strong-redis-pw-0123456789", fallback_mode="degrade")
        sc = SessionClient(session_config=cfg, auto_initialize=False)

        sid = await sc.get_or_create_session(user_id="a")
        assert sid
        assert isinstance(sc._provider, MemorySessionProvider)
        assert sc.persistence_degraded is True

    async def test_weak_secret_refuses_even_in_degrade_mode(self, monkeypatch):
        # An insecure secret is an operator misconfiguration, not an outage:
        # it must hard-fail (refuse-to-boot) rather than silently degrade,
        # regardless of fallback_mode. (Phase 5)
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        cfg = SessionConfig(enabled=True, redis_host="localhost",
                            redis_password="miniosecret", fallback_mode="degrade")
        sc = SessionClient(session_config=cfg, auto_initialize=False)
        with pytest.raises(InsecureConfigurationError):
            await sc.get_or_create_session(user_id="a")
        assert sc.persistence_degraded is False

    async def test_escape_hatch_allows_degrade_with_weak_secret(self, monkeypatch):
        # With the escape hatch, the guard downgrades to a warning and normal
        # outage handling resumes (here: unreachable -> degrade).
        monkeypatch.setenv("CONTINUUM_ALLOW_INSECURE", "1")
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=False))
        cfg = SessionConfig(enabled=True, redis_host="localhost",
                            redis_password="miniosecret", fallback_mode="degrade")
        sc = SessionClient(session_config=cfg, auto_initialize=False)
        await sc.get_or_create_session(user_id="a")
        assert isinstance(sc._provider, MemorySessionProvider)
        assert sc.persistence_degraded is True

    async def test_explicit_memory_is_not_marked_degraded(self):
        cfg = SessionConfig(enabled=True, provider="memory")
        sc = SessionClient(session_config=cfg, auto_initialize=False)

        await sc.get_or_create_session(user_id="a")
        assert isinstance(sc._provider, MemorySessionProvider)
        # Chosen on purpose — not a degradation.
        assert sc.persistence_degraded is False


class TestFailMode:
    async def test_unreachable_raises_instead_of_degrading(self, monkeypatch):
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=False))
        cfg = SessionConfig(enabled=True, redis_host="localhost",
                            redis_password="ut-strong-redis-pw-0123456789", fallback_mode="fail")
        sc = SessionClient(session_config=cfg, auto_initialize=False)

        with pytest.raises(SessionConnectionError):
            await sc.get_or_create_session(user_id="a")
        assert sc.persistence_degraded is False

    async def test_unconfigured_raises(self, monkeypatch):
        cfg = SessionConfig(enabled=True, redis_host="", fallback_mode="fail")
        sc = SessionClient(session_config=cfg, auto_initialize=False)

        with pytest.raises(SessionConnectionError):
            await sc.get_or_create_session(user_id="a")

    async def test_midsession_failure_raises_not_degrades(self, monkeypatch):
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=True))
        monkeypatch.setattr(
            RedisSessionProvider,
            "get_or_create_session",
            AsyncMock(side_effect=SessionConnectionError("timeout")),
        )
        cfg = SessionConfig(enabled=True, redis_host="localhost",
                            redis_password="ut-strong-redis-pw-0123456789", fallback_mode="fail")
        sc = SessionClient(session_config=cfg, auto_initialize=False)

        with pytest.raises(SessionConnectionError):
            await sc.get_or_create_session(user_id="a")
        assert not isinstance(sc._provider, MemorySessionProvider)
        assert sc.persistence_degraded is False


class TestDefaultMode:
    def test_fallback_mode_defaults_from_settings(self):
        # Env-independent: the field must default from global settings (whatever
        # the ambient SESSION_FALLBACK_MODE is), not a hard-coded literal.
        from continuum.config import settings

        assert SessionConfig(enabled=True).fallback_mode == settings.session_fallback_mode

    def test_settings_default_is_degrade(self, monkeypatch):
        # The Settings *default* (no env override) is 'degrade'.
        from continuum.config import Settings

        monkeypatch.delenv("SESSION_FALLBACK_MODE", raising=False)
        assert Settings().session_fallback_mode == "degrade"
