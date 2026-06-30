"""
Make persistence_degraded observable:
  #1 a 'session_persistence' health check (DEGRADED when the client fell back),
  #2 a 'session_persistence_degraded' gauge metric emitted on degrade / recovery.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.providers.redis import RedisSessionProvider


class TestHealthCheck:
    async def test_reports_degraded_when_client_degraded(self, monkeypatch):
        from continuum.core import health as health_mod
        from continuum.core.container import get_container, reset_container
        from continuum.core.health import HealthCheck, HealthStatus

        monkeypatch.setattr(health_mod.settings, "session_enabled", True)
        reset_container()
        try:
            fake = MagicMock()
            fake.persistence_degraded = True
            get_container().set_session_client(fake)

            result = await HealthCheck()._check_session_persistence()
            assert result.status is HealthStatus.DEGRADED
            assert result.details["persistence_degraded"] is True
        finally:
            reset_container()

    async def test_reports_healthy_when_not_degraded(self, monkeypatch):
        from continuum.core import health as health_mod
        from continuum.core.container import get_container, reset_container
        from continuum.core.health import HealthCheck, HealthStatus

        monkeypatch.setattr(health_mod.settings, "session_enabled", True)
        reset_container()
        try:
            fake = MagicMock()
            fake.persistence_degraded = False
            get_container().set_session_client(fake)

            result = await HealthCheck()._check_session_persistence()
            assert result.status is HealthStatus.HEALTHY
            assert result.details["persistence_degraded"] is False
        finally:
            reset_container()

    async def test_disabled_is_healthy(self, monkeypatch):
        from continuum.core import health as health_mod
        from continuum.core.health import HealthCheck, HealthStatus

        monkeypatch.setattr(health_mod.settings, "session_enabled", False)
        result = await HealthCheck()._check_session_persistence()
        assert result.status is HealthStatus.HEALTHY

    def test_registered_by_default(self):
        from continuum.core.health import HealthCheck

        assert "session_persistence" in HealthCheck()._checks

    async def test_does_not_create_session_client(self, monkeypatch):
        # A health check must observe, not initialize: it must NOT force-create
        # the session client (which would cascade to the memory client and
        # eagerly connect the vector store — a real hang risk during startup).
        from continuum.core import health as health_mod
        from continuum.core.container import get_container, reset_container
        from continuum.core.health import HealthCheck, HealthStatus

        monkeypatch.setattr(health_mod.settings, "session_enabled", True)
        reset_container()
        try:
            assert get_container().has_session_client() is False
            result = await HealthCheck()._check_session_persistence()
            assert result.status is HealthStatus.HEALTHY
            # Still not created — the check observed without initializing.
            assert get_container().has_session_client() is False
        finally:
            reset_container()


class TestMetric:
    async def test_degrade_emits_gauge_1(self, monkeypatch):
        collector = MagicMock()
        monkeypatch.setattr(
            "continuum.observability.metrics.get_metrics_collector", lambda: collector
        )
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=False))

        cfg = SessionConfig(enabled=True, redis_host="localhost", fallback_mode="degrade")
        sc = SessionClient(session_config=cfg, auto_initialize=False)
        await sc.get_or_create_session(user_id="a")

        collector.record_metric.assert_any_call("session_persistence_degraded", 1.0)

    async def test_healthy_resolution_emits_gauge_0(self, monkeypatch):
        collector = MagicMock()
        monkeypatch.setattr(
            "continuum.observability.metrics.get_metrics_collector", lambda: collector
        )
        monkeypatch.setattr(RedisSessionProvider, "aping", AsyncMock(return_value=True))
        monkeypatch.setattr(
            RedisSessionProvider, "get_or_create_session", AsyncMock(return_value="sess-1")
        )

        cfg = SessionConfig(enabled=True, redis_host="localhost", fallback_mode="degrade")
        sc = SessionClient(session_config=cfg, auto_initialize=False)
        await sc.get_or_create_session(user_id="a")

        collector.record_metric.assert_any_call("session_persistence_degraded", 0.0)
