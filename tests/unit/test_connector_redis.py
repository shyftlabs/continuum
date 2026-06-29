"""
Phase 2 contract: RedisConnector owns the live Redis client build, infers the
connection mode (local-docker / cloud / custom), probes quietly, and masks
secrets in describe(). RedisSessionProvider sources its client from it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.connectors.base import ConnectionMode
from continuum.connectors.redis import RedisConnector
from continuum.session.config import SessionConfig


class TestModeInference:
    def test_localhost_is_local_docker(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost"))
        assert c.mode is ConnectionMode.LOCAL_DOCKER

    def test_ssl_is_cloud(self):
        c = RedisConnector(
            SessionConfig(enabled=True, redis_host="my.redis.cloud", redis_ssl=True)
        )
        assert c.mode is ConnectionMode.CLOUD

    def test_remote_without_ssl_is_custom(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="10.0.0.5", redis_ssl=False))
        assert c.mode is ConnectionMode.CUSTOM

    def test_disabled_is_disabled(self):
        c = RedisConnector(SessionConfig(enabled=False, redis_host="localhost"))
        assert c.mode is ConnectionMode.DISABLED


class TestConfiguredAndEnabled:
    def test_enabled_and_configured(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost"))
        assert c.is_enabled is True
        assert c.is_configured() is True

    def test_unconfigured_when_host_empty(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host=""))
        assert c.is_configured() is False


class TestBuildClient:
    def test_build_client_does_not_connect_and_is_cached(self):
        # Constructing a redis.asyncio client/pool does not open a socket.
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost", redis_port=6380))
        client = c.build_client()
        assert client is not None
        assert c.pool is not None
        # Cached — same instance on repeat.
        assert c.build_client() is client


class TestApingIsQuiet:
    async def test_aping_true_when_ping_ok(self, monkeypatch):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost"))
        fake = MagicMock()
        fake.ping = AsyncMock(return_value=True)
        monkeypatch.setattr(c, "connect", AsyncMock(return_value=fake))
        assert await c.aping() is True

    async def test_aping_false_when_unreachable(self, monkeypatch):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost"))
        monkeypatch.setattr(c, "connect", AsyncMock(side_effect=ConnectionError("down")))
        assert await c.aping() is False  # never raises


class TestDescribeMasksSecrets:
    def test_password_is_masked(self):
        c = RedisConnector(
            SessionConfig(enabled=True, redis_host="localhost", redis_password="supersecret")
        )
        d = c.describe()
        assert d["name"] == "redis"
        assert d["host"] == "localhost"
        assert "supersecret" not in str(d)


class TestProviderUsesConnector:
    def test_redis_session_provider_sources_client_from_connector(self):
        from continuum.session.providers.redis import RedisSessionProvider

        provider = RedisSessionProvider(
            SessionConfig(enabled=True, redis_host="localhost", redis_port=6380),
            auto_initialize=True,
        )
        assert provider.is_initialized is True
        assert isinstance(provider._connector, RedisConnector)
        # The provider's live client is the one the connector built.
        assert provider._redis is provider._connector.build_client()
