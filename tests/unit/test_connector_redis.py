"""
Phase 2 contract: RedisConnector owns the live Redis client build, infers the
connection mode (local-docker / cloud / custom), probes quietly, and masks
secrets in describe(). RedisSessionProvider sources its client from it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio.connection import Connection, SSLConnection

from continuum.connectors.base import ConnectionMode
from continuum.connectors.redis import RedisConnector
from continuum.exceptions import InsecureConfigurationError
from continuum.session.config import SessionConfig

# A strong (non-placeholder) password so build_client() passes the fail-closed
# credential guard — these tests exercise connector mechanics, not the secret.
_PW = "ut-strong-redis-pw-0123456789"


class TestModeInference:
    def test_localhost_is_local_docker(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost"))
        assert c.mode is ConnectionMode.LOCAL_DOCKER

    def test_ssl_is_cloud(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="my.redis.cloud", redis_ssl=True, redis_password=_PW))
        assert c.mode is ConnectionMode.CLOUD

    def test_remote_without_ssl_is_custom(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="10.0.0.5", redis_ssl=False, redis_password=_PW))
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
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost", redis_port=6380, redis_password=_PW))
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


class TestTlsConnection:
    """Regression: a pool selects TLS via connection_class=SSLConnection, not an
    `ssl=True` kwarg. Passing `ssl=True` to the pool reaches AbstractConnection
    and raises TypeError lazily on first command (S-TLS). Also guards the cert
    verification default from silently weakening.
    """

    def test_tls_uses_ssl_connection_class(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="my.redis.cloud", redis_ssl=True, redis_password=_PW))
        pool = c.build_client().connection_pool
        assert pool.connection_class is SSLConnection

    def test_tls_connection_object_builds_without_typeerror(self):
        # The original bug surfaced lazily when the pool first built a connection.
        c = RedisConnector(SessionConfig(enabled=True, redis_host="my.redis.cloud", redis_ssl=True, redis_password=_PW))
        pool = c.build_client().connection_pool
        conn = pool.connection_class(**pool.connection_kwargs)  # must not raise
        assert isinstance(conn, SSLConnection)
        # No stray `ssl` kwarg leaked into connection_kwargs.
        assert "ssl" not in pool.connection_kwargs

    def test_non_tls_path_untouched(self):
        c = RedisConnector(SessionConfig(enabled=True, redis_host="10.0.0.5", redis_ssl=False, redis_password=_PW))
        pool = c.build_client().connection_pool
        assert pool.connection_class is Connection
        assert "ssl" not in pool.connection_kwargs
        assert "ssl_cert_reqs" not in pool.connection_kwargs

    def test_cert_kwargs_omitted_when_unset(self):
        # Default (None) must NOT pass ssl_cert_reqs=None (which maps to CERT_NONE,
        # disabling verification). Omitting keeps redis-py's verifying default.
        c = RedisConnector(SessionConfig(enabled=True, redis_host="my.redis.cloud", redis_ssl=True, redis_password=_PW))
        kwargs = c.build_client().connection_pool.connection_kwargs
        assert "ssl_cert_reqs" not in kwargs
        assert "ssl_ca_certs" not in kwargs

    def test_cert_kwargs_passed_through_when_set(self):
        c = RedisConnector(
            SessionConfig(
                enabled=True,
                redis_host="my.redis.cloud",
                redis_ssl=True,
                redis_password=_PW,
                redis_ssl_cert_reqs="none",
                redis_ssl_ca_certs="/etc/ssl/my-ca.pem",
            )
        )
        kwargs = c.build_client().connection_pool.connection_kwargs
        assert kwargs["ssl_cert_reqs"] == "none"
        assert kwargs["ssl_ca_certs"] == "/etc/ssl/my-ca.pem"


class TestProviderProbeReason:
    """The session provider's aping() records WHY it failed (last_probe_error) so
    the caller can report the real cause instead of a generic 'unreachable'. The
    never-raise bool contract is preserved.
    """

    def _provider(self):
        from continuum.session.providers.redis import RedisSessionProvider

        return RedisSessionProvider(
            SessionConfig(enabled=True, redis_host="localhost", redis_port=6380, redis_password=_PW),
            auto_initialize=True,
        )

    async def test_success_clears_reason(self, monkeypatch):
        p = self._provider()
        p._redis = MagicMock()
        p._redis.ping = AsyncMock(return_value=True)
        assert await p.aping() is True
        assert p.last_probe_error is None

    async def test_failure_records_reason(self, monkeypatch):
        p = self._provider()
        p._redis = MagicMock()
        p._redis.ping = AsyncMock(side_effect=ValueError("WRONGPASS bad auth"))
        assert await p.aping() is False  # never raises
        assert p.last_probe_error is not None
        assert "ValueError" in p.last_probe_error
        assert "WRONGPASS" in p.last_probe_error

    async def test_prior_error_cleared_on_later_success(self):
        p = self._provider()
        p._redis = MagicMock()
        p._redis.ping = AsyncMock(side_effect=ValueError("boom"))
        await p.aping()
        assert p.last_probe_error is not None
        p._redis.ping = AsyncMock(return_value=True)
        await p.aping()
        assert p.last_probe_error is None

    async def test_none_client_sets_reason(self):
        p = self._provider()
        p._redis = None
        assert await p.aping() is False
        assert p.last_probe_error is not None


class TestFailClosedCredential:
    """build_client() refuses a missing/weak session secret (F8/D2) unless the
    CONTINUUM_ALLOW_INSECURE escape hatch is set."""

    def test_weak_password_refuses_build(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = RedisConnector(
            SessionConfig(enabled=True, redis_host="localhost", redis_password="myredissecret")
        )
        with pytest.raises(InsecureConfigurationError, match="SESSION_REDIS_PASSWORD"):
            c.build_client()

    def test_blank_password_refuses_build(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        # Force blank explicitly — don't rely on the default, which reads the
        # ambient SESSION_REDIS_PASSWORD env and would make this env-dependent.
        c = RedisConnector(
            SessionConfig(enabled=True, redis_host="localhost", redis_password="")
        )
        with pytest.raises(InsecureConfigurationError):
            c.build_client()

    def test_escape_hatch_allows_weak(self, monkeypatch):
        monkeypatch.setenv("CONTINUUM_ALLOW_INSECURE", "1")
        c = RedisConnector(SessionConfig(enabled=True, redis_host="localhost"))
        assert c.build_client() is not None


class TestProviderUsesConnector:
    def test_redis_session_provider_sources_client_from_connector(self):
        from continuum.session.providers.redis import RedisSessionProvider

        provider = RedisSessionProvider(
            SessionConfig(enabled=True, redis_host="localhost", redis_port=6380, redis_password=_PW),
            auto_initialize=True,
        )
        assert provider.is_initialized is True
        assert isinstance(provider._connector, RedisConnector)
        # The provider's live client is the one the connector built.
        assert provider._redis is provider._connector.build_client()
