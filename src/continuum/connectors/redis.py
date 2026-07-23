"""
Redis connector — owns the live Redis client/connection-pool build.

This is a "we own the client" connector: ``connect()`` returns the actual
``redis.asyncio.Redis`` client, so the connector is the single place Redis
connections are constructed. ``RedisSessionProvider`` (and any other Redis
consumer) sources its client here instead of hand-rolling pool setup.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import redis.asyncio as redis

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger
from continuum.security.secrets_guard import enforce_credential
from continuum.session.config import SessionConfig
from continuum.utils.secrets import mask_value

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


class RedisConnector(BaseConnector["Redis"]):
    """Connector that builds and probes the Redis client used for sessions."""

    name = "redis"

    # Hosts treated as a local `continuum up` / Docker deployment.
    _LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "redis", "redis-sdk"}

    def __init__(self, config: SessionConfig | None = None) -> None:
        self._config = config or SessionConfig()
        self._client: Redis | None = None
        self._pool: Any = None

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def is_configured(self) -> bool:
        return bool(self._config.redis_host)

    @property
    def mode(self) -> ConnectionMode:
        if not self._config.enabled:
            return ConnectionMode.DISABLED
        if self._config.redis_host in self._LOCAL_HOSTS:
            return ConnectionMode.LOCAL_DOCKER
        # TLS implies a managed/cloud endpoint.
        if self._config.redis_ssl:
            return ConnectionMode.CLOUD
        return ConnectionMode.CUSTOM

    @property
    def pool(self) -> Any:
        """The underlying connection pool (None until build_client is called)."""
        return self._pool

    def build_client(self) -> Redis:
        """Construct (once) and return the async Redis client.

        Building the client/pool does not open a socket — redis-py connects
        lazily on first command — so this is cheap and connection-free.
        """
        if self._client is not None:
            return self._client

        if not self.is_configured():
            raise RuntimeError(
                "Redis is not configured. Set SESSION_REDIS_HOST / SESSION_REDIS_PORT."
            )

        # Fail-closed on a missing/weak session secret (F8/D2). Refuses to build
        # the client unless CONTINUUM_ALLOW_INSECURE=1 is set (local/testing).
        enforce_credential(
            service="Session Redis",
            credential=self._config.redis_password,
            env_var="SESSION_REDIS_PASSWORD",
        )

        conn_kwargs: dict[str, Any] = {
            "host": self._config.redis_host,
            "port": self._config.redis_port,
            "password": self._config.redis_password,
            "db": self._config.redis_db,
            "max_connections": self._config.redis_max_connections,
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 5,
            # Queue (up to `timeout`s) for a free connection under burst instead
            # of raising MaxConnectionsError — prevents silent write loss (S-001).
            "timeout": 5,
        }
        if self._config.redis_ssl:
            # A pool selects TLS via connection_class=SSLConnection — NOT an
            # `ssl=True` kwarg. `ssl=True` is a convenience only on redis.Redis()/
            # from_url(); forwarded to a pool it reaches AbstractConnection.__init__
            # and raises TypeError lazily on first command (S-TLS).
            conn_kwargs["connection_class"] = redis.SSLConnection
            # Pass cert knobs ONLY when explicitly set. Passing ssl_cert_reqs=None
            # would map to CERT_NONE (verification OFF); omitting it keeps redis-py's
            # verifying default ('required'). ssl_ca_certs=None means system CA store.
            if self._config.redis_ssl_cert_reqs is not None:
                conn_kwargs["ssl_cert_reqs"] = self._config.redis_ssl_cert_reqs
            if self._config.redis_ssl_ca_certs is not None:
                conn_kwargs["ssl_ca_certs"] = self._config.redis_ssl_ca_certs

        self._pool = redis.BlockingConnectionPool(**conn_kwargs)
        self._client = redis.Redis(connection_pool=self._pool, decode_responses=True)
        return self._client

    async def connect(self) -> Redis:
        return self.build_client()

    async def aping(self) -> bool:
        """PING the server — returns False on any failure, never raises."""
        try:
            client = await self.connect()
            pong = client.ping()
            if inspect.isawaitable(pong):
                pong = await pong
            return bool(pong)
        except Exception as e:  # noqa: BLE001 — probe must never propagate
            logger.debug("Redis connector aping failed: %s", e)
            return False

    async def aclose(self) -> None:
        """Close the pool/client and drop references."""
        try:
            if self._pool is not None:
                await self._pool.aclose()
            if self._client is not None:
                await self._client.aclose()
        except Exception as e:  # noqa: BLE001
            logger.debug("Error closing Redis connector: %s", e)
        finally:
            self._client = None
            self._pool = None

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        pwd = self._config.redis_password
        d.update(
            host=self._config.redis_host,
            port=self._config.redis_port,
            db=self._config.redis_db,
            ssl=self._config.redis_ssl,
            password=mask_value(pwd) if pwd else None,
        )
        return d
