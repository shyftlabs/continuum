"""
PostgreSQL connector — probes the database behind the postgres session provider.

``PostgresSessionProvider`` owns its own pool (or uses the application's), so
this connector does not hand out connections for sessions. ``connect()`` opens
a short-lived probe connection, which is what ``health_check_all`` and the
core health check need. asyncpg is imported lazily: the ``[postgres]`` extra
is optional and a Redis-only install must still import this module.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger
from continuum.session.config import SessionConfig
from continuum.session.providers.postgres import DSN_PARSE_ERROR

logger = get_logger(__name__)


class PostgresConnector(BaseConnector[Any]):
    """Connector that probes the PostgreSQL database used for sessions."""

    name = "postgres"

    _LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres", "db"}

    def __init__(self, config: SessionConfig | None = None) -> None:
        self._config = config or SessionConfig()

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled and self._config.provider == "postgres"

    def is_configured(self) -> bool:
        return bool(self._config.postgres_dsn)

    def _target(self) -> dict[str, Any]:
        """Host, port and database from the DSN. Never the user or password."""
        dsn = self._config.postgres_dsn
        empty = {"host": None, "port": None, "database": None}
        if not dsn:
            return empty
        if "://" not in dsn:
            # libpq keyword form ("host=db dbname=app ..."): asyncpg does not accept
            # it, so there is no target to report.
            return empty
        # An unencoded "#", "?" or "/" in the password makes urlsplit cut the URL
        # inside it, so keep only what follows the last "@". That split is safe
        # only when every "@" comes before the first "?" or "#": an "@" after
        # one is either in a query value (password=...) or in a password that
        # also holds "?" or "#", and the two cannot be told apart.
        scheme, _, rest = dsn.partition("://")
        query_at = min((i for i in (rest.find("?"), rest.find("#")) if i >= 0), default=len(rest))
        if "@" in rest[query_at:]:
            return empty
        try:
            parts = urlsplit(f"{scheme}://{rest.rpartition('@')[2]}")
            return {
                "host": parts.hostname,
                "port": parts.port,
                "database": parts.path.lstrip("/") or None,
            }
        except ValueError:
            return empty

    @property
    def mode(self) -> ConnectionMode:
        if not self.is_enabled:
            return ConnectionMode.DISABLED
        if self._target()["host"] in self._LOCAL_HOSTS:
            return ConnectionMode.LOCAL_DOCKER
        return ConnectionMode.CUSTOM

    async def connect(self) -> Any:
        """Open a probe connection. The caller closes it."""
        if not self.is_configured():
            raise RuntimeError("PostgreSQL is not configured. Set SESSION_POSTGRES_DSN.")
        import asyncpg  # type: ignore[import-untyped]

        try:
            return await asyncpg.connect(
                self._config.postgres_dsn,
                timeout=self._config.postgres_timeout_seconds,
                command_timeout=self._config.postgres_timeout_seconds,
            )
        except ValueError:
            # asyncpg's DSN parser quotes the text it failed on, which can be part
            # of the password; report a fixed message instead.
            raise ValueError(DSN_PARSE_ERROR) from None

    async def server_version(self) -> str:
        """Run one query and return the server version. Raises on any failure."""
        conn = await self.connect()
        try:
            return str(await conn.fetchval("SHOW server_version"))
        finally:
            await conn.close()

    async def aping(self) -> bool:
        """Connect and query once — returns False on any failure, never raises."""
        try:
            await self.server_version()
            return True
        except Exception as e:  # noqa: BLE001 — probe must never propagate
            logger.debug("PostgreSQL connector aping failed: %s", e)
            return False

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(self._target())
        return d
