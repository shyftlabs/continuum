"""
Temporal connector — owns the Temporal client connect.

A "we own the client" connector: ``connect()`` returns a live
``temporalio.client.Client``, so this is the single place Temporal connections
are opened — supporting local Docker, a custom host, or Temporal Cloud (TLS +
API key). ``TemporalClient.connect`` delegates here.

``temporalio`` is an optional dependency, so it is imported lazily inside
``connect()`` — the connector module itself imports cleanly without it (mode /
describe / config still work).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger
from continuum.temporal.config import TemporalConfig
from continuum.temporal.exceptions import TemporalConnectionError
from continuum.utils.secrets import mask_value

if TYPE_CHECKING:
    from temporalio.client import Client

logger = get_logger(__name__)


class TemporalConnector(BaseConnector["Client"]):
    """Connector that opens and probes the Temporal client."""

    name = "temporal"

    _LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "temporal"}

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self._config = config or TemporalConfig.from_settings()
        self._client: Client | None = None

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def is_configured(self) -> bool:
        return bool(self._config.host)

    @property
    def mode(self) -> ConnectionMode:
        if not self._config.enabled:
            return ConnectionMode.DISABLED
        # TLS or an API key means a managed/cloud endpoint (Temporal Cloud).
        if self._config.tls or self._config.api_key:
            return ConnectionMode.CLOUD
        hostname = self._config.host.split(":", 1)[0]
        if hostname in self._LOCAL_HOSTS:
            return ConnectionMode.LOCAL_DOCKER
        return ConnectionMode.CUSTOM

    async def connect(self, host: str | None = None, namespace: str | None = None) -> Client:
        """Open and return a Temporal client (uses config defaults if not given)."""
        from temporalio.client import Client

        try:
            from temporalio.contrib.pydantic import pydantic_data_converter
        except ImportError:
            pydantic_data_converter = None  # type: ignore[assignment]

        target_host = host or self._config.host
        target_ns = namespace or self._config.namespace

        connect_kw: dict[str, Any] = {"namespace": target_ns}
        if pydantic_data_converter is not None:
            connect_kw["data_converter"] = pydantic_data_converter
        if self._config.tls:
            connect_kw["tls"] = True
        if self._config.api_key:
            connect_kw["api_key"] = self._config.api_key

        try:
            self._client = await Client.connect(target_host, **connect_kw)
            logger.info(f"Connected to Temporal at {target_host} (ns={target_ns})")
            return self._client
        except Exception as e:
            raise TemporalConnectionError(
                f"Failed to connect to Temporal at {target_host}: {e}",
                host=target_host,
                namespace=target_ns,
                original_error=e,
            ) from e

    async def aping(self) -> bool:
        """Connect as the liveness check — returns False on failure, never raises."""
        try:
            await self.connect()
            return True
        except Exception as e:  # noqa: BLE001 — probe must never propagate
            logger.debug("Temporal connector aping failed: %s", e)
            return False

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        key = self._config.api_key
        d.update(
            host=self._config.host,
            namespace=self._config.namespace,
            tls=self._config.tls,
            api_key=mask_value(key) if key else None,
        )
        return d
