"""
Langfuse connector — brings observability/tracing under the connector module.

A "we own the client" connector: ``connect()`` returns a live ``Langfuse``
client (authenticated by public/secret keys), so observability is configured and
probed through the same uniform interface as Redis/Temporal — self-hosted via
local Docker or the managed cloud.

Note: the LLM providers are intentionally NOT modeled as a connector. They are a
per-request *router* (provider chosen by model prefix, optional Smart Gateway,
fallback chains), not a persistent connection — there is no single client to
own. They keep their existing routing layer.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger
from continuum.utils.secrets import mask_value

if TYPE_CHECKING:
    from continuum.observability.config import ObservabilityConfig

logger = get_logger(__name__)


class LangfuseConnector(BaseConnector[Any]):
    """Connector for the Langfuse observability backend."""

    name = "langfuse"

    _LOCAL_HINTS = ("localhost", "127.0.0.1", "://langfuse")

    def __init__(self, config: ObservabilityConfig | None = None) -> None:
        if config is None:
            from continuum.observability.config import ObservabilityConfig

            config = ObservabilityConfig()
        self._config = config

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def is_configured(self) -> bool:
        # Langfuse authenticates with a public/secret key pair (its "API keys"),
        # required for both self-hosted and cloud.
        return bool(self._config.public_key and self._config.secret_key)

    @property
    def mode(self) -> ConnectionMode:
        if not self._config.enabled:
            return ConnectionMode.DISABLED
        host = (self._config.host or "").lower()
        if any(hint in host for hint in self._LOCAL_HINTS):
            return ConnectionMode.LOCAL_DOCKER
        # Remote host with API keys → managed/cloud (or remote self-host).
        return ConnectionMode.CLOUD

    async def connect(self) -> Any:
        """Build and return a Langfuse client from the configured keys/host."""
        from langfuse import Langfuse

        return Langfuse(**self._config.to_langfuse_kwargs())

    async def aping(self) -> bool:
        """Verify credentials via auth_check — returns False on failure, never raises.

        The Langfuse client is synchronous, so the check runs in a worker thread.
        """

        def _check() -> bool:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=self._config.public_key,
                secret_key=self._config.secret_key,
                host=self._config.host,
            )
            try:
                return bool(client.auth_check())
            finally:
                try:
                    client.shutdown()
                except Exception:  # noqa: BLE001
                    pass

        try:
            return await asyncio.to_thread(_check)
        except Exception as e:  # noqa: BLE001 — probe must never propagate
            logger.debug("Langfuse connector aping failed: %s", e)
            return False

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            host=self._config.host,
            public_key=mask_value(self._config.public_key) if self._config.public_key else None,
            secret_key=mask_value(self._config.secret_key) if self._config.secret_key else None,
        )
        return d
