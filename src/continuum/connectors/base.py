"""
Connector base contract — the uniform interface every external-service
connection plugs into.

A *connector* is the single, consistent place that knows how to reach one
external service (Redis, a vector store, Temporal, …): whether it is enabled,
how it is configured, which connection *mode* it is using (local Docker vs a
cloud/API-key endpoint vs a custom host), how to open a connection, how to
probe it, and how to describe it (with secrets masked) for diagnostics.

There are two flavors, both expressed through this one interface:

* **We own the client** (Redis, Temporal): ``connect()`` returns the live
  client and the connector *is* the connection.
* **A library owns the client** (e.g. mem0 builds its own vector-store client
  from a config dict): ``connect()`` returns a standalone probe client used
  only for health checks, and the connector additionally exposes a
  config-producing method the library consumes. The connector remains the
  single source of truth for connection params, mode, and auth.

Adding a new service
--------------------
1. Create ``connectors/<service>.py`` with a ``BaseConnector`` subclass that
   sets ``name`` and implements ``is_enabled`` / ``is_configured`` / ``mode`` /
   ``connect`` (override ``aping`` / ``describe`` when a cheaper or richer
   implementation exists).
2. Register it in ``connectors/registry.py`` via ``register_connector(...)``.

Once registered it is discoverable via ``get_connector``, included in
``health_check_all``, and gains uniform mode/auth/masking for free. The cost of
adding a service is always one file plus one registration line — it does not
grow as more connectors are added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from continuum.logging import get_logger

logger = get_logger(__name__)


class ConnectionMode(str, Enum):
    """How a connector reaches its service.

    Inferred from configuration (presence of an API key/token/TLS vs. default
    local ports), with an optional explicit override per service.
    """

    LOCAL_DOCKER = "local_docker"  # default host/ports — e.g. `continuum up`
    CLOUD = "cloud"  # api-key / token / TLS endpoint
    CUSTOM = "custom"  # explicit non-default host, no cloud credential
    DISABLED = "disabled"  # turned off by config


class BaseConnector[T](ABC):
    """Abstract base for all external-service connectors.

    Subclasses set the class attribute ``name`` and implement the abstract
    members. ``aping`` and ``describe`` have sensible defaults that most
    connectors can keep.
    """

    name: str = "connector"

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether this connector is turned on by configuration."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether the minimum settings needed to connect are present."""
        ...

    @property
    @abstractmethod
    def mode(self) -> ConnectionMode:
        """The connection mode inferred from configuration."""
        ...

    @abstractmethod
    async def connect(self) -> T:
        """Open and return a connection/client. May raise on failure."""
        ...

    async def aping(self) -> bool:
        """Best-effort connectivity probe — returns False on any failure, never raises.

        Default implementation simply attempts ``connect()``. Override when the
        service offers a cheaper liveness check (e.g. PING / list-namespaces).
        """
        try:
            await self.connect()
            return True
        except Exception as e:  # noqa: BLE001 — probe must never propagate
            logger.debug("Connector '%s' aping failed: %s", self.name, e)
            return False

    def describe(self) -> dict[str, Any]:
        """Return a diagnostics summary. Override to add masked service specifics."""
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "configured": self.is_configured(),
            "mode": self.mode.value,
        }
