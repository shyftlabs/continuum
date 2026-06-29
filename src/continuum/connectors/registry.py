"""
Connector registry — the single discovery point for all external-service
connectors.

Concrete connectors register here so callers can look them up by name, list
them, and run a uniform health check across every enabled service.
"""

from __future__ import annotations

from typing import Any

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger

logger = get_logger(__name__)

_REGISTRY: dict[str, BaseConnector[Any]] = {}


def register_connector(
    name: str, connector: BaseConnector[Any], *, replace: bool = False
) -> None:
    """Register a connector under ``name``.

    Raises ValueError if the name is already taken unless ``replace=True``.
    """
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"Connector '{name}' is already registered. Pass replace=True to override."
        )
    _REGISTRY[name] = connector


def get_connector(name: str) -> BaseConnector[Any]:
    """Return the connector registered under ``name``.

    Raises KeyError with the available names if not found.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"Unknown connector: '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_connectors() -> list[str]:
    """List registered connector names (sorted)."""
    return sorted(_REGISTRY)


def all_connectors() -> dict[str, BaseConnector[Any]]:
    """Return a snapshot mapping of all registered connectors."""
    return dict(_REGISTRY)


def register_default_connectors(*, replace: bool = True) -> None:
    """Register the built-in connectors (redis, vector_store, temporal).

    Idempotent by default (``replace=True``). Each is constructed from the
    global settings; consumers can re-register with a custom-config instance.
    """
    from continuum.connectors.redis import RedisConnector
    from continuum.connectors.temporal import TemporalConnector
    from continuum.connectors.vector_store import VectorStoreConnector

    register_connector("redis", RedisConnector(), replace=replace)
    register_connector("vector_store", VectorStoreConnector(), replace=replace)
    register_connector("temporal", TemporalConnector(), replace=replace)


def unregister_connector(name: str) -> None:
    """Remove a connector from the registry (no error if absent). For tests."""
    _REGISTRY.pop(name, None)


async def health_check_all() -> dict[str, dict[str, Any]]:
    """Probe every registered connector and return a per-connector report.

    Disabled connectors are reported as ``disabled`` and never probed, so a
    turned-off service costs no connection attempt.
    """
    report: dict[str, dict[str, Any]] = {}
    for name, connector in _REGISTRY.items():
        if not connector.is_enabled:
            report[name] = {"status": "disabled", "mode": ConnectionMode.DISABLED.value}
            continue
        ok = await connector.aping()
        report[name] = {"status": "healthy" if ok else "unhealthy", **connector.describe()}
    return report
