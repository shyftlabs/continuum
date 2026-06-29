"""
Connectors — a uniform, pluggable layer for connecting to external services
(Redis, vector stores, Temporal, …).

Each connector exposes the same interface (enabled / configured / mode /
connect / aping / describe) and registers in a shared registry, so connections
are configured and probed consistently — via API keys, local Docker, or custom
hosts — and new services can be added with one file plus one registration.
"""

from __future__ import annotations

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.connectors.registry import (
    all_connectors,
    get_connector,
    health_check_all,
    list_connectors,
    register_connector,
    register_default_connectors,
    unregister_connector,
)

__all__ = [
    "BaseConnector",
    "ConnectionMode",
    "all_connectors",
    "get_connector",
    "health_check_all",
    "list_connectors",
    "register_connector",
    "register_default_connectors",
    "unregister_connector",
]
