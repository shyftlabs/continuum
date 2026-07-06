"""
Phase 5 contract: the three built-in connectors register under uniform names,
are reachable via the container, and health_check_all aggregates them without
probing disabled ones.
"""

from __future__ import annotations

import pytest

from continuum.connectors.base import BaseConnector
from continuum.connectors.registry import (
    get_connector,
    list_connectors,
    register_default_connectors,
    unregister_connector,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    for n in list_connectors():
        unregister_connector(n)
    yield
    for n in list_connectors():
        unregister_connector(n)


class TestDefaultRegistration:
    def test_registers_the_three_named_services(self):
        register_default_connectors()
        names = set(list_connectors())
        assert {"redis", "vector_store", "temporal"} <= names

    def test_each_default_is_a_connector_with_describe(self):
        register_default_connectors()
        for name in ("redis", "vector_store", "temporal"):
            c = get_connector(name)
            assert isinstance(c, BaseConnector)
            d = c.describe()  # must not touch the network
            assert d["name"] == name
            assert "mode" in d

    def test_register_default_is_idempotent(self):
        register_default_connectors()
        register_default_connectors()  # replace=True → no duplicate error
        assert len([n for n in list_connectors() if n == "redis"]) == 1


class TestContainerExposesConnectors:
    def test_container_connectors_returns_the_three(self):
        from continuum.core.container import get_container, reset_container

        reset_container()
        try:
            conns = get_container().connectors
            assert {"redis", "vector_store", "temporal"} <= set(conns)
            assert all(isinstance(c, BaseConnector) for c in conns.values())
        finally:
            reset_container()


class TestHealthCheckAllSkipsDisabled:
    async def test_disabled_connector_reported_not_probed(self, monkeypatch):
        from unittest.mock import AsyncMock

        from continuum.connectors.registry import health_check_all

        register_default_connectors()
        # Force temporal disabled (default) and ensure no connector is dialed for it.
        temporal = get_connector("temporal")
        # If temporal is disabled in this env, it must be reported without a probe.
        if not temporal.is_enabled:
            spy = AsyncMock(return_value=True)
            monkeypatch.setattr(temporal, "aping", spy)
            report = await health_check_all()
            assert report["temporal"]["status"] == "disabled"
            spy.assert_not_called()
        else:
            pytest.skip("temporal enabled in this environment")
