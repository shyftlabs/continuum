"""
Phase 1 contract for the connector module (Part B): the uniform interface and
registry that every external-service connector plugs into.

No real services here — a dummy connector validates the base contract and the
registry/health-aggregation behavior that all concrete connectors rely on.
"""

from __future__ import annotations

import pytest

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.connectors.registry import (
    get_connector,
    health_check_all,
    list_connectors,
    register_connector,
    unregister_connector,
)


class _DummyConnector(BaseConnector[str]):
    name = "dummy"

    def __init__(
        self, *, enabled=True, configured=True, reachable=True, mode=ConnectionMode.CUSTOM
    ):
        self._enabled = enabled
        self._configured = configured
        self._reachable = reachable
        self._mode = mode
        self.connect_calls = 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def is_configured(self) -> bool:
        return self._configured

    @property
    def mode(self) -> ConnectionMode:
        return self._mode

    async def connect(self) -> str:
        self.connect_calls += 1
        if not self._reachable:
            raise ConnectionError("unreachable")
        return "client"


@pytest.fixture(autouse=True)
def _clean_registry():
    for n in list_connectors():
        unregister_connector(n)
    yield
    for n in list_connectors():
        unregister_connector(n)


class TestBaseContract:
    async def test_default_aping_true_when_connect_succeeds(self):
        assert await _DummyConnector().aping() is True

    async def test_default_aping_false_and_quiet_when_connect_fails(self):
        # Never raises — returns False on any connect error.
        assert await _DummyConnector(reachable=False).aping() is False

    def test_describe_reports_common_fields(self):
        d = _DummyConnector(mode=ConnectionMode.CLOUD).describe()
        assert d["name"] == "dummy"
        assert d["enabled"] is True
        assert d["configured"] is True
        assert d["mode"] == "cloud"


class TestRegistry:
    def test_register_get_list(self):
        c = _DummyConnector()
        register_connector("dummy", c)
        assert get_connector("dummy") is c
        assert "dummy" in list_connectors()

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_connector("nope")

    def test_duplicate_register_raises_unless_replace(self):
        register_connector("dummy", _DummyConnector())
        with pytest.raises(ValueError):
            register_connector("dummy", _DummyConnector())
        # replace=True is allowed
        new = _DummyConnector()
        register_connector("dummy", new, replace=True)
        assert get_connector("dummy") is new


class TestHealthCheckAll:
    async def test_aggregates_enabled_and_skips_disabled(self):
        register_connector("ok", _DummyConnector(reachable=True))
        register_connector("down", _DummyConnector(reachable=False))
        register_connector("off", _DummyConnector(enabled=False))

        report = await health_check_all()

        assert report["ok"]["status"] == "healthy"
        assert report["down"]["status"] == "unhealthy"
        assert report["off"]["status"] == "disabled"
        # Disabled connectors are never probed (no connection attempt).
        assert get_connector("off").connect_calls == 0
