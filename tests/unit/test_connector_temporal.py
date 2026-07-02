"""
Phase 4 contract: TemporalConnector owns the Temporal client connect (host /
namespace / TLS / API key), infers mode, probes quietly, masks the API key, and
is what TemporalClient.connect delegates to.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from continuum.connectors.base import ConnectionMode
from continuum.connectors.temporal import TemporalConnector
from continuum.temporal.config import TemporalConfig
from continuum.temporal.exceptions import TemporalConnectionError


class TestModeInference:
    def test_localhost_is_local_docker(self):
        c = TemporalConnector(TemporalConfig(enabled=True, host="localhost:7233"))
        assert c.mode is ConnectionMode.LOCAL_DOCKER

    def test_tls_is_cloud(self):
        c = TemporalConnector(TemporalConfig(enabled=True, host="ns.tmprl.cloud:7233", tls=True))
        assert c.mode is ConnectionMode.CLOUD

    def test_api_key_is_cloud(self):
        c = TemporalConnector(
            TemporalConfig(enabled=True, host="ns.tmprl.cloud:7233", api_key="key")
        )
        assert c.mode is ConnectionMode.CLOUD

    def test_remote_plain_is_custom(self):
        c = TemporalConnector(TemporalConfig(enabled=True, host="10.0.0.7:7233"))
        assert c.mode is ConnectionMode.CUSTOM

    def test_disabled_is_disabled(self):
        c = TemporalConnector(TemporalConfig(enabled=False, host="localhost:7233"))
        assert c.mode is ConnectionMode.DISABLED


class TestConfigured:
    def test_configured_when_host_present(self):
        assert TemporalConnector(
            TemporalConfig(enabled=True, host="localhost:7233")
        ).is_configured()

    def test_unconfigured_when_host_empty(self):
        assert not TemporalConnector(TemporalConfig(enabled=True, host="")).is_configured()


class TestConnect:
    async def test_connect_passes_tls_and_api_key(self, monkeypatch):
        sentinel = object()
        connect_mock = AsyncMock(return_value=sentinel)
        monkeypatch.setattr("temporalio.client.Client.connect", connect_mock)

        c = TemporalConnector(
            TemporalConfig(
                enabled=True, host="ns.tmprl.cloud:7233", namespace="prod", api_key="sk", tls=True
            )
        )
        client = await c.connect()

        assert client is sentinel
        kwargs = connect_mock.await_args.kwargs
        assert kwargs["api_key"] == "sk"
        assert kwargs["tls"] is True
        assert kwargs["namespace"] == "prod"

    async def test_connect_wraps_errors(self, monkeypatch):
        monkeypatch.setattr(
            "temporalio.client.Client.connect", AsyncMock(side_effect=RuntimeError("boom"))
        )
        c = TemporalConnector(TemporalConfig(enabled=True, host="localhost:7233"))
        with pytest.raises(TemporalConnectionError):
            await c.connect()


class TestApingIsQuiet:
    async def test_aping_true_on_connect(self, monkeypatch):
        monkeypatch.setattr("temporalio.client.Client.connect", AsyncMock(return_value=object()))
        c = TemporalConnector(TemporalConfig(enabled=True, host="localhost:7233"))
        assert await c.aping() is True

    async def test_aping_false_when_unreachable(self, monkeypatch):
        monkeypatch.setattr(
            "temporalio.client.Client.connect", AsyncMock(side_effect=RuntimeError("down"))
        )
        c = TemporalConnector(TemporalConfig(enabled=True, host="localhost:7233"))
        assert await c.aping() is False  # never raises


class TestDescribeMasksSecrets:
    def test_api_key_masked(self):
        c = TemporalConnector(
            TemporalConfig(enabled=True, host="ns.tmprl.cloud:7233", api_key="supersecretkey")
        )
        d = c.describe()
        assert d["name"] == "temporal"
        assert d["host"] == "ns.tmprl.cloud:7233"
        assert "supersecretkey" not in str(d)


class TestTemporalClientDelegates:
    async def test_client_connect_uses_connector(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr("temporalio.client.Client.connect", AsyncMock(return_value=sentinel))

        from continuum.temporal.client import TemporalClient

        tc = TemporalClient(TemporalConfig(enabled=True, host="localhost:7233"))
        await tc.connect()
        assert tc.is_connected
        assert tc.raw_client is sentinel
