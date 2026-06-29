"""
Phase 6 contract: LangfuseConnector brings observability under the connector
module — infers mode (self-hosted docker vs cloud), probes via auth_check, and
masks the public/secret keys in describe().
"""

from __future__ import annotations

import sys
import types

from continuum.connectors.base import ConnectionMode
from continuum.connectors.langfuse import LangfuseConnector
from continuum.observability.config import ObservabilityConfig


def _cfg(**kw) -> ObservabilityConfig:
    base = {"enabled": True, "public_key": "pk", "secret_key": "sk"}
    base.update(kw)
    return ObservabilityConfig(**base)


class TestModeInference:
    def test_localhost_is_local_docker(self):
        c = LangfuseConnector(_cfg(host="http://localhost:3000"))
        assert c.mode is ConnectionMode.LOCAL_DOCKER

    def test_remote_host_is_cloud(self):
        c = LangfuseConnector(_cfg(host="https://cloud.langfuse.com"))
        assert c.mode is ConnectionMode.CLOUD

    def test_disabled_is_disabled(self):
        c = LangfuseConnector(_cfg(enabled=False, host="http://localhost:3000"))
        assert c.mode is ConnectionMode.DISABLED


class TestConfigured:
    def test_configured_with_both_keys(self):
        assert LangfuseConnector(_cfg()).is_configured() is True

    def test_unconfigured_without_keys(self):
        assert LangfuseConnector(_cfg(public_key=None, secret_key=None)).is_configured() is False


class TestDescribeMasksSecrets:
    def test_keys_masked(self):
        c = LangfuseConnector(
            _cfg(public_key="pk-publicsecret", secret_key="sk-supersecret", host="http://localhost:3000")
        )
        d = c.describe()
        assert d["name"] == "langfuse"
        assert d["host"] == "http://localhost:3000"
        assert "pk-publicsecret" not in str(d)
        assert "sk-supersecret" not in str(d)


def _install_fake_langfuse(monkeypatch, *, auth_result=True, raise_on_init=False):
    """Install a fake `langfuse` module so the connector can be probed offline."""
    mod = types.ModuleType("langfuse")

    class FakeLangfuse:
        def __init__(self, **kwargs):
            if raise_on_init:
                raise RuntimeError("cannot reach langfuse")

        def auth_check(self):
            return auth_result

        def shutdown(self):
            pass

    mod.Langfuse = FakeLangfuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    return mod


class TestApingIsQuiet:
    async def test_aping_true_on_auth_success(self, monkeypatch):
        _install_fake_langfuse(monkeypatch, auth_result=True)
        c = LangfuseConnector(_cfg(host="http://localhost:3000"))
        assert await c.aping() is True

    async def test_aping_false_on_auth_failure(self, monkeypatch):
        _install_fake_langfuse(monkeypatch, auth_result=False)
        c = LangfuseConnector(_cfg(host="http://localhost:3000"))
        assert await c.aping() is False

    async def test_aping_false_and_quiet_when_unreachable(self, monkeypatch):
        _install_fake_langfuse(monkeypatch, raise_on_init=True)
        c = LangfuseConnector(_cfg(host="http://localhost:3000"))
        assert await c.aping() is False  # never raises


class TestRegisteredAsDefault:
    def test_langfuse_in_default_connectors(self):
        from continuum.connectors.registry import (
            list_connectors,
            register_default_connectors,
            unregister_connector,
        )

        for n in list_connectors():
            unregister_connector(n)
        try:
            register_default_connectors()
            assert "langfuse" in list_connectors()
        finally:
            for n in list_connectors():
                unregister_connector(n)
