"""
Phase 3 contract: VectorStoreConnector is the single source of truth for the
Milvus/Qdrant *connection* (params, mode, auth) and produces the mem0
vector_store config block. mem0 still builds the live client, so the connector
exposes a standalone health-probe via connect()/aping().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.connectors.base import ConnectionMode
from continuum.connectors.vector_store import VectorStoreConnector
from continuum.exceptions import InsecureConfigurationError
from continuum.memory.config import MemoryConfig


def _cfg(**kw) -> MemoryConfig:
    base = {"enabled": True}
    base.update(kw)
    return MemoryConfig(**base)


class TestFailClosedCredentialRemoteOnly:
    """to_mem0_block() refuses a REMOTE vector store with a missing/weak token
    (F8/D4, Option B). A local/loopback store is exempt — the normal tokenless
    posture — matching Redis protected mode. Escape hatch: CONTINUUM_ALLOW_INSECURE."""

    def test_remote_qdrant_without_key_refuses(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = VectorStoreConnector(
            _cfg(vector_store_provider="qdrant", qdrant_host="q.internal.example")
        )
        assert c.mode is ConnectionMode.CUSTOM
        with pytest.raises(InsecureConfigurationError, match="QDRANT_API_KEY"):
            c.to_mem0_block()

    def test_remote_milvus_without_token_refuses(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = VectorStoreConnector(
            _cfg(
                vector_store_provider="milvus", milvus_host="m.internal.example", milvus_token=None
            )
        )
        assert c.mode is ConnectionMode.CUSTOM
        with pytest.raises(InsecureConfigurationError, match="MILVUS_TOKEN"):
            c.to_mem0_block()

    def test_local_qdrant_without_key_is_allowed(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = VectorStoreConnector(_cfg(vector_store_provider="qdrant", qdrant_host="localhost"))
        assert c.mode is ConnectionMode.LOCAL_DOCKER
        # Exempt — must not raise.
        assert c.to_mem0_block()["provider"] == "qdrant"

    def test_remote_qdrant_with_key_is_allowed(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = VectorStoreConnector(
            _cfg(
                vector_store_provider="qdrant",
                qdrant_host="q.cloud",
                qdrant_api_key="strong-key-xyz",
            )
        )
        assert c.to_mem0_block()["config"]["api_key"] == "strong-key-xyz"

    def test_escape_hatch_allows_remote_without_token(self, monkeypatch):
        monkeypatch.setenv("CONTINUUM_ALLOW_INSECURE", "1")
        c = VectorStoreConnector(
            _cfg(vector_store_provider="qdrant", qdrant_host="q.internal.example")
        )
        assert c.to_mem0_block()["provider"] == "qdrant"


class TestModeInference:
    def test_milvus_local_is_local_docker(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="localhost", milvus_token=None)
        )
        assert c.mode is ConnectionMode.LOCAL_DOCKER

    def test_milvus_token_is_cloud(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="zilliz.example", milvus_token="tok")
        )
        assert c.mode is ConnectionMode.CLOUD

    def test_qdrant_api_key_is_cloud(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="qdrant", qdrant_host="q.cloud", qdrant_api_key="k")
        )
        assert c.mode is ConnectionMode.CLOUD

    def test_qdrant_local_is_local_docker(self):
        c = VectorStoreConnector(_cfg(vector_store_provider="qdrant", qdrant_host="localhost"))
        assert c.mode is ConnectionMode.LOCAL_DOCKER

    def test_qdrant_remote_without_key_is_custom(self):
        c = VectorStoreConnector(_cfg(vector_store_provider="qdrant", qdrant_host="10.0.0.9"))
        assert c.mode is ConnectionMode.CUSTOM

    def test_disabled_is_disabled(self):
        c = VectorStoreConnector(MemoryConfig(enabled=False, vector_store_provider="qdrant"))
        assert c.mode is ConnectionMode.DISABLED


class TestMem0Block:
    def test_qdrant_block(self):
        c = VectorStoreConnector(
            _cfg(
                vector_store_provider="qdrant",
                qdrant_host="localhost",
                qdrant_port=6333,
                qdrant_collection="mems",
                embedding_dims=1536,
            )
        )
        block = c.to_mem0_block()
        assert block["provider"] == "qdrant"
        assert block["config"]["host"] == "localhost"
        assert block["config"]["port"] == 6333
        assert block["config"]["collection_name"] == "mems"
        assert block["config"]["embedding_model_dims"] == 1536
        assert "api_key" not in block["config"]  # omitted when unset

    def test_qdrant_block_includes_api_key_when_set(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="qdrant", qdrant_host="q.cloud", qdrant_api_key="secretkey")
        )
        assert c.to_mem0_block()["config"]["api_key"] == "secretkey"

    def test_milvus_block(self):
        c = VectorStoreConnector(
            _cfg(
                vector_store_provider="milvus",
                milvus_host="localhost",
                milvus_port=19530,
                milvus_collection="mems",
                embedding_dims=768,
                milvus_token=None,
            )
        )
        block = c.to_mem0_block()
        assert block["provider"] == "milvus"
        assert block["config"]["url"] == "http://localhost:19530"
        assert block["config"]["embedding_model_dims"] == 768
        # An unset token is sent as "" and never omitted — see
        # TestMilvusTokenSurvivesMem0Revalidation for why.
        assert block["config"]["token"] == ""

    def test_milvus_block_includes_token_when_set(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="zilliz", milvus_token="zztok")
        )
        assert c.to_mem0_block()["config"]["token"] == "zztok"

    def test_milvus_empty_token_is_not_dropped(self):
        """An explicitly empty MILVUS_TOKEN must behave like an unset one, not
        fall back to omitting the key (the old truthiness check dropped both)."""
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="localhost", milvus_token="")
        )
        assert c.to_mem0_block()["config"]["token"] == ""


class TestMilvusTokenSurvivesMem0Revalidation:
    """Regression: a tokenless Milvus (the default config) must not kill mem0.

    mem0's MilvusDBConfig declares ``token: str`` but defaults it to None.
    Pydantic skips validation of defaults, so the first construction succeeds —
    then mem0 rebuilds the config from ``model_dump()`` for its telemetry store
    (mem0/memory/main.py), where None is validated against ``str`` and raises,
    taking down the whole MemoryClient. Sending "" keeps that round-trip valid.
    """

    def test_tokenless_block_round_trips_through_milvus_db_config(self):
        milvus_cfg_mod = pytest.importorskip("mem0.configs.vector_stores.milvus")
        config_cls = milvus_cfg_mod.MilvusDBConfig

        block = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="localhost", milvus_token=None)
        ).to_mem0_block()

        validated = config_cls(**block["config"])
        # This is the exact mem0 telemetry-init path that used to raise.
        config_cls(**validated.model_dump())


class TestMilvusUri:
    """MILVUS_URI is the only way to express TLS / a portless endpoint (Zilliz
    Cloud). It must drive every place a Milvus address is built, not just the
    mem0 config block."""

    ZILLIZ = "https://in03-abc.serverless.gcp-us-west1.cloud.zilliz.com"

    def test_host_port_used_when_uri_unset(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="localhost", milvus_port=19530)
        )
        assert c.milvus_uri() == "http://localhost:19530"

    def test_uri_overrides_host_and_port(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_uri=self.ZILLIZ, milvus_token="zztok")
        )
        assert c.milvus_uri() == self.ZILLIZ
        assert c.to_mem0_block()["config"]["url"] == self.ZILLIZ

    def test_uri_drives_health_probe_client(self, monkeypatch):
        """connect() must use the same URI — otherwise aping() reports a healthy
        Zilliz endpoint as down while memory works fine."""
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_uri=self.ZILLIZ, milvus_token="zztok")
        )
        seen: dict[str, object] = {}

        class FakeMilvusClient:
            def __init__(self, uri, token):
                seen["uri"] = uri

        fake_pymilvus = MagicMock()
        fake_pymilvus.MilvusClient = FakeMilvusClient
        monkeypatch.setitem(__import__("sys").modules, "pymilvus", fake_pymilvus)

        import asyncio

        asyncio.run(c.connect())
        assert seen["uri"] == self.ZILLIZ

    def test_scheme_in_host_is_honored_not_mangled(self):
        """MILVUS_HOST=https://... previously produced http://https://host:19530."""
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host=self.ZILLIZ, milvus_token="zztok")
        )
        assert c.milvus_uri() == self.ZILLIZ

    def test_remote_uri_without_token_still_refuses(self, monkeypatch):
        """Regression guard: mode must be derived from the URI's host, not the
        (default 'localhost') milvus_host, or the F8/D4 fail-closed credential
        check silently stops running for a remote endpoint."""
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_uri=self.ZILLIZ, milvus_token=None)
        )
        assert c.mode is ConnectionMode.CUSTOM
        with pytest.raises(InsecureConfigurationError, match="MILVUS_TOKEN"):
            c.to_mem0_block()

    def test_local_uri_stays_exempt(self, monkeypatch):
        monkeypatch.delenv("CONTINUUM_ALLOW_INSECURE", raising=False)
        c = VectorStoreConnector(
            _cfg(
                vector_store_provider="milvus",
                milvus_uri="http://localhost:19530",
                milvus_token=None,
            )
        )
        assert c.mode is ConnectionMode.LOCAL_DOCKER
        assert c.to_mem0_block()["config"]["url"] == "http://localhost:19530"

    def test_describe_reports_effective_uri(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_uri=self.ZILLIZ, milvus_token="zztoksecret")
        )
        d = c.describe()
        assert d["uri"] == self.ZILLIZ
        assert "zztoksecret" not in str(d)


class TestDescribeMasksSecrets:
    def test_qdrant_api_key_masked(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="qdrant", qdrant_host="q.cloud", qdrant_api_key="secretkey")
        )
        assert "secretkey" not in str(c.describe())

    def test_milvus_token_masked(self):
        c = VectorStoreConnector(
            _cfg(vector_store_provider="milvus", milvus_host="z", milvus_token="zztoksecret")
        )
        assert "zztoksecret" not in str(c.describe())


class TestApingIsQuiet:
    async def test_qdrant_aping_true(self, monkeypatch):
        c = VectorStoreConnector(_cfg(vector_store_provider="qdrant", qdrant_host="localhost"))
        client = MagicMock()
        client.get_collections = MagicMock(return_value=[])
        monkeypatch.setattr(c, "connect", AsyncMock(return_value=client))
        assert await c.aping() is True

    async def test_aping_false_when_unreachable(self, monkeypatch):
        c = VectorStoreConnector(_cfg(vector_store_provider="qdrant", qdrant_host="localhost"))
        monkeypatch.setattr(c, "connect", AsyncMock(side_effect=ConnectionError("down")))
        assert await c.aping() is False


class TestMemoryConfigUsesConnector:
    def test_to_mem0_config_vector_block_comes_from_connector(self):
        cfg = _cfg(vector_store_provider="qdrant", qdrant_host="localhost", qdrant_api_key="k")
        expected = VectorStoreConnector(cfg).to_mem0_block()
        assert cfg.to_mem0_config()["vector_store"] == expected
