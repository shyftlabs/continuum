"""
Vector store connector — single source of truth for the Milvus/Qdrant
*connection* (params, mode, auth).

This is a "library owns the client" connector: mem0 builds the live vector-store
client itself from a config dict (``Memory.from_config``), so this connector
cannot hand mem0 a live client. Instead it:

* produces the mem0 ``vector_store`` config block (``to_mem0_block``) that
  ``MemoryConfig`` consumes — making this the one place connection params live;
* exposes a standalone health-probe client via ``connect()`` / ``aping()`` for
  diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from continuum.config import resolve_milvus_uri
from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger
from continuum.security.secrets_guard import enforce_credential
from continuum.utils.secrets import mask_value

if TYPE_CHECKING:
    from continuum.memory.config import MemoryConfig

logger = get_logger(__name__)


class VectorStoreConnector(BaseConnector[Any]):
    """Connector for the long-term-memory vector store (Milvus or Qdrant)."""

    name = "vector_store"

    _LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "milvus", "qdrant"}

    def __init__(self, config: MemoryConfig | None = None) -> None:
        if config is None:
            from continuum.memory.config import MemoryConfig

            config = MemoryConfig()
        self._config = config

    @property
    def provider(self) -> str:
        """The active vector store provider: 'milvus' or 'qdrant'."""
        return self._config.vector_store_provider

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def is_configured(self) -> bool:
        if self.provider == "milvus":
            return bool(self._config.milvus_uri or self._config.milvus_host)
        return bool(self._config.qdrant_host)

    def milvus_uri(self) -> str:
        """The effective Milvus endpoint — MILVUS_URI when set, else host:port."""
        return resolve_milvus_uri(
            self._config.milvus_uri,
            self._config.milvus_host,
            self._config.milvus_port,
        )

    def _milvus_effective_host(self) -> str:
        """Hostname of the effective Milvus endpoint.

        Derived from the resolved URI, not ``milvus_host`` — otherwise a remote
        endpoint set via ``MILVUS_URI`` would be classified by the *default*
        ``milvus_host`` ("localhost"), read as LOCAL_DOCKER, and skip the
        fail-closed credential check in ``_enforce_credential`` (F8/D4).
        """
        return urlparse(self.milvus_uri()).hostname or self._config.milvus_host

    @property
    def mode(self) -> ConnectionMode:
        if not self._config.enabled:
            return ConnectionMode.DISABLED
        if self.provider == "milvus":
            host, cloud_cred = self._milvus_effective_host(), self._config.milvus_token
        else:
            host, cloud_cred = self._config.qdrant_host, self._config.qdrant_api_key
        if cloud_cred:
            return ConnectionMode.CLOUD
        if host in self._LOCAL_HOSTS:
            return ConnectionMode.LOCAL_DOCKER
        return ConnectionMode.CUSTOM

    def _enforce_credential(self) -> None:
        """Fail-closed on a missing/weak credential for a REMOTE store (F8/D4).

        A loopback/local vector store (``LOCAL_DOCKER``) legitimately runs
        without a token — the same posture as Redis's own protected mode — so it
        is exempt. Only a remote ``CUSTOM`` host or a ``CLOUD`` endpoint must
        authenticate. Override with ``CONTINUUM_ALLOW_INSECURE=1`` (local/testing).
        """
        if self.mode not in (ConnectionMode.CUSTOM, ConnectionMode.CLOUD):
            return
        if self.provider == "milvus":
            enforce_credential(
                service="Milvus vector store",
                credential=self._config.milvus_token,
                env_var="MILVUS_TOKEN",
            )
        else:
            enforce_credential(
                service="Qdrant vector store",
                credential=self._config.qdrant_api_key,
                env_var="QDRANT_API_KEY",
            )

    def to_mem0_block(self) -> dict[str, Any]:
        """Build the mem0 ``vector_store`` config block from connection settings.

        This is the single source of truth for the vector-store connection that
        ``MemoryConfig.to_mem0_config`` consumes.
        """
        self._enforce_credential()
        if self.provider == "milvus":
            milvus_config: dict[str, Any] = {
                "collection_name": self._config.milvus_collection,
                "embedding_model_dims": self._config.embedding_dims,
                "url": self.milvus_uri(),
            }
            # Always emit a *string* token, never omit the key. mem0's
            # MilvusDBConfig declares `token: str` but defaults it to None;
            # Pydantic tolerates that default on first construction, then mem0
            # re-creates the config from model_dump() for its telemetry store
            # (mem0/memory/main.py) — where None is validated against `str` and
            # raises, taking down the whole MemoryClient. An unauthenticated
            # local/self-hosted Milvus (the default config) therefore cannot come
            # up unless we send "". Matches connect() below, which already does
            # `token=... or ""`.
            milvus_config["token"] = self._config.milvus_token or ""
            return {"provider": "milvus", "config": milvus_config}

        qdrant_config: dict[str, Any] = {
            "host": self._config.qdrant_host,
            "port": self._config.qdrant_port,
            "collection_name": self._config.qdrant_collection,
            "embedding_model_dims": self._config.embedding_dims,
        }
        if self._config.qdrant_api_key:
            qdrant_config["api_key"] = self._config.qdrant_api_key
        return {"provider": "qdrant", "config": qdrant_config}

    async def connect(self) -> Any:
        """Return a standalone client for health probing.

        Note: this is NOT the client mem0 uses (mem0 builds its own from
        ``to_mem0_block``); it exists only for diagnostics.
        """
        if self.provider == "milvus":
            from pymilvus import MilvusClient

            return MilvusClient(
                uri=self.milvus_uri(),
                token=self._config.milvus_token or "",
            )
        from qdrant_client import QdrantClient

        return QdrantClient(
            host=self._config.qdrant_host,
            port=self._config.qdrant_port,
            api_key=self._config.qdrant_api_key,
            timeout=5,
        )

    async def aping(self) -> bool:
        """Probe the vector store — returns False on any failure, never raises."""
        client = None
        try:
            client = await self.connect()
            if self.provider == "milvus":
                client.list_collections()
            else:
                client.get_collections()
            return True
        except Exception as e:  # noqa: BLE001 — probe must never propagate
            logger.debug("Vector store connector aping failed: %s", e)
            return False
        finally:
            try:
                if client is not None and hasattr(client, "close"):
                    client.close()
            except Exception:  # noqa: BLE001
                pass

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["provider"] = self.provider
        if self.provider == "milvus":
            token = self._config.milvus_token
            d.update(
                uri=self.milvus_uri(),
                host=self._milvus_effective_host(),
                port=self._config.milvus_port,
                collection=self._config.milvus_collection,
                token=mask_value(token) if token else None,
            )
        else:
            api_key = self._config.qdrant_api_key
            d.update(
                host=self._config.qdrant_host,
                port=self._config.qdrant_port,
                collection=self._config.qdrant_collection,
                api_key=mask_value(api_key) if api_key else None,
            )
        return d
