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

from continuum.connectors.base import BaseConnector, ConnectionMode
from continuum.logging import get_logger
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
            return bool(self._config.milvus_host)
        return bool(self._config.qdrant_host)

    @property
    def mode(self) -> ConnectionMode:
        if not self._config.enabled:
            return ConnectionMode.DISABLED
        if self.provider == "milvus":
            host, cloud_cred = self._config.milvus_host, self._config.milvus_token
        else:
            host, cloud_cred = self._config.qdrant_host, self._config.qdrant_api_key
        if cloud_cred:
            return ConnectionMode.CLOUD
        if host in self._LOCAL_HOSTS:
            return ConnectionMode.LOCAL_DOCKER
        return ConnectionMode.CUSTOM

    def to_mem0_block(self) -> dict[str, Any]:
        """Build the mem0 ``vector_store`` config block from connection settings.

        This is the single source of truth for the vector-store connection that
        ``MemoryConfig.to_mem0_config`` consumes.
        """
        if self.provider == "milvus":
            milvus_config: dict[str, Any] = {
                "collection_name": self._config.milvus_collection,
                "embedding_model_dims": self._config.embedding_dims,
                "url": f"http://{self._config.milvus_host}:{self._config.milvus_port}",
            }
            if self._config.milvus_token:
                milvus_config["token"] = self._config.milvus_token
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
                uri=f"http://{self._config.milvus_host}:{self._config.milvus_port}",
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
                host=self._config.milvus_host,
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
