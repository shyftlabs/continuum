"""
Milvus-backed registry of tool summary embeddings for semantic routing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.tools.tool_attention.config import ToolAttentionConfig

logger = get_logger(__name__)


def _tool_summary(tool: Any) -> tuple[str, str]:
    """Return (tool_name, summary_text) for embedding."""
    if isinstance(tool, dict):
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        props = (fn.get("parameters") or {}).get("properties", {})
    else:
        fn = tool.function
        name = fn.name
        desc = fn.description or ""
        props = (fn.parameters or {}).get("properties", {})

    inputs = ", ".join(f"{k} ({v.get('type', 'any')})" for k, v in props.items())
    summary = f"{name}: {desc}. Inputs: {inputs}." if inputs else f"{name}: {desc}."
    return name, summary


class ToolSummaryRegistry:
    """
    Embeds tool summaries and stores them in Milvus for cosine similarity search.

    Call initialize() once at agent startup. The collection persists across
    restarts so embeddings are not recomputed unless tools change.
    """

    def __init__(self, config: ToolAttentionConfig) -> None:
        self._config = config
        self._client: Any = None
        self._encoder: Any = None
        self._ready = False

    async def initialize(self, tool_defs: list[Any]) -> None:
        """Connect to Milvus, create collection if needed, upsert embeddings."""
        try:
            await asyncio.to_thread(self._sync_init, tool_defs)
            self._ready = True
            logger.info(
                f"ToolSummaryRegistry ready: {len(tool_defs)} tools "
                f"in collection '{self._config.collection_name}'"
            )
        except Exception as e:
            logger.warning(f"ToolSummaryRegistry init failed (tool-attention disabled): {e}")

    def _sync_init(self, tool_defs: list[Any]) -> None:
        from pymilvus import DataType, MilvusClient
        from sentence_transformers import SentenceTransformer

        from orchestrator.config import settings

        uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        token = settings.milvus_token or ""
        self._client = MilvusClient(uri=uri, token=token)
        self._encoder = SentenceTransformer(self._config.embedding_model)

        col = self._config.collection_name
        dim = self._config.embedding_dim

        if not self._client.has_collection(col):
            self._client.create_collection(
                collection_name=col,
                dimension=dim,
                primary_field_name="tool_name",
                id_type=DataType.VARCHAR,
                metric_type="COSINE",
                max_length=256,
                auto_id=False,
            )
            logger.info(f"Created Milvus collection '{col}' (dim={dim})")

        if tool_defs:
            self._sync_upsert(tool_defs)

    def _sync_upsert(self, tool_defs: list[Any]) -> None:
        pairs = [_tool_summary(t) for t in tool_defs]
        names = [p[0] for p in pairs if p[0]]
        summaries = [p[1] for p, n in zip(pairs, [p[0] for p in pairs], strict=False) if n]

        embeddings = self._encoder.encode(summaries, normalize_embeddings=True).tolist()
        data = [
            {"tool_name": n, "vector": e, "summary": s}
            for n, e, s in zip(names, embeddings, summaries, strict=False)
        ]
        self._client.upsert(collection_name=self._config.collection_name, data=data)
        logger.debug(f"Upserted {len(data)} tool embeddings")

    def search(self, query: str, k: int) -> list[str]:
        """Return top-k tool names by cosine similarity. Synchronous."""
        if not self._ready or not self._client or not self._encoder:
            return []
        try:
            emb = self._encoder.encode([query], normalize_embeddings=True).tolist()
            results = self._client.search(
                collection_name=self._config.collection_name,
                data=emb,
                limit=k,
                output_fields=["tool_name"],
            )
            return [hit["entity"]["tool_name"] for hit in results[0]]
        except Exception as e:
            logger.warning(f"Milvus search error: {e}")
            return []

    def refresh(self, tool_defs: list[Any]) -> None:
        """Re-upsert tool embeddings after tool list changes."""
        if not self._ready or not tool_defs:
            return
        try:
            self._sync_upsert(tool_defs)
        except Exception as e:
            logger.warning(f"Registry refresh failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ready
