"""HeadroomClient — thin async HTTP client for the Headroom compression sidecar.

The sidecar is a locally-run `headroom proxy` (loopback-only API, default
http://127.0.0.1:8787). Contract verified against v0.29.0:

  POST /v1/compress  {"messages": [...], "model": "..."} ->
      {"messages": [...compressed...], "tokens_before", "tokens_after",
       "tokens_saved", "compression_ratio", "transforms_applied",
       "ccr_hashes": [...]}
  GET  /v1/retrieve/{hash}?query=... -> {"original_content": "..."}

This module never applies fail-open/fail-closed policy — errors propagate so
the orchestrating compressor decides. Continuum never stores originals; they
live only in the sidecar's own store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from continuum.logging import get_logger

logger = get_logger(__name__)

# Per-message content preview length for DEBUG payload logs. Full blobs (a
# 44k-char tool result) flood the terminal, so each message's content is
# truncated to this many chars with its true length noted — enough to see the
# structure and the before/after effect without the wall of text.
_DEBUG_CONTENT_PREVIEW = 220


def _compact_messages(messages: list[dict[str, Any]]) -> str:
    """One line per message: role, content preview (+true length), tool info.

    Compressed blocks (small) print in full; big originals show a preview
    tagged with how much was elided — so the before/after contrast is obvious
    at a glance in the DEBUG log.
    """
    lines: list[str] = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        parts = [f"[{i}] {role}"]
        tcs = m.get("tool_calls")
        if tcs:
            names = ", ".join(
                (tc.get("function", {}) or {}).get("name", "?")
                for tc in tcs
                if isinstance(tc, dict)
            )
            parts.append(f"tool_calls=[{names}]")
        if m.get("tool_call_id"):
            parts.append(f"tcid={m['tool_call_id']}")
        content = m.get("content")
        if content is not None:
            s = str(content)
            if len(s) > _DEBUG_CONTENT_PREVIEW:
                preview = s[:_DEBUG_CONTENT_PREVIEW].replace("\n", "⏎")
                parts.append(
                    f'content({len(s)} chars): "{preview}…(+{len(s) - _DEBUG_CONTENT_PREVIEW} more)"'
                )
            else:
                parts.append(f'content({len(s)} chars): "{s}"')
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


@dataclass
class CompressionStats:
    """Token-savings metrics returned by /v1/compress."""

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    compression_ratio: float
    transforms_applied: list[str]


class HeadroomClient:
    """Async client for the Headroom sidecar's compress/retrieve API.

    One instance per process (shares the httpx connection pool). Raises on
    transport/HTTP errors — fail-open/closed policy belongs to the caller.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str | None = None,
        timeout: float = 5.0,
    ):
        self._base = api_base.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout)

    async def compress(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
    ) -> tuple[list[dict[str, Any]], CompressionStats, list[str]]:
        """POST /v1/compress. Returns (compressed_messages, stats, ccr_hashes).

        ``ccr_hashes`` is the authoritative list of hashes the sidecar issued on
        this call — use it for retrieve authorization (never regex-scrape the
        marker text). Empty for lossless compression (the common JSON case).
        """
        payload: dict[str, Any] = {"messages": messages}
        if model:
            payload["model"] = model

        import logging as _logging

        if logger.isEnabledFor(_logging.DEBUG):
            logger.debug(
                "headroom REQUEST (%d messages):\n%s",
                len(messages),
                _compact_messages(messages),
            )

        resp = await self._client.post(
            f"{self._base}/v1/compress", json=payload, headers=self._headers
        )
        resp.raise_for_status()
        body = resp.json()

        if logger.isEnabledFor(_logging.DEBUG):
            logger.debug(
                "headroom RESPONSE (%d messages, %d→%d tokens):\n%s",
                len(body.get("messages", [])),
                body.get("tokens_before", 0),
                body.get("tokens_after", 0),
                _compact_messages(body.get("messages", [])),
            )
        stats = CompressionStats(
            tokens_before=body.get("tokens_before", 0),
            tokens_after=body.get("tokens_after", 0),
            tokens_saved=body.get("tokens_saved", 0),
            compression_ratio=body.get("compression_ratio", 1.0),
            transforms_applied=body.get("transforms_applied", []),
        )
        return body.get("messages", messages), stats, body.get("ccr_hashes", [])

    async def retrieve(self, hash_value: str, query: str | None = None) -> str:
        """GET /v1/retrieve/{hash}. Returns the original uncompressed content."""
        params = {"query": query} if query else None
        resp = await self._client.get(
            f"{self._base}/v1/retrieve/{hash_value}",
            params=params,
            headers=self._headers,
        )
        resp.raise_for_status()
        body = resp.json()
        return str(body.get("original_content") or body.get("content", ""))

    async def health_check(self) -> bool:
        """True when the sidecar responds on /health. Never raises."""
        try:
            resp = await self._client.get(f"{self._base}/health")
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Headroom sidecar health check failed: {e}")
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
