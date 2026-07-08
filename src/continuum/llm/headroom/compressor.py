"""HeadroomCompressor — pre-call compression orchestration (Phase 1).

Owns the fail-open/fail-closed policy and per-run hash bookkeeping. One
instance per run/agent (the ``issued_hashes`` scope is the anti-forgery
boundary for the future Phase-2 retrieve tool).

Phase-1 scope is compress-only: no ``continuum_retrieve`` tool injection.
Phase 2 (CCR retrieval) is evidence-gated — see
``gap-analysis/headroom-native-integration-plan.md``.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from continuum.llm.headroom.client import CompressionStats, HeadroomClient
from continuum.logging import get_logger

logger = get_logger(__name__)

# CCR retrieval markers embedded in compressed content, e.g.
#   "[2501 lines compressed to 7. Retrieve more: hash=7e443033ad1ff3f9ca0b8c49]"
# Needed because the /v1/compress `ccr_hashes` response field is UNRELIABLE —
# verified empty even when markers are inserted (log/search compressor path,
# sidecar v0.29.0). Hashes are the union of the field and this scan.
_MARKER_HASH_RE = re.compile(r"hash=([0-9a-f]{24})")


def _scan_marker_hashes(messages: list[dict[str, Any]]) -> set[str]:
    """Extract CCR marker hashes from message content (any content shape)."""
    found: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if content:
            found.update(_MARKER_HASH_RE.findall(str(content)))
    return found


class HeadroomCompressor:
    """Applies sidecar compression to an outgoing message list.

    The contract with the caller (the ``LLMClient.chat()`` seam) is
    rebind-not-mutate: ``apply`` returns a *new* list from the sidecar's JSON
    round-trip and never mutates the input, so the executor's own message list
    (and anything persisted from it) stays pristine.
    """

    def __init__(self, client: HeadroomClient, fail_open: bool = True):
        self._client = client
        self._fail_open = fail_open
        # Hashes the sidecar issued for THIS run only. Populated from the
        # response's `ccr_hashes` field (authoritative — never regex-scraped).
        # SECURITY (Phase 2): retrieve authorization must check this set.
        self._issued_hashes: set[str] = set()
        self._last_stats: CompressionStats | None = None

    @property
    def issued_hashes(self) -> set[str]:
        """Hashes issued by the sidecar for this run (retrieve authorization)."""
        return self._issued_hashes

    @property
    def last_stats(self) -> CompressionStats | None:
        """Stats from the most recent successful compress call."""
        return self._last_stats

    async def apply(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
    ) -> list[dict[str, Any]]:
        """Compress ``messages`` via the sidecar.

        On sidecar error: fail-open returns the original list unchanged;
        fail-closed re-raises for the caller to surface.
        """
        try:
            compressed, stats, ccr_hashes = await self._client.compress(messages, model)
        except Exception as e:
            if self._fail_open:
                logger.warning(
                    f"Headroom compress failed (fail-open, forwarding uncompressed): {e}"
                )
                return messages
            logger.error(f"Headroom compress failed (fail-closed): {e}")
            raise

        self._last_stats = stats
        if stats.tokens_saved > 0:
            logger.info(
                "headroom: %d→%d tokens (%.0f%% saved, transforms=%s)",
                stats.tokens_before,
                stats.tokens_after,
                (1 - stats.compression_ratio) * 100,
                stats.transforms_applied,
            )
        # Union of both hash sources (decision #6): the response field alone
        # is unreliable — it was observed empty while markers WERE inserted.
        issued = set(ccr_hashes) | _scan_marker_hashes(compressed)
        if issued:
            self._issued_hashes.update(issued)
            logger.info("headroom: CCR issued %d retrievable hash(es)", len(issued))
        return compressed


# Process-global compressor (Phase 1: hash bookkeeping is observability-only,
# so a shared instance is fine; Phase 2's retrieve authorization will move to
# a per-run instance on RunContext). Mirrors get_progressive_context_manager().
_global_compressor: HeadroomCompressor | None = None
_global_lock = threading.Lock()


def get_headroom_compressor() -> HeadroomCompressor:
    """Get the global HeadroomCompressor, built from settings on first use."""
    global _global_compressor

    if _global_compressor is None:
        with _global_lock:
            if _global_compressor is None:
                from continuum.config import settings

                client = HeadroomClient(
                    api_base=settings.headroom_api_base,
                    api_key=settings.headroom_api_key,
                    timeout=settings.headroom_timeout_seconds,
                )
                _global_compressor = HeadroomCompressor(
                    client=client, fail_open=settings.headroom_fail_open
                )
    return _global_compressor


def reset_headroom_compressor() -> None:
    """Reset the global compressor (tests / settings changes)."""
    global _global_compressor
    with _global_lock:
        _global_compressor = None
