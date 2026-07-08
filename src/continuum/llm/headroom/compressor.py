"""HeadroomCompressor — pre-call compression orchestration (Phase 1).

Owns the fail-open/fail-closed policy and per-run hash bookkeeping. One
instance per run/agent (the ``issued_hashes`` scope is the anti-forgery
boundary for the future Phase-2 retrieve tool).

Phase-1 scope is compress-only: no ``continuum_retrieve`` tool injection.
Phase 2 (CCR retrieval) is evidence-gated — see
``gap-analysis/headroom-native-integration-plan.md``.
"""

from __future__ import annotations

import threading
from typing import Any

from continuum.llm.headroom.client import CompressionStats, HeadroomClient
from continuum.logging import get_logger

logger = get_logger(__name__)


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
        if ccr_hashes:
            self._issued_hashes.update(ccr_hashes)
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
