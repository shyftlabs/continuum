"""HeadroomCompressor — pre-call compression orchestration (Phase 1).

Owns the fail-open/fail-closed policy and per-run hash bookkeeping. One
instance per run/agent (the ``issued_hashes`` scope is the anti-forgery
boundary for the future Phase-2 retrieve tool).

Phase-1 scope is compress-only: no ``continuum_headroom_retrieve`` tool injection.
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


# Reserved internal tool the model calls to fetch originals of compressed
# content. Registered at agent startup (stable tool list => stable prompt
# cache) and INTERCEPTED in the tool loop — never dispatched to ToolService.
RETRIEVE_TOOL_NAME = "continuum_headroom_retrieve"

RETRIEVE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": RETRIEVE_TOOL_NAME,
        "description": (
            "Retrieve the original uncompressed content behind a compression "
            "marker like '[... compressed ... Retrieve more: hash=abc123]'. "
            "Call this only when the compressed view is insufficient to answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hash": {
                    "type": "string",
                    "description": "The hex hash from the compression marker",
                },
                "query": {
                    "type": "string",
                    "description": "Optional: what you are looking for in the original",
                },
            },
            "required": ["hash"],
        },
    },
}


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
        # Message snapshots for glassbox inspection (before/after compression).
        self._last_messages_before: list[dict[str, Any]] | None = None
        self._last_messages_after: list[dict[str, Any]] | None = None
        # Cumulative counters since the last reset_run_counters(). A single run
        # can compress multiple times (e.g. once per turn), so last_stats alone
        # under-reports a run's total; these sum every apply() call. Observability
        # only — never read on the request path.
        self._cum_tokens_before = 0
        self._cum_tokens_after = 0
        self._cum_calls = 0

    @property
    def issued_hashes(self) -> set[str]:
        """Hashes issued by the sidecar for this run (retrieve authorization)."""
        return self._issued_hashes

    @property
    def last_stats(self) -> CompressionStats | None:
        """Stats from the most recent successful compress call."""
        return self._last_stats

    @property
    def last_messages_before(self) -> list[dict[str, Any]] | None:
        """Pre-compression message list from the most recent apply() call."""
        return self._last_messages_before

    @property
    def last_messages_after(self) -> list[dict[str, Any]] | None:
        """Post-compression message list from the most recent apply() call."""
        return self._last_messages_after

    @property
    def run_totals(self) -> dict[str, int]:
        """Cumulative compression totals since the last reset_run_counters().
        Summed across every apply() call — the honest per-run figure."""
        return {
            "calls": self._cum_calls,
            "tokens_before": self._cum_tokens_before,
            "tokens_after": self._cum_tokens_after,
            "tokens_removed": self._cum_tokens_before - self._cum_tokens_after,
        }

    def reset_run_counters(self) -> None:
        """Zero the cumulative counters (call at the start of a run)."""
        self._cum_tokens_before = 0
        self._cum_tokens_after = 0
        self._cum_calls = 0

    async def apply(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
    ) -> list[dict[str, Any]]:
        """Compress ``messages`` via the sidecar.

        On sidecar error: fail-open returns the original list unchanged;
        fail-closed re-raises for the caller to surface.
        """
        self._last_messages_before = messages
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
        self._cum_tokens_before += stats.tokens_before
        self._cum_tokens_after += stats.tokens_after
        self._cum_calls += 1
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

        # Anti-doom-loop: NEVER re-compress a continuum_headroom_retrieve result. The
        # model retrieved the original precisely because the compressed view
        # was insufficient — compressing it again before the model sees it
        # would erase the retrieval (observed live: retrieve → recompress →
        # empty answer). Restore those tool messages' original content.
        compressed = self._restore_retrieve_results(messages, compressed)
        self._last_messages_after = compressed
        return compressed

    @staticmethod
    def _restore_retrieve_results(
        original: list[dict[str, Any]], compressed: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Restore original content of retrieve-result tool messages."""

        def _tc_fields(tc: Any) -> tuple[str, str]:
            if hasattr(tc, "function"):
                return getattr(tc, "id", ""), tc.function.name
            fn = tc.get("function") or {}
            return tc.get("id", ""), fn.get("name", "")

        retrieve_ids = {
            tc_id
            for m in original
            if m.get("role") == "assistant"
            for tc_id, name in (_tc_fields(tc) for tc in (m.get("tool_calls") or []))
            if name == RETRIEVE_TOOL_NAME
        }
        if not retrieve_ids or len(original) != len(compressed):
            return compressed
        for i, msg in enumerate(original):
            if msg.get("role") == "tool" and msg.get("tool_call_id") in retrieve_ids:
                if compressed[i].get("content") != msg.get("content"):
                    compressed[i] = {**compressed[i], "content": msg.get("content")}
                    logger.debug("headroom: protected retrieve result at index %d", i)
        return compressed

    async def resolve_retrieve(self, hash_value: str, query: str | None = None) -> str:
        """Resolve a model-issued `continuum_headroom_retrieve` call. Never raises.

        SECURITY (anti-forgery — do NOT remove): the LLM chooses this hash and
        can hallucinate or replay one from another context. Only serve hashes
        this compressor itself recorded as issued, else a fabricated hash could
        pull back another request's cached originals from the shared sidecar
        store. Failures are fail-open text so the agent loop continues.
        """
        if hash_value not in self._issued_hashes:
            logger.warning(f"headroom: rejected un-issued retrieve hash {hash_value!r}")
            return (
                f"[continuum_headroom_retrieve: hash {hash_value!r} was not issued in this "
                "context. If the data came from a tool, re-run that tool instead.]"
            )
        try:
            content = await self._client.retrieve(hash_value, query)
        except Exception as e:
            logger.warning(f"headroom: retrieve failed for {hash_value}: {e}")
            return (
                "[continuum_headroom_retrieve: retrieval failed (content may have expired). "
                "If the data came from a tool, re-run that tool instead.]"
            )
        logger.info("headroom: retrieved original for hash %s (%d chars)", hash_value, len(content))
        return content


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
