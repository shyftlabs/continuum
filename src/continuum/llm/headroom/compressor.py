"""HeadroomCompressor — pre-call compression orchestration.

Owns the fail-open/fail-closed policy and per-run hash bookkeeping. One
instance is published per agent run into a contextvar (see
:func:`use_run_compressor_if_enabled`), so ``issued_hashes`` — the anti-forgery
boundary for the ``continuum_headroom_retrieve`` tool — is isolated per run:
one agent cannot authorize a retrieve of another agent's cached originals from
the shared sidecar store. The expensive resource (the httpx pool inside
``HeadroomClient``) stays process-global and is shared by every run.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
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
        # Hashes the sidecar issued for THIS run only — the retrieve
        # authorization set. The compressor is published per run in a contextvar
        # (use_run_compressor_if_enabled), so this set is NOT shared across
        # concurrent agents: resolve_retrieve() checks it, giving a genuine
        # per-run anti-forgery boundary. Populated from both the response
        # `ccr_hashes` field and a marker scan (the field alone is unreliable).
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
        self._log_role_effect(messages, compressed, model)
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
        if not retrieve_ids:
            return compressed
        # Map each retrieve-result's full content by tool_call_id, then overwrite
        # the matching compressed message wherever it sits. Keyed by the stable
        # id — NOT by list index — so a sidecar reorder or a count change can't
        # write the original into the wrong message (or silently skip the
        # restore, reviving the doom loop).
        originals_by_id = {
            m["tool_call_id"]: m.get("content")
            for m in original
            if m.get("role") == "tool" and m.get("tool_call_id") in retrieve_ids
        }
        for cmsg in compressed:
            tc_id = cmsg.get("tool_call_id")
            if tc_id in originals_by_id and cmsg.get("content") != originals_by_id[tc_id]:
                cmsg["content"] = originals_by_id[tc_id]
                logger.debug("headroom: protected retrieve result tcid=%s", tc_id)
        return compressed

    @staticmethod
    def _log_role_effect(
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        model: str | None,
    ) -> None:
        """INFO log of Headroom's effect on the payload, bucketed by role, in
        TOKENS (tiktoken — the same counter Continuum uses for context
        management). This is Continuum's own estimate, so its TOTAL will not
        match the sidecar's authoritative token stat exactly; it complements
        that role-blind number by showing WHICH role shrank.

        Tokenizing is skipped entirely when INFO is off or when nothing changed
        (detected cheaply by char length first), so the common no-op turn costs
        only a couple of len() calls. Falls back to chars if tiktoken is
        unavailable. Observability only — wrapped so it can never disturb the
        request path.
        """
        if not logger.isEnabledFor(logging.INFO):
            return
        try:

            def _chars(msgs: list[dict[str, Any]]) -> int:
                return sum(
                    len(str(m["content"])) for m in msgs if m.get("content") is not None
                )

            # Cheap change detection: if no char changed, no token changed either.
            if _chars(before) == _chars(after):
                logger.info(
                    "headroom: no content change (%d msgs, %d chars unchanged)",
                    len(after),
                    _chars(after),
                )
                return

            # Something compressed — measure per role in tokens.
            enc = None
            try:
                import tiktoken

                try:
                    enc = tiktoken.encoding_for_model((model or "").split("/")[-1])
                except KeyError:
                    enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None  # tiktoken unavailable — degrade to chars

            def _measure(text: str) -> int:
                return len(enc.encode(text)) if enc is not None else len(text)

            unit = "tokens, tiktoken est." if enc is not None else "chars (tiktoken unavailable)"

            def _totals(msgs: list[dict[str, Any]]) -> dict[str, list[int]]:
                out: dict[str, list[int]] = {}
                for m in msgs:
                    role = m.get("role", "?")
                    content = m.get("content")
                    entry = out.setdefault(role, [0, 0])
                    entry[0] += _measure(str(content)) if content is not None else 0
                    entry[1] += 1
                return out

            b, a = _totals(before), _totals(after)
            tot_b = sum(v[0] for v in b.values())
            tot_a = sum(v[0] for v in a.values())
            lines = [f"headroom effect ({unit} by role):"]
            for role in sorted(set(b) | set(a)):
                cb = b.get(role, [0, 0])[0]
                ca, na = a.get(role, [0, 0])
                pct = f"{(ca - cb) / cb * 100:+.1f}%" if cb else "—"
                lines.append(f"  {role:<10}: {cb:>9,} → {ca:>9,}  ({pct:>7})  [{na} msg]")
            pct_t = f"{(tot_a - tot_b) / tot_b * 100:+.1f}%" if tot_b else "—"
            lines.append(f"  {'TOTAL':<10}: {tot_b:>9,} → {tot_a:>9,}  ({pct_t:>7})")
            logger.info("\n".join(lines))
        except Exception as e:  # observability must never break the call path
            logger.debug("headroom: role-effect logging failed: %s", e)

    async def resolve_retrieve(self, hash_value: str, query: str | None = None) -> str:
        """Resolve a model-issued `continuum_headroom_retrieve` call. Never raises.

        SECURITY (anti-forgery — do NOT remove): the LLM chooses this hash and
        can hallucinate or replay one from another context. Only serve hashes
        this compressor itself recorded as issued, else a fabricated hash could
        pull back another request's cached originals from the shared sidecar
        store. Because the compressor is published per run (contextvar-scoped),
        ``_issued_hashes`` holds only THIS run's hashes — so this check is a
        real per-run boundary, not a process-wide one. Failures are fail-open
        text so the agent loop continues.
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


# ---------------------------------------------------------------------------
# Compressor scoping
#
# The httpx connection pool (inside HeadroomClient) is the expensive, shareable
# resource — one per process. The per-run *state* (issued_hashes, the retrieve
# anti-forgery boundary) must NOT be shared, so each agent run gets its own
# HeadroomCompressor wrapping the shared client, published into a contextvar.
#
# contextvars are per-async-task: parallel agents spawned via create_task/gather
# each inherit their own value and cannot see a sibling's issued hashes, so one
# agent can no longer authorize a retrieve of another's cached originals.
# Mirrors the ambient pattern in continuum.security.policy_context. Outside a
# run (bare LLMClient.chat), calls fall back to a process-global compressor.
# ---------------------------------------------------------------------------

_global_client: HeadroomClient | None = None
_global_compressor: HeadroomCompressor | None = None
_global_lock = threading.Lock()

_run_compressor: ContextVar[HeadroomCompressor | None] = ContextVar(
    "headroom_run_compressor", default=None
)


def get_headroom_client() -> HeadroomClient:
    """Process-global HeadroomClient (shared httpx connection pool)."""
    global _global_client
    if _global_client is None:
        with _global_lock:
            if _global_client is None:
                from continuum.config import settings

                _global_client = HeadroomClient(
                    api_base=settings.headroom_api_base,
                    api_key=settings.headroom_api_key,
                    timeout=settings.headroom_timeout_seconds,
                )
    return _global_client


def new_run_compressor() -> HeadroomCompressor:
    """A fresh compressor for one agent run, wrapping the shared httpx client.
    Its ``issued_hashes`` set is private to the run — the retrieve boundary."""
    from continuum.config import settings

    return HeadroomCompressor(
        client=get_headroom_client(), fail_open=settings.headroom_fail_open
    )


def get_headroom_compressor() -> HeadroomCompressor:
    """The active compressor: the per-run instance published for this async task
    if any, else a process-global fallback (the most-recently-finished run's
    compressor, or a fresh one before any run).

    Call sites (compress in llm/client.py, retrieve in runner/executor) don't
    need to know which they got — a run publishes one via
    :func:`use_run_compressor_if_enabled` / :func:`enter_run_compressor`, and
    every call within that task then shares its issued-hash set. Post-run
    inspectors read the fallback, which reflects the run that just finished.
    """
    run = _run_compressor.get()
    if run is not None:
        return run
    global _global_compressor
    if _global_compressor is not None:
        return _global_compressor
    # Build the fallback OUTSIDE _global_lock: new_run_compressor() acquires it
    # via get_headroom_client(), and threading.Lock is non-reentrant, so nesting
    # would self-deadlock. Double-check on assignment; a lost startup race just
    # discards an unused, never-connected client.
    candidate = new_run_compressor()
    with _global_lock:
        if _global_compressor is None:
            _global_compressor = candidate
    return _global_compressor


def enter_run_compressor() -> Token[HeadroomCompressor | None] | None:
    """Publish a fresh per-run compressor if none is active in this task.

    Returns a token for :func:`exit_run_compressor`, or None when Headroom is
    disabled or a compressor is already active (a nested handoff/executor —
    which must KEEP the run's issued-hash set, so it neither rebinds nor resets).

    Never raises: a binding failure (e.g. a misconfigured client) returns None
    so the run proceeds on the global/fail-open compressor. Critically, this
    keeps a Headroom hiccup from skipping the caller's policy-context teardown
    (the set/reset sites bind this right before their try/finally), which would
    otherwise leak the data-label enforcement context across runs.
    """
    try:
        from continuum.config import settings

        if not settings.headroom_enabled or _run_compressor.get() is not None:
            return None
        return _run_compressor.set(new_run_compressor())
    except Exception as e:
        logger.warning("headroom: per-run compressor bind failed (%s); using fallback", e)
        return None


def exit_run_compressor(token: Token[HeadroomCompressor | None] | None) -> None:
    """Restore the previous compressor (no-op when :func:`enter_run_compressor`
    returned None). Tolerates the cross-context reset that happens when a
    streaming generator is finalized in a different context than it started in
    (mirrors reset_active_policy's ValueError guard)."""
    if token is None:
        return
    # Observability: keep the just-finished run's compressor as the global
    # fallback so post-run inspectors (glassbox stats, issued-hash deltas) that
    # read get_headroom_compressor() AFTER the run see the run that ran. Safe for
    # isolation: a new run always binds a FRESH compressor via
    # enter_run_compressor, so these hashes can never authorize a later retrieve.
    global _global_compressor
    finished = _run_compressor.get()
    if finished is not None:
        _global_compressor = finished
    try:
        _run_compressor.reset(token)
    except ValueError:
        # Token created in a different context (abandoned stream, GC finalizer).
        # The per-task context copy means there's nothing to leak; best-effort.
        pass


@contextmanager
def use_run_compressor_if_enabled() -> Iterator[None]:
    """Bind a per-run compressor for the block (nesting-safe). Use on the same
    ``with`` line as ``use_active_policy`` at run/handoff entry points."""
    token = enter_run_compressor()
    try:
        yield
    finally:
        exit_run_compressor(token)


def reset_headroom_compressor() -> None:
    """Reset process-global state (tests / settings changes). Does not touch a
    per-run compressor published in the current task."""
    global _global_compressor, _global_client
    with _global_lock:
        _global_compressor = None
        _global_client = None
