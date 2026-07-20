"""LocalHeadroomClient — in-process Headroom backend (``headroom_mode=local``).

Duck-typed to :class:`~continuum.llm.headroom.client.HeadroomClient` (compress /
retrieve / health_check / aclose) so :class:`HeadroomCompressor` works unchanged:
fail-open, per-run anti-forgery, and marker-hash capture all live above this
seam. Instead of HTTP to the sidecar, this calls the pip-installed ``headroom``
library directly:

  compress  -> headroom.compress(messages, model=...)   (same pipeline as the
               sidecar's /v1/compress; the shared transforms write originals to
               the process-global CCR store)
  retrieve  -> headroom.cache.compression_store.get_compression_store().retrieve()

Requires the ``headroom-local`` extra (``pip install "shyftlabs-continuum[headroom-local]"``).
Contract notes (verified live against the published headroom-ai v0.29.0):

  - ``CompressResult.compression_ratio`` is the FRACTION SAVED (1.0 = all
    saved); the sidecar — and every Continuum call site — uses after/before
    (1.0 = nothing saved). We recompute from token counts, never pass it through.
  - ``CompressResult`` has no ccr_hashes field; we return ``[]`` and the
    compressor's marker scan of the returned messages (the authoritative
    source either way — see _scan_marker_hashes) supplies the issued hashes.
  - The library is synchronous and CPU-bound, so calls run in a worker thread
    bounded by ``timeout`` — a hung/cold path (e.g. first Kompress ONNX model
    load) raises TimeoutError into the compressor's fail-open, matching the
    sidecar mode's HTTP-timeout semantics. The orphaned thread finishes the
    model load, so subsequent calls are warm.

Multi-worker note: the CCR store is per-process by default; set
``HEADROOM_CCR_BACKEND=sqlite`` so retrieves resolve across workers.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from continuum.llm.headroom.client import CompressionStats
from continuum.logging import get_logger

logger = get_logger(__name__)

_INSTALL_HINT = (
    "headroom_mode='local' requires the Headroom library. Install it with: "
    'pip install "shyftlabs-continuum[headroom-local]"'
)


class LocalHeadroomClient:
    """In-process Headroom backend. One instance per process (the library's
    pipeline is a process-global singleton; per-run state lives in the
    compressor, exactly as with the HTTP client)."""

    def __init__(
        self,
        timeout: float = 30.0,
        kompress_prewarm: bool = False,
        kompress_execution_timeout_ms: int = 5000,
    ):
        self._timeout = timeout
        self._prose_enabled = kompress_prewarm
        # Set once the background pre-warm finishes (success OR failure), so the
        # first compress can wait for the model instead of racing it and
        # silently skipping prose. Pre-set when prose is off (never wait).
        self._kompress_ready = threading.Event()
        if not kompress_prewarm:
            self._kompress_ready.set()
        try:
            import headroom  # noqa: F401 — fail at construction, not first call
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e
        self._align_router_to_sidecar()
        if kompress_prewarm:
            self._enable_prose_kompress(kompress_execution_timeout_ms)

    def _align_router_to_sidecar(self) -> None:
        """Match the sidecar's router config so local mode compresses identically.

        The library's ``compress()`` builds the ContentRouter from simpler
        defaults than the proxy; these diverge on real payloads. There's no
        public ``compress()`` knob for either, so we set the router config
        directly, at construction, BEFORE any compress runs (the router caches
        results, and the SmartCrusher/relevance path are lazy — so the config
        must be right on the first call). Duck-typed and best-effort: a future
        headroom that renames/removes a field is a no-op, not a crash.

        Two alignments, both verified against headroom-ai 0.29.0:

        - ``relevance_split = False``. Default ON, it intercepts LOG/SEARCH
          content whenever Kompress is resident and Kompresses the
          "low-relevance tail". Its query is the LAST user message (e.g.
          "answer using the data above"), which matches nothing, so the whole
          log scores low-relevance and goes through the ML model — 14.5% vs the
          SearchCompressor's 99% — and the split never compares against the
          crusher, so the bad result wins. The sidecar doesn't hit this.

        - ``smart_crusher_max_items_after_crush = 50``. The library default is
          15; the proxy uses 50. On a 43-row table that's drop-28-rows (49.7%,
          lossy-recoverable) vs keep-all (19%). Pin 50 to match the sidecar's
          more conservative keep-inline policy.
        """
        try:
            from headroom.compress import _get_pipeline

            for transform in getattr(_get_pipeline(), "transforms", []):
                cfg = getattr(transform, "config", None)
                if cfg is None:
                    continue
                if hasattr(cfg, "relevance_split"):
                    cfg.relevance_split = False
                if hasattr(cfg, "smart_crusher_max_items_after_crush"):
                    cfg.smart_crusher_max_items_after_crush = 50
        except Exception as e:  # never fatal — worst case is the library defaults
            logger.warning("headroom: could not align router to sidecar defaults (%s)", e)

    def _enable_prose_kompress(self, execution_timeout_ms: int) -> None:
        """Make in-process prose (Kompress ML `text`) compression actually fire.

        Headroom skips the ML model on the hot path by default (background load
        + ~25ms per-call budget), so prose never compresses in-process without
        this. Two moves: (1) raise the execution budget so a call WAITS for a
        slot instead of skipping (setdefault — an explicit env wins); (2) load
        the model now, in a daemon thread, so it's ready by the first real
        request without blocking startup. The load sets ``_kompress_ready`` when
        done, so :meth:`compress` can block briefly on the FIRST prose call
        rather than racing the load (which is why an immediate first query after
        a cold start otherwise shows 0% — see the incident-desk cold-start
        window). Best-effort: a missing [headroom-local-ml] extra or a load
        failure still sets the event (so compress proceeds and prose just skips)
        and never crashes. Needs onnxruntime + transformers (the extra) + a
        ~1.4 GB model download on first use.
        """
        import os

        os.environ.setdefault("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", str(execution_timeout_ms))

        def _warm() -> None:
            try:
                from headroom.transforms.kompress_compressor import KompressCompressor

                backend = KompressCompressor().preload(allow_download=True)
                logger.info("headroom: Kompress prose model pre-warmed (backend=%s)", backend)
            except Exception as e:
                logger.warning(
                    "headroom: Kompress pre-warm failed (%s); prose compression will "
                    "skip until warm. Install the [headroom-local-ml] extra for prose.",
                    e,
                )
            finally:
                self._kompress_ready.set()

        threading.Thread(target=_warm, name="headroom-kompress-prewarm", daemon=True).start()

    async def compress(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
    ) -> tuple[list[dict[str, Any]], CompressionStats, list[str]]:
        """Run the full Headroom pipeline in-process.

        Returns (compressed_messages, stats, ccr_hashes) — the same tuple shape
        as the HTTP client. ``ccr_hashes`` is always empty here (see module
        docstring); the compressor derives hashes from markers.

        SAFETY: the library's ``compress()`` defaults ``compress_system_messages
        =True`` — it would compress system/developer prompts. The sidecar
        protects them (its default is False), and Continuum's whole "compression
        can't corrupt instructions" guarantee depends on that. So we pin the
        sidecar's protective defaults here; without this, local mode compresses
        system messages (verified: a protected RAG-context system message that
        the sidecar leaves at 0% got compressed ~62% in local mode).
        """
        from headroom import compress as headroom_compress

        # Match the sidecar's protect-system/protect-user defaults.
        _protect = {"compress_system_messages": False, "compress_user_messages": False}

        def _run() -> Any:
            # On the first prose call, wait for the background pre-warm so we
            # compress with a ready model instead of racing it (the model load
            # is ~3-5s; once ready the event is set forever, so later calls
            # don't wait). Bounded by the outer wait_for → fail-open on timeout.
            if self._prose_enabled and not self._kompress_ready.is_set():
                self._kompress_ready.wait(self._timeout)
            if model:
                return headroom_compress(messages, model=model, **_protect)
            return headroom_compress(messages, **_protect)

        result = await asyncio.wait_for(asyncio.to_thread(_run), self._timeout)

        before = result.tokens_before
        after = result.tokens_after
        stats = CompressionStats(
            tokens_before=before,
            tokens_after=after,
            tokens_saved=result.tokens_saved,
            # after/before, NOT the library's saved-fraction (inverted meaning).
            compression_ratio=(after / before) if before else 1.0,
            transforms_applied=list(result.transforms_applied),
        )
        return result.messages, stats, []

    async def retrieve(self, hash_value: str, query: str | None = None) -> str:
        """Fetch the original content behind a CCR marker from the in-process
        store. Raises KeyError on miss/expiry — the compressor's
        resolve_retrieve maps any exception to its fail-open 'expired' text,
        matching the sidecar's 404 path."""
        from headroom.cache.compression_store import get_compression_store

        def _run() -> Any:
            return get_compression_store().retrieve(hash_value, query)

        entry = await asyncio.wait_for(asyncio.to_thread(_run), self._timeout)
        if entry is None:
            raise KeyError(f"no CCR entry for hash {hash_value!r} (missing or expired)")
        return str(entry.original_content)

    def ccr_backend_info(self) -> str:
        """Ground-truth description of the CCR store's ACTUAL backend — not the
        HEADROOM_CCR_BACKEND env var, which is only a request. Building the store
        here also surfaces a SQLite-init failure (→ silent in-memory fallback) at
        startup instead of mid-run. For SQLiteBackend, includes the db path so
        operators can confirm where originals persist / which file workers share.
        """
        try:
            from headroom.cache.compression_store import get_compression_store

            backend = get_compression_store()._backend
            name = type(backend).__name__
            if name == "SQLiteBackend":
                from headroom.cache.backends.sqlite import default_db_path

                return f"{name} ({default_db_path()})"
            return name
        except Exception as e:  # pragma: no cover — diagnostic only, never fatal
            return f"unknown ({e})"

    async def health_check(self) -> bool:
        """True when the library imports. Never raises."""
        try:
            import headroom  # noqa: F401

            return True
        except Exception as e:
            logger.debug(f"Headroom library health check failed: {e}")
            return False

    async def aclose(self) -> None:
        """No connection pool to close — present for interface parity."""
