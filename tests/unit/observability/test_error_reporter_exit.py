"""Regression: the error-reporter atexit flush must be bounded.

Langfuse's ``flush()`` ends in an unbounded ``queue.join()``. If the upload
queue can't drain (no network / no server — the normal case in unit tests),
an unbounded exit flush hangs the whole process after all tests pass.
_cleanup must give up after a bounded window instead.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from continuum.observability import error_reporter as er


class TestBoundedExitFlush:
    def test_cleanup_returns_when_flush_blocks_forever(self):
        def blocking_flush() -> None:
            time.sleep(60)  # simulate Langfuse queue.join() never draining

        start = time.monotonic()
        with patch.object(er, "flush_errors", side_effect=blocking_flush):
            er._cleanup()
        elapsed = time.monotonic() - start

        # Must give up at ~_CLEANUP_FLUSH_TIMEOUT_SECONDS, never block 60s.
        assert elapsed < er._CLEANUP_FLUSH_TIMEOUT_SECONDS + 2

    def test_cleanup_fast_when_flush_is_fast(self):
        with patch.object(er, "flush_errors", return_value=None):
            start = time.monotonic()
            er._cleanup()
            elapsed = time.monotonic() - start
        assert elapsed < 1

    def test_cleanup_swallows_flush_exceptions(self):
        with patch.object(er, "flush_errors", side_effect=RuntimeError("boom")):
            er._cleanup()  # must not raise
