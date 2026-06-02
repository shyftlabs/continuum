"""
Background task registry — fire-and-forget tasks with ownership + shutdown draining.

asyncio has two well-known footguns for fire-and-forget work:

1. A task created with ``asyncio.create_task`` may be garbage-collected mid-run if
   nothing holds a strong reference to it (see the asyncio docs warning). The task
   then silently disappears.
2. If the event loop is torn down (process/worker shutdown) while a task is still
   running, the task is cancelled and its work is lost.

This registry addresses both: it holds a strong reference to every spawned task
until it completes, logs any exception instead of letting it vanish, and exposes
``drain()`` so a shutdown path can wait for in-flight work to finish (with a
timeout, so a stuck task can't hang shutdown forever).

It is intentionally tiny and dependency-free so it can be owned by the DI container
and shared across services (e.g. the SessionClient backgrounds long-term memory
writes onto it).
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from orchestrator.logging import get_logger

logger = get_logger(__name__)


class BackgroundTaskRegistry:
    """Owns fire-and-forget asyncio tasks and drains them on shutdown.

    Example:
        ```python
        registry = BackgroundTaskRegistry()
        registry.spawn(some_coroutine())   # returns immediately
        ...
        await registry.drain(timeout=5.0)  # at shutdown
        ```
    """

    def __init__(self, name: str = "background") -> None:
        self._name = name
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        label: str | None = None,
    ) -> asyncio.Task[Any] | None:
        """Schedule a coroutine as an owned background task.

        Holds a strong reference until the task completes (preventing GC), and
        logs any exception the task raises so failures are never silent.

        Args:
            coro: The coroutine to run in the background.
            label: Optional human-readable label for logging.

        Returns:
            The created task, or None if no running event loop is available
            (in which case the coroutine is closed to avoid a "never awaited"
            warning and the caller should fall back to awaiting it inline).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — cannot background. Close the coroutine so it
            # doesn't emit a "coroutine was never awaited" warning. The caller
            # is responsible for falling back to inline execution.
            coro.close()
            logger.debug(
                "BackgroundTaskRegistry[%s]: no running loop, cannot spawn %s",
                self._name,
                label or "task",
            )
            return None

        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        if label:
            # Best-effort: name the task for easier debugging in tracebacks.
            try:
                task.set_name(label)
            except Exception:  # pragma: no cover - defensive
                pass
        return task

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        """Drop the task reference and surface any exception."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "BackgroundTaskRegistry[%s]: task %s failed: %s",
                self._name,
                task.get_name(),
                exc,
                exc_info=exc,
            )

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for all in-flight tasks to complete.

        Args:
            timeout: Max seconds to wait. None waits indefinitely. On timeout,
                outstanding tasks are left running and a warning is logged
                (they are NOT cancelled here — shutdown of the loop will handle
                that — but the caller should treat unfinished work as at-risk).
        """
        if not self._tasks:
            return
        pending = list(self._tasks)
        logger.debug("BackgroundTaskRegistry[%s]: draining %d task(s)", self._name, len(pending))
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        if still_pending:
            logger.warning(
                "BackgroundTaskRegistry[%s]: %d task(s) did not finish within %ss",
                self._name,
                len(still_pending),
                timeout,
            )

    @property
    def pending_count(self) -> int:
        """Number of tasks currently in flight."""
        return len(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)
