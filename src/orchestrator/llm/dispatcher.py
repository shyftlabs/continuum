"""
Priority dispatchers for LLM call scheduling.

Two implementations:

PriorityDispatcher — for external LLM APIs (Anthropic, OpenAI, Bedrock).
    Runs N concurrent workers. When all workers are busy, incoming calls
    queue up ordered by RunContext.priority (highest first). High-priority
    requests jump ahead of low-priority ones under load.

TwoLevelDispatcher — for internal/self-hosted models (vLLM, SGLang, etc.).
    Same mechanism, but the priority key is a two-tuple of
    (stage_priority, request_priority), matching Orla's two-level scheduler:
    - stage_priority (from AgentConfig.stage_priority): static weight of the
      agent type — a "reply" agent outranks a "summarize" agent regardless of
      individual request urgency.
    - request_priority (from RunContext.priority): runtime weight of this
      specific request — set by the RouterAgent based on user tier or urgency.

Usage::

    # External API (e.g. Anthropic) — 10 concurrent calls max
    dispatcher = PriorityDispatcher(max_concurrent=10)
    await dispatcher.start()

    # In LLMClient.chat():
    response = await dispatcher.dispatch(
        lambda: provider.acomplete(...),
        priority=context.priority,
    )

    # Internal model — 4 GPU workers, two-level scheduling
    dispatcher = TwoLevelDispatcher(max_workers=4)
    await dispatcher.start()

    response = await dispatcher.dispatch(
        lambda: provider.acomplete(...),
        stage_priority=agent.config.stage_priority,
        request_priority=context.priority,
    )
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from orchestrator.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Thread-safe sequence counter for FIFO tiebreaking within the same priority.
_seq_counter = itertools.count()


class PriorityDispatcher:
    """
    Priority-ordered dispatcher for external LLM APIs.

    Runs ``max_concurrent`` worker coroutines.  When all workers are busy,
    queued calls are served highest-priority-first.  Equal priorities are
    served FIFO.

    Priority scale: 1 (lowest / batch) … 5 (normal) … 10 (highest / urgent).
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self.max_concurrent = max_concurrent
        # Items: (-priority, seq, call_fn, future)
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, Callable[..., Coroutine[Any, Any, Any]], asyncio.Future[Any]]
        ] = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start background worker tasks. Call once after the event loop is running."""
        for _ in range(self.max_concurrent):
            self._workers.append(asyncio.create_task(self._worker_loop()))
        logger.debug(f"PriorityDispatcher started with {self.max_concurrent} workers")

    async def stop(self) -> None:
        """Cancel all workers. Call on shutdown."""
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def dispatch(
        self,
        call_fn: Callable[[], Coroutine[Any, Any, T]],
        priority: int = 5,
    ) -> T:
        """
        Enqueue ``call_fn`` and await its result.

        Args:
            call_fn: Zero-argument async callable that returns the LLM response.
            priority: Request priority (1-10). Higher values are served first.
        """
        if not self._workers:
            await self.start()

        future: asyncio.Future[T] = asyncio.get_event_loop().create_future()
        seq = next(_seq_counter)
        await self._queue.put((-priority, seq, call_fn, future))

        queue_depth = self._queue.qsize()
        if queue_depth > 5:
            logger.warning(f"PriorityDispatcher queue depth={queue_depth} (priority={priority})")

        return await future

    async def _worker_loop(self) -> None:
        while True:
            try:
                _neg_priority, _seq, call_fn, future = await self._queue.get()
                start = time.monotonic()
                try:
                    result = await call_fn()
                    if not future.done():
                        future.set_result(result)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    logger.debug(f"PriorityDispatcher: call completed in {elapsed_ms:.0f}ms")
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"PriorityDispatcher worker error: {exc}")


class TwoLevelDispatcher:
    """
    Two-level priority dispatcher for internal/self-hosted LLM models.

    Encodes Orla's two-level scheduler as a single priority queue with a
    composite key ``(-stage_priority, -request_priority, seq)``:

    - **Stage level**: ``AgentConfig.stage_priority`` — static weight of the
      agent type.  A "reply" agent (stage_priority=8) is always served ahead
      of a "summarize" agent (stage_priority=3), regardless of request urgency.
    - **Request level**: ``RunContext.priority`` — runtime weight of this
      specific request.  Within the same agent type, premium user requests
      (priority=8) jump ahead of free-tier requests (priority=3).

    Args:
        max_workers: Number of parallel GPU/inference workers.  Set this to
            match your internal model's actual concurrency capacity.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        # Items: (-stage_priority, -request_priority, seq, call_fn, future)
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, int, Callable[..., Coroutine[Any, Any, Any]], asyncio.Future[Any]]
        ] = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start background worker tasks."""
        for _ in range(self.max_workers):
            self._workers.append(asyncio.create_task(self._worker_loop()))
        logger.debug(f"TwoLevelDispatcher started with {self.max_workers} workers")

    async def stop(self) -> None:
        """Cancel all workers."""
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def dispatch(
        self,
        call_fn: Callable[[], Coroutine[Any, Any, T]],
        stage_priority: int = 5,
        request_priority: int = 5,
    ) -> T:
        """
        Enqueue ``call_fn`` and await its result.

        Args:
            call_fn: Zero-argument async callable that returns the LLM response.
            stage_priority: Static priority of the agent type (AgentConfig.stage_priority).
            request_priority: Runtime priority of this request (RunContext.priority).
        """
        if not self._workers:
            await self.start()

        future: asyncio.Future[T] = asyncio.get_event_loop().create_future()
        seq = next(_seq_counter)
        await self._queue.put((-stage_priority, -request_priority, seq, call_fn, future))

        queue_depth = self._queue.qsize()
        if queue_depth > 5:
            logger.warning(
                f"TwoLevelDispatcher queue depth={queue_depth} "
                f"(stage={stage_priority}, request={request_priority})"
            )

        return await future

    async def _worker_loop(self) -> None:
        while True:
            try:
                _neg_stage, _neg_req, _seq, call_fn, future = await self._queue.get()
                start = time.monotonic()
                try:
                    result = await call_fn()
                    if not future.done():
                        future.set_result(result)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    logger.debug(f"TwoLevelDispatcher: call completed in {elapsed_ms:.0f}ms")
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"TwoLevelDispatcher worker error: {exc}")
