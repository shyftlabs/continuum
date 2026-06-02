"""Tests for BackgroundTaskRegistry — task ownership, exception isolation, draining.

Covers the two asyncio footguns the registry exists to prevent:
  - Problem A: a fire-and-forget task being garbage-collected mid-run.
  - Problem B: in-flight tasks being lost when the loop/process shuts down.
"""

from __future__ import annotations

import asyncio


def _make_registry():
    from orchestrator.core.background_tasks import BackgroundTaskRegistry

    return BackgroundTaskRegistry(name="test")


async def _append_after(out, value, delay):
    await asyncio.sleep(delay)
    out.append(value)


async def _raise(message="boom"):
    raise ValueError(message)


class TestSpawn:
    async def test_spawned_task_runs_to_completion(self):
        reg = _make_registry()
        out: list[int] = []
        reg.spawn(_append_after(out, 1, 0.0))
        await reg.drain(timeout=1.0)
        assert out == [1]

    async def test_reference_held_while_running_then_released(self):
        # Problem A: registry keeps a strong ref so the GC can't collect mid-run.
        reg = _make_registry()
        reg.spawn(asyncio.sleep(0.05))
        assert len(reg) == 1  # held while in flight
        await reg.drain(timeout=1.0)
        assert len(reg) == 0  # auto-removed on completion

    async def test_multiple_tasks_all_complete(self):
        reg = _make_registry()
        out: list[int] = []
        for i in range(5):
            reg.spawn(_append_after(out, i, 0.01))
        assert len(reg) == 5
        await reg.drain(timeout=1.0)
        assert sorted(out) == [0, 1, 2, 3, 4]
        assert len(reg) == 0

    async def test_exception_in_task_is_isolated(self):
        # A failing task must not propagate or leave a dangling reference.
        reg = _make_registry()
        reg.spawn(_raise("nope"))
        await reg.drain(timeout=1.0)  # does not raise
        assert len(reg) == 0

    async def test_spawn_without_running_loop_returns_none(self):
        # No running loop → cannot schedule; returns None and closes the coro.
        reg = _make_registry()
        coro = asyncio.sleep(0)

        def _call():
            return reg.spawn(coro)

        # Run the (sync) spawn call outside any event loop.
        task = await asyncio.get_running_loop().run_in_executor(None, _call)
        assert task is None
        assert len(reg) == 0


class TestDrain:
    async def test_drain_waits_for_inflight_work(self):
        # Problem B: shutdown waits for pending writes to finish.
        reg = _make_registry()
        flag = {"done": False}

        async def _work():
            await asyncio.sleep(0.05)
            flag["done"] = True

        reg.spawn(_work())
        await reg.drain(timeout=1.0)
        assert flag["done"] is True

    async def test_drain_with_no_tasks_is_noop(self):
        reg = _make_registry()
        await reg.drain(timeout=1.0)  # returns immediately, no error

    async def test_drain_respects_timeout_on_stuck_task(self):
        # A stuck task must not hang shutdown forever.
        reg = _make_registry()
        stuck = reg.spawn(asyncio.sleep(10))
        await reg.drain(timeout=0.05)  # returns despite the task still running
        assert stuck is not None
        # Cleanup so the test loop doesn't warn about a pending task.
        stuck.cancel()
        try:
            await stuck
        except asyncio.CancelledError:
            pass
