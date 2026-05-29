"""
Tests for PriorityDispatcher and TwoLevelDispatcher.

Key invariant: higher-priority calls are served before lower-priority ones
when workers are saturated.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.llm.dispatcher import PriorityDispatcher, TwoLevelDispatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _slow_call(value, delay=0.02):
    """Simulate a slow LLM call that returns a tagged value."""
    await asyncio.sleep(delay)
    return value


# ---------------------------------------------------------------------------
# PriorityDispatcher
# ---------------------------------------------------------------------------


class TestPriorityDispatcher:
    @pytest.mark.asyncio
    async def test_basic_dispatch_returns_result(self):
        d = PriorityDispatcher(max_concurrent=2)
        await d.start()
        try:
            result = await d.dispatch(lambda: _slow_call("hello"), priority=5)
            assert result == "hello"
        finally:
            await d.stop()

    @pytest.mark.asyncio
    async def test_higher_priority_served_first_under_load(self):
        """
        Saturate all workers so calls queue up.
        Then submit low and high priority — high should finish first.
        """
        d = PriorityDispatcher(max_concurrent=1)
        await d.start()
        order = []

        async def tagged(tag, delay=0.05):
            await asyncio.sleep(delay)
            order.append(tag)
            return tag

        try:
            # Fill the single worker
            blocker = asyncio.create_task(d.dispatch(lambda: tagged("blocker", 0.1), priority=5))

            # Give blocker time to grab the worker
            await asyncio.sleep(0.01)

            # Queue low then high (high should jump ahead)
            low = asyncio.create_task(d.dispatch(lambda: tagged("low"), priority=1))
            high = asyncio.create_task(d.dispatch(lambda: tagged("high"), priority=9))

            await asyncio.gather(blocker, low, high)
        finally:
            await d.stop()

        # high must appear before low in completed order (after blocker)
        assert order.index("high") < order.index("low")

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self):
        d = PriorityDispatcher(max_concurrent=1)
        await d.start()
        order = []

        async def tagged(tag):
            order.append(tag)
            return tag

        try:
            blocker = asyncio.create_task(
                d.dispatch(lambda: _slow_call("blocker", 0.1), priority=5)
            )
            await asyncio.sleep(0.01)

            t1 = asyncio.create_task(d.dispatch(lambda: tagged("first"), priority=5))
            await asyncio.sleep(0.001)
            t2 = asyncio.create_task(d.dispatch(lambda: tagged("second"), priority=5))

            await asyncio.gather(blocker, t1, t2)
        finally:
            await d.stop()

        assert order.index("first") < order.index("second")

    @pytest.mark.asyncio
    async def test_exception_propagated_to_caller(self):
        d = PriorityDispatcher(max_concurrent=1)
        await d.start()
        try:
            with pytest.raises(RuntimeError, match="boom"):
                await d.dispatch(
                    lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                    priority=5,
                )
        finally:
            await d.stop()

    @pytest.mark.asyncio
    async def test_auto_start_on_first_dispatch(self):
        d = PriorityDispatcher(max_concurrent=2)
        # Don't call start() — dispatch should auto-start
        try:
            result = await d.dispatch(lambda: _slow_call(42), priority=5)
            assert result == 42
        finally:
            await d.stop()

    @pytest.mark.asyncio
    async def test_stop_and_restart(self):
        d = PriorityDispatcher(max_concurrent=1)
        await d.start()
        result1 = await d.dispatch(lambda: _slow_call("a"), priority=5)
        await d.stop()
        await d.start()
        result2 = await d.dispatch(lambda: _slow_call("b"), priority=5)
        await d.stop()
        assert result1 == "a"
        assert result2 == "b"


# ---------------------------------------------------------------------------
# TwoLevelDispatcher
# ---------------------------------------------------------------------------


class TestTwoLevelDispatcher:
    @pytest.mark.asyncio
    async def test_basic_dispatch_returns_result(self):
        d = TwoLevelDispatcher(max_workers=2)
        await d.start()
        try:
            result = await d.dispatch(
                lambda: _slow_call("hello"),
                stage_priority=5,
                request_priority=5,
            )
            assert result == "hello"
        finally:
            await d.stop()

    @pytest.mark.asyncio
    async def test_stage_priority_dominates(self):
        """High stage_priority beats low stage_priority regardless of request_priority."""
        d = TwoLevelDispatcher(max_workers=1)
        await d.start()
        order = []

        async def tagged(tag):
            order.append(tag)
            return tag

        try:
            blocker = asyncio.create_task(
                d.dispatch(lambda: _slow_call("blocker", 0.1), stage_priority=5, request_priority=5)
            )
            await asyncio.sleep(0.01)

            # low stage, high request
            low_stage = asyncio.create_task(
                d.dispatch(lambda: tagged("low_stage"), stage_priority=1, request_priority=10)
            )
            # high stage, low request — should win
            high_stage = asyncio.create_task(
                d.dispatch(lambda: tagged("high_stage"), stage_priority=9, request_priority=1)
            )

            await asyncio.gather(blocker, low_stage, high_stage)
        finally:
            await d.stop()

        assert order.index("high_stage") < order.index("low_stage")

    @pytest.mark.asyncio
    async def test_request_priority_tiebreaks_same_stage(self):
        """Within the same stage_priority, request_priority decides order."""
        d = TwoLevelDispatcher(max_workers=1)
        await d.start()
        order = []

        async def tagged(tag):
            order.append(tag)
            return tag

        try:
            blocker = asyncio.create_task(
                d.dispatch(lambda: _slow_call("blocker", 0.1), stage_priority=5, request_priority=5)
            )
            await asyncio.sleep(0.01)

            low_req = asyncio.create_task(
                d.dispatch(lambda: tagged("low_req"), stage_priority=5, request_priority=1)
            )
            high_req = asyncio.create_task(
                d.dispatch(lambda: tagged("high_req"), stage_priority=5, request_priority=9)
            )

            await asyncio.gather(blocker, low_req, high_req)
        finally:
            await d.stop()

        assert order.index("high_req") < order.index("low_req")

    @pytest.mark.asyncio
    async def test_exception_propagated(self):
        d = TwoLevelDispatcher(max_workers=1)
        await d.start()
        try:
            with pytest.raises(ValueError, match="internal error"):
                await d.dispatch(
                    lambda: (_ for _ in ()).throw(ValueError("internal error")),
                    stage_priority=5,
                    request_priority=5,
                )
        finally:
            await d.stop()

    @pytest.mark.asyncio
    async def test_auto_start_on_first_dispatch(self):
        d = TwoLevelDispatcher(max_workers=1)
        result = await d.dispatch(lambda: _slow_call(99), stage_priority=5, request_priority=5)
        await d.stop()
        assert result == 99
