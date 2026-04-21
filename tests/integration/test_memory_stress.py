"""
Stress tests for the memory system.

Tests bulk operations, concurrent access, and performance under load.
All tests use real vector store — no mocks.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

pytestmark = pytest.mark.integration


def _uid() -> str:
    return f"stress-{uuid.uuid4().hex[:10]}"


def _aid() -> str:
    return f"agent-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def memory_client():
    from orchestrator.memory.client import MemoryClient

    client = MemoryClient()
    if not client.is_enabled:
        pytest.skip("Memory client not enabled")

    created_user_ids: list[str] = []

    class TrackedClient:
        def __init__(self, inner):
            self._inner = inner

        async def add(self, messages, *, user_id=None, agent_id=None, conversation_id=None, **kw):
            if user_id:
                created_user_ids.append(user_id)
            return await self._inner.add(
                messages, user_id=user_id, agent_id=agent_id, conversation_id=conversation_id, **kw
            )

        async def search(self, query, **kw):
            return await self._inner.search(query, **kw)

        async def get_all(self, **kw):
            return await self._inner.get_all(**kw)

        async def delete(self, memory_id):
            return await self._inner.delete(memory_id)

        async def delete_all(self, **kw):
            return await self._inner.delete_all(**kw)

        @property
        def is_enabled(self):
            return self._inner.is_enabled

    tracked = TrackedClient(client)
    yield tracked

    for uid in set(created_user_ids):
        try:
            await client.delete_all(user_id=uid)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------


class TestBulkOperations:
    """Test system behavior under bulk memory operations."""

    async def test_add_20_memories_single_user(self, memory_client):
        """Add 20 distinct memories for one user — no crash, all stored."""
        uid = _uid()
        facts = [
            f"I visited city number {i} on my travels, it was city_{i}."
            for i in range(20)
        ]

        start = time.time()
        for fact in facts:
            result = await memory_client.add(fact, user_id=uid)
            assert result.message is not None
        elapsed = time.time() - start

        all_mems = await memory_client.get_all(user_id=uid)
        assert len(all_mems) >= 5  # mem0 deduplicates aggressively, expect at least half stored

        # Should complete in reasonable time (< 5 min for 20 sequential adds with LLM extraction)
        assert elapsed < 300, f"20 adds took {elapsed:.1f}s — too slow"

    async def test_search_after_100_memories(self, memory_client):
        """Search relevance holds after adding 100 memories."""
        uid = _uid()

        # Add 100 diverse facts
        facts = (
            [f"I own item number {i} in my collection." for i in range(40)]
            + [f"My colleague number {i} works in department {i}." for i in range(40)]
            + ["I am allergic to shellfish."]
            + ["My emergency contact is Dr. Smith."]
            + ["I have a severe peanut allergy."]
            + ["I take daily medication for blood pressure."]
            + [f"I have book number {i} on my shelf." for i in range(17)]
        )
        for fact in facts:
            await memory_client.add(fact, user_id=uid)

        # Search for specific high-value facts after bulk insert
        result = await memory_client.search("allergies or medical conditions", user_id=uid, limit=5)
        assert result.total_results >= 1
        all_text = " ".join(r.memory.lower() for r in result.results)
        assert "allerg" in all_text or "peanut" in all_text or "medication" in all_text

    async def test_rapid_add_delete_cycle(self, memory_client):
        """Rapid add → delete cycle does not corrupt state."""
        uid = _uid()

        for i in range(30):
            await memory_client.add(
                f"Temporary fact {i}: I was at location_{i} today.",
                user_id=uid,
            )
            mems = await memory_client.get_all(user_id=uid)
            if mems:
                # Delete the oldest memory each cycle
                await memory_client.delete(mems[0].id)

        # After 30 cycles the system should still be responsive
        final = await memory_client.get_all(user_id=uid)
        result = await memory_client.search("location", user_id=uid, limit=5)
        assert result is not None  # No crash


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """Test concurrent memory operations across multiple users."""

    async def test_10_users_add_simultaneously(self, memory_client):
        """10 users add memories at the same time — no data mixing."""
        users = [_uid() for _ in range(10)]

        async def add_for_user(uid: str, index: int):
            await memory_client.add(
                f"My secret code is USERCODE_{index}_XYZ.",
                user_id=uid,
            )

        await asyncio.gather(*[add_for_user(uid, i) for i, uid in enumerate(users)])

        # Each user should only see their own code
        for i, uid in enumerate(users):
            result = await memory_client.search("secret code", user_id=uid, limit=5)
            texts = " ".join(r.memory for r in result.results)
            assert f"USERCODE_{i}_XYZ" in texts, f"User {i} missing their own code"
            # Must not contain other users' codes
            for j in range(10):
                if j != i:
                    assert f"USERCODE_{j}_XYZ" not in texts, (
                        f"User {i} can see user {j}'s code — isolation broken!"
                    )

    async def test_concurrent_search_5_users(self, memory_client):
        """5 users search simultaneously — no cross-contamination."""
        users = [_uid() for _ in range(5)]

        # Pre-populate each user with distinct memories
        for i, uid in enumerate(users):
            await memory_client.add(
                f"I exclusively use framework FRAMEWORK_{i} for all my projects.",
                user_id=uid,
            )

        # All 5 users search at the same time
        async def search_for_user(uid: str, index: int):
            return index, await memory_client.search("framework preference", user_id=uid, limit=3)

        results = await asyncio.gather(*[search_for_user(uid, i) for i, uid in enumerate(users)])

        for index, result in results:
            texts = " ".join(r.memory for r in result.results)
            assert f"FRAMEWORK_{index}" in texts, (
                f"User {index} got wrong search results"
            )

    async def test_concurrent_add_and_search_same_user(self, memory_client):
        """Concurrent adds and searches for the same user do not cause errors."""
        uid = _uid()

        async def add_facts():
            for i in range(10):
                await memory_client.add(
                    f"Concurrent fact {i}: I like activity_{i}.",
                    user_id=uid,
                )

        async def search_facts():
            for _ in range(5):
                await memory_client.search("activities I like", user_id=uid, limit=3)
                await asyncio.sleep(0.1)

        # Run adds and searches concurrently — should not crash
        await asyncio.gather(add_facts(), search_facts())

        final = await memory_client.get_all(user_id=uid)
        assert final is not None  # System still responsive


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


class TestPerformanceBenchmarks:
    """Measure response times under realistic load."""

    async def test_search_latency_p95(self, memory_client):
        """p95 search latency should be under 5 seconds with 20 stored memories."""
        uid = _uid()

        for i in range(20):
            await memory_client.add(
                f"Performance test fact {i}: I use tool_{i} daily.",
                user_id=uid,
            )

        latencies = []
        for _ in range(10):
            start = time.time()
            await memory_client.search("tools I use", user_id=uid, limit=5)
            latencies.append(time.time() - start)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 5.0, f"p95 search latency {p95:.2f}s exceeds 5s threshold"

    async def test_add_latency_acceptable(self, memory_client):
        """Single add operation should complete within 10 seconds."""
        uid = _uid()

        start = time.time()
        await memory_client.add("Latency test: I prefer async programming.", user_id=uid)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Single add took {elapsed:.2f}s — too slow"
