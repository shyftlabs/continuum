"""
Memory leak tests for the memory system.

Verifies that repeated operations do not grow unboundedly and that
cleanup operations fully remove data from the vector store.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.integration


def _uid() -> str:
    return f"leak-{uuid.uuid4().hex[:10]}"


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

        async def update(self, memory_id, data, **kw):
            return await self._inner.update(memory_id, data, **kw)

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
# Cleanup correctness
# ---------------------------------------------------------------------------


class TestCleanupCorrectness:
    """Verify delete operations fully remove data."""

    async def test_delete_all_leaves_no_orphans(self, memory_client):
        """After delete_all, get_all and search both return empty."""
        uid = _uid()

        await memory_client.add("I work as a pilot.", user_id=uid)
        await memory_client.add("I live in Berlin.", user_id=uid)
        await memory_client.add("I speak three languages.", user_id=uid)

        # Confirm data is there
        before = await memory_client.get_all(user_id=uid)
        assert len(before) >= 1

        # Full cleanup
        success = await memory_client.delete_all(user_id=uid)
        assert success is True

        # get_all must be empty
        after = await memory_client.get_all(user_id=uid)
        assert len(after) == 0, f"Expected 0 memories after delete_all, got {len(after)}"

        # search must also return empty
        sr = await memory_client.search("pilot or Berlin or languages", user_id=uid, limit=10)
        assert sr.total_results == 0, (
            f"Search still found {sr.total_results} results after delete_all"
        )

    async def test_delete_specific_memory_fully_removed(self, memory_client):
        """A deleted memory ID cannot be retrieved via get_all or search."""
        uid = _uid()

        await memory_client.add("My credit card PIN is 1234.", user_id=uid)
        mems = await memory_client.get_all(user_id=uid)
        assert len(mems) >= 1

        target_id = mems[0].id
        await memory_client.delete(target_id)

        # Must not appear in get_all
        remaining = await memory_client.get_all(user_id=uid)
        assert target_id not in [m.id for m in remaining]

        # Must not appear in search
        sr = await memory_client.search("credit card PIN", user_id=uid, limit=5)
        result_ids = [r.id for r in sr.results]
        assert target_id not in result_ids

    async def test_20_users_created_then_all_deleted(self, memory_client):
        """Create 20 users, delete all — no orphan data remains."""
        users = [_uid() for _ in range(20)]

        # Populate all users
        for uid in users:
            await memory_client.add(
                f"Orphan test user {uid}: I have private data.",
                user_id=uid,
            )

        # Delete all users
        for uid in users:
            await memory_client.delete_all(user_id=uid)

        # Verify all gone
        for uid in users:
            remaining = await memory_client.get_all(user_id=uid)
            assert len(remaining) == 0, (
                f"User {uid} still has {len(remaining)} memories after delete"
            )


# ---------------------------------------------------------------------------
# Growth stability
# ---------------------------------------------------------------------------


class TestGrowthStability:
    """Verify memory count does not grow unboundedly with repeated operations."""

    async def test_same_fact_added_10_times_no_unbounded_growth(self, memory_client):
        """Adding the same fact 10 times should not create 10 copies."""
        uid = _uid()
        fact = "I am the CEO of my company."

        for _ in range(10):
            await memory_client.add(fact, user_id=uid)

        all_mems = await memory_client.get_all(user_id=uid)
        # mem0 deduplication should keep this well below 10
        ceo_mems = [
            m for m in all_mems if "ceo" in m.memory.lower() or "company" in m.memory.lower()
        ]
        assert len(ceo_mems) <= 3, (
            f"Expected deduplication, but found {len(ceo_mems)} copies of the same fact"
        )

    async def test_add_delete_cycle_count_stable(self, memory_client):
        """Repeated add/delete cycles keep total memory count stable."""
        uid = _uid()

        # Seed with baseline facts
        await memory_client.add("I enjoy reading books.", user_id=uid)
        await memory_client.add("I play chess on weekends.", user_id=uid)

        baseline = await memory_client.get_all(user_id=uid)
        baseline_count = len(baseline)

        # 20 add → delete cycles
        for i in range(20):
            await memory_client.add(
                f"Temporary note {i}: I visited location_{i} today.",
                user_id=uid,
            )
            mems = await memory_client.get_all(user_id=uid)
            # Delete any memory beyond the baseline
            new_mems = [m for m in mems if m.id not in {b.id for b in baseline}]
            for m in new_mems:
                await memory_client.delete(m.id)

        # Total count should be close to baseline
        final = await memory_client.get_all(user_id=uid)
        assert len(final) <= baseline_count + 3, (
            f"Memory count grew from {baseline_count} to {len(final)} after add/delete cycles"
        )

    async def test_update_cycle_no_duplication(self, memory_client):
        """Updating a memory 10 times should not create 10 separate entries."""
        uid = _uid()

        await memory_client.add("My favorite sport is tennis.", user_id=uid)
        mems = await memory_client.get_all(user_id=uid)
        assert len(mems) >= 1
        mem_id = mems[0].id

        # Update the same memory 10 times
        sports = [
            "tennis",
            "golf",
            "swimming",
            "cycling",
            "rowing",
            "running",
            "boxing",
            "climbing",
            "surfing",
            "skiing",
        ]
        for sport in sports:
            try:
                await memory_client.update(mem_id, f"My favorite sport is {sport}.")
            except Exception:
                break  # mem0 may reject some updates — that's ok

        # Should not have exploded into many entries
        final = await memory_client.get_all(user_id=uid)
        sport_mems = [
            m for m in final if "sport" in m.memory.lower() or "favorite" in m.memory.lower()
        ]
        assert len(sport_mems) <= 3, (
            f"Update cycle created {len(sport_mems)} entries — expected deduplication"
        )


# ---------------------------------------------------------------------------
# Concurrent cleanup
# ---------------------------------------------------------------------------


class TestConcurrentCleanup:
    """Verify cleanup works correctly under concurrent conditions."""

    async def test_concurrent_delete_all_multiple_users(self, memory_client):
        """Deleting multiple users concurrently does not leave orphans."""
        users = [_uid() for _ in range(10)]

        # Populate
        for uid in users:
            await memory_client.add(f"User {uid} has private data.", user_id=uid)

        # Delete all concurrently
        await asyncio.gather(*[memory_client.delete_all(user_id=uid) for uid in users])

        # Verify all gone
        checks = await asyncio.gather(*[memory_client.get_all(user_id=uid) for uid in users])
        for uid, remaining in zip(users, checks, strict=False):
            assert len(remaining) == 0, f"User {uid} still has {len(remaining)} memories"

    async def test_add_then_immediate_delete_all(self, memory_client):
        """Add 10 memories then immediately delete_all — result must be empty."""
        uid = _uid()

        async def add_facts():
            for i in range(10):
                await memory_client.add(f"Fact {i}: I have item_{i}.", user_id=uid)

        await add_facts()
        await memory_client.delete_all(user_id=uid)

        remaining = await memory_client.get_all(user_id=uid)
        assert len(remaining) == 0, (
            f"Expected empty after delete_all, got {len(remaining)} memories"
        )
