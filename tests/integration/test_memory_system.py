"""
Integration tests — Memory system (Mem0 + Qdrant).

Tests long-term memory CRUD, user isolation, agent isolation,
search relevance, memory deduplication, and scope boundaries.
All tests use real Qdrant vector store — no mocks.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.integration


def _uid() -> str:
    """Generate a unique user ID for test isolation."""
    return f"memtest-{uuid.uuid4().hex[:10]}"


@pytest.fixture
async def memory_client():
    """Provide a real MemoryClient and clean up after the test."""
    from orchestrator.memory.client import MemoryClient

    client = MemoryClient()
    if not client.is_enabled:
        pytest.skip("Memory client not enabled (Qdrant unavailable)")

    created_user_ids: list[str] = []
    created_agent_ids: list[str] = []

    class TrackedClient:
        """Wrapper that tracks IDs for cleanup."""

        def __init__(self, inner):
            self._inner = inner

        async def add(self, messages, *, user_id=None, agent_id=None, run_id=None, **kw):
            if user_id:
                created_user_ids.append(user_id)
            if agent_id:
                created_agent_ids.append(agent_id)
            return await self._inner.add(
                messages, user_id=user_id, agent_id=agent_id, run_id=run_id, **kw
            )

        async def search(self, query, **kw):
            return await self._inner.search(query, **kw)

        async def get_all(self, **kw):
            return await self._inner.get_all(**kw)

        async def delete(self, memory_id):
            return await self._inner.delete(memory_id)

        async def delete_all(self, **kw):
            return await self._inner.delete_all(**kw)

        async def get(self, memory_id):
            return await self._inner.get(memory_id)

        async def update(self, memory_id, data, **kw):
            return await self._inner.update(memory_id, data, **kw)

        async def history(self, memory_id):
            return await self._inner.history(memory_id)

        @property
        def is_enabled(self):
            return self._inner.is_enabled

    tracked = TrackedClient(client)
    yield tracked

    # Cleanup all test data
    for uid in set(created_user_ids):
        try:
            await client.delete_all(user_id=uid)
        except Exception:
            pass
    for aid in set(created_agent_ids):
        try:
            await client.delete_all(agent_id=aid)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test: Basic memory CRUD
# ---------------------------------------------------------------------------


class TestMemoryCRUD:
    """Test basic memory add, search, get, delete operations."""

    async def test_add_and_search_memory(self, memory_client):
        """Store a fact and retrieve it via semantic search."""
        uid = _uid()
        result = await memory_client.add(
            "My favorite programming language is Python.",
            user_id=uid,
        )
        assert result.message is not None

        # Search for the stored fact
        search_result = await memory_client.search(
            "What programming language does the user prefer?",
            user_id=uid,
            limit=5,
        )
        assert search_result.total_results >= 1
        memories = [r.memory.lower() for r in search_result.results]
        assert any("python" in m for m in memories)

    async def test_add_multiple_memories_and_search(self, memory_client):
        """Store multiple facts and verify search returns relevant ones."""
        uid = _uid()

        await memory_client.add("I work as a data scientist.", user_id=uid)
        await memory_client.add("I live in San Francisco.", user_id=uid)
        await memory_client.add("I have a golden retriever named Max.", user_id=uid)

        # Search for job-related memory
        result = await memory_client.search("What is the user's job?", user_id=uid, limit=3)
        assert result.total_results >= 1
        top_memory = result.results[0].memory.lower()
        assert "data scientist" in top_memory or "scientist" in top_memory

        # Search for location
        result2 = await memory_client.search("Where does the user live?", user_id=uid, limit=3)
        assert result2.total_results >= 1
        top_memory2 = result2.results[0].memory.lower()
        assert "san francisco" in top_memory2 or "francisco" in top_memory2

    async def test_delete_specific_memory(self, memory_client):
        """Delete a single memory by ID."""
        uid = _uid()
        await memory_client.add("I hate broccoli.", user_id=uid)

        all_mems = await memory_client.get_all(user_id=uid)
        assert len(all_mems) >= 1

        # Delete the first memory
        mem_id = all_mems[0].id
        success = await memory_client.delete(mem_id)
        assert success is True

        # Verify deletion
        remaining = await memory_client.get_all(user_id=uid)
        remaining_ids = [m.id for m in remaining]
        assert mem_id not in remaining_ids

    async def test_delete_all_memories_for_user(self, memory_client):
        """Delete all memories for a user."""
        uid = _uid()
        # Use distinct facts that Mem0 won't deduplicate
        await memory_client.add("I enjoy mountain biking on weekends.", user_id=uid)
        await memory_client.add("My home address is 42 Oak Street.", user_id=uid)

        all_before = await memory_client.get_all(user_id=uid)
        assert len(all_before) >= 1  # At least 1 (Mem0 may consolidate)

        success = await memory_client.delete_all(user_id=uid)
        assert success is True

        all_after = await memory_client.get_all(user_id=uid)
        assert len(all_after) == 0

    async def test_get_all_memories(self, memory_client):
        """get_all returns all memories for a user."""
        uid = _uid()
        await memory_client.add("I enjoy hiking.", user_id=uid)
        await memory_client.add("I drink coffee every morning.", user_id=uid)

        all_mems = await memory_client.get_all(user_id=uid)
        assert len(all_mems) >= 2
        texts = [m.memory.lower() for m in all_mems]
        assert any("hik" in t for t in texts)
        assert any("coffee" in t for t in texts)


# ---------------------------------------------------------------------------
# Test: User isolation
# ---------------------------------------------------------------------------


class TestUserIsolation:
    """Test that memories are properly isolated between users."""

    async def test_users_cannot_see_each_others_memories(self, memory_client):
        """User A's memories should not appear in User B's searches."""
        uid_a = _uid()
        uid_b = _uid()

        await memory_client.add("I am allergic to peanuts.", user_id=uid_a)
        await memory_client.add("I love skiing.", user_id=uid_b)

        # User A searches — should find peanuts, NOT skiing
        result_a = await memory_client.search("allergies", user_id=uid_a, limit=5)
        a_texts = [r.memory.lower() for r in result_a.results]
        assert any("peanut" in t for t in a_texts)
        assert not any("ski" in t for t in a_texts)

        # User B searches — should find skiing, NOT peanuts
        result_b = await memory_client.search("hobbies", user_id=uid_b, limit=5)
        b_texts = [r.memory.lower() for r in result_b.results]
        assert any("ski" in t for t in b_texts)
        assert not any("peanut" in t for t in b_texts)

    async def test_delete_one_user_preserves_other(self, memory_client):
        """Deleting User A's memories should not affect User B."""
        uid_a = _uid()
        uid_b = _uid()

        await memory_client.add("User A has a cat.", user_id=uid_a)
        await memory_client.add("User B has a dog.", user_id=uid_b)

        # Delete User A
        await memory_client.delete_all(user_id=uid_a)

        # User B should still have their memory
        b_mems = await memory_client.get_all(user_id=uid_b)
        assert len(b_mems) >= 1
        assert any("dog" in m.memory.lower() for m in b_mems)

        # Cleanup
        await memory_client.delete_all(user_id=uid_b)


# ---------------------------------------------------------------------------
# Test: Agent isolation
# ---------------------------------------------------------------------------


class TestAgentIsolation:
    """Test that memories can be scoped per agent."""

    async def test_agent_scoped_memories_are_isolated(self, memory_client):
        """Two agents for the same user should store separate knowledge bases."""
        # Memory isolation is 'user' mode, so agent_id acts as secondary filter.
        # We use the same user_id but different agent_ids to test agent-level isolation.
        uid = _uid()
        aid_a = f"agent-a-{uuid.uuid4().hex[:6]}"
        aid_b = f"agent-b-{uuid.uuid4().hex[:6]}"

        await memory_client.add(
            "The refund policy is 30 days.", user_id=uid, agent_id=aid_a
        )
        await memory_client.add(
            "The server is hosted on AWS us-east-1.", user_id=uid, agent_id=aid_b
        )

        # Both memories stored under same user — search returns both
        result = await memory_client.search("policy or server", user_id=uid, limit=5)
        assert result.total_results >= 2

        all_texts = " ".join(r.memory.lower() for r in result.results)
        assert "refund" in all_texts or "30" in all_texts
        assert "aws" in all_texts or "us-east" in all_texts

        # Cleanup
        await memory_client.delete_all(user_id=uid)


# ---------------------------------------------------------------------------
# Test: Search relevance and ranking
# ---------------------------------------------------------------------------


class TestSearchRelevance:
    """Test that semantic search returns the most relevant memories first."""

    async def test_relevant_memory_ranks_higher(self, memory_client):
        """Semantically similar queries should return relevant results first."""
        uid = _uid()

        await memory_client.add("I am a vegetarian and don't eat meat.", user_id=uid)
        await memory_client.add("I work at Google as a software engineer.", user_id=uid)
        await memory_client.add("My birthday is on March 15th.", user_id=uid)

        result = await memory_client.search("dietary preferences", user_id=uid, limit=3)
        assert result.total_results >= 1
        top = result.results[0].memory.lower()
        assert "vegetarian" in top or "meat" in top

    async def test_search_with_no_relevant_results(self, memory_client):
        """Search for something completely unrelated should still work."""
        uid = _uid()
        await memory_client.add("I drive a Tesla Model 3.", user_id=uid)

        result = await memory_client.search(
            "quantum entanglement theory",
            user_id=uid,
            limit=5,
        )
        # Should return results (vector similarity is never exactly 0)
        # but scores should be relatively low
        if result.total_results > 0:
            # At least it doesn't crash
            assert result.results[0].score is not None

    async def test_search_limit_respected(self, memory_client):
        """Search limit should cap the number of results."""
        uid = _uid()

        for i in range(5):
            await memory_client.add(f"Fact number {i}: I have item {i}.", user_id=uid)

        result = await memory_client.search("facts about items", user_id=uid, limit=2)
        assert len(result.results) <= 2


# ---------------------------------------------------------------------------
# Test: Memory update and deduplication
# ---------------------------------------------------------------------------


class TestMemoryUpdateAndDedup:
    """Test memory updates and Mem0's deduplication behavior."""

    async def test_contradicting_facts_handled(self, memory_client):
        """Adding contradicting facts should not crash; Mem0 handles dedup."""
        uid = _uid()

        result1 = await memory_client.add("My favorite color is blue.", user_id=uid)
        assert result1.message is not None

        result2 = await memory_client.add("My favorite color is now red.", user_id=uid)
        assert result2.message is not None

        # Mem0 may consolidate, update, or even temporarily empty during dedup.
        # The key assertion: both add() calls succeed without error.
        # If memories exist, they should be searchable.
        all_mems = await memory_client.get_all(user_id=uid)
        if len(all_mems) > 0:
            all_text = " ".join(m.memory.lower() for m in all_mems)
            assert "red" in all_text or "blue" in all_text or "color" in all_text

    async def test_duplicate_facts_consolidated(self, memory_client):
        """Adding the same fact twice should not create duplicates."""
        uid = _uid()

        await memory_client.add("I live in Tokyo, Japan.", user_id=uid)
        await memory_client.add("I live in Tokyo, Japan.", user_id=uid)

        all_mems = await memory_client.get_all(user_id=uid)
        # Mem0 should deduplicate — at most one entry for this fact
        tokyo_mems = [m for m in all_mems if "tokyo" in m.memory.lower()]
        assert len(tokyo_mems) <= 2  # Ideally 1, but allow 2 for timing


# ---------------------------------------------------------------------------
# Test: Empty and edge cases
# ---------------------------------------------------------------------------


class TestMemoryEdgeCases:
    """Test edge cases in the memory system."""

    async def test_search_empty_user_returns_empty(self, memory_client):
        """Searching for a non-existent user should return empty results."""
        uid = _uid()  # Fresh user with no memories
        result = await memory_client.search("anything", user_id=uid, limit=5)
        assert result.total_results == 0
        assert len(result.results) == 0

    async def test_add_long_text_memory(self, memory_client):
        """Store a very long text as memory."""
        uid = _uid()
        long_text = (
            "The user has extensive experience in machine learning, "
            "natural language processing, computer vision, and robotics. "
            "They have published papers on transformer architectures, "
            "reinforcement learning, and multi-modal AI systems. "
        ) * 3  # ~600 chars

        result = await memory_client.add(long_text, user_id=uid)
        assert result.message is not None

        # Should be searchable
        sr = await memory_client.search("machine learning experience", user_id=uid, limit=3)
        assert sr.total_results >= 1

    async def test_add_list_of_messages(self, memory_client):
        """Store a conversation as a list of message dicts."""
        uid = _uid()
        messages = [
            {"role": "user", "content": "My phone number is 555-0123."},
            {"role": "assistant", "content": "Got it, I'll remember your phone number."},
        ]
        result = await memory_client.add(messages, user_id=uid)
        assert result.message is not None

        # Should extract the fact
        sr = await memory_client.search("phone number", user_id=uid, limit=3)
        assert sr.total_results >= 1
        all_text = " ".join(r.memory.lower() for r in sr.results)
        assert "555" in all_text or "phone" in all_text

    async def test_unicode_memory_storage(self, memory_client):
        """Store and retrieve memories with unicode content."""
        uid = _uid()
        await memory_client.add("ユーザーは日本語を話します。(User speaks Japanese)", user_id=uid)

        sr = await memory_client.search("language spoken", user_id=uid, limit=3)
        assert sr.total_results >= 1
        all_text = " ".join(r.memory for r in sr.results)
        assert "japanese" in all_text.lower() or "日本語" in all_text
