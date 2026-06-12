"""
What happens when two people use the same user_id?

Real scenario: two people open the local-shop CLI and BOTH type "alice"
when asked for a user ID.

  Terminal 1: Enter user ID: alice   → person A
  Terminal 2: Enter user ID: alice   → person B

Question: does the framework keep them separate, or do they share data?

Short answer proven by the tests below:
  The framework has NO concept of "who is behind the user_id".
  user_id is purely a scope key — a label on a bucket.
  Two people with the same user_id ARE the same bucket.
  They share memories AND session history.
  This is by design: the framework trusts the caller to supply unique ids.
  Uniqueness is the responsibility of the auth layer (JWT / OAuth),
  not the framework.

No external services needed — all tests are pure logic / mock-based.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.memory import (
    MemoryAddResult,
    MemoryClient,
    MemoryConfig,
    MemoryEntry,
    MemorySearchResult,
)

# ── shared helpers ────────────────────────────────────────────────────────────

def _mock_provider():
    from continuum.memory import BaseMemoryProvider
    p = MagicMock(spec=BaseMemoryProvider)
    p.is_initialized = True
    p.add    = AsyncMock(return_value=MemoryAddResult(message="Added", results=[]))
    p.search = AsyncMock(return_value=MemorySearchResult(
        results=[], query="", limit=5, total_results=0
    ))
    p.delete     = AsyncMock(return_value=True)
    p.delete_all = AsyncMock(return_value=True)
    p.get        = AsyncMock(return_value=None)
    p.get_all    = AsyncMock(return_value=[])
    p.update     = AsyncMock(return_value=MemoryEntry(id="m-1", memory="updated"))
    p.history    = AsyncMock(return_value=[])
    p.reset      = AsyncMock(return_value=True)
    p.close      = AsyncMock()
    return p


def _client(provider=None):
    config = MemoryConfig(enabled=True, memory_isolation="user")
    return MemoryClient(config=config, provider=provider or _mock_provider())


# =============================================================================
# 1. Memory layer — same user_id → same bucket
# =============================================================================

class TestMemoryBucketCollision:
    """
    The memory client uses user_id as the only key to scope reads and writes.
    Two people using the same user_id read and write the same bucket.
    """

    @pytest.mark.asyncio
    async def test_person_A_and_B_write_to_identical_provider_kwargs(self):
        """
        Person A adds "I own a golden retriever".
        Person B adds "I prefer cats".
        Both calls reach the provider with user_id="alice" — same bucket.
        The framework cannot tell them apart.
        """
        received: list[dict] = []

        async def capture_add(messages, **kwargs):
            received.append({"messages": messages, "user_id": kwargs.get("user_id")})
            return MemoryAddResult(message="Added", results=[])

        provider = _mock_provider()
        provider.add.side_effect = capture_add

        client_A = _client(provider=provider)
        client_B = _client(provider=provider)  # same provider = same store

        await client_A.add("I own a golden retriever", user_id="alice")
        await client_B.add("I prefer cats",            user_id="alice")

        # Both wrote to exactly the same user_id key
        assert received[0]["user_id"] == "alice"
        assert received[1]["user_id"] == "alice"

        # The provider has no way to distinguish the two callers
        assert received[0]["user_id"] == received[1]["user_id"]

    @pytest.mark.asyncio
    async def test_person_B_search_returns_person_A_memories(self):
        """
        If Person A stored a memory under user_id="alice", Person B searching
        with user_id="alice" will find it — the framework returns everything
        in the bucket regardless of who wrote it.
        """
        # Provider pretends Person A's memory exists in the bucket
        person_A_memory = MemoryEntry(id="m-1", memory="I own a golden retriever", score=0.95)
        provider = _mock_provider()
        provider.search.return_value = MemorySearchResult(
            results=[person_A_memory], query="pets", limit=5, total_results=1
        )

        client_B = _client(provider=provider)
        result = await client_B.search("pets", user_id="alice")

        # Person B receives Person A's memory — shared bucket, no separation
        assert result.total_results == 1
        assert "golden retriever" in result.results[0].memory

        # The provider was called with user_id="alice" — same key Person A wrote to
        assert provider.search.call_args.kwargs["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_person_B_delete_all_wipes_person_A_memories_too(self):
        """
        Person B calling delete_all(user_id="alice") deletes every memory
        in the bucket — including memories Person A stored.
        There is no per-author ownership inside a bucket.
        """
        provider = _mock_provider()
        client_B = _client(provider=provider)

        await client_B.delete_all(user_id="alice")

        provider.delete_all.assert_awaited_once()
        # The key sent to the provider is the shared bucket key
        assert provider.delete_all.call_args.kwargs.get("user_id") == "alice"


# =============================================================================
# 2. Session layer — same user_id → same Redis key
# =============================================================================

class TestSessionKeyCollision:
    """
    The Redis provider computes a deterministic session ID from user_id.
    Same user_id → same Redis key → same session (same chat history).
    """

    def _compute_session_id(self, session_id, user_id, conversation_id):
        from continuum.session.types import generate_session_id
        if session_id:
            return session_id
        if conversation_id and user_id:
            return f"c:{conversation_id}:u:{user_id}"
        if user_id:
            return f"u:{user_id}"
        return generate_session_id()

    def test_same_user_id_produces_identical_redis_key(self):
        """
        Person A and Person B both use user_id="alice".
        They get the exact same Redis key → they share the same session.
        """
        key_A = self._compute_session_id(None, "alice", None)
        key_B = self._compute_session_id(None, "alice", None)

        assert key_A == key_B == "u:alice"
        # One session in Redis. Both people read/write the same history.

    def test_different_user_ids_produce_different_keys(self):
        """Normal case: unique user_ids → completely separate sessions."""
        key_alice = self._compute_session_id(None, "alice", None)
        key_bob   = self._compute_session_id(None, "bob",   None)

        assert key_alice != key_bob
        assert key_alice == "u:alice"
        assert key_bob   == "u:bob"

    def test_conversation_id_separates_sessions_for_same_user(self):
        """
        Adding a conversation_id creates a separate session key even
        for the same user_id.  This is the escape hatch for multi-tab
        or multi-device use by the same identity.
        BUT: memory isolation (user mode) is still shared — see next class.
        """
        key_tab1 = self._compute_session_id(None, "alice", "conv-tab1")
        key_tab2 = self._compute_session_id(None, "alice", "conv-tab2")

        assert key_tab1 != key_tab2          # separate sessions
        assert "alice" in key_tab1           # same user embedded in key
        assert "alice" in key_tab2

    def test_no_user_id_generates_random_uuid_per_caller(self):
        """
        If neither person types a user_id (just presses Enter in the CLI),
        each gets a unique random UUID — they are automatically isolated.
        This is why pressing Enter is the safe default.
        """
        key_A = self._compute_session_id(None, None, None)
        key_B = self._compute_session_id(None, None, None)

        assert key_A != key_B   # random UUIDs — always different


# =============================================================================
# 3. Memory isolation with conversation_id — partial fix only
# =============================================================================

class TestConversationIdAsPartialFix:
    """
    conversation_id separates SESSIONS but NOT MEMORIES in user isolation mode.
    The memory scope is determined by memory_isolation, not by conversation_id.
    In user isolation mode, all conversations for the same user_id share one
    memory bucket regardless of conversation_id.
    """

    @pytest.mark.asyncio
    async def test_user_mode_ignores_conversation_id_for_memory_scope(self):
        """
        In user isolation mode, conversation_id is dropped by _build_scope().
        Two people sharing user_id="alice" but with different conversation_ids
        still write to the same memory bucket.
        """
        received_kwargs: list[dict] = []

        async def capture(messages, **kwargs):
            received_kwargs.append(kwargs)
            return MemoryAddResult(message="Added", results=[])

        provider = _mock_provider()
        provider.add.side_effect = capture

        client = _client(provider=provider)

        # Person A: has conversation_id="conv-A"
        await client.add("I like dogs", user_id="alice", conversation_id="conv-A")
        # Person B: has conversation_id="conv-B"
        await client.add("I like cats", user_id="alice", conversation_id="conv-B")

        # Both calls reach the provider with user_id="alice" and NO conversation_id
        # because user isolation mode DROPS conversation_id (scope only uses user_id)
        assert received_kwargs[0] == {"user_id": "alice", "metadata": None,
                                       "custom_prompt": None, "infer": True}
        assert received_kwargs[1] == {"user_id": "alice", "metadata": None,
                                       "custom_prompt": None, "infer": True}

        # Both land in the same bucket despite different conversation_ids
        assert received_kwargs[0]["user_id"] == received_kwargs[1]["user_id"]

    @pytest.mark.asyncio
    async def test_conversation_mode_separates_memory_buckets(self):
        """
        Switching to conversation isolation mode DOES separate memories
        by conversation_id — this is the correct isolation level when
        you want per-chat separation even for the same user.
        """
        received_kwargs: list[dict] = []

        async def capture(messages, **kwargs):
            received_kwargs.append(kwargs)
            return MemoryAddResult(message="Added", results=[])

        provider = _mock_provider()
        provider.add.side_effect = capture

        config = MemoryConfig(enabled=True, memory_isolation="conversation")
        client = MemoryClient(config=config, provider=provider)

        await client.add("I like dogs", conversation_id="conv-A")
        await client.add("I like cats", conversation_id="conv-B")

        # Now conversation_id IS kept — different buckets
        assert received_kwargs[0]["conversation_id"] == "conv-A"
        assert received_kwargs[1]["conversation_id"] == "conv-B"
        assert received_kwargs[0]["conversation_id"] != received_kwargs[1]["conversation_id"]


# =============================================================================
# 4. The playground scenario end-to-end
# =============================================================================

class TestPlaygroundScenario:
    """
    Concrete simulation of the CLI scenario:
    two people open gateway-local-shop and both type "alice".
    """

    def test_cli_with_empty_enter_gives_unique_ids(self):
        """
        Safest path: both press Enter (empty input).
        CLI: user_id = "".strip() or None → None
        Framework: None → random UUID per caller → isolated automatically.
        """
        from continuum.session.types import generate_session_id

        # Both press Enter — both get None from the CLI
        uid_A = "".strip() or None
        uid_B = "".strip() or None
        assert uid_A is None
        assert uid_B is None

        # Framework generates separate UUIDs for each
        session_A = generate_session_id()
        session_B = generate_session_id()
        assert session_A != session_B  # random UUIDs never collide

    def test_cli_both_type_same_name_shares_everything(self):
        """
        Dangerous path: both type "alice".
        CLI: "alice".strip() or None → "alice"
        Framework: "alice" → same Redis key "u:alice" → shared session & memories.
        """
        uid_A = "alice".strip() or None
        uid_B = "alice".strip() or None

        assert uid_A == uid_B == "alice"

        # Same session key
        key_A = f"u:{uid_A}"
        key_B = f"u:{uid_B}"
        assert key_A == key_B  # ONE session in Redis — both share history

        # Same memory scope key
        from continuum.memory.scopes import MemoryScope
        scope_A = MemoryScope.user(uid_A).to_identifiers()
        scope_B = MemoryScope.user(uid_B).to_identifiers()
        assert scope_A == scope_B  # ONE memory bucket — both share memories

    def test_framework_has_no_uniqueness_enforcement(self):
        """
        The framework does NOT check whether a user_id is already in use
        by someone else.  It cannot — it has no user registry.
        Uniqueness is entirely the caller's responsibility.

        Correct approach:
          - In production: use JWT sub (always unique per real user)
          - In CLI/dev:    press Enter to get a random UUID, or
                           use a unique name like "alice-laptop-2026"
        """
        from continuum.memory.scopes import MemoryScope

        # Framework builds identical scopes for both — no check possible
        scope_person1 = MemoryScope.user("alice")
        scope_person2 = MemoryScope.user("alice")

        assert scope_person1.to_identifiers() == scope_person2.to_identifiers()
        # The framework sees one user "alice" — it cannot know there are two people
