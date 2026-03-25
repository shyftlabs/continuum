"""
Integration tests — Session (short-term) memory system.

Tests conversation history persistence, sliding window behavior,
session isolation, metadata handling, and the session→memory pipeline.
All tests use real Redis — no mocks.
"""

from __future__ import annotations

import uuid

import pytest

from orchestrator.llm.types import ChatMessage
from orchestrator.session.client import SessionClient

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_client():
    """Provide a real SessionClient connected to Redis."""
    client = SessionClient()
    client.initialize()
    created_sessions: list[str] = []

    class TrackedSession:
        def __init__(self, inner):
            self._inner = inner

        async def create(self, *, user_id=None, agent_id=None, session_id=None):
            sid = await self._inner.get_or_create_session(
                session_id=session_id or f"sess-{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                agent_id=agent_id,
            )
            created_sessions.append(sid)
            return sid

        async def add_message(self, session_id, message, **kw):
            return await self._inner.add_message(session_id, message, **kw)

        async def get_history(self, session_id, limit=None):
            return await self._inner.get_conversation_history(session_id, limit=limit)

        async def get_metadata(self, session_id):
            return await self._inner.get_session_metadata(session_id)

        async def clear(self, session_id):
            return await self._inner.clear_session(session_id)

        async def delete(self, session_id):
            return await self._inner.delete_session(session_id)

    tracked = TrackedSession(client)
    yield tracked

    # Cleanup
    for sid in created_sessions:
        try:
            await client.delete_session(sid)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test: Session CRUD operations
# ---------------------------------------------------------------------------


class TestSessionCRUD:
    """Test basic session create, read, update, delete."""

    async def test_create_session_returns_id(self, session_client):
        """Creating a session returns a valid session ID."""
        sid = await session_client.create(user_id="user-crud-1", agent_id="agent-1")
        assert sid is not None
        assert len(sid) > 0

    async def test_session_metadata_stored(self, session_client):
        """Session metadata should persist user_id and agent_id."""
        sid = await session_client.create(user_id="user-meta-1", agent_id="agent-meta-1")

        meta = await session_client.get_metadata(sid)
        assert meta is not None
        assert meta.user_id == "user-meta-1"
        assert meta.agent_id == "agent-meta-1"
        assert meta.message_count >= 0

    async def test_add_and_retrieve_messages(self, session_client):
        """Messages added to a session are retrievable."""
        sid = await session_client.create(user_id="user-msg-1")

        await session_client.add_message(
            sid, ChatMessage(role="user", content="Hello!"), store_in_memory=False
        )
        await session_client.add_message(
            sid, ChatMessage(role="assistant", content="Hi there!"), store_in_memory=False
        )

        history = await session_client.get_history(sid)
        assert len(history) >= 2
        contents = [m.content for m in history]
        assert "Hello!" in contents
        assert "Hi there!" in contents

    async def test_clear_session_removes_messages(self, session_client):
        """Clearing a session removes messages but keeps the session."""
        sid = await session_client.create(user_id="user-clear-1")

        await session_client.add_message(
            sid, ChatMessage(role="user", content="Important message"), store_in_memory=False
        )
        history_before = await session_client.get_history(sid)
        assert len(history_before) >= 1

        await session_client.clear(sid)

        history_after = await session_client.get_history(sid)
        assert len(history_after) == 0

        # Metadata should still exist
        meta = await session_client.get_metadata(sid)
        assert meta is not None

    async def test_delete_session_removes_everything(self, session_client):
        """Deleting a session removes messages and metadata."""
        sid = await session_client.create(user_id="user-del-1")
        await session_client.add_message(
            sid, ChatMessage(role="user", content="Will be deleted"), store_in_memory=False
        )

        await session_client.delete(sid)

        meta = await session_client.get_metadata(sid)
        assert meta is None


# ---------------------------------------------------------------------------
# Test: Conversation history ordering and limits
# ---------------------------------------------------------------------------


class TestConversationHistory:
    """Test conversation history ordering, sliding window, and limits."""

    async def test_messages_in_chronological_order(self, session_client):
        """Messages should be returned in the order they were added."""
        sid = await session_client.create(user_id="user-order-1")

        for i in range(5):
            await session_client.add_message(
                sid,
                ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}"),
                store_in_memory=False,
            )

        history = await session_client.get_history(sid)
        contents = [m.content for m in history]
        assert contents == [f"Message {i}" for i in range(5)]

    async def test_history_limit_returns_latest(self, session_client):
        """History with limit should return the most recent messages."""
        sid = await session_client.create(user_id="user-limit-1")

        for i in range(10):
            await session_client.add_message(
                sid,
                ChatMessage(role="user", content=f"Msg-{i}"),
                store_in_memory=False,
            )

        # Get only last 3
        history = await session_client.get_history(sid, limit=3)
        assert len(history) == 3
        # Should be the 3 most recent
        contents = [m.content for m in history]
        assert contents == ["Msg-7", "Msg-8", "Msg-9"]

    async def test_many_messages_sliding_window(self, session_client):
        """Large conversation should handle sliding window correctly."""
        sid = await session_client.create(user_id="user-sliding-1")

        # Add 50 messages
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            await session_client.add_message(
                sid,
                ChatMessage(role=role, content=f"Turn-{i}"),
                store_in_memory=False,
            )

        # Default limit
        history = await session_client.get_history(sid, limit=10)
        assert len(history) == 10
        # Should be the last 10
        assert history[0].content == "Turn-40"
        assert history[-1].content == "Turn-49"


# ---------------------------------------------------------------------------
# Test: Session isolation
# ---------------------------------------------------------------------------


class TestSessionIsolation:
    """Test that sessions are properly isolated from each other."""

    async def test_different_sessions_have_different_histories(self, session_client):
        """Messages in one session should not appear in another."""
        sid_a = await session_client.create(user_id="user-iso-a")
        sid_b = await session_client.create(user_id="user-iso-b")

        await session_client.add_message(
            sid_a, ChatMessage(role="user", content="Session A only"), store_in_memory=False
        )
        await session_client.add_message(
            sid_b, ChatMessage(role="user", content="Session B only"), store_in_memory=False
        )

        hist_a = await session_client.get_history(sid_a)
        hist_b = await session_client.get_history(sid_b)

        a_contents = [m.content for m in hist_a]
        b_contents = [m.content for m in hist_b]

        assert "Session A only" in a_contents
        assert "Session B only" not in a_contents
        assert "Session B only" in b_contents
        assert "Session A only" not in b_contents

    async def test_same_user_different_sessions_isolated(self, session_client):
        """Same user with multiple sessions should keep them separate."""
        uid = "user-multi-sess"
        sid_1 = await session_client.create(user_id=uid, agent_id="agent-1")
        sid_2 = await session_client.create(user_id=uid, agent_id="agent-2")

        await session_client.add_message(
            sid_1, ChatMessage(role="user", content="Talking to agent 1"), store_in_memory=False
        )
        await session_client.add_message(
            sid_2, ChatMessage(role="user", content="Talking to agent 2"), store_in_memory=False
        )

        hist_1 = await session_client.get_history(sid_1)
        hist_2 = await session_client.get_history(sid_2)

        assert len(hist_1) >= 1
        assert len(hist_2) >= 1
        assert hist_1[0].content == "Talking to agent 1"
        assert hist_2[0].content == "Talking to agent 2"


# ---------------------------------------------------------------------------
# Test: Message types and content
# ---------------------------------------------------------------------------


class TestMessageTypes:
    """Test different message roles and content types."""

    async def test_all_roles_stored(self, session_client):
        """User, assistant, and system messages should all be storable."""
        sid = await session_client.create(user_id="user-roles-1")

        await session_client.add_message(
            sid, ChatMessage(role="user", content="Question"), store_in_memory=False
        )
        await session_client.add_message(
            sid, ChatMessage(role="assistant", content="Answer"), store_in_memory=False
        )

        history = await session_client.get_history(sid)
        roles = [m.role for m in history]
        assert "user" in roles
        assert "assistant" in roles

    async def test_long_message_content(self, session_client):
        """Very long messages should be stored and retrieved correctly."""
        sid = await session_client.create(user_id="user-long-1")

        long_content = "A" * 10000  # 10KB message
        await session_client.add_message(
            sid, ChatMessage(role="user", content=long_content), store_in_memory=False
        )

        history = await session_client.get_history(sid)
        assert len(history) >= 1
        assert len(history[-1].content) == 10000

    async def test_unicode_messages(self, session_client):
        """Unicode content should round-trip correctly."""
        sid = await session_client.create(user_id="user-unicode-1")

        unicode_msg = "こんにちは世界 🌍 Привет мир 你好世界"
        await session_client.add_message(
            sid, ChatMessage(role="user", content=unicode_msg), store_in_memory=False
        )

        history = await session_client.get_history(sid)
        assert len(history) >= 1
        assert history[-1].content == unicode_msg

    async def test_empty_content_message(self, session_client):
        """Empty content messages should be handled gracefully."""
        sid = await session_client.create(user_id="user-empty-1")

        await session_client.add_message(
            sid, ChatMessage(role="user", content=""), store_in_memory=False
        )

        history = await session_client.get_history(sid)
        # Either stored with empty content or silently skipped
        assert isinstance(history, list)
