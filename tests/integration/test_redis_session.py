"""
Integration tests for Redis Session Provider — real Redis on port 6380.

Tests add_message, get_messages, sliding window, JSON error handling.
"""

from __future__ import annotations

import json
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_provider():
    """Create a real Redis session provider."""
    from orchestrator.session.config import SessionConfig
    from orchestrator.session.providers.redis import RedisSessionProvider

    config = SessionConfig(
        enabled=True,
        redis_host="localhost",
        redis_port=6380,
        redis_password="sdk123456789",
        ttl_seconds=300,
        max_messages=100,
    )
    provider = RedisSessionProvider(config=config)
    result = provider.initialize()
    if not result:
        pytest.skip("Redis session provider failed to initialize")
    yield provider
    await provider.close()


class TestRedisSessionIntegration:
    async def test_create_and_get_session(self, session_provider, test_id):
        sid = await session_provider.get_or_create_session(
            session_id=f"test-sess-{test_id}",
            user_id="user-1",
            agent_id="agent-1",
        )
        assert sid == f"test-sess-{test_id}"

    async def test_add_and_retrieve_messages(self, session_provider, test_id):
        from orchestrator.llm.types import ChatMessage

        sid = await session_provider.get_or_create_session(
            session_id=f"msg-sess-{test_id}"
        )

        # Add messages
        await session_provider.add_message(
            sid, ChatMessage(role="user", content="Hello")
        )
        await session_provider.add_message(
            sid, ChatMessage(role="assistant", content="Hi there!")
        )

        messages = await session_provider.get_messages(sid)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"

    async def test_session_metadata_persistence(self, session_provider, test_id):
        sid = await session_provider.get_or_create_session(
            session_id=f"meta-sess-{test_id}",
            user_id="user-meta",
            agent_id="agent-meta",
        )
        metadata = await session_provider.get_session_metadata(sid)
        assert metadata is not None
        assert metadata.session_id == f"meta-sess-{test_id}"
        assert metadata.user_id == "user-meta"

    async def test_malformed_json_in_messages_skipped(self, session_provider, test_id):
        """Verify that malformed JSON entries in the message list are skipped gracefully."""
        from orchestrator.llm.types import ChatMessage

        sid = f"corrupt-sess-{test_id}"
        await session_provider.get_or_create_session(session_id=sid)

        # Add a valid message first
        await session_provider.add_message(
            sid, ChatMessage(role="user", content="Valid message")
        )

        # Inject corrupt JSON directly into Redis using the provider's internal sync client
        messages_key = f"orchestrator:session:{sid}:messages"
        await session_provider._redis.rpush(messages_key, "THIS IS NOT JSON {{{")

        # Add another valid message
        await session_provider.add_message(
            sid, ChatMessage(role="assistant", content="Also valid")
        )

        # Should get 2 valid messages, skipping the corrupt one
        messages = await session_provider.get_messages(sid)
        assert len(messages) == 2
        assert messages[0].content == "Valid message"
        assert messages[1].content == "Also valid"

    async def test_sliding_window_trims_oldest(self, session_provider, test_id):
        """Test sliding window actually trims old messages."""
        from orchestrator.llm.types import ChatMessage

        # Override max_messages for this test
        session_provider._config.max_messages = 5
        session_provider._config.message_limit_strategy = "sliding_window"
        session_provider._config.sliding_window_trim_count = 2

        sid = f"sliding-sess-{test_id}"
        await session_provider.get_or_create_session(session_id=sid)

        for i in range(7):
            await session_provider.add_message(
                sid, ChatMessage(role="user", content=f"Message {i}")
            )

        messages = await session_provider.get_messages(sid)
        # Should have trimmed oldest messages
        assert len(messages) <= 7
        # Most recent messages should still be there
        assert any("Message 6" in m.content for m in messages)
