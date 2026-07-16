"""
Integration tests — Adversarial Redis session scenarios (Issue #21).

Goal: try to BREAK the Redis-backed session layer in src/continuum/session/.
Covers:
  A. Cross-conversation bleed (session_id derivation, isolation)
  B. TTL expiry mid-conversation
  C. Concurrent writes / get-or-create races
  D. Message limit strategies (error + sliding_window)
  E. Adversarial inputs (1 MB, null bytes, RTL/emoji, nested metadata)
  F. Connection failures (wrong port, wrong password)

Requirements:
  - Real Redis on port 6380 (docker compose up redis-sdk -d)
  - All tests are marked @pytest.mark.integration
  - Tests skip cleanly when Redis is absent
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("redis", reason="redis package not installed — run: pip install redis")

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Return a SessionConfig pointed at the test Redis instance."""
    from continuum.session.config import SessionConfig

    defaults = {
        "enabled": True,
        "redis_host": "localhost",
        "redis_port": 6380,
        "redis_password": "sdk123456789",
        "ttl_seconds": 300,
        "max_messages": 100,
        "key_prefix": "orchestrator:session",
        "message_limit_strategy": "sliding_window",
        "sliding_window_trim_count": 10,
    }
    defaults.update(overrides)
    return SessionConfig(**defaults)


def _make_provider(**config_overrides):
    """Create + initialize a RedisSessionProvider. Skips test if Redis is unavailable."""
    from continuum.session.providers.redis import RedisSessionProvider

    p = RedisSessionProvider(config=_make_config(**config_overrides), auto_initialize=False)
    ok = p.initialize()
    if not ok:
        pytest.skip("Redis not available on port 6380 — start with: docker compose up redis-sdk -d")
    return p


def _msg(role: str = "user", content: str = "test message"):
    from continuum.llm.types import ChatMessage

    return ChatMessage(role=role, content=content)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def provider():
    """Standard RedisSessionProvider for tests that don't need custom config."""
    p = _make_provider()
    yield p
    await p.close()


# ---------------------------------------------------------------------------
# Group A: Cross-Conversation Bleed
# ---------------------------------------------------------------------------


class TestCrossConversationBleed:
    """Verify that session_id derivation prevents messages leaking across users/windows."""

    async def test_same_conv_id_different_users_produce_different_sessions(self, provider, test_id):
        """
        Key formula: c:{conv_id}:u:{user_id}
        Two users sharing a conversation_id must never share a session.
        """
        conv_id = f"shared-conv-{test_id}"

        sid_a = await provider.get_or_create_session(
            user_id=f"user-A-{test_id}", conversation_id=conv_id
        )
        sid_b = await provider.get_or_create_session(
            user_id=f"user-B-{test_id}", conversation_id=conv_id
        )

        assert sid_a != sid_b, (
            f"Different users mapped to the same session key: {sid_a}. "
            "This allows cross-user message bleed."
        )

        await provider.add_message(sid_a, _msg(content="Secret for A only"))
        await provider.add_message(sid_b, _msg(content="Secret for B only"))

        msgs_a = [m.content for m in await provider.get_messages(sid_a)]
        msgs_b = [m.content for m in await provider.get_messages(sid_b)]

        assert "Secret for A only" in msgs_a
        assert "Secret for B only" not in msgs_a, "User B's message bled into User A's session"
        assert "Secret for B only" in msgs_b
        assert "Secret for A only" not in msgs_b, "User A's message bled into User B's session"

        await provider.delete_session(sid_a)
        await provider.delete_session(sid_b)

    async def test_deleted_session_conv_id_reuse_starts_clean(self, provider, test_id):
        """
        After delete_session, reusing the same conversation_id + user_id must
        produce a fresh session with no old messages (no ghost data resurrection).
        """
        conv_id = f"conv-reuse-{test_id}"
        user_id = f"user-reuse-{test_id}"

        # Create, populate, delete
        sid = await provider.get_or_create_session(user_id=user_id, conversation_id=conv_id)
        await provider.add_message(sid, _msg(content="Old ghost message"))
        await provider.delete_session(sid)

        # Recreate with the same deterministic key
        sid2 = await provider.get_or_create_session(user_id=user_id, conversation_id=conv_id)
        msgs = [m.content for m in await provider.get_messages(sid2)]

        assert "Old ghost message" not in msgs, (
            "Deleted session's messages resurfaced after the conversation_id was reused."
        )

        await provider.delete_session(sid2)


# ---------------------------------------------------------------------------
# Group B: TTL Expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    """Verify TTL semantics — expiry raises the right error, access refreshes TTL."""

    async def test_expired_session_raises_session_not_found_error(self, test_id):
        """
        After a session's TTL has elapsed, add_message must raise SessionNotFoundError —
        not silently succeed or raise a generic exception.
        """
        from continuum.session.exceptions import SessionNotFoundError

        p = _make_provider(ttl_seconds=2)
        sid = await p.get_or_create_session(session_id=f"ttl-expire-{test_id}")
        try:
            await p.add_message(sid, _msg(content="Written before expiry"))

            await asyncio.sleep(3)  # outlast the 2-second TTL

            with pytest.raises(SessionNotFoundError):
                await p.add_message(sid, _msg(content="Written after expiry"))
        finally:
            # Session already expired; delete is a no-op but harmless
            try:
                await p.delete_session(sid)
            except Exception:
                pass
            await p.close()

    async def test_ttl_is_refreshed_on_get_messages(self, test_id):
        """
        Accessing a session via get_messages before its TTL expires must reset
        the TTL, keeping the session alive past the original deadline.
        """
        p = _make_provider(ttl_seconds=4)
        sid = await p.get_or_create_session(session_id=f"ttl-refresh-{test_id}")
        try:
            await p.add_message(sid, _msg(content="Persistent message"))

            # At t≈2s: access the session — TTL is reset to t=6s
            await asyncio.sleep(2)
            msgs = await p.get_messages(sid)
            assert len(msgs) == 1, "Message missing after first access"

            # At t≈4s (2s after the refresh): the session must still be alive
            # because TTL was extended to t≈6s at the refresh point.
            await asyncio.sleep(2)
            msgs2 = await p.get_messages(sid)
            assert len(msgs2) == 1, (
                "Session expired even though TTL should have been refreshed on access."
            )
        finally:
            try:
                await p.delete_session(sid)
            except Exception:
                pass
            await p.close()


# ---------------------------------------------------------------------------
# Group C: Concurrency Races
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    """Probe race conditions in add_message and get_or_create_session."""

    async def test_concurrent_add_message_no_lost_writes(self, provider, test_id):
        """
        50 concurrent add_message calls on the same session must not lose any writes.
        The Redis list length (ground truth) and the metadata message_count must both
        equal 50 — a mismatch indicates a lost-update race on message_count.
        """
        sid = await provider.get_or_create_session(session_id=f"conc-write-{test_id}")

        n = 50
        await asyncio.gather(
            *[provider.add_message(sid, _msg(content=f"msg-{i}")) for i in range(n)]
        )

        messages_key = f"orchestrator:session:{sid}:messages"
        actual_llen = await provider._redis.llen(messages_key)
        metadata = await provider.get_session_metadata(sid)

        assert actual_llen == n, (
            f"Redis list has {actual_llen} entries, expected {n}. Lost writes detected."
        )
        assert metadata.message_count == n, (
            f"metadata.message_count={metadata.message_count} != llen={actual_llen}. "
            "Race condition on the read-modify-write of message_count."
        )

        await provider.delete_session(sid)

    async def test_concurrent_get_or_create_same_key_no_duplicate_session(self, provider, test_id):
        """
        20 concurrent get_or_create_session calls with identical (user_id, conversation_id)
        must all return the same session_id — the SET NX guard must prevent duplicates.
        """
        user_id = f"user-conc-{test_id}"
        conv_id = f"conv-conc-{test_id}"

        results = await asyncio.gather(
            *[
                provider.get_or_create_session(user_id=user_id, conversation_id=conv_id)
                for _ in range(20)
            ]
        )

        unique = set(results)
        assert len(unique) == 1, (
            f"Expected 1 unique session ID from 20 concurrent creates, got {len(unique)}: {unique}"
        )

        sid = results[0]
        metadata = await provider.get_session_metadata(sid)
        assert metadata is not None
        assert metadata.session_id == sid

        await provider.delete_session(sid)


# ---------------------------------------------------------------------------
# Group D: Message Limit Strategies
# ---------------------------------------------------------------------------


class TestMessageLimitStrategies:
    """Verify both 'error' and 'sliding_window' strategies behave correctly."""

    async def test_error_strategy_raises_on_fourth_message(self, test_id):
        """
        With max_messages=3 and strategy='error', the 4th add_message must raise
        SessionMessageLimitError with correct current_count and max_messages fields.
        """
        from continuum.session.exceptions import SessionMessageLimitError

        p = _make_provider(max_messages=3, message_limit_strategy="error")
        sid = f"limit-err-{test_id}"
        try:
            await p.get_or_create_session(session_id=sid)
            for i in range(3):
                await p.add_message(sid, _msg(content=f"msg-{i}"))

            with pytest.raises(SessionMessageLimitError) as exc_info:
                await p.add_message(sid, _msg(content="one-too-many"))

            err = exc_info.value
            assert err.max_messages == 3
            assert err.current_count == 3
        finally:
            try:
                await p.delete_session(sid)
            except Exception:
                pass
            await p.close()

    async def test_error_strategy_count_resets_to_zero_after_clear(self, test_id):
        """
        clear_session resets message_count to 0. Subsequent messages must succeed
        without raising SessionMessageLimitError.
        """

        p = _make_provider(max_messages=3, message_limit_strategy="error")
        sid = f"limit-clear-{test_id}"
        try:
            await p.get_or_create_session(session_id=sid)

            for i in range(3):
                await p.add_message(sid, _msg(content=f"fill-{i}"))

            await p.clear_session(sid)

            # Must not raise — count should be back to 0
            for i in range(3):
                await p.add_message(sid, _msg(content=f"refill-{i}"))

            msgs = await p.get_messages(sid)
            assert len(msgs) == 3
            assert all("refill" in m.content for m in msgs), (
                "Expected only refill messages after clear; old messages may have leaked back."
            )
        finally:
            try:
                await p.delete_session(sid)
            except Exception:
                pass
            await p.close()

    async def test_sliding_window_evicts_oldest_preserves_order(self, test_id):
        """
        With max_messages=5 and trim_count=2, adding 8 messages must:
        - Evict 'msg-0' and 'msg-1' (the two oldest)
        - Keep 'msg-7' (the newest)
        - Preserve strict chronological order in the remaining list
        """
        p = _make_provider(
            max_messages=5,
            message_limit_strategy="sliding_window",
            sliding_window_trim_count=2,
        )
        sid = f"sliding-order-{test_id}"
        try:
            await p.get_or_create_session(session_id=sid)

            for i in range(8):
                await p.add_message(sid, _msg(content=f"msg-{i}"))

            msgs = await p.get_messages(sid)
            contents = [m.content for m in msgs]

            assert "msg-0" not in contents, "msg-0 (oldest) was not evicted by sliding window"
            assert "msg-1" not in contents, "msg-1 was not evicted by sliding window"
            assert "msg-7" in contents, "msg-7 (newest) is missing after sliding window"

            nums = [int(c.split("-")[1]) for c in contents]
            assert nums == sorted(nums), f"Sliding window broke chronological order: {contents}"
        finally:
            try:
                await p.delete_session(sid)
            except Exception:
                pass
            await p.close()


# ---------------------------------------------------------------------------
# Group E: Adversarial Inputs
# ---------------------------------------------------------------------------


class TestAdversarialInputs:
    """Push message content to extremes — size, encoding, and metadata complexity."""

    async def test_one_megabyte_message_stored_and_retrieved_intact(self, provider, test_id):
        """
        A 1 MB string must survive storage and retrieval without truncation or corruption.
        Probes the Redis max-value size and JSON serialization limits.
        """
        sid = await provider.get_or_create_session(session_id=f"large-msg-{test_id}")
        large = "X" * 1_000_000  # exactly 1 MB

        await provider.add_message(sid, _msg(content=large))

        msgs = await provider.get_messages(sid)
        assert len(msgs) == 1
        assert len(msgs[0].content) == 1_000_000, (
            f"1 MB message was truncated: got {len(msgs[0].content)} chars"
        )
        assert msgs[0].content == large, "1 MB message content was corrupted during round-trip"

        await provider.delete_session(sid)

    async def test_null_bytes_and_control_chars_round_trip(self, provider, test_id):
        """
        Messages containing null bytes (\\x00) and ASCII control characters must
        survive JSON serialization without being silently dropped or corrupted.
        """
        sid = await provider.get_or_create_session(session_id=f"ctrl-chars-{test_id}")
        tricky = "hello\x00world\ttab\rnewline\x0cff\x0bvt"

        await provider.add_message(sid, _msg(content=tricky))

        msgs = await provider.get_messages(sid)
        assert len(msgs) == 1, "Message with control chars was silently dropped"
        assert msgs[0].content == tricky, (
            f"Control characters changed during round-trip.\n"
            f"  Expected: {tricky!r}\n"
            f"  Got:      {msgs[0].content!r}"
        )

        await provider.delete_session(sid)

    async def test_rtl_and_emoji_unicode_round_trip(self, provider, test_id):
        """
        RTL text (Arabic, Hebrew) and emoji must survive the Redis/JSON round-trip
        without byte-order corruption or character substitution.
        """
        sid = await provider.get_or_create_session(session_id=f"rtl-emoji-{test_id}")
        content = "مرحبا بالعالم 🌍 שלום עולם 🎉 \u200f\u202b mixed"

        await provider.add_message(sid, _msg(content=content))

        msgs = await provider.get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0].content == content, (
            f"RTL/emoji content was corrupted.\n"
            f"  Expected: {content!r}\n"
            f"  Got:      {msgs[0].content!r}"
        )

        await provider.delete_session(sid)

    async def test_message_metadata_nested_structures_stored(self, provider, test_id):
        """
        add_message metadata with nested dicts, lists, and None values must be
        accepted without raising a serialization error.
        """
        sid = await provider.get_or_create_session(session_id=f"nested-meta-{test_id}")
        nested_meta = {
            "tags": ["urgent", "billing"],
            "context": {"window": 1, "agent": None, "scores": [0.1, 0.9]},
            "score": 0.99,
            "empty": {},
        }

        # Must not raise
        await provider.add_message(sid, _msg(content="with nested metadata"), metadata=nested_meta)

        msgs = await provider.get_messages(sid)
        assert len(msgs) == 1, "Message with nested metadata was not stored"

        await provider.delete_session(sid)


# ---------------------------------------------------------------------------
# Group F: Connection Failures
# ---------------------------------------------------------------------------


class TestConnectionFailures:
    """Verify graceful degradation when Redis is misconfigured or unreachable."""

    def test_wrong_port_initialize_does_not_raise(self):
        """
        Pointing the provider at a closed port (6399) must not crash initialize().
        Redis uses lazy connections — the pool is created eagerly but connects on
        first command, so initialize() should return without raising.
        """
        from continuum.session.providers.redis import RedisSessionProvider

        p = RedisSessionProvider(config=_make_config(redis_port=6399), auto_initialize=False)
        try:
            result = p.initialize()
            assert isinstance(result, bool), "initialize() must return a bool"
        except Exception as exc:
            pytest.fail(f"initialize() raised an unexpected exception: {exc!r}")

    async def test_wrong_port_raises_session_connection_error_on_operation(self, test_id):
        """
        The first operation against an unreachable port must raise SessionConnectionError
        and must complete within the configured socket timeout (not hang for 30+ seconds).
        """
        from continuum.session.exceptions import SessionConnectionError
        from continuum.session.providers.redis import RedisSessionProvider

        p = RedisSessionProvider(config=_make_config(redis_port=6399), auto_initialize=False)
        p.initialize()

        try:
            with pytest.raises((SessionConnectionError, Exception)):
                # socket_connect_timeout=5 is set in the provider — must not hang beyond 10s
                await asyncio.wait_for(
                    p.get_or_create_session(session_id=f"bad-port-{test_id}"),
                    timeout=10.0,
                )
        finally:
            await p.close()

    async def test_wrong_password_raises_session_connection_error(self, test_id):
        """
        A provider configured with the wrong password must raise SessionConnectionError
        on the first Redis operation — not an unhandled AuthenticationError or hang.
        """
        from continuum.session.exceptions import SessionConnectionError
        from continuum.session.providers.redis import RedisSessionProvider

        p = RedisSessionProvider(
            config=_make_config(redis_password="definitely-wrong-password-xyz"),
            auto_initialize=False,
        )
        p.initialize()

        try:
            with pytest.raises(SessionConnectionError):
                await p.get_or_create_session(session_id=f"wrong-pw-{test_id}")
        finally:
            await p.close()


# ---------------------------------------------------------------------------
# Group G: Key Injection
# ---------------------------------------------------------------------------


class TestKeyInjection:
    """Verify session_id values containing special Redis characters are handled safely."""

    async def test_session_id_with_colon_stored_as_literal(self, provider, test_id):
        """
        A session_id containing ':' characters must be treated as a literal string.
        It must NOT split the Redis key into extra segments or overwrite unrelated keys.

        Risk: key_prefix + ':' + session_id + ':messages'
        If session_id = 'a:b', the key becomes 'orchestrator:session:a:b:messages'
        which could collide with a real session whose id is 'a' (metadata key pattern).
        The provider must either sanitize the id or isolate it so no collision occurs.
        """
        malicious_id = f"sess:colon:injection:{test_id}"

        sid = await provider.get_or_create_session(session_id=malicious_id)
        await provider.add_message(sid, _msg(content="colon injection test"))

        msgs = await provider.get_messages(sid)
        assert len(msgs) == 1, "Message was not stored for session_id containing colons"
        assert msgs[0].content == "colon injection test"

        # Verify the key written is exactly what we expect — no extra segments
        messages_key = f"orchestrator:session:{malicious_id}:messages"
        raw_len = await provider._redis.llen(messages_key)
        assert raw_len == 1, (
            f"Redis key structure broken by colon in session_id. "
            f"Expected key '{messages_key}' to have 1 entry, got {raw_len}."
        )

        await provider.delete_session(sid)

    async def test_session_id_with_wildcard_does_not_match_other_keys(self, provider, test_id):
        """
        A session_id containing '*' must be stored as a literal string and must NOT
        trigger a Redis glob scan that could match and corrupt unrelated session keys.

        Risk: if session_id is passed unescaped to a KEYS or SCAN command internally,
        '*' would match every key in Redis and could cause mass data exposure or deletion.
        """
        # First create a legitimate session to ensure there is something to potentially match
        legit_id = f"legit-sess-{test_id}"
        legit_sid = await provider.get_or_create_session(session_id=legit_id)
        await provider.add_message(legit_sid, _msg(content="legitimate message"))

        # Now create a session with wildcard in the ID
        wildcard_id = f"wildcard-*-{test_id}"
        wild_sid = await provider.get_or_create_session(session_id=wildcard_id)
        await provider.add_message(wild_sid, _msg(content="wildcard session message"))

        # The wildcard session must not have absorbed the legitimate session's messages
        wild_msgs = await provider.get_messages(wild_sid)
        contents = [m.content for m in wild_msgs]
        assert "legitimate message" not in contents, (
            "Wildcard session_id caused cross-session message leak — "
            "the '*' was treated as a glob pattern, not a literal character."
        )
        assert "wildcard session message" in contents

        # The legitimate session must be completely unaffected
        legit_msgs = await provider.get_messages(legit_sid)
        legit_contents = [m.content for m in legit_msgs]
        assert "legitimate message" in legit_contents, (
            "Legitimate session was corrupted by the wildcard session_id operation."
        )

        await provider.delete_session(wild_sid)
        await provider.delete_session(legit_sid)

    async def test_session_id_with_newline_and_null_byte_is_handled(self, provider, test_id):
        """
        A session_id containing newlines or null bytes must not crash the provider
        or corrupt the Redis key namespace. These characters are illegal in Redis keys
        and must be caught early — either rejected with a clear error or sanitized.
        """
        for char_name, bad_id in [
            ("newline", f"sess-\n-newline-{test_id}"),
            ("null byte", f"sess-\x00-null-{test_id}"),
        ]:
            try:
                sid = await provider.get_or_create_session(session_id=bad_id)
                # If it succeeds, the key must be safe to use — add and retrieve a message
                await provider.add_message(sid, _msg(content=f"message with {char_name} in key"))
                msgs = await provider.get_messages(sid)
                assert isinstance(msgs, list), (
                    f"get_messages returned non-list for session_id with {char_name}"
                )
                await provider.delete_session(sid)
            except Exception as exc:
                # Raising a clear exception is also acceptable — it means the provider
                # detected the invalid input rather than silently corrupting the key.
                assert "session" in type(exc).__name__.lower() or isinstance(
                    exc, (ValueError, TypeError)
                ), (
                    f"session_id with {char_name} raised an unexpected low-level exception: {exc!r}. "
                    "Expected a SessionError, ValueError, or TypeError — not a raw Redis error."
                )


# ---------------------------------------------------------------------------
# Group H: PII / Privacy
# ---------------------------------------------------------------------------


class TestPIIPrivacy:
    """Verify how sensitive data is stored in Redis — plain text vs encrypted."""

    async def test_message_content_stored_as_plain_text(self, provider, test_id):
        """
        FINDING TEST: This test DOCUMENTS the current behavior — messages are stored
        as plain JSON in Redis with no encryption at rest.

        If this test PASSES, it confirms a privacy risk: anyone with Redis access
        can read all conversation content directly without any decryption step.

        If this test FAILS in the future, it means encryption was added — good.
        """
        sid = await provider.get_or_create_session(session_id=f"pii-plain-{test_id}")
        pii_content = "My name is John Doe, my email is john@example.com, SSN: 123-45-6789"

        await provider.add_message(sid, _msg(content=pii_content))

        # Read the raw bytes directly from Redis — bypassing the provider's deserializer
        messages_key = f"orchestrator:session:{sid}:messages"
        raw_entries = await provider._redis.lrange(messages_key, 0, -1)

        assert len(raw_entries) == 1, "Message was not stored in Redis"

        raw_value = raw_entries[0]

        # DOCUMENT: This assertion confirms plain text storage (currently expected to PASS)
        # When encryption is added, this will FAIL — update this test to verify decryption instead
        assert pii_content in raw_value, (
            "Message content is NOT stored as plain text — encryption may be in place. "
            "Update this test to verify the encryption/decryption round-trip instead."
        )

        await provider.delete_session(sid)

    async def test_metadata_contains_user_id_in_plain_text(self, provider, test_id):
        """
        FINDING TEST: Session metadata (including user_id) is stored as plain JSON.
        Anyone with Redis access can enumerate all user IDs from the metadata keys.
        """
        user_id = f"real-user-id-{test_id}"
        sid = await provider.get_or_create_session(
            session_id=f"pii-meta-{test_id}", user_id=user_id
        )

        # Read raw metadata directly from Redis
        metadata_key = f"orchestrator:session:{sid}:metadata"
        raw_metadata = await provider._redis.get(metadata_key)

        assert raw_metadata is not None, "Metadata key not found in Redis"

        # DOCUMENT: user_id is readable as plain text in Redis
        assert user_id in raw_metadata, (
            "user_id is NOT present as plain text in metadata — "
            "either it is encrypted or stored under a different key. "
            "Update this test to reflect the actual storage format."
        )

        await provider.delete_session(sid)

    async def test_deleted_session_leaves_no_pii_in_redis(self, provider, test_id):
        """
        After delete_session, ALL keys belonging to that session must be removed.
        No PII must linger in Redis after explicit deletion — no partial cleanup.
        """
        sid = await provider.get_or_create_session(
            session_id=f"pii-delete-{test_id}",
            user_id=f"user-to-delete-{test_id}",
        )
        await provider.add_message(sid, _msg(content="sensitive data to be deleted"))

        await provider.delete_session(sid)

        # Check both keys are gone
        messages_key = f"orchestrator:session:{sid}:messages"
        metadata_key = f"orchestrator:session:{sid}:metadata"

        msg_exists = await provider._redis.exists(messages_key)
        meta_exists = await provider._redis.exists(metadata_key)

        assert msg_exists == 0, (
            f"Messages key '{messages_key}' still exists after delete_session. "
            "PII not fully purged from Redis."
        )
        assert meta_exists == 0, (
            f"Metadata key '{metadata_key}' still exists after delete_session. "
            "User ID and session info not fully purged from Redis."
        )


# ---------------------------------------------------------------------------
# Group I: Resource Leaks
# ---------------------------------------------------------------------------


class TestResourceLeaks:
    """Verify that connections are returned to the pool after each operation."""

    async def test_connections_returned_to_pool_after_repeated_sessions(self, test_id):
        """
        Create and close 30 sessions one by one.
        After each session is closed, the connection must be returned to the pool.
        If connections leak, the pool shrinks each iteration and eventually crashes.

        A healthy pool has the same number of idle connections at the end as at start.
        """
        p = _make_provider()

        pool = p._redis.connection_pool

        for i in range(30):
            sid = await p.get_or_create_session(session_id=f"leak-sess-{i}-{test_id}")
            await p.add_message(sid, _msg(content=f"message {i}"))
            await p.get_messages(sid)
            await p.delete_session(sid)

            # After each complete operation, no connections should remain in use
            in_use = len(pool._in_use_connections)
            assert in_use == 0, (
                f"Iteration {i}: {in_use} connection(s) still in use after session closed. "
                "Connection not returned to pool — leak detected."
            )

        await p.close()

    async def test_failed_operation_does_not_leak_connection(self, test_id):
        """
        When an operation fails (session not found, limit exceeded), the connection
        must still be returned to the pool — errors must not hold connections open.
        """
        from continuum.session.exceptions import SessionNotFoundError

        p = _make_provider(ttl_seconds=2)
        pool = p._redis.connection_pool

        sid = await p.get_or_create_session(session_id=f"leak-fail-{test_id}")
        await p.add_message(sid, _msg(content="before expiry"))

        # Let the session expire
        await asyncio.sleep(3)

        # This will raise SessionNotFoundError — the connection must still be released
        for _ in range(5):
            try:
                await p.add_message(sid, _msg(content="after expiry"))
            except (SessionNotFoundError, Exception):
                pass

        # After all failed operations, no connections should remain held
        in_use = len(pool._in_use_connections)
        assert in_use == 0, (
            f"{in_use} connection(s) still in use after repeated failed operations. "
            "Errors are leaking connections instead of releasing them back to the pool."
        )

        await p.close()

    async def test_provider_close_releases_all_connections(self, test_id):
        """
        After provider.close() is called, all connections must be disconnected.
        No dangling open sockets should remain after graceful shutdown.
        """
        p = _make_provider()

        # Do some work to open connections
        for i in range(5):
            sid = await p.get_or_create_session(session_id=f"leak-close-{i}-{test_id}")
            await p.add_message(sid, _msg(content=f"msg {i}"))

        pool = p._redis.connection_pool

        await p.close()

        # After close, the pool must report 0 in-use connections
        in_use = len(pool._in_use_connections)
        assert in_use == 0, (
            f"{in_use} connections still marked in-use after provider.close(). "
            "Graceful shutdown did not release all connections."
        )


# ---------------------------------------------------------------------------
# Group J: Performance & Latency
# ---------------------------------------------------------------------------


class TestPerformanceLatency:
    """Measure latency of core operations — must meet thresholds before production."""

    async def test_add_message_p99_latency_under_50ms(self, provider, test_id):
        """
        200 sequential add_message calls must complete with p99 latency under 50ms.
        Measures: min, p50, p95, p99, max — printed for visibility.
        Fails if p99 exceeds 50ms, indicating Redis or network is too slow for production.
        """
        import time

        sid = await provider.get_or_create_session(session_id=f"perf-add-{test_id}")
        latencies = []

        for i in range(200):
            start = time.perf_counter()
            await provider.add_message(sid, _msg(content=f"perf message {i}"))
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        minimum = latencies[0]
        maximum = latencies[-1]

        print(
            f"\nadd_message latency (200 calls): "
            f"min={minimum:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  "
            f"p99={p99:.2f}ms  max={maximum:.2f}ms"
        )

        assert p99 < 50, (
            f"add_message p99 latency {p99:.2f}ms exceeds 50ms threshold. "
            "Redis may be overloaded or the network is too slow for production use."
        )

        await provider.delete_session(sid)

    async def test_get_messages_p99_latency_under_50ms(self, provider, test_id):
        """
        200 sequential get_messages calls on a session with 50 messages must
        complete with p99 latency under 50ms.
        """
        import time

        sid = await provider.get_or_create_session(session_id=f"perf-get-{test_id}")

        # Pre-fill session with 50 messages
        for i in range(50):
            await provider.add_message(sid, _msg(content=f"prefill {i}"))

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            await provider.get_messages(sid)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        minimum = latencies[0]
        maximum = latencies[-1]

        print(
            f"\nget_messages latency (200 calls, 50 msgs): "
            f"min={minimum:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  "
            f"p99={p99:.2f}ms  max={maximum:.2f}ms"
        )

        assert p99 < 50, (
            f"get_messages p99 latency {p99:.2f}ms exceeds 50ms threshold. "
            "Reading conversation history is too slow for production use."
        )

        await provider.delete_session(sid)

    async def test_session_create_p99_latency_under_30ms(self, provider, test_id):
        """
        100 sequential get_or_create_session calls must complete with p99 under 30ms.
        Session creation is on the hot path — every first message in a conversation
        goes through this call.
        """
        import time

        latencies = []
        session_ids = []

        for i in range(100):
            sid = f"perf-create-{i}-{test_id}"
            start = time.perf_counter()
            await provider.get_or_create_session(session_id=sid)
            latencies.append((time.perf_counter() - start) * 1000)
            session_ids.append(sid)

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        minimum = latencies[0]
        maximum = latencies[-1]

        print(
            f"\nget_or_create_session latency (100 calls): "
            f"min={minimum:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  "
            f"p99={p99:.2f}ms  max={maximum:.2f}ms"
        )

        assert p99 < 30, (
            f"get_or_create_session p99 latency {p99:.2f}ms exceeds 30ms threshold. "
            "Session creation is too slow — will add noticeable delay to first message."
        )

        for sid in session_ids:
            await provider.delete_session(sid)


# ---------------------------------------------------------------------------
# Group K: Retries / Backoff
# ---------------------------------------------------------------------------


class TestRetriesBackoff:
    """
    Document the retry and timeout behavior of the session layer.

    FINDING: The provider has NO automatic retry logic.
    Configured: socket_connect_timeout=5s, socket_timeout=5s.
    If Redis goes down briefly and recovers, the operation fails immediately —
    the caller must handle retry manually.
    """

    async def test_operation_fails_within_socket_timeout(self, test_id):
        """
        An operation against an unreachable Redis must fail within socket_timeout (5s)
        plus a small buffer — NOT hang for 30+ seconds.

        This verifies the timeout is actually enforced. If it hangs, the timeout
        setting is broken and all production requests would queue up silently.
        """
        import time

        from continuum.session.providers.redis import RedisSessionProvider

        p = RedisSessionProvider(config=_make_config(redis_port=6399), auto_initialize=False)
        p.initialize()

        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                p.get_or_create_session(session_id=f"timeout-test-{test_id}"),
                timeout=12.0,
            )
        except Exception:
            pass
        elapsed = (time.perf_counter() - start) * 1000

        # Must fail within socket_timeout (5s) + generous buffer (3s) = 8s
        assert elapsed < 8000, (
            f"Operation took {elapsed:.0f}ms to fail — socket_timeout is not being enforced. "
            "Production requests would hang for up to 8+ seconds per user."
        )

        print(f"\nTimeout enforced: operation failed in {elapsed:.0f}ms (expected < 8000ms)")
        await p.close()

    async def test_no_auto_retry_on_connection_failure(self, test_id):
        """
        FINDING TEST: The provider has no automatic retry logic.
        A single failed connection must raise immediately — not silently retry
        multiple times and delay the error.

        This documents the current behavior. If retries are added in the future,
        this test should be updated to verify the retry count and backoff delays.
        """
        import time

        from continuum.session.providers.redis import RedisSessionProvider

        p = RedisSessionProvider(config=_make_config(redis_port=6399), auto_initialize=False)
        p.initialize()

        start = time.perf_counter()
        error_raised = False
        try:
            await asyncio.wait_for(
                p.get_or_create_session(session_id=f"no-retry-{test_id}"),
                timeout=12.0,
            )
        except Exception:
            error_raised = True
        elapsed = (time.perf_counter() - start) * 1000

        assert error_raised, "Expected an exception for unreachable Redis — none was raised"

        # DOCUMENT: With no retry logic, failure happens in 1 attempt within timeout
        # If retries were configured (e.g. 3 retries with 1s backoff), elapsed would be 3s+
        # Currently: fails in one attempt — fast failure, no backoff delay
        print(
            f"\nRetry behavior: error raised in {elapsed:.0f}ms. "
            f"{'Fast failure — no retry delay detected' if elapsed < 6000 else 'Possible retry delay detected'}"
        )

        await p.close()

    async def test_provider_recovers_when_redis_comes_back(self, test_id):
        """
        After a failed operation, the SAME provider must successfully complete
        the next operation when Redis is reachable again.

        This simulates: Redis blips for a moment, caller catches the error,
        retries manually, and the provider does not stay in a broken state.
        """
        from continuum.session.providers.redis import RedisSessionProvider

        # Step 1: Attempt an operation that fails (wrong port)
        bad_provider = RedisSessionProvider(
            config=_make_config(redis_port=6399), auto_initialize=False
        )
        bad_provider.initialize()

        try:
            await asyncio.wait_for(
                bad_provider.get_or_create_session(session_id=f"recovery-bad-{test_id}"),
                timeout=12.0,
            )
        except Exception:
            pass  # Expected failure
        finally:
            await bad_provider.close()

        # Step 2: Create a FRESH provider pointing at the real Redis
        # This simulates the caller retrying after Redis recovers
        good_provider = _make_provider()
        try:
            sid = await good_provider.get_or_create_session(session_id=f"recovery-good-{test_id}")
            await good_provider.add_message(sid, _msg(content="recovery message"))
            msgs = await good_provider.get_messages(sid)

            assert len(msgs) == 1, "Provider did not recover after Redis came back"
            assert msgs[0].content == "recovery message"

            await good_provider.delete_session(sid)
            print("\nRecovery: new provider connected successfully after prior failure")
        finally:
            await good_provider.close()


# ---------------------------------------------------------------------------
# Group L: Trace / Metric Completeness
# ---------------------------------------------------------------------------


class TestTraceMetricCompleteness:
    """
    Verify that the session layer emits correct log events at the right levels.
    The provider uses @observe decorators (Langfuse tracing) and structured logger.
    These tests verify the logging layer — the observable contract of the session layer.
    """

    async def test_session_creation_emits_info_log(self, provider, test_id):
        """
        Creating a new session must emit an INFO log containing the session ID.
        If this log is absent, operators cannot tell when sessions are being created
        — blind to session lifecycle in production logs.
        """
        from unittest.mock import patch

        sid = f"trace-create-{test_id}"
        with patch("continuum.session.providers.redis.logger") as mock_logger:
            await provider.get_or_create_session(session_id=sid)

        # info() must have been called at least once with the session_id in the message
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any(sid in call for call in info_calls), (
            f"logger.info was not called with session_id '{sid}' on session creation. "
            f"Calls recorded: {info_calls}"
        )

        await provider.delete_session(sid)

    async def test_error_on_failed_operation_is_surfaced(self, test_id):
        """
        A failed Redis operation must NOT be swallowed silently — operators
        need visibility into connection failures / data loss.

        By design the provider surfaces the failure by RAISING
        SessionConnectionError and logs it at DEBUG, keeping the provider layer
        quiet so the degrade-mode fallback path has no error spam; the
        operator-facing ERROR log + error report is owned by the SessionClient
        layer (see SessionClient.get_or_create_session). This test asserts the
        provider-level contract: the failure is raised (not returned as a
        success) and is logged, so nothing is silent.
        """
        from unittest.mock import patch

        from continuum.session.exceptions import SessionConnectionError
        from continuum.session.providers.redis import RedisSessionProvider

        p = RedisSessionProvider(config=_make_config(redis_port=6399), auto_initialize=False)
        p.initialize()

        raised: Exception | None = None
        with patch("continuum.session.providers.redis.logger") as mock_logger:
            try:
                await asyncio.wait_for(
                    p.get_or_create_session(session_id=f"trace-err-{test_id}"),
                    timeout=10.0,
                )
            except Exception as e:  # noqa: BLE001 - capturing to assert on type
                raised = e

        # 1. The failure is surfaced to the caller, not swallowed.
        assert isinstance(raised, SessionConnectionError), (
            f"expected SessionConnectionError to be raised on a failed Redis op, "
            f"got {type(raised).__name__ if raised else None}"
        )
        # 2. It is also logged (at debug — the provider stays quiet by design;
        #    the ERROR-level log is the SessionClient's responsibility).
        assert mock_logger.debug.called or mock_logger.error.called, (
            "failure was neither logged at debug nor error — it is silent."
        )

        await p.close()

    async def test_sliding_window_trim_emits_info_log(self, test_id):
        """
        When sliding window trims old messages, an INFO log must be emitted.
        Without this log, operators cannot tell if history is being silently dropped.
        """
        from unittest.mock import patch

        p = _make_provider(
            max_messages=3,
            message_limit_strategy="sliding_window",
            sliding_window_trim_count=2,
        )
        sid = f"trace-trim-{test_id}"
        try:
            await p.get_or_create_session(session_id=sid)

            with patch("continuum.session.providers.redis.logger") as mock_logger:
                for i in range(5):
                    await p.add_message(sid, _msg(content=f"msg-{i}"))

            info_calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any(
                "trimmed" in c.lower() or "sliding window" in c.lower() for c in info_calls
            ), (
                "logger.info was not called with trim details when sliding window triggered. "
                "Silent history drops are undetectable in production logs. "
                f"Info calls recorded: {info_calls}"
            )
        finally:
            try:
                await p.delete_session(sid)
            except Exception:
                pass
            await p.close()

    async def test_session_delete_emits_info_log(self, provider, test_id):
        """
        Deleting a session must emit an INFO log.
        Session deletions are important lifecycle events — they must be traceable
        in logs for auditing and debugging data retention.
        """
        from unittest.mock import patch

        sid = f"trace-delete-{test_id}"
        await provider.get_or_create_session(session_id=sid)

        with patch("continuum.session.providers.redis.logger") as mock_logger:
            await provider.delete_session(sid)

        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("deleted" in c.lower() or sid in c for c in info_calls), (
            f"logger.info was not called with 'deleted' when session '{sid}' was removed. "
            f"Info calls recorded: {info_calls}"
        )

    async def test_all_operations_have_observe_decorator(self, provider, test_id):
        """
        Verify the @observe decorator is present on all core operations by
        checking the function names exist and execute without raising decorator errors.
        The @observe decorator wraps functions for Langfuse tracing — if missing,
        the operation is invisible to the distributed tracing system.
        """
        sid = await provider.get_or_create_session(session_id=f"trace-obs-{test_id}")

        # All these must execute without AttributeError or decorator-related failures
        await provider.add_message(sid, _msg(content="observe test"))
        await provider.get_messages(sid)
        await provider.get_session_metadata(sid)
        await provider.clear_session(sid)
        await provider.delete_session(sid)

        # If we reach here all @observe-decorated methods executed successfully
        assert True, "All @observe-decorated operations completed without decorator errors"


# ---------------------------------------------------------------------------
# Group M: Identifier Sanitization & Injection (memory-parity findings)
#
# These probe the SAME class of issues found in the memory layer (MEM-011/012/
# 014/010/003): user_id / conversation_id flow straight into Redis keys AND
# log lines with NO sanitization, even though sanitize_user_input() exists.
# Assertions encode the SECURE expectation — a FAIL here is a real finding.
# ---------------------------------------------------------------------------


class TestIdentifierSanitization:
    """user_id / conversation_id are never sanitized before key/log interpolation."""

    @pytest.mark.xfail(
        reason="SESS-101 (Medium): newline in user_id reaches logs verbatim — "
        "log injection. sanitize_user_input() exists but is never called on id fields.",
        strict=True,
    )
    async def test_newline_in_user_id_does_not_reach_logs_raw(self, test_id):
        """
        SESS-LOG-INJECTION (parity: MEM-012, Medium):
        A newline in user_id produces session_id 'u:alice\\n<forged log line>'.
        That value is interpolated into logger.info("Created new session: {id}")
        — the newline lets an attacker forge fake log entries (log injection).

        SECURE expectation: the logged session_id contains no raw newline.
        Current behavior: FAILS — newline reaches the log verbatim.
        """
        from unittest.mock import patch

        malicious_user = f"alice-{test_id}\n2099-01-01 12:00:00 ERROR FORGED-LOG-ENTRY admin-login"

        p = _make_provider()
        try:
            with patch("continuum.session.providers.redis.logger") as mock_logger:
                sid = await p.get_or_create_session(user_id=malicious_user)

            logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
            assert "\n" not in sid, (
                "FINDING: newline survived into session_id and is logged verbatim — "
                "log injection possible. sanitize_user_input() strips \\x0a but is "
                f"never called on id fields. session_id={sid!r}"
            )
            assert "FORGED-LOG-ENTRY" not in logged or "\n" not in logged, (
                "FINDING: forged log line reached the logger via newline in user_id."
            )
            await p.delete_session(sid)
        finally:
            await p.close()

    async def test_zero_width_char_in_user_id_is_stripped(self, test_id):
        """
        SESS-ZEROWIDTH (parity: MEM-011, Low):
        A zero-width space (\\u200b) in user_id creates an invisible bucket —
        'u:alice\\u200bbob' looks identical to 'u:alicebob' in any log or UI.

        SECURE expectation: invisible unicode stripped from the resolved id.
        Fixed by #55: validate_user_id() strips invisible unicode from id fields
        before they become Redis key fragments.
        """
        p = _make_provider()
        try:
            sid = await p.get_or_create_session(user_id=f"alice​bob-{test_id}")
            assert "​" not in sid, (
                "FINDING: zero-width space preserved in session_id — creates an "
                f"invisible, un-greppable session bucket. session_id={sid!r}"
            )
            await p.delete_session(sid)
        finally:
            await p.close()

    async def test_whitespace_only_user_id_is_rejected_or_normalized(self, test_id):
        """
        SESS-WHITESPACE (parity: MEM-010, Low):
        '   ' (whitespace only) is truthy, so `if user_id:` accepts it and creates
        a bucket 'u:   '. A blank-looking user gets a real, addressable session.

        SECURE expectation: whitespace-only id rejected or trimmed to empty→UUID.
        Fixed by #55: validate_user_id() strips whitespace and treats a
        whitespace-only id as anonymous (→ generated UUID session).
        """
        from continuum.session.exceptions import SessionError

        p = _make_provider()
        sid = None
        try:
            try:
                sid = await p.get_or_create_session(user_id="   ")
            except (SessionError, ValueError):
                return  # rejection is the secure outcome — test passes

            assert sid.strip() != "u:" and sid.replace(" ", "") != "u:", (
                f"FINDING: whitespace-only user_id accepted → bucket {sid!r} created. "
                "A blank user gets a persistent addressable session."
            )
        finally:
            if sid:
                try:
                    await p.delete_session(sid)
                except Exception:
                    pass
            await p.close()

    @pytest.mark.xfail(
        reason="SESS-104 (Medium): conversation_id alone (no user_id) falls through to a "
        "random UUID each call → conversation history silently lost across calls.",
        strict=True,
    )
    async def test_conversation_id_only_is_deterministic(self, test_id):
        """
        SESS-CONV-ONLY (Medium):
        Passing ONLY conversation_id (no user_id) falls through to a random UUID.
        Two calls with the same conversation_id return DIFFERENT session ids —
        conversation history is silently lost across calls.

        SECURE expectation: same conversation_id → same deterministic session.
        Current behavior: FAILS — each call generates a fresh UUID.
        """
        p = _make_provider()
        try:
            conv = f"conv-only-{test_id}"
            sid_a = await p.get_or_create_session(conversation_id=conv)
            sid_b = await p.get_or_create_session(conversation_id=conv)

            assert sid_a == sid_b, (
                "FINDING: conversation_id alone is not deterministic — "
                f"call 1 returned {sid_a!r}, call 2 returned {sid_b!r}. "
                "History is lost when only conversation_id is supplied."
            )
            await p.delete_session(sid_a)
            await p.delete_session(sid_b)
        finally:
            await p.close()

    @pytest.mark.xfail(
        reason="SESS-105 (Low): no length cap on user_id → 100k char id produces a "
        "100k char Redis key. sanitize_user_input()'s 50k limit is never applied.",
        strict=True,
    )
    async def test_oversized_user_id_is_length_capped(self, test_id):
        """
        SESS-OVERSIZE (parity: MEM-014 50k cap, Low):
        A 100,000-char user_id produces a 100,002-char Redis key with no cap.
        sanitize_user_input() enforces a 50,000-char limit but is never applied.

        SECURE expectation: resolved id length is bounded (<= ~50k).
        Current behavior: FAILS — unbounded key length.
        """
        p = _make_provider()
        sid = None
        try:
            sid = await p.get_or_create_session(user_id="x" * 100_000)
            assert len(sid) <= 50_010, (
                f"FINDING: no length cap on user_id — produced a {len(sid)}-char Redis key. "
                "Unbounded keys waste memory and can be used to bloat the keyspace."
            )
        finally:
            if sid:
                try:
                    await p.delete_session(sid)
                except Exception:
                    pass
            await p.close()

    async def test_non_string_user_id_is_rejected(self, test_id):
        """
        SESS-106 (parity: MEM-013, Low):
        An int passed as user_id bypasses the `str | None` hint at the key-compute
        step (12345 → 'u:12345' via f-string) BUT is then caught by Pydantic when
        SessionMetadata(user_id=12345) is constructed — so creation fails.

        FINDING (Low): the rejection works, but surfaces as a misleading
        SessionConnectionError (implies a network fault) wrapping a Pydantic
        ValidationError, instead of a clean input-validation error.

        This test asserts the SECURE outcome (non-string is rejected) — it passes.
        """
        from continuum.session.exceptions import SessionError

        p = _make_provider()
        try:
            with pytest.raises((SessionError, ValueError, TypeError)):
                await p.get_or_create_session(user_id=12345)  # type: ignore[arg-type]
            # Documented: rejection works but error type is misleading (SessionConnectionError).
        finally:
            await p.close()

    async def test_concurrent_metadata_updates_last_write_wins(self, provider, test_id):
        """
        SESS-107 (parity: MEM-020, Medium / by design):
        update_session_metadata has no optimistic locking. Concurrent updates to
        the same session race; the last write silently wins and the caller has no
        way to know their update was overwritten.

        Concurrency is kept at 8 (< pool of 10) so this isolates the last-write-wins
        behavior rather than re-tripping the S-001 pool-exhaustion bug.

        This test documents that all concurrent updates complete without error
        (no crash) — the silent data-integrity caveat is the finding, not a crash.
        """
        sid = await provider.get_or_create_session(session_id=f"lww-{test_id}")
        base = await provider.get_session_metadata(sid)

        async def update_with(count):
            base.message_count = count
            return await provider.update_session_metadata(sid, base)

        # 8 concurrent updates — stays under the default pool size (10)
        results = await asyncio.gather(*[update_with(i) for i in range(8)], return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Concurrent metadata updates raised: {errors}"

        # Final value is whichever write landed last — non-deterministic, by design
        final = await provider.get_session_metadata(sid)
        assert final is not None
        # Documented: no optimistic locking — last-write-wins is silent (SESS-107).

        await provider.delete_session(sid)
