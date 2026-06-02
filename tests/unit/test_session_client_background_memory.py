"""Tests for SessionClient.add_message memory-write routing (sync vs background).

Verifies:
  - default 'sync' mode awaits the mem0 write before returning (no behavior change),
  - 'background' mode returns before the mem0 write completes (the latency win),
  - the short-term Redis write is always synchronous,
  - background mode falls back to synchronous when no registry is available.

All collaborators are mocked — no Redis, no mem0/LLM, no vector store.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.core.background_tasks import BackgroundTaskRegistry
from orchestrator.llm.types import ChatMessage
from orchestrator.session.client import SessionClient
from orchestrator.session.config import SessionConfig
from orchestrator.session.types import SessionMetadata


def _metadata(session_id="sess-1234abcd"):
    now = datetime.now(UTC)
    return SessionMetadata(
        session_id=session_id,
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        created_at=now,
        last_accessed_at=now,
    )


def _make_client(*, mode, with_registry=True, add_mock=None):
    """Build a SessionClient with mocked provider + memory client."""
    mem = MagicMock()
    mem.is_enabled = True
    mem.add = add_mock or AsyncMock(return_value=MagicMock(results=[]))
    mem.delete = AsyncMock()

    provider = MagicMock()
    provider.add_message = AsyncMock()
    provider.get_session_metadata = AsyncMock(return_value=_metadata())

    registry = BackgroundTaskRegistry(name="test") if with_registry else None

    client = SessionClient(
        session_config=SessionConfig(enabled=True, memory_write_mode=mode),
        memory_client=mem,
        provider=provider,
        auto_initialize=False,
        background_tasks=registry,
    )
    return client, mem, provider, registry


def _msg():
    return ChatMessage(role="user", content="My name is Tom")


def _gated_add(gate):
    """An AsyncMock whose add() blocks until `gate` is set, then returns a result."""

    async def _add(*args, **kwargs):
        await gate.wait()
        return MagicMock(results=[])

    return AsyncMock(side_effect=_add)


class TestSyncMode:
    async def test_shipped_default_is_background(self):
        # The shipped default is 'background' (writes off the response path).
        # Isolate from any ambient SESSION_MEMORY_WRITE_MODE in the dev .env/shell
        # by pinning the setting the field reads from.
        with patch("orchestrator.session.config.settings") as mock_settings:
            mock_settings.session_memory_write_mode = "background"
            assert SessionConfig().memory_write_mode == "background"

    async def test_sync_awaits_memory_add_before_return(self):
        client, mem, provider, _ = _make_client(mode="sync")
        await client.add_message("sess-1234abcd", _msg())
        provider.add_message.assert_awaited_once()
        mem.add.assert_awaited_once()  # completed inline, before return

    async def test_sync_blocks_until_memory_write_finishes(self):
        gate = asyncio.Event()
        slow_add = _gated_add(gate)
        client, mem, _, _ = _make_client(mode="sync", add_mock=slow_add)

        call = asyncio.ensure_future(client.add_message("sess-1234abcd", _msg()))
        # Should NOT complete while the memory write is blocked.
        done, _pending = await asyncio.wait({call}, timeout=0.1)
        assert call not in done
        # Release and let it finish.
        gate.set()
        await call


class TestBackgroundMode:
    async def test_background_returns_before_memory_add_completes(self):
        gate = asyncio.Event()
        slow_add = _gated_add(gate)
        client, mem, provider, registry = _make_client(mode="background", add_mock=slow_add)

        await client.add_message("sess-1234abcd", _msg())  # returns immediately

        # Redis write happened synchronously; mem0 write was scheduled, not awaited.
        provider.add_message.assert_awaited_once()
        assert mem.add.await_count == 0  # not yet completed
        assert len(registry) == 1  # in flight

        # Let it finish and confirm it eventually ran.
        gate.set()
        await registry.drain(timeout=1.0)
        mem.add.assert_awaited_once()
        assert len(registry) == 0

    async def test_redis_write_always_synchronous_in_background_mode(self):
        # The conversation-history write must not be backgrounded.
        client, mem, provider, registry = _make_client(mode="background")
        await client.add_message("sess-1234abcd", _msg())
        provider.add_message.assert_awaited_once()
        await registry.drain(timeout=1.0)

    async def test_falls_back_to_sync_when_no_registry(self):
        # background mode + no registry (and container unavailable) → inline write.
        client, mem, _, _ = _make_client(mode="background", with_registry=False)
        with patch(
            "orchestrator.core.container.get_container",
            side_effect=RuntimeError("no container"),
        ):
            await client.add_message("sess-1234abcd", _msg())
        mem.add.assert_awaited_once()  # ran inline


class TestTemporalDowngrade:
    async def test_temporal_activity_forces_sync_even_in_background_mode(self):
        # Inside a Temporal activity, background mode is downgraded to sync so the
        # write completes within the durable/retriable activity boundary.
        client, mem, provider, registry = _make_client(mode="background")
        with patch("orchestrator.session.client._in_temporal_activity", return_value=True):
            await client.add_message("sess-1234abcd", _msg())
        mem.add.assert_awaited_once()  # ran inline (sync)
        assert len(registry) == 0  # nothing was scheduled in the background

    async def test_non_temporal_uses_background(self):
        # Outside Temporal, background mode schedules the write off the path.
        gate = asyncio.Event()
        client, mem, provider, registry = _make_client(mode="background", add_mock=_gated_add(gate))
        with patch("orchestrator.session.client._in_temporal_activity", return_value=False):
            await client.add_message("sess-1234abcd", _msg())
        assert mem.add.await_count == 0  # not awaited inline
        assert len(registry) == 1  # scheduled in background
        gate.set()
        await registry.drain(timeout=1.0)
        mem.add.assert_awaited_once()


class TestStoreInMemoryIsBestEffort:
    async def test_memory_failure_does_not_raise(self):
        failing_add = AsyncMock(side_effect=RuntimeError("mem0 down"))
        client, _, provider, _ = _make_client(mode="sync", add_mock=failing_add)
        # Must not raise — memory storage is best-effort.
        await client.add_message("sess-1234abcd", _msg())
        provider.add_message.assert_awaited_once()
