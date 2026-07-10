"""Integration: memory-client wiring through the DI container.

Covers the container-level contracts the unit tests can't:
  1. A background memory write scheduled through the container-built SessionClient
     lands on the container's shared BackgroundTaskRegistry.
  2. container.shutdown() drains in-flight background memory writes BEFORE closing
     the memory client (so a write scheduled in 'background' mode is not lost),
     and stays clean even when a pending write errors.
  3. A container-derived memory client is resolved LIVE on every use, so
     Container.set_memory_client() applied after a SessionClient exists takes
     effect (TL-82 fix); an explicitly injected client is trusted for life.

Uses mocked provider + memory client — no Redis, no mem0/LLM, no vector store.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.core.container import (
    Container,
    ContainerConfig,
    get_container,
    reset_container,
)
from continuum.llm.types import ChatMessage
from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.types import SessionMetadata

pytestmark = pytest.mark.integration


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


def _mock_memory_client(add_mock=None):
    mem = MagicMock()
    mem.is_enabled = True
    mem.add = add_mock or AsyncMock(return_value=MagicMock(results=[]))
    mem.search = AsyncMock(return_value=MagicMock(results=[]))
    mem.delete = AsyncMock()
    mem.close = AsyncMock()
    return mem


def _mock_provider():
    provider = MagicMock()
    provider.add_message = AsyncMock()
    provider.get_session_metadata = AsyncMock(return_value=_metadata())
    provider.close = AsyncMock()
    return provider


def _msg():
    return ChatMessage(role="user", content="My name is Tom")


class TestContainerWiring:
    async def test_container_injects_registry_into_session_client(self):
        # Behavioral (not a private-attribute check): a memory write scheduled in
        # background mode through the container-built session client must land on
        # the container's own shared BackgroundTaskRegistry.
        reset_container()
        container = get_container(
            ContainerConfig(
                enable_memory=True,
                enable_session=True,
                session_config={"enabled": True, "memory_write_mode": "background"},
            )
        )
        container.set_memory_client(_mock_memory_client())
        try:
            session_client = container.session_client
            assert session_client is not None
            # Inject a mock provider onto the container-built singleton (public
            # API) so ops run without Redis.
            session_client.set_provider(_mock_provider())

            await session_client.add_message("sess-1234abcd", _msg())

            # The background mem0 write was scheduled on the container's registry.
            assert len(container.background_tasks) == 1
            await container.background_tasks.drain(timeout=5.0)
            assert len(container.background_tasks) == 0
        finally:
            reset_container()


class TestShutdownDrain:
    async def test_pending_background_write_completes_on_shutdown(self):
        # Ordering + no-hang: an in-flight background write must be drained to
        # completion BEFORE the memory client is closed. The gate is released
        # shortly after shutdown starts, and shutdown is bounded so a drain
        # regression fails loudly instead of hanging CI.
        events: list[str] = []
        gate = asyncio.Event()

        async def _gated_add(*args, **kwargs):
            await gate.wait()
            events.append("add_done")
            return MagicMock(results=[])

        async def _close():
            events.append("close")

        container = Container(ContainerConfig())
        mem = _mock_memory_client(add_mock=AsyncMock(side_effect=_gated_add))
        mem.close = AsyncMock(side_effect=_close)
        container.set_memory_client(mem)

        # Build a session client on the container's shared registry (background mode).
        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="background"),
            memory_client=mem,
            provider=_mock_provider(),
            auto_initialize=False,
            background_tasks=container.background_tasks,
        )
        container.set_session_client(session_client)

        try:
            await session_client.add_message("sess-1234abcd", _msg())

            # Returned before the gated write ran — it's in flight on the registry.
            assert mem.add.await_count == 0
            assert len(container.background_tasks) == 1

            loop = asyncio.get_running_loop()
            loop.call_later(0.05, gate.set)  # release shortly after shutdown starts

            # Shutdown must drain the in-flight write before closing the client.
            await asyncio.wait_for(container.shutdown(), timeout=5)

            # The write finished, and it finished BEFORE the client was closed.
            assert events == ["add_done", "close"]
            assert mem.add.await_count == 1
            assert mem.close.await_count == 1
            assert len(container.background_tasks) == 0
        finally:
            container.reset()

    async def test_shutdown_stays_clean_when_background_write_raises(self):
        # A pending background write that raises must not break shutdown: drain
        # swallows/observes the failure, and the memory client is still closed
        # exactly once.
        gate = asyncio.Event()

        async def _boom(*args, **kwargs):
            await gate.wait()
            raise RuntimeError("mem0 add failed")

        container = Container(ContainerConfig())
        mem = _mock_memory_client(add_mock=AsyncMock(side_effect=_boom))
        container.set_memory_client(mem)

        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="background"),
            memory_client=mem,
            provider=_mock_provider(),
            auto_initialize=False,
            background_tasks=container.background_tasks,
        )
        container.set_session_client(session_client)

        try:
            await session_client.add_message("sess-1234abcd", _msg())
            assert len(container.background_tasks) == 1

            loop = asyncio.get_running_loop()
            loop.call_later(0.05, gate.set)  # let the in-flight write reach its raise

            # Must complete without propagating the background write's error.
            await asyncio.wait_for(container.shutdown(), timeout=5)

            assert mem.close.await_count == 1
            assert len(container.background_tasks) == 0
        finally:
            container.reset()

    async def test_shutdown_with_no_pending_writes_is_clean(self):
        container = Container(ContainerConfig())
        container.set_memory_client(_mock_memory_client())
        try:
            # No writes scheduled — shutdown should not error or hang.
            await container.shutdown()
            assert len(container.background_tasks) == 0
        finally:
            container.reset()


class TestMemoryClientLiveResolution:
    """Regression guard: a SessionClient that resolved its memory client from
    the container must NOT cache it at construction time.

    The bug this guards against: Container.set_memory_client() applied AFTER a
    SessionClient was already built silently had no effect — the SessionClient
    kept using the stale memory client captured when it was first initialized,
    so session memory reads/writes hit the OLD backend with no error or warning.

    These tests assert on observable behavior (which mock's add()/search() is
    actually invoked), not on private attributes, so the contract survives
    internal refactors and can't be quietly weakened.

    SessionClient resolves the container-derived memory client through the
    GLOBAL get_container(), so the autouse fixture resets the global container
    around each test.
    """

    @pytest.fixture(autouse=True)
    def _reset_global_container(self):
        reset_container()
        yield
        reset_container()

    async def test_bug_report_repro_through_container_singleton(self):
        """The exact bug-report sequence, through the cached
        container.session_client singleton — no manually constructed
        SessionClient. Fails against the pre-fix caching impl: the singleton
        would keep using the memory client captured when it was first built."""
        container = get_container(
            ContainerConfig(
                enable_memory=True,
                enable_session=True,
                session_config={"enabled": True, "memory_write_mode": "sync"},
            )
        )
        old_mem = _mock_memory_client()
        container.set_memory_client(old_mem)

        # The cached singleton. Inject a mock provider (public API) so ops run
        # without Redis; the memory client is left to resolve from the container.
        session_client = container.session_client
        session_client.set_provider(_mock_provider())

        await session_client.add_message("sess-1234abcd", _msg())
        assert old_mem.add.await_count == 1

        # Swap the container's memory client after the singleton was built + used.
        new_mem = _mock_memory_client()
        container.set_memory_client(new_mem)

        # Same singleton instance now reflects the swapped client...
        assert container.session_client.memory_client is new_mem
        # ...and routes subsequent writes to it, not the stale one.
        await container.session_client.add_message("sess-1234abcd", _msg())
        assert new_mem.add.await_count == 1
        assert old_mem.add.await_count == 1  # unchanged — no longer used

    async def test_swap_after_build_routes_writes_to_new_client(self):
        """Fails against the pre-fix impl: after the swap the write would still
        go to the memory client cached at construction time."""
        container = get_container(ContainerConfig(enable_memory=True))
        old_mem = _mock_memory_client()
        container.set_memory_client(old_mem)

        # Built WITHOUT an explicit memory client → resolves from the container
        # (the default/common path). Sync write mode keeps the mem0 write inline
        # so the assertion doesn't race a background task.
        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="sync"),
            provider=_mock_provider(),
            auto_initialize=True,
        )
        # First write goes to the client the container had at build time.
        await session_client.add_message("sess-1234abcd", _msg())
        assert old_mem.add.await_count == 1

        # Swap the container's memory client AFTER the session client was
        # built and already used.
        new_mem = _mock_memory_client()
        container.set_memory_client(new_mem)

        # The next write must route to the NEW client — proving the session
        # client re-resolves live instead of using the stale reference.
        await session_client.add_message("sess-1234abcd", _msg())
        assert new_mem.add.await_count == 1
        assert old_mem.add.await_count == 1  # unchanged — no longer used

    async def test_explicit_memory_client_is_not_overridden_by_container(self):
        # The other half of the contract: an explicitly injected memory client
        # is trusted for the client's lifetime and is NEVER swapped out by the
        # container — this is the pre-existing behavior and must not regress.
        # (Passes both pre- and post-fix; guards against over-correcting.)
        container = get_container(ContainerConfig(enable_memory=True))
        container.set_memory_client(_mock_memory_client())

        explicit_mem = _mock_memory_client()
        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="sync"),
            memory_client=explicit_mem,
            provider=_mock_provider(),
            auto_initialize=True,
        )
        await session_client.add_message("sess-1234abcd", _msg())
        # Writes go to the explicitly injected client, never the container's.
        assert explicit_mem.add.await_count == 1

        # Even after a container swap, the explicit client is still honored.
        container.set_memory_client(_mock_memory_client())
        await session_client.add_message("sess-1234abcd", _msg())
        assert explicit_mem.add.await_count == 2

    async def test_swap_reflected_in_read_path(self):
        """Fails against the pre-fix impl: the read path would keep searching the
        memory client cached at construction time."""
        container = get_container(ContainerConfig(enable_memory=True))
        old_mem = _mock_memory_client()
        container.set_memory_client(old_mem)

        session_client = SessionClient(
            session_config=SessionConfig(enabled=True),
            provider=_mock_provider(),
            auto_initialize=True,
        )
        await session_client.get_relevant_memories("sess-1234abcd", query="who am i")
        assert old_mem.search.await_count == 1

        new_mem = _mock_memory_client()
        container.set_memory_client(new_mem)

        await session_client.get_relevant_memories("sess-1234abcd", query="who am i")
        assert new_mem.search.await_count == 1
        assert old_mem.search.await_count == 1  # unchanged

    async def test_swap_reflected_in_background_write_mode(self):
        """Fails against the pre-fix impl: in background mode too, the write
        would resolve the stale cached client."""
        container = get_container(ContainerConfig(enable_memory=True))
        old_mem = _mock_memory_client()
        container.set_memory_client(old_mem)

        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="background"),
            provider=_mock_provider(),
            auto_initialize=True,
            background_tasks=container.background_tasks,
        )
        await session_client.add_message("sess-1234abcd", _msg())
        await container.background_tasks.drain(timeout=5.0)
        assert old_mem.add.await_count == 1

        new_mem = _mock_memory_client()
        container.set_memory_client(new_mem)

        await session_client.add_message("sess-1234abcd", _msg())
        await container.background_tasks.drain(timeout=5.0)
        assert new_mem.add.await_count == 1
        assert old_mem.add.await_count == 1  # unchanged

    async def test_write_path_skips_disabled_memory_client(self):
        # Guard coverage: a memory client with is_enabled=False must be skipped
        # on the write path — no add(), no raise — while the short-term session
        # write (provider.add_message) still succeeds.
        container = get_container(ContainerConfig(enable_memory=True))
        disabled = _mock_memory_client()
        disabled.is_enabled = False
        container.set_memory_client(disabled)

        provider = _mock_provider()
        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="sync"),
            provider=provider,
            auto_initialize=True,
        )
        # Must not raise.
        await session_client.add_message("sess-1234abcd", _msg())

        assert disabled.add.await_count == 0  # memory skipped
        assert provider.add_message.await_count == 1  # session write still happened

    async def test_read_path_returns_empty_with_disabled_memory_client(self):
        # Guard coverage on the read path: a disabled memory client makes
        # get_relevant_memories return an empty list without calling search()
        # (asserting the actual implementation behavior at client.py:823-825).
        container = get_container(ContainerConfig(enable_memory=True))
        disabled = _mock_memory_client()
        disabled.is_enabled = False
        container.set_memory_client(disabled)

        session_client = SessionClient(
            session_config=SessionConfig(enabled=True),
            provider=_mock_provider(),
            auto_initialize=True,
        )
        result = await session_client.get_relevant_memories("sess-1234abcd", query="q")

        assert result == []
        assert disabled.search.await_count == 0

    async def test_swap_to_none_disables_memory_writes_gracefully(self):
        """Fails against the pre-fix impl: the stale cached client would keep
        receiving writes after the container was set to None."""
        container = get_container(ContainerConfig(enable_memory=True))
        old_mem = _mock_memory_client()
        container.set_memory_client(old_mem)

        session_client = SessionClient(
            session_config=SessionConfig(enabled=True, memory_write_mode="sync"),
            provider=_mock_provider(),
            auto_initialize=True,
        )
        await session_client.add_message("sess-1234abcd", _msg())
        assert old_mem.add.await_count == 1

        container.set_memory_client(None)

        # Must not raise; the memory write is simply skipped.
        await session_client.add_message("sess-1234abcd", _msg())
        assert old_mem.add.await_count == 1  # unchanged — no client to write to

    async def test_property_resolves_live_each_access(self):
        """Fails against the pre-fix impl: the property would return the first
        resolved client forever. Direct property-level check complementing the
        behavioral tests."""
        container = get_container(ContainerConfig(enable_memory=True))

        session_client = SessionClient(
            session_config=SessionConfig(enabled=True),
            provider=_mock_provider(),
            auto_initialize=True,
        )
        old_mem = _mock_memory_client()
        container.set_memory_client(old_mem)
        assert session_client.memory_client is old_mem

        new_mem = _mock_memory_client()
        container.set_memory_client(new_mem)
        assert session_client.memory_client is new_mem

        # No client available → the property raises a clear error.
        container.set_memory_client(None)
        with pytest.raises(RuntimeError, match="MemoryClient is not available"):
            _ = session_client.memory_client
