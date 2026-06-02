"""Integration: background memory writes wired through the DI container.

Verifies the two container-level contracts that the unit tests can't cover:
  1. The container injects its shared BackgroundTaskRegistry into the SessionClient.
  2. container.shutdown() drains in-flight background memory writes (so a write
     scheduled in 'background' mode is not lost at shutdown).

Uses mocked provider + memory client — no Redis, no mem0/LLM, no vector store.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.core.container import Container, ContainerConfig
from orchestrator.llm.types import ChatMessage
from orchestrator.session.client import SessionClient
from orchestrator.session.config import SessionConfig
from orchestrator.session.types import SessionMetadata

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
        container = Container(
            ContainerConfig(session_config={"enabled": True, "provider": "redis"})
        )
        try:
            with patch(
                "orchestrator.session.providers.create_provider",
                return_value=_mock_provider(),
            ):
                session_client = container.session_client

            assert session_client is not None
            # The session client shares the container's single registry instance.
            assert session_client._background_tasks is container.background_tasks
        finally:
            container.reset()


class TestShutdownDrain:
    async def test_pending_background_write_completes_on_shutdown(self):
        completed = {"done": False}

        async def _slow_add(*args, **kwargs):
            await asyncio.sleep(0.2)
            completed["done"] = True
            return MagicMock(results=[])

        container = Container(ContainerConfig())
        mem = _mock_memory_client(add_mock=AsyncMock(side_effect=_slow_add))
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

            # Returned before the slow write finished — it's in flight on the registry.
            assert mem.add.await_count == 0
            assert len(container.background_tasks) == 1
            assert completed["done"] is False

            # Shutdown must drain the in-flight write before closing the memory client.
            await container.shutdown()

            assert completed["done"] is True
            assert mem.add.await_count == 1
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
