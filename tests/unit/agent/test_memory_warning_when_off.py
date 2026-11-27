"""The memory-provenance warning is not raised when long-term memory is off.

AgentMemoryConfig.search_memories / store_memories default to True and nothing
under agent/ reads MEMORY_ENABLED, so with memory switched off every agent still
logged "has long-term memory enabled but declares no data provenance" although
no memory client existed and nothing was read or written.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from continuum.agent.base import BaseAgent
from continuum.agent.services import memory_service
from continuum.agent.services.memory_service import MemoryService
from continuum.agent.types import RunContext
from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig


def _warned(memory_client, session_client=None) -> bool:
    # Patch the module logger rather than using caplog: other tests reconfigure
    # continuum's logging (propagation, handlers), which makes caplog order-dependent.
    with patch.object(memory_service, "logger") as log:
        MemoryService(
            memory_client=memory_client, session_client=session_client
        )._warn_if_provenance_undeclared(
            BaseAgent(name="ask", instructions="x", model="m"), RunContext(run_id="r")
        )
    return any(
        "declares no data provenance" in str(call.args[0]) for call in log.warning.call_args_list
    )


def test_no_warning_without_a_memory_client():
    assert not _warned(None)


def _memory(enabled: bool) -> MagicMock:
    client = MagicMock()
    client.is_enabled = enabled
    return client


def _session(memory_client, enabled: bool = True) -> SessionClient:
    return SessionClient(session_config=SessionConfig(enabled=enabled), memory_client=memory_client)


def test_warning_kept_when_memory_is_live():
    assert _warned(_memory(True))


def test_no_warning_for_a_disabled_memory_client():
    # MEMORY_ENABLED=false: get_global_memory_client() returns a client whose
    # is_enabled is False, and AgentRunner keeps it because it is truthy.
    assert not _warned(_memory(False))


def test_warning_for_session_only_memory():
    # SessionClient(memory_client=...) writes long-term memory in add_message even
    # when the container's memory is off, so the warning still applies.
    assert _warned(None, _session(_memory(True)))


def test_no_warning_when_session_memory_is_off():
    assert not _warned(None, _session(None))
    assert not _warned(None, _session(_memory(False)))
    assert not _warned(None, _session(_memory(True), enabled=False))


async def test_disabled_client_skips_memory_search():
    client = _memory(False)
    service = MemoryService(memory_client=client)
    agent = BaseAgent(name="ask", instructions="x", model="m")
    assert await service.retrieve_memories(agent, "q", RunContext(run_id="r")) == []
    client.search.assert_not_called()


async def test_protocol_session_client_keeps_memory_recall():
    # A session client that follows ISessionClient without subclassing
    # SessionClient has no _resolve_memory_client. The warning check used to call
    # it anyway, so retrieve_memories raised before searching and nothing was
    # recalled.
    from continuum.protocols import ISessionClient

    client = _memory(True)
    client.config.memory_isolation = "user"
    client.search = AsyncMock(return_value=[])
    session = MagicMock(spec=ISessionClient)
    session.is_enabled = True
    service = MemoryService(memory_client=client, session_client=session)
    agent = BaseAgent(name="ask", instructions="x", model="m")
    with patch.object(memory_service, "logger"):
        await service.retrieve_memories(agent, "q", RunContext(run_id="r", user_id="u"))
    client.search.assert_called_once()
    assert not _warned(None, session)  # unknown memory path: no warning, no error


async def test_duck_typed_memory_client_without_is_enabled_is_searched():
    class DuckMemory:
        def __init__(self):
            self.config = MagicMock(memory_isolation="user")
            self.search = AsyncMock(return_value=[])

    client = DuckMemory()
    service = MemoryService(memory_client=client)
    agent = BaseAgent(name="ask", instructions="x", model="m")
    with patch.object(memory_service, "logger"):
        await service.retrieve_memories(agent, "q", RunContext(run_id="r", user_id="u"))
    client.search.assert_called_once()


def test_warning_check_does_not_create_the_container():
    # A container-resolving SessionClient with memory off in the app's own
    # container: deciding on a log line must not create the global container or
    # build a MemoryClient.
    from continuum.core import container as container_module

    container_module.reset_container()
    try:
        with patch.object(container_module, "Container") as container_cls:
            assert not _warned(None, _session(None))
        container_cls.assert_not_called()
        assert container_module._global_container is None
    finally:
        container_module.reset_container()


def test_warning_uses_an_existing_container_client():
    from continuum.core import container as container_module

    container_module.reset_container()
    try:
        existing = MagicMock()
        existing.has_memory_client.return_value = True
        container_module._global_container = existing
        assert _warned(None, _session(None))
    finally:
        container_module._global_container = None


class _AdapterWithoutIsEnabled:
    """A memory adapter with search/add/config but no is_enabled."""

    def __init__(self):
        from continuum.memory.types import MemoryEntry, MemorySearchResult

        self.config = MagicMock(memory_isolation="user")
        self.search = AsyncMock(
            return_value=MemorySearchResult(
                results=[MemoryEntry(id="m1", memory="prefers tea", user_id="u")],
                query="q",
                limit=5,
            )
        )
        self.add = AsyncMock()


async def test_container_client_without_is_enabled_is_recalled_by_default_runner():
    # get_container().set_memory_client(adapter) + AgentRunner(): the warning
    # check fell through to has_memory_client(), which read adapter.is_enabled
    # and raised before the search, so nothing was recalled on any turn.
    from continuum.agent.runner import AgentRunner
    from continuum.core import container as container_module
    from continuum.core.container import ContainerConfig, get_container

    container_module.reset_container()
    try:
        container = get_container(ContainerConfig(enable_memory=True, enable_session=True))
        adapter = _AdapterWithoutIsEnabled()
        container.set_llm_client(MagicMock())
        container.set_tool_executor(MagicMock())
        container.set_memory_client(adapter)
        container.set_session_client(_session(None))  # resolves memory from the container
        runner = AgentRunner()
        agent = BaseAgent(name="ask", instructions="x", model="m")
        with patch.object(memory_service, "logger") as log:
            for turn in range(3):
                recalled = await runner._memory_service.retrieve_memories(
                    agent, "q", RunContext(run_id=f"r{turn}", user_id="u")
                )
                assert [m["memory"] for m in recalled] == ["prefers tea"]
        assert adapter.search.await_count == 3
        assert not any("Failed to retrieve" in str(c.args[0]) for c in log.error.call_args_list)
        warnings = [
            c for c in log.warning.call_args_list if "declares no data provenance" in str(c.args[0])
        ]
        assert len(warnings) == 1
    finally:
        container_module.reset_container()


def test_own_client_without_is_enabled_still_warns():
    # AgentRunner(memory_client=adapter) with container memory off: the search
    # runs, so the provenance warning applies.
    assert _warned(_AdapterWithoutIsEnabled())


def test_warning_check_survives_a_failing_container_check():
    from continuum.core import container as container_module

    container_module.reset_container()
    try:
        existing = MagicMock()
        existing.has_memory_client.side_effect = AttributeError("is_enabled")
        container_module._global_container = existing
        assert not _warned(None, _session(None))
    finally:
        container_module._global_container = None
