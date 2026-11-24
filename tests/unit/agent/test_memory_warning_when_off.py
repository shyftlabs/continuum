"""The memory-provenance warning is not raised when long-term memory is off.

AgentMemoryConfig.search_memories / store_memories default to True and nothing
under agent/ reads MEMORY_ENABLED, so with memory switched off every agent still
logged "has long-term memory enabled but declares no data provenance" although
no memory client existed and nothing was read or written.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
