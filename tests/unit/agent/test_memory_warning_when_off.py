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


def _warned(memory_client) -> bool:
    # Patch the module logger rather than using caplog: other tests reconfigure
    # continuum's logging (propagation, handlers), which makes caplog order-dependent.
    with patch.object(memory_service, "logger") as log:
        MemoryService(memory_client=memory_client)._warn_if_provenance_undeclared(
            BaseAgent(name="ask", instructions="x", model="m"), RunContext(run_id="r")
        )
    return any(
        "declares no data provenance" in str(call.args[0]) for call in log.warning.call_args_list
    )


def test_no_warning_without_a_memory_client():
    assert not _warned(None)


def test_warning_kept_when_memory_is_live():
    assert _warned(MagicMock())
