"""The memory-provenance warning is not raised when long-term memory is off.

AgentMemoryConfig.search_memories / store_memories default to True and nothing
under agent/ reads MEMORY_ENABLED, so with memory switched off every agent still
logged "has long-term memory enabled but declares no data provenance" although
no memory client existed and nothing was read or written.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from continuum.agent.base import BaseAgent
from continuum.agent.services.memory_service import MemoryService
from continuum.agent.types import RunContext

WARNING = "declares no data provenance"


def _agent() -> BaseAgent:
    return BaseAgent(name="ask", instructions="x", model="m")


def test_no_warning_without_a_memory_client(caplog):
    caplog.set_level(logging.WARNING)
    MemoryService(memory_client=None)._warn_if_provenance_undeclared(
        _agent(), RunContext(run_id="r")
    )
    assert WARNING not in caplog.text


def test_warning_kept_when_memory_is_live(caplog):
    caplog.set_level(logging.WARNING)
    MemoryService(memory_client=MagicMock())._warn_if_provenance_undeclared(
        _agent(), RunContext(run_id="r")
    )
    assert WARNING in caplog.text
