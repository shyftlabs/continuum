"""Dependency initialization boundaries for :class:`AgentRunner`."""

from __future__ import annotations

from unittest.mock import MagicMock

from continuum.agent.runner import AgentRunner


class _MemoryTrapContainer:
    """Container double that fails if runner construction resolves memory."""

    @property
    def memory_client(self):
        raise AssertionError("memory_client was initialized eagerly")


def test_runner_construction_does_not_initialize_memory_backend() -> None:
    dependency = MagicMock()

    runner = AgentRunner(
        container=_MemoryTrapContainer(),
        llm_client=dependency,
        session_client=dependency,
        tool_executor=dependency,
    )

    assert runner is not None
