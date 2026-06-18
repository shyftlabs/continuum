"""
Executor interfaces for agent execution.

Defines contracts for execution components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from continuum.agent.base import BaseAgent
    from continuum.agent.types import (
        AgentEvent,
        AgentResponse,
        RunContext,
        RunState,
    )


class IExecutor(ABC):
    """Interface for core execution logic."""

    @abstractmethod
    async def execute_loop(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> AgentResponse:
        """Execute the main conversation loop.

        Contract: implementations should publish the agent's data-label policy
        for the duration of the loop so per-agent gating is correct across
        handoffs (the run switches agents, and each agent may have a different
        ``policy_store``). The shipped ``Executor`` does this via
        ``continuum.security.policy_context.use_active_policy``::

            with use_active_policy(agent.policy_store, agent.name, context):
                ...  # run the loop

        Omitting it is not a security hole — ``AgentRunner.run``/``run_stream``
        still publish the *entry* agent's policy run-wide, so the gate still
        fires — but a handed-off agent's turns would then be gated by the entry
        agent's policy rather than its own.
        """
        pass


class IStreamExecutor(ABC):
    """Interface for streaming execution."""

    @abstractmethod
    async def execute_stream(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> AsyncIterator[AgentEvent]:
        """Execute with streaming output."""
        pass
