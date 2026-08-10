"""
Handler interfaces for agent execution components.

Defines contracts for specialized handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.types import (
        HandoffResult,
        RunContext,
        RunState,
    )
    from continuum.llm.types import ToolCallInput
    from continuum.tools.types import ToolContextState


class IMessageBuilder(ABC):
    """Interface for message preparation."""

    @abstractmethod
    async def prepare_messages(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[Any],
        context: RunContext,
        tool_context_state: ToolContextState | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Prepare messages for agent execution.

        Returns ``(messages, user_message_index)``. The index marks where this
        turn's new messages begin, and the runner unpacks both — an
        implementation returning a bare list breaks it. The interface previously
        declared only the list, so anyone writing to this contract would have
        produced exactly that.
        """
        pass


class IToolHandler(ABC):
    """Interface for tool execution handling."""

    @abstractmethod
    async def execute_tool_call(
        self,
        agent: BaseAgent,
        tool_call: ToolCallInput,
        context: RunContext,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute a single tool call."""
        pass

    @abstractmethod
    async def execute_tools_batch(
        self,
        agent: BaseAgent,
        tool_calls: list[ToolCallInput],
        context: RunContext,
    ) -> list[dict[str, Any]]:
        """Execute multiple tool calls."""
        pass


class IHandoffExecutor(ABC):
    """Interface for handoff execution."""

    @abstractmethod
    async def execute_handoff(
        self,
        agent: BaseAgent,
        target_name: str,
        tool_call: Any,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> HandoffResult:
        """Execute a handoff to another agent."""
        pass
