"""
Service interfaces for agent execution.

Defines contracts for service layer components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.types import (
        AgentResponse,
        RunContext,
        RunState,
        ToolExecutionSummary,
    )
    from orchestrator.llm.types import ChatMessage, ToolCallInput
    from orchestrator.tools.types import ToolContextState


class IExecutionService(ABC):
    """Interface for execution orchestration service."""

    @abstractmethod
    async def execute(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[ChatMessage],
        context: RunContext,
    ) -> AgentResponse:
        """Execute an agent to completion."""
        pass

    @abstractmethod
    async def execute_stream(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]],
        context: RunContext,
    ) -> Any:  # AsyncIterator[AgentEvent]
        """Execute an agent with streaming output."""
        pass


class IContextService(ABC):
    """Interface for context and state management."""

    @abstractmethod
    async def create_run_state(
        self,
        agent: BaseAgent,
        context: RunContext,
    ) -> RunState:
        """Create initial run state."""
        pass

    @abstractmethod
    async def save_run_state(self, state: RunState) -> None:
        """Save run state."""
        pass

    @abstractmethod
    async def load_run_state(self, run_id: str) -> RunState | None:
        """Load run state."""
        pass


class ISessionService(ABC):
    """Interface for session integration."""

    @abstractmethod
    async def save_messages(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        user_message_index: int,
        session_id: str,
        trace_id: str | None = None,
        tool_execution_summary: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """Save messages to session."""
        pass

    @abstractmethod
    async def load_tool_context_state(
        self,
        session_id: str,
        trace_id: str | None = None,
    ) -> ToolContextState:
        """Load tool context state from session."""
        pass

    @abstractmethod
    async def save_tool_context_state(
        self,
        session_id: str,
        context_state: ToolContextState,
        trace_id: str | None = None,
    ) -> None:
        """Save tool context state to session."""
        pass

    @abstractmethod
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get conversation history from session."""
        pass


class IMemoryService(ABC):
    """Interface for memory integration."""

    @abstractmethod
    async def retrieve_memories(
        self,
        agent: BaseAgent,
        query: str,
        context: RunContext,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories for the agent."""
        pass

    @abstractmethod
    async def store_memories(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
    ) -> None:
        """Store memories from conversation."""
        pass


class IToolService(ABC):
    """Interface for tool execution."""

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
        tool_summary: ToolExecutionSummary | None = None,
    ) -> list[dict[str, Any]]:
        """Execute multiple tool calls."""
        pass
