"""
Agent types and data structures.

Defines all type definitions for the agent orchestration module.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.llm.types import ChatMessage, ToolCall

_logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class RunStatus(str, Enum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_TOOL = "waiting_for_tool"
    HANDOFF_PENDING = "handoff_pending"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResponseStatus(str, Enum):
    """Status of an agent response."""

    SUCCESS = "success"
    ERROR = "error"
    HANDOFF = "handoff"
    TOOL_CALL = "tool_call"
    MAX_TURNS_REACHED = "max_turns_reached"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """Types of events emitted during agent execution."""

    # Run lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"
    RUN_ERROR = "run_error"

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"

    # Content events
    CONTENT_DELTA = "content_delta"
    CONTENT_COMPLETE = "content_complete"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"

    # Handoff events
    HANDOFF_START = "handoff_start"
    HANDOFF_END = "handoff_end"
    HANDOFF_RETURN = "handoff_return"

    # Memory events
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_STORAGE = "memory_storage"

    # Workflow events
    WORKFLOW_STEP = "workflow_step"
    LOOP_ITERATION = "loop_iteration"


class MemoryScope(str, Enum):
    """Scope for memory operations."""

    SHARED = "shared"  # Shared across all users/agents
    USER = "user"  # Scoped to user
    AGENT = "agent"  # Scoped to agent
    RUN = "run"  # Scoped to current run/session


class MergeStrategy(str, Enum):
    """Strategy for merging parallel agent outputs."""

    CONCATENATE = "concatenate"  # Simple concatenation
    LLM_SUMMARIZE = "llm_summarize"  # LLM summarizes all outputs
    STRUCTURED = "structured"  # Return structured dict of outputs
    FIRST_SUCCESS = "first_success"  # Return first successful result


class FailStrategy(str, Enum):
    """Strategy for handling failures in parallel/sequential execution."""

    FAIL_FAST = "fail_fast"  # Stop on first failure
    CONTINUE_ON_ERROR = "continue"  # Continue with remaining agents
    REQUIRE_ALL = "require_all"  # Require all to succeed


class TerminationType(str, Enum):
    """Type of termination condition for loops."""

    LLM_DECISION = "llm_decision"  # LLM decides when complete
    TOOL_CALL = "tool_call"  # Terminates on specific tool call
    OUTPUT_MATCH = "output_match"  # Terminates when output matches pattern
    CUSTOM = "custom"  # Custom callable condition


class HistorySummarizationMode(str, Enum):
    """Mode for summarizing conversation history during handoffs."""

    FULL = "full"  # Pass entire history
    SUMMARY = "summary"  # LLM-generated summary
    RECENT_N = "recent_n"  # Last N messages only
    HYBRID = "hybrid"  # Summary + recent N messages


# =============================================================================
# ID Generation
# =============================================================================


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid.uuid4().hex[:16]}"


def generate_handoff_id() -> str:
    """Generate a unique handoff ID."""
    return f"handoff_{uuid.uuid4().hex[:12]}"


# =============================================================================
# Token Usage
# =============================================================================


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Per-model breakdown
    model_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, other: TokenUsage) -> TokenUsage:
        """Add token usage from another instance. Coerces values to int for safety."""
        all_models = set(self.model_usage.keys()) | set(other.model_usage.keys())
        merged_usage: dict[str, dict[str, int]] = {}
        for model in all_models:
            self_usage = self.model_usage.get(model, {})
            other_usage = other.model_usage.get(model, {})
            merged_usage[model] = {
                "prompt_tokens": int(self_usage.get("prompt_tokens", 0) or 0)
                + int(other_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(self_usage.get("completion_tokens", 0) or 0)
                + int(other_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(self_usage.get("total_tokens", 0) or 0)
                + int(other_usage.get("total_tokens", 0) or 0),
            }
        return TokenUsage(
            prompt_tokens=int(self.prompt_tokens or 0) + int(other.prompt_tokens or 0),
            completion_tokens=int(self.completion_tokens or 0) + int(other.completion_tokens or 0),
            total_tokens=int(self.total_tokens or 0) + int(other.total_tokens or 0),
            model_usage=merged_usage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_usage": self.model_usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        """Create from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            model_usage=data.get("model_usage", {}),
        )


# =============================================================================
# Handoff Types
# =============================================================================


@dataclass
class Handoff:
    """
    Definition of a handoff to another agent.

    Handoffs allow an agent to delegate to another agent when appropriate.
    The LLM sees handoffs as special tools it can call.
    """

    target_agent: str  # Name of agent to hand off to
    description: str  # Description for LLM (when to use)
    condition: str | Callable[..., bool] | None = None  # Optional programmatic condition
    transfer_history: bool = True  # Whether to pass conversation history
    summarize_history: bool = True  # Whether to summarize history
    summarization_mode: HistorySummarizationMode = HistorySummarizationMode.HYBRID
    recent_messages: int = 5  # Number of recent messages for hybrid mode
    return_to_parent: bool = True  # Whether to return control after completion

    def to_tool_definition(self) -> dict[str, Any]:
        """Convert handoff to a tool definition for LLM."""
        return {
            "type": "function",
            "function": {
                "name": f"handoff_to_{self.target_agent}",
                "description": f"Hand off the conversation to {self.target_agent}. {self.description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for the handoff",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for the target agent",
                        },
                    },
                    "required": ["reason"],
                },
            },
        }


def _log_missing_timestamp(handoff_id: str) -> datetime:
    """Log a warning and return current time when HandoffData timestamp is missing."""
    _logger.warning(
        f"HandoffData '{handoff_id}' missing timestamp field during deserialization. "
        f"Using current time as fallback — this may indicate data loss."
    )
    return datetime.now(UTC)


@dataclass
class HandoffData:
    """Data passed during a handoff."""

    handoff_id: str
    from_agent: str
    to_agent: str
    reason: str
    context: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    history_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handoff_id": self.handoff_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "context": self.context,
            "history": self.history,
            "history_summary": self.history_summary,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffData:
        """Create from dictionary."""
        return cls(
            handoff_id=data["handoff_id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            reason=data["reason"],
            context=data.get("context"),
            history=data.get("history", []),
            history_summary=data.get("history_summary"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else _log_missing_timestamp(data.get("handoff_id", "unknown")),
        )


@dataclass
class HandoffResult:
    """Result of a handoff execution."""

    handoff_id: str
    from_agent: str
    to_agent: str
    success: bool
    response: AgentResponse | None = None
    error: str | None = None
    returned_to_parent: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handoff_id": self.handoff_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "success": self.success,
            "response": self.response.to_dict() if self.response else None,
            "error": self.error,
            "returned_to_parent": self.returned_to_parent,
        }


# =============================================================================
# Run State
# =============================================================================


@dataclass
class RunState:
    """
    State of an agent run.

    Stored in Redis for active runs to allow pause/resume and
    state recovery on failures.
    """

    run_id: str
    session_id: str | None = None
    user_id: str | None = None

    # Agent state
    current_agent: str = ""
    agent_stack: list[str] = field(default_factory=list)  # Handoff call stack
    entry_agent: str = ""  # Original entry point

    # Conversation state
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Handoff state (stored as dicts for Redis serialization)
    handoff_chain: list[dict[str, Any]] = field(default_factory=list)
    current_handoff: dict[str, Any] | None = None

    def get_handoff_chain_as_data(self) -> list[HandoffData]:
        """Convert handoff_chain dicts to HandoffData objects."""
        return [HandoffData.from_dict(h) for h in self.handoff_chain]

    def add_handoff(self, handoff: HandoffData) -> None:
        """Add a handoff (stores as dict for serialization)."""
        self.handoff_chain.append(handoff.to_dict())

    # Execution state
    status: RunStatus = RunStatus.PENDING
    turn_count: int = 0
    max_turns: int = 25

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    parent_span_id: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Token tracking
    usage: TokenUsage = field(default_factory=TokenUsage)

    # Thread-safety lock for mutable state (agent_stack, handoff_chain)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def push_agent(self, agent_name: str) -> None:
        """Thread-safe push to agent_stack."""
        with self._lock:
            self.agent_stack.append(agent_name)

    def pop_agent(self) -> str | None:
        """Thread-safe pop from agent_stack."""
        with self._lock:
            return self.agent_stack.pop() if self.agent_stack else None

    def get_agent_stack_snapshot(self) -> list[str]:
        """Thread-safe snapshot of agent_stack."""
        with self._lock:
            return list(self.agent_stack)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "current_agent": self.current_agent,
            "agent_stack": self.agent_stack,
            "entry_agent": self.entry_agent,
            "messages": self.messages,
            "pending_tool_calls": self.pending_tool_calls,
            "handoff_chain": self.handoff_chain,
            "current_handoff": self.current_handoff,
            "status": self.status.value,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "metadata": self.metadata,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            current_agent=data.get("current_agent", ""),
            agent_stack=data.get("agent_stack", []),
            entry_agent=data.get("entry_agent", ""),
            messages=data.get("messages", []),
            pending_tool_calls=data.get("pending_tool_calls", []),
            handoff_chain=data.get("handoff_chain", []),
            current_handoff=data.get("current_handoff"),
            status=RunStatus(data.get("status", "pending")),
            turn_count=data.get("turn_count", 0),
            max_turns=data.get("max_turns", 25),
            metadata=data.get("metadata", {}),
            trace_id=data.get("trace_id"),
            parent_span_id=data.get("parent_span_id"),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else datetime.now(UTC),
            usage=TokenUsage.from_dict(data.get("usage", {})),
        )

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(UTC)


# =============================================================================
# Agent Response
# =============================================================================


@dataclass
class AgentResponse:
    """
    Response from an agent execution.

    Contains the output, execution metadata, and full audit trail.
    """

    # Core output
    content: str
    structured_output: BaseModel | None = None

    # Execution info
    run_id: str = ""
    agent_name: str = ""
    status: ResponseStatus = ResponseStatus.SUCCESS

    # Tool calls (if any pending/executed)
    tool_calls: list[ToolCall] | None = None
    tool_results: list[dict[str, Any]] | None = None

    # Run artifacts - full MCP responses including widgets/structured data (per-run)
    run_artifacts: dict[str, Any] | None = None
    """
    Full artifacts from MCP tool calls during this run.

    Includes meta (widget templates), structured_content (rendering data),
    and text_content for each tool call. This is CLEARED at the start of
    each run - use this for frontend rendering, widget display, etc.

    Structure:
        {
            "run_id": "...",
            "tool_artifacts": [
                {
                    "tool_name": "get_cart",
                    "server_name": "petco-mcp",
                    "meta": {"openai/outputTemplate": "ui://widget/cart.html", ...},
                    "structured_content": {"items": [...], "subtotal": 29.99},
                    "text_content": "...",
                    ...
                },
                ...
            ]
        }
    """

    # Handoff info
    handoff: HandoffData | None = None
    handoff_result: HandoffResult | None = None

    # Conversation
    messages: list[ChatMessage] = field(default_factory=list)

    # Metrics
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0
    turn_count: int = 0

    # Tracing
    trace_id: str | None = None
    span_id: str | None = None

    # Multi-agent info
    agents_used: list[str] = field(default_factory=list)
    handoff_chain: list[str] = field(default_factory=list)

    # Error info
    error: str | None = None
    error_type: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @classmethod
    def error_response(
        cls,
        error: str | Exception,
        *,
        agent_name: str = "",
        run_id: str = "",
        trace_id: str | None = None,
        error_type: str | None = None,
    ) -> AgentResponse:
        """
        Factory for consistent error responses across the codebase.

        Args:
            error: Error message or exception
            agent_name: Agent that produced the error
            run_id: Run ID for tracing
            trace_id: Trace ID for observability
            error_type: Error classification (defaults to exception class name)
        """
        error_str = str(error)
        return cls(
            content=f"An error occurred: {error_str}",
            agent_name=agent_name,
            run_id=run_id,
            status=ResponseStatus.ERROR,
            error=error_str,
            error_type=error_type or (type(error).__name__ if isinstance(error, Exception) else "Error"),
            trace_id=trace_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "content": self.content,
            "structured_output": self.structured_output.model_dump()
            if self.structured_output
            else None,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in self.tool_calls]
            if self.tool_calls
            else None,
            "tool_results": self.tool_results,
            "run_artifacts": self.run_artifacts,
            "handoff": self.handoff.to_dict() if self.handoff else None,
            "handoff_result": self.handoff_result.to_dict() if self.handoff_result else None,
            "messages": [m.to_dict() if hasattr(m, "to_dict") else m for m in self.messages],
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "turn_count": self.turn_count,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agents_used": self.agents_used,
            "handoff_chain": self.handoff_chain,
            "error": self.error,
            "error_type": self.error_type,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# Agent Event (for streaming)
# =============================================================================


@dataclass
class AgentEvent:
    """
    Event emitted during agent execution.

    Used for streaming responses and real-time observability.
    """

    type: EventType
    agent_name: str
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Optional trace context
    trace_id: str | None = None
    span_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


# =============================================================================
# Step Result (for manual stepping)
# =============================================================================


@dataclass
class StepResult:
    """
    Result of a single step in agent execution.

    Used for manual step-by-step control.
    """

    run_state: RunState
    events: list[AgentEvent] = field(default_factory=list)
    is_complete: bool = False
    requires_input: bool = False
    requires_tool_execution: bool = False
    response: AgentResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_state": self.run_state.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "is_complete": self.is_complete,
            "requires_input": self.requires_input,
            "requires_tool_execution": self.requires_tool_execution,
            "response": self.response.to_dict() if self.response else None,
        }


# =============================================================================
# Termination Config (for loops)
# =============================================================================


@dataclass
class TerminationConfig:
    """
    Configuration for loop termination.

    Defines when a LoopAgent should stop iterating.
    """

    type: TerminationType = TerminationType.LLM_DECISION

    # For LLM_DECISION: Prompt for LLM to decide
    decision_prompt: str = "Is the task complete? Respond with 'COMPLETE' if done, or 'CONTINUE' if more work is needed."

    # For TOOL_CALL: Tool name that triggers termination
    tool_name: str | None = None

    # For OUTPUT_MATCH: Pattern to match
    pattern: str | None = None

    # For CUSTOM: Callable that returns True when done
    condition: Callable[[str, list[dict[str, Any]]], bool] | None = None

    # Max iterations (safety limit)
    max_iterations: int = 10


# =============================================================================
# Route (for RouterAgent)
# =============================================================================


@dataclass
class Route:
    """
    A route definition for RouterAgent.

    Maps conditions to target agents.
    """

    agent_name: str
    description: str  # Description for LLM routing
    condition: str | Callable[..., bool] | None = None  # Optional programmatic condition
    priority: int = 0  # Higher priority routes checked first

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "description": self.description,
            "condition": str(self.condition) if callable(self.condition) else self.condition,
            "priority": self.priority,
        }


# =============================================================================
# Tool Execution Summary (for session storage without tool_call_ids)
# =============================================================================


@dataclass
class ToolExecutionSummary:
    """
    Lightweight summary of tool executions during a turn.

    This is stored as metadata on the final assistant message instead of
    storing full tool_calls and tool results. This approach:
    - Eliminates cross-model tool_call_id compatibility issues (Gemini vs OpenAI)
    - Reduces token usage in session history
    - Keeps detailed tool traces in Langfuse for debugging

    Example:
        ```python
        summary = ToolExecutionSummary(
            tools_used=["product_search", "get_cart"],
            tool_count=2,
            total_latency_ms=150,
            servers_used=["petco-mcp"],
            success_count=2,
            error_count=0,
        )
        ```
    """

    # Tools executed in this turn
    tools_used: list[str] = field(default_factory=list)
    tool_count: int = 0

    # Performance metrics
    total_latency_ms: float = 0.0
    tool_latencies: dict[str, float] = field(default_factory=dict)  # tool_name -> latency_ms

    # MCP server info
    servers_used: list[str] = field(default_factory=list)

    # Success/error tracking
    success_count: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)  # Brief error messages

    # Auth/internal details (masked for security)
    auth_info: dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"jwt_exp": "2024-...", "api_key_id": "sk-...abc"}

    # Token info if available
    input_tokens: int = 0
    output_tokens: int = 0

    def add_tool_execution(
        self,
        tool_name: str,
        latency_ms: float,
        server_name: str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Add a tool execution to the summary."""
        self.tools_used.append(tool_name)
        self.tool_count += 1
        self.total_latency_ms += latency_ms
        self.tool_latencies[tool_name] = latency_ms

        if server_name and server_name not in self.servers_used:
            self.servers_used.append(server_name)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                # Keep error messages brief
                self.errors.append(f"{tool_name}: {error[:100]}")

    def set_auth_info(
        self,
        jwt_expiry: str | None = None,
        api_key_id: str | None = None,
        token_type: str | None = None,
        **kwargs: str,
    ) -> None:
        """Set masked auth information."""
        if jwt_expiry:
            self.auth_info["jwt_exp"] = jwt_expiry
        if api_key_id:
            # Mask API key - show only last 4 chars
            masked = f"...{api_key_id[-4:]}" if len(api_key_id) > 4 else "****"
            self.auth_info["api_key_id"] = masked
        if token_type:
            self.auth_info["token_type"] = token_type
        # Allow custom auth info
        for key, value in kwargs.items():
            self.auth_info[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "tools_used": self.tools_used,
            "tool_count": self.tool_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tool_latencies": {k: round(v, 2) for k, v in self.tool_latencies.items()},
            "servers_used": self.servers_used,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "errors": self.errors,
            "auth_info": self.auth_info,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExecutionSummary:
        """Create from dictionary."""
        return cls(
            tools_used=data.get("tools_used", []),
            tool_count=data.get("tool_count", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            tool_latencies=data.get("tool_latencies", {}),
            servers_used=data.get("servers_used", []),
            success_count=data.get("success_count", 0),
            error_count=data.get("error_count", 0),
            errors=data.get("errors", []),
            auth_info=data.get("auth_info", {}),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )

    def is_empty(self) -> bool:
        """Check if no tools were executed."""
        return self.tool_count == 0


# =============================================================================
# Run Context
# =============================================================================


@dataclass
class RunContext:
    """
    Context passed through agent execution.

    Contains all shared state and configuration for a run.
    """

    run_id: str
    session_id: str | None = None
    user_id: str | None = None

    # Trace context
    trace_id: str | None = None
    parent_span_id: str | None = None

    # Agent context
    agent_stack: list[str] = field(default_factory=list)
    handoff_chain: list[HandoffData] = field(default_factory=list)

    # Memory context
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)

    # Configuration
    max_turns: int = 25
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # State
    usage: TokenUsage = field(default_factory=TokenUsage)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "agent_stack": self.agent_stack,
            "handoff_chain": [h.to_dict() for h in self.handoff_chain],
            "retrieved_memories": self.retrieved_memories,
            "max_turns": self.max_turns,
            "metadata": self.metadata,
            "tags": self.tags,
            "usage": self.usage.to_dict(),
        }


@dataclass
class PrepareRunResult:
    """Result of AgentRunner._prepare_run()."""

    success: bool
    context: RunContext | None = None
    run_state: RunState | None = None
    initial_message_count: int = 0
    tool_context_state: Any = None
    error_response: AgentResponse | None = None
