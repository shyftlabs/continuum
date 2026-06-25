"""
Async-Safe Trace Context - Foundation for observability tracing.

Provides async-safe context propagation using contextvars for:
- Trace ID propagation
- Span hierarchy management
- User/session context
- Automatic parent-child span relationships

This replaces global mutable state with async-safe context variables
that work correctly with concurrent async operations.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from continuum.logging import get_logger

if TYPE_CHECKING:
    from langfuse.client import StatefulSpanClient, StatefulTraceClient

logger = get_logger(__name__)

# Type variable for generic return types
T = TypeVar("T")

# =============================================================================
# Context Variables (Async-Safe)
# =============================================================================

# Current trace context
_trace_id: ContextVar[str | None] = ContextVar("observability_trace_id", default=None)
_trace_client: ContextVar[StatefulTraceClient | None] = ContextVar(
    "observability_trace_client", default=None
)

# Current span context (for parent-child relationships)
_span_id: ContextVar[str | None] = ContextVar("observability_span_id", default=None)
_span_client: ContextVar[StatefulSpanClient | None] = ContextVar(
    "observability_span_client", default=None
)

# User/session context
_user_id: ContextVar[str | None] = ContextVar("observability_user_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("observability_session_id", default=None)

# Agent context
_agent_name: ContextVar[str | None] = ContextVar("observability_agent_name", default=None)
_run_id: ContextVar[str | None] = ContextVar("observability_run_id", default=None)

# Sampling control
_sampling_enabled: ContextVar[bool] = ContextVar("observability_sampling", default=True)

# Maximum size for trace data (bytes)
MAX_TRACE_DATA_SIZE = 10 * 1024  # 10KB


# =============================================================================
# Context Token for Restoration
# =============================================================================


@dataclass
class TraceContextToken:
    """Token for restoring previous trace context."""

    trace_id_token: Token[str | None]
    trace_client_token: Token[StatefulTraceClient | None]
    span_id_token: Token[str | None]
    span_client_token: Token[StatefulSpanClient | None]
    user_id_token: Token[str | None]
    session_id_token: Token[str | None]
    agent_name_token: Token[str | None]
    run_id_token: Token[str | None]


# =============================================================================
# Context Getters
# =============================================================================


def get_current_trace_id() -> str | None:
    """Get current trace ID (async-safe)."""
    return _trace_id.get()


def get_current_trace_client() -> StatefulTraceClient | None:
    """Get current Langfuse trace client (async-safe)."""
    return _trace_client.get()


def get_current_span_id() -> str | None:
    """Get current span ID (async-safe)."""
    return _span_id.get()


def get_current_span_client() -> StatefulSpanClient | None:
    """Get current Langfuse span client (async-safe)."""
    return _span_client.get()


def get_current_user_id() -> str | None:
    """Get current user ID (async-safe)."""
    return _user_id.get()


def get_current_session_id() -> str | None:
    """Get current session ID (async-safe)."""
    return _session_id.get()


def get_current_agent_name() -> str | None:
    """Get current agent name (async-safe)."""
    return _agent_name.get()


def get_current_run_id() -> str | None:
    """Get current run ID (async-safe)."""
    return _run_id.get()


def is_sampling_enabled() -> bool:
    """Check if sampling is enabled for current context."""
    return _sampling_enabled.get()


def get_parent_observation_id() -> str | None:
    """
    Get the parent observation ID for creating child spans.

    Returns span_id if in a span context, otherwise trace_id.
    """
    span_id = _span_id.get()
    if span_id:
        return span_id
    return _trace_id.get()


def get_parent_client() -> StatefulSpanClient | StatefulTraceClient | None:
    """
    Get the parent client for creating child spans.

    Returns span client if in a span context, otherwise trace client.
    """
    span_client = _span_client.get()
    if span_client:
        return span_client
    return _trace_client.get()


# =============================================================================
# Context Setters
# =============================================================================


def set_trace_context(
    trace_id: str | None = None,
    trace_client: StatefulTraceClient | None = None,
    span_id: str | None = None,
    span_client: StatefulSpanClient | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_name: str | None = None,
    run_id: str | None = None,
) -> TraceContextToken:
    """
    Set trace context values and return token for restoration.

    This is async-safe and works correctly with concurrent operations.

    Args:
        trace_id: Trace ID for observability
        trace_client: Provider trace client
        span_id: Current span ID
        span_client: Provider span client
        user_id: User identifier
        session_id: Session identifier
        agent_name: Current agent name
        run_id: Current run ID

    Returns:
        Token that can be used to restore previous context
    """
    tokens = TraceContextToken(
        trace_id_token=_trace_id.set(trace_id)
        if trace_id is not None
        else _trace_id.set(_trace_id.get()),
        trace_client_token=_trace_client.set(trace_client)
        if trace_client is not None
        else _trace_client.set(_trace_client.get()),
        span_id_token=_span_id.set(span_id)
        if span_id is not None
        else _span_id.set(_span_id.get()),
        span_client_token=_span_client.set(span_client)
        if span_client is not None
        else _span_client.set(_span_client.get()),
        user_id_token=_user_id.set(user_id)
        if user_id is not None
        else _user_id.set(_user_id.get()),
        session_id_token=_session_id.set(session_id)
        if session_id is not None
        else _session_id.set(_session_id.get()),
        agent_name_token=_agent_name.set(agent_name)
        if agent_name is not None
        else _agent_name.set(_agent_name.get()),
        run_id_token=_run_id.set(run_id) if run_id is not None else _run_id.set(_run_id.get()),
    )
    return tokens


def restore_trace_context(token: TraceContextToken) -> None:
    """Restore previous trace context from token."""
    _trace_id.reset(token.trace_id_token)
    _trace_client.reset(token.trace_client_token)
    _span_id.reset(token.span_id_token)
    _span_client.reset(token.span_client_token)
    _user_id.reset(token.user_id_token)
    _session_id.reset(token.session_id_token)
    _agent_name.reset(token.agent_name_token)
    _run_id.reset(token.run_id_token)


def clear_trace_context() -> None:
    """Clear all trace context values."""
    _trace_id.set(None)
    _trace_client.set(None)
    _span_id.set(None)
    _span_client.set(None)
    _user_id.set(None)
    _session_id.set(None)
    _agent_name.set(None)
    _run_id.set(None)


def set_sampling_enabled(enabled: bool) -> Token[bool]:
    """Set sampling enabled state."""
    return _sampling_enabled.set(enabled)


# =============================================================================
# Context Managers
# =============================================================================


class TraceScope:
    """
    Context manager for trace scope.

    Automatically sets up trace context and restores on exit.

    Example:
        ```python
        with TraceScope(trace_id="trace-123", user_id="user-456") as ctx:
            # All operations in this scope will use this trace context
            await some_operation()
        # Context automatically restored
        ```
    """

    def __init__(
        self,
        trace_id: str | None = None,
        trace_client: StatefulTraceClient | None = None,
        span_id: str | None = None,
        span_client: StatefulSpanClient | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        run_id: str | None = None,
    ):
        self._trace_id = trace_id
        self._trace_client = trace_client
        self._span_id = span_id
        self._span_client = span_client
        self._user_id = user_id
        self._session_id = session_id
        self._agent_name = agent_name
        self._run_id = run_id
        self._token: TraceContextToken | None = None

    def __enter__(self) -> TraceScope:
        self._token = set_trace_context(
            trace_id=self._trace_id,
            trace_client=self._trace_client,
            span_id=self._span_id,
            span_client=self._span_client,
            user_id=self._user_id,
            session_id=self._session_id,
            agent_name=self._agent_name,
            run_id=self._run_id,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token:
            restore_trace_context(self._token)

    async def __aenter__(self) -> TraceScope:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class SpanScope:
    """
    Context manager for creating a child span.

    Automatically creates a span as child of current context and updates context.

    Example:
        ```python
        async with SpanScope("session.add_message", input={"content": msg}) as span:
            result = await do_work()
            span.set_output(result)
        ```
    """

    def __init__(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
    ):
        self.name = name
        self.input = input
        self.metadata = metadata or {}
        self.level = level
        self._token: TraceContextToken | None = None
        self._span_client: StatefulSpanClient | None = None
        self._span_id: str | None = None
        self._start_time: float = 0
        self._output: Any = None
        self._error: str | None = None

    def _create_span(self) -> None:
        """Create the span using provider manager or parent client."""
        import uuid

        parent = get_parent_client()
        self._span_id = str(uuid.uuid4())
        self._start_time = time.time()

        try:
            # Truncate input if too large, then apply label-aware redaction
            # (ambient run policy). SpanScope is the single chokepoint every span
            # passes through — @observe, tool calls, workflow stages, executor —
            # so redacting here covers them all uniformly. mask_secrets=False:
            # span payloads are arbitrary structured data (a deny -> placeholder;
            # otherwise pass through, so token-usage fields aren't masked).
            from continuum.observability.data_redaction import redact_for_telemetry

            truncated_input = redact_for_telemetry(truncate_data(self.input), mask_secrets=False)

            # If we have a parent client, use it directly
            if parent is not None:
                self._span_client = parent.span(
                    id=self._span_id,
                    name=self.name,
                    input=truncated_input,
                    metadata=self.metadata,
                    level=self.level,
                )

                # Update context with new span
                self._token = set_trace_context(
                    span_id=self._span_id,
                    span_client=self._span_client,
                )
            else:
                # No parent client, use ProviderManager
                # CRITICAL: Spans require a parent trace. Never create a trace here.
                from continuum.observability.provider_manager import get_provider_manager

                manager = get_provider_manager()
                trace_id = get_current_trace_id()

                if not trace_id:
                    # No trace context exists - cannot create span without parent trace
                    # This is expected for operations that happen before trace creation
                    # (e.g., session creation in API layer, initial setup)
                    logger.debug(
                        f"Skipping span '{self.name}' - no trace context exists yet. "
                        "This is normal for operations that occur before trace creation."
                    )
                    self._span_client = None
                    return

                if manager.is_enabled:
                    parent_obs_id = get_parent_observation_id()

                    # Create span via provider manager (under existing trace)
                    span_result = manager.span(
                        trace_id=trace_id,
                        parent_observation_id=parent_obs_id,
                        name=self.name,
                        input=truncated_input,
                        metadata=self.metadata,
                        level=self.level,
                    )

                    # Store the result (may be a provider client or None)
                    self._span_client = span_result

                    # Update context if we got a client back
                    if span_result is not None:
                        self._token = set_trace_context(
                            span_id=self._span_id,
                            span_client=span_result,
                        )
                        logger.debug(
                            f"Created span '{self.name}' under trace {trace_id} "
                            f"(parent_obs_id={parent_obs_id})"
                        )
                    else:
                        logger.debug(
                            f"No provider created span '{self.name}' (may be disabled or not sampled)"
                        )
                else:
                    logger.debug(f"Provider manager not enabled, skipping span '{self.name}'")

        except Exception as e:
            logger.warning(f"Failed to create span '{self.name}': {e}")

    def _end_span(self, error: Exception | None = None) -> None:
        """End the span."""
        if self._span_client is None:
            # Restore previous context even if no client
            if self._token:
                restore_trace_context(self._token)
            return

        try:
            latency_ms = (time.time() - self._start_time) * 1000

            # Truncate output if too large
            truncated_output = truncate_data(self._output)

            # Try to call end() on the client (works for Langfuse clients)
            if hasattr(self._span_client, "end"):
                self._span_client.end(
                    output=truncated_output,
                    metadata={
                        **self.metadata,
                        "latency_ms": round(latency_ms, 2),
                    },
                    level="ERROR" if error else self.level,
                    status_message=str(error) if error else None,
                )
            else:
                # If it doesn't have end(), it might be a different provider type
                # Try to update it if it has an update method
                if hasattr(self._span_client, "update"):
                    self._span_client.update(
                        output=truncated_output,
                        metadata={
                            **self.metadata,
                            "latency_ms": round(latency_ms, 2),
                        },
                        level="ERROR" if error else self.level,
                        status_message=str(error) if error else None,
                    )

        except Exception as e:
            logger.warning(f"Failed to end span '{self.name}': {e}")

        # Restore previous context
        if self._token:
            restore_trace_context(self._token)

    def set_output(self, output: Any) -> None:
        """Set span output (label-aware redaction applied — see _create_span)."""
        try:
            from continuum.observability.data_redaction import redact_for_telemetry

            self._output = redact_for_telemetry(output, mask_secrets=False)
        except Exception:
            # Fail safe: never let a redaction error leak raw output to telemetry.
            self._output = {"_redacted": "redaction error"}

    def set_error(self, error: str) -> None:
        """Set span error."""
        self._error = error

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to span."""
        self.metadata[key] = value

    def __enter__(self) -> SpanScope:
        self._create_span()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end_span(self._span_error(exc_type, exc_val))

    async def __aenter__(self) -> SpanScope:
        self._create_span()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end_span(self._span_error(exc_type, exc_val))

    @staticmethod
    def _span_error(exc_type: Any, exc_val: Any) -> Any:
        """The exception to flag this span with — or None.

        An EXPECTED access-control denial (``PolicyDeniedError`` — a data-label
        gate) is propagating by design, not failing: the span is NOT marked
        ERROR for it (it flushes at its normal/declared level, e.g. the WARNING
        set by the observe decorator). Every other exception marks the span
        ERROR as before.
        """
        if not exc_type:
            return None
        from continuum.exceptions import PolicyDeniedError

        if isinstance(exc_val, PolicyDeniedError):
            return None
        return exc_val


# =============================================================================
# Utility Functions
# =============================================================================


def truncate_data(data: Any, max_size: int = MAX_TRACE_DATA_SIZE) -> Any:
    """
    Truncate data to fit within size limits for Langfuse.

    Args:
        data: Data to truncate
        max_size: Maximum size in bytes

    Returns:
        Truncated data
    """
    if data is None:
        return None

    import json

    try:
        # Convert to JSON string to measure size
        json_str = json.dumps(data, default=str)

        if len(json_str) <= max_size:
            return data

        # Truncate string representation
        truncated = json_str[: max_size - 50]  # Leave room for truncation message
        return {
            "_truncated": True,
            "_original_size": len(json_str),
            "preview": truncated + "...[TRUNCATED]",
        }

    except (TypeError, ValueError):
        # Can't serialize, return string representation
        str_data = str(data)
        if len(str_data) <= max_size:
            return str_data
        return str_data[: max_size - 20] + "...[TRUNCATED]"


def get_trace_metadata() -> dict[str, Any]:
    """
    Get current trace metadata for Langfuse calls.

    Returns metadata dict with all current context values.
    """
    return {
        "trace_id": get_current_trace_id(),
        "parent_observation_id": get_parent_observation_id(),
        "user_id": get_current_user_id(),
        "session_id": get_current_session_id(),
        "agent_name": get_current_agent_name(),
        "run_id": get_current_run_id(),
    }


def build_langfuse_metadata(
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build metadata dict for Langfuse operations.

    Combines current context with extra metadata.
    """
    metadata = get_trace_metadata()
    if extra:
        metadata.update(extra)
    return {k: v for k, v in metadata.items() if v is not None}


# =============================================================================
# Decorator for Automatic Span Creation
# =============================================================================


def traced_operation(
    name: str | None = None,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
    level: str = "DEFAULT",
):
    """
    Decorator for automatic span creation around operations.

    Args:
        name: Span name (defaults to function name)
        capture_input: Whether to capture function arguments
        capture_output: Whether to capture return value
        level: Span level (DEFAULT, DEBUG, WARNING, ERROR)

    Example:
        ```python
        @traced_operation("memory.search")
        async def search_memories(query: str, limit: int):
            return await mem0.search(query, limit=limit)
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        operation_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Build input
            input_data = None
            if capture_input:
                input_data = {
                    "args": str(args)[:500],
                    "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
                }

            async with SpanScope(operation_name, input=input_data, level=level) as span:
                result = await func(*args, **kwargs)
                if capture_output:
                    span.set_output(result)
                return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Build input
            input_data = None
            if capture_input:
                input_data = {
                    "args": str(args)[:500],
                    "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
                }

            with SpanScope(operation_name, input=input_data, level=level) as span:
                result = func(*args, **kwargs)
                if capture_output:
                    span.set_output(result)
                return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
