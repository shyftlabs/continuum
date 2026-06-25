"""
Tracing decorators for easy instrumentation.

Provides decorators to automatically trace functions, tools, and agents.

NOTE: Decorators now use async-safe SpanScope from trace_context.py.
Spans are automatically linked to the current trace context if one exists.
If no trace context exists, spans are created via ProviderManager.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from continuum.logging import get_logger
from continuum.observability.trace_context import (
    SpanScope,
    get_current_trace_id,
    truncate_data,
)
from continuum.observability.tracing import SpanLevel

if TYPE_CHECKING:
    from continuum.observability.tracing import TracingManager

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _record_observe_exception(span: Any, span_name: str, e: BaseException) -> None:
    """Record an exception on the span before it re-raises.

    Expected access-control denials (``PolicyDeniedError`` — the data-label
    model/tool/memory gates) are NOT failures: they are recorded as a
    WARNING-level "policy denied" event (audit-visible in the trace) and are NOT
    escalated to error reporting, so they don't pollute error dashboards or
    trigger alerts. Every other exception keeps the normal ERROR + report path.
    """
    from continuum.exceptions import PolicyDeniedError

    span.add_metadata("error_type", type(e).__name__)
    if isinstance(e, PolicyDeniedError):
        span.level = "WARNING"
        span.add_metadata("policy_denied", True)
        span.add_metadata("denial", str(e))
        return

    span.set_error(str(e))
    from continuum.observability.error_reporter import report_error

    report_error(e, context=f"observe.{span_name}", trace_id=get_current_trace_id())


def _get_function_input(func: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    """Extract function input as a dictionary."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _serialize_output(output: Any) -> Any:
    """Serialize output for tracing."""
    if output is None:
        return None

    # Handle common types
    if isinstance(output, (str, int, float, bool)):
        return output

    if isinstance(output, (list, tuple)):
        return [_serialize_output(item) for item in output[:100]]  # Limit list size

    if isinstance(output, dict):
        return {k: _serialize_output(v) for k, v in list(output.items())[:50]}

    # Try to convert to dict if possible
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "__dict__"):
        return {k: v for k, v in output.__dict__.items() if not k.startswith("_")}

    return str(output)[:1000]  # Truncate long strings


def observe(
    name: str | None = None,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: dict[str, Any] | None = None,
    level: SpanLevel = SpanLevel.DEFAULT,
    manager: TracingManager | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace a function execution.

    Creates a span for the function call with input/output capture.

    Args:
        name: Name for the span (defaults to function name)
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        metadata: Additional metadata to include
        level: Log level for the span
        manager: Optional TracingManager instance

    Example:
        ```python
        @observe()
        def process_data(data: dict) -> dict:
            return transform(data)

        @observe(name="custom-name", capture_output=False)
        async def fetch_data(url: str) -> Response:
            return await client.get(url)
        ```
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Prepare input
            input_data = None
            if capture_input:
                try:
                    # Redaction is applied centrally in SpanScope (see _create_span).
                    input_data = truncate_data(_get_function_input(func, args, kwargs))
                except Exception as e:
                    logger.debug(f"Failed to capture input: {e}")

            # Use SpanScope from trace_context (async-safe, links to current trace)
            with SpanScope(
                span_name,
                input=input_data,
                metadata=metadata or {},
                level=level.value if isinstance(level, SpanLevel) else level,
            ) as span:
                try:
                    result = func(*args, **kwargs)

                    if capture_output:
                        # Redaction is applied centrally in SpanScope.set_output.
                        span.set_output(truncate_data(_serialize_output(result)))

                    return result
                except Exception as e:
                    _record_observe_exception(span, span_name, e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Prepare input
            input_data = None
            if capture_input:
                try:
                    # Redaction is applied centrally in SpanScope (see _create_span).
                    input_data = truncate_data(_get_function_input(func, args, kwargs))
                except Exception as e:
                    logger.debug(f"Failed to capture input: {e}")

            # Use SpanScope from trace_context (async-safe, links to current trace)
            async with SpanScope(
                span_name,
                input=input_data,
                metadata=metadata or {},
                level=level.value if isinstance(level, SpanLevel) else level,
            ) as span:
                try:
                    result = await func(*args, **kwargs)

                    if capture_output:
                        # Redaction is applied centrally in SpanScope.set_output.
                        span.set_output(truncate_data(_serialize_output(result)))

                    return result
                except Exception as e:
                    _record_observe_exception(span, span_name, e)
                    raise

        @functools.wraps(func)
        async def async_gen_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            # Async generators MUST be driven inside the span scope: calling the
            # function only creates the generator — its body (and any exception,
            # e.g. a PolicyDeniedError gate) runs lazily as the consumer iterates.
            # The plain async_wrapper would close the span before the first
            # __anext__, recording ~0s and never seeing the deny — so streaming
            # spans (llm_chat_stream) would silently drop the WARNING the
            # non-streaming path records. Iterate here so the span stays open
            # across the whole stream and exceptions are recorded.
            input_data = None
            if capture_input:
                try:
                    input_data = truncate_data(_get_function_input(func, args, kwargs))
                except Exception as e:
                    logger.debug(f"Failed to capture input: {e}")

            async with SpanScope(
                span_name,
                input=input_data,
                metadata=metadata or {},
                level=level.value if isinstance(level, SpanLevel) else level,
            ) as span:
                collected: list[Any] = []
                try:
                    async for item in func(*args, **kwargs):
                        if capture_output:
                            collected.append(item)
                        yield item
                    if capture_output and collected:
                        span.set_output(truncate_data(_serialize_output(collected)))
                except Exception as e:
                    _record_observe_exception(span, span_name, e)
                    raise

        @functools.wraps(func)
        def sync_gen_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            # Sync analogue of async_gen_wrapper (see note there): drive the
            # generator inside the span so lazy bodies/exceptions are captured.
            input_data = None
            if capture_input:
                try:
                    input_data = truncate_data(_get_function_input(func, args, kwargs))
                except Exception as e:
                    logger.debug(f"Failed to capture input: {e}")

            with SpanScope(
                span_name,
                input=input_data,
                metadata=metadata or {},
                level=level.value if isinstance(level, SpanLevel) else level,
            ) as span:
                collected: list[Any] = []
                try:
                    for item in func(*args, **kwargs):
                        if capture_output:
                            collected.append(item)
                        yield item
                    if capture_output and collected:
                        span.set_output(truncate_data(_serialize_output(collected)))
                except Exception as e:
                    _record_observe_exception(span, span_name, e)
                    raise

        # Order matters: an async generator is neither a coroutine function nor a
        # (sync) generator function, so check the most specific kinds first.
        if inspect.isasyncgenfunction(func):
            return async_gen_wrapper  # type: ignore
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        if inspect.isgeneratorfunction(func):
            return sync_gen_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def trace_tool(
    name: str | None = None,
    *,
    tool_type: str = "function",
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: dict[str, Any] | None = None,
    manager: TracingManager | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace a tool/function call.

    Creates a span specifically for tool invocations with tool-specific metadata.

    Args:
        name: Name for the span (defaults to function name)
        tool_type: Type of tool (function, api, database, etc.)
        capture_input: Whether to capture tool input
        capture_output: Whether to capture tool output
        metadata: Additional metadata
        manager: Optional TracingManager instance

    Example:
        ```python
        @trace_tool()
        def search_database(query: str) -> list[dict]:
            return db.search(query)

        @trace_tool(tool_type="api")
        async def call_weather_api(location: str) -> dict:
            return await weather_client.get(location)
        ```
    """
    tool_metadata = {"tool_type": tool_type}
    if metadata:
        tool_metadata.update(metadata)

    return observe(
        name=name,
        capture_input=capture_input,
        capture_output=capture_output,
        metadata=tool_metadata,
        level=SpanLevel.DEFAULT,
        manager=manager,
    )


def trace_agent(
    name: str | None = None,
    *,
    agent_type: str = "agent",
    capture_input: bool = True,
    capture_output: bool = True,
    metadata: dict[str, Any] | None = None,
    create_new_trace: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace an agent execution.

    UPDATED: Now creates a SPAN under the existing trace context by default.
    Only creates a new trace if create_new_trace=True or no trace context exists.

    This prevents the issue of having two disconnected traces when AgentRunner
    creates its own trace.

    Args:
        name: Name for the span/trace (defaults to function name)
        agent_type: Type of agent
        capture_input: Whether to capture agent input
        capture_output: Whether to capture agent output
        metadata: Additional metadata
        create_new_trace: If True, always create new trace. If False, create span under existing trace.

    Example:
        ```python
        @trace_agent()
        async def run_assistant(user_message: str, user_id: str) -> str:
            # Agent logic here
            return response

        @trace_agent(name="research-agent", agent_type="research")
        async def research(query: str) -> ResearchResult:
            # Research agent logic
            return result
        ```
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        trace_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Prepare input
            input_data = None
            if capture_input:
                try:
                    # Redaction is applied centrally in SpanScope (see _create_span).
                    input_data = truncate_data(_get_function_input(func, args, kwargs))
                except Exception as e:
                    logger.debug(f"Failed to capture input: {e}")

            # Create agent metadata
            agent_metadata = {"agent_type": agent_type}
            if metadata:
                agent_metadata.update(metadata)

            # Check if there's an existing trace context
            existing_trace_id = get_current_trace_id()

            # CRITICAL: Always prefer creating span under existing trace.
            # Only create new trace if explicitly requested AND no trace exists.
            if create_new_trace and not existing_trace_id:
                # Explicitly requested new trace and no trace exists - create trace
                user_id = kwargs.get("user_id") or (
                    input_data.get("user_id") if input_data else None
                )
                session_id = kwargs.get("session_id") or (
                    input_data.get("session_id") if input_data else None
                )

                from continuum.observability.provider_manager import get_provider_manager

                manager = get_provider_manager()

                trace = None
                if manager.is_enabled:
                    trace = manager.trace(
                        name=trace_name,
                        user_id=user_id,
                        session_id=session_id,
                        input=input_data,
                        metadata=agent_metadata,
                    )
                    # Set trace context for child operations
                    if trace:
                        from continuum.observability.trace_context import (
                            get_current_trace_client,
                            set_trace_context,
                        )

                        trace_client = get_current_trace_client()
                        set_trace_context(
                            trace_id=trace.id if hasattr(trace, "id") else None,
                            trace_client=trace_client,
                            user_id=user_id,
                            session_id=session_id,
                        )

                try:
                    result = func(*args, **kwargs)
                    if trace and capture_output:
                        if hasattr(trace, "update"):
                            trace.update(output=_serialize_output(result))
                    return result
                except Exception as e:
                    from continuum.exceptions import PolicyDeniedError

                    policy_denied = isinstance(e, PolicyDeniedError)
                    if trace and hasattr(trace, "update"):
                        trace.update(
                            metadata={
                                **agent_metadata,
                                # expected denial → audit field, not an "error"
                                ("policy_denied" if policy_denied else "error"): str(e),
                                "error_type": type(e).__name__,
                            }
                        )
                    # Expected policy denials are not escalated to error reporting.
                    if not policy_denied:
                        from continuum.observability.error_reporter import report_error

                        trace_id = trace.id if hasattr(trace, "id") and trace.id else None
                        report_error(
                            e, context=f"agent.{trace_name}", trace_id=trace_id, user_id=user_id
                        )
                    raise
                finally:
                    # Flush ProviderManager
                    try:
                        from continuum.observability.provider_manager import get_provider_manager

                        get_provider_manager().flush()
                    except Exception:
                        pass
            else:
                # Default behavior: Create span under existing trace (or skip if no trace)
                if existing_trace_id:
                    # Create span under existing trace
                    with SpanScope(
                        f"agent.{trace_name}",
                        input=input_data,
                        metadata=agent_metadata,
                    ) as span:
                        try:
                            result = func(*args, **kwargs)
                            if capture_output:
                                span.set_output(truncate_data(_serialize_output(result)))
                            return result
                        except Exception as e:
                            span.set_error(str(e))
                            span.add_metadata("error_type", type(e).__name__)
                            from continuum.observability.error_reporter import report_error

                            report_error(
                                e, context=f"agent.{trace_name}", trace_id=existing_trace_id
                            )
                            raise
                else:
                    # No trace context and not explicitly creating one - skip tracing
                    logger.warning(
                        f"@trace_agent on '{trace_name}' called without trace context. "
                        "Skipping trace creation (use create_new_trace=True to create trace). "
                        "Function will execute without tracing."
                    )
                    return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Prepare input
            input_data = None
            if capture_input:
                try:
                    # Redaction is applied centrally in SpanScope (see _create_span).
                    input_data = truncate_data(_get_function_input(func, args, kwargs))
                except Exception as e:
                    logger.debug(f"Failed to capture input: {e}")

            # Create agent metadata
            agent_metadata = {"agent_type": agent_type}
            if metadata:
                agent_metadata.update(metadata)

            # Check if there's an existing trace context
            existing_trace_id = get_current_trace_id()

            if existing_trace_id and not create_new_trace:
                # Create span under existing trace
                async with SpanScope(
                    f"agent.{trace_name}",
                    input=input_data,
                    metadata=agent_metadata,
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        if capture_output:
                            span.set_output(truncate_data(_serialize_output(result)))
                        return result
                    except Exception as e:
                        span.set_error(str(e))
                        span.add_metadata("error_type", type(e).__name__)
                        from continuum.observability.error_reporter import report_error

                        report_error(e, context=f"agent.{trace_name}", trace_id=existing_trace_id)
                        raise
            else:
                # Create new trace using ProviderManager
                user_id = kwargs.get("user_id") or (
                    input_data.get("user_id") if input_data else None
                )
                session_id = kwargs.get("session_id") or (
                    input_data.get("session_id") if input_data else None
                )

                from continuum.observability.provider_manager import get_provider_manager

                manager = get_provider_manager()

                trace = None
                if manager.is_enabled:
                    trace = manager.trace(
                        name=trace_name,
                        user_id=user_id,
                        session_id=session_id,
                        input=input_data,
                        metadata=agent_metadata,
                    )

            try:
                result = await func(*args, **kwargs)
                if trace and capture_output:
                    if hasattr(trace, "update"):
                        trace.update(output=_serialize_output(result))
                return result
            except Exception as e:
                if trace:
                    if hasattr(trace, "update"):
                        trace.update(
                            metadata={
                                **agent_metadata,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            }
                        )
                from continuum.observability.error_reporter import report_error

                trace_id = trace.id if hasattr(trace, "id") and trace.id else None
                report_error(e, context=f"agent.{trace_name}", trace_id=trace_id, user_id=user_id)
                raise
            finally:
                # Flush ProviderManager
                try:
                    from continuum.observability.provider_manager import get_provider_manager

                    get_provider_manager().flush()
                except Exception:
                    pass

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class ObservationContext:
    """
    Context manager for manual observation control.

    Use this when decorators don't fit your use case and you need
    manual control over span lifecycle.

    Example:
        ```python
        from continuum.observability import ObservationContext

        async def process_request(request):
            with ObservationContext("process-request") as ctx:
                ctx.set_input(request)

                # Do processing...
                result = await do_work(request)

                ctx.set_output(result)
                ctx.add_metadata({"items_processed": len(result)})

            return result
        ```
    """

    def __init__(
        self,
        name: str,
        *,
        span_type: str = "span",
        metadata: dict[str, Any] | None = None,
        manager: TracingManager | None = None,
    ):
        self.name = name
        self.span_type = span_type
        self._metadata = metadata or {}
        self._manager = manager
        self._input: Any = None
        self._output: Any = None
        self._span: Any = None

    def set_input(self, input_data: Any) -> None:
        """Set the input data for this observation."""
        self._input = input_data
        if self._span:
            self._span.update(input=input_data)

    def set_output(self, output_data: Any) -> None:
        """Set the output data for this observation."""
        self._output = output_data

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        """Add metadata to this observation."""
        self._metadata.update(metadata)
        if self._span:
            self._span.update(metadata=metadata)

    def set_level(self, level: SpanLevel) -> None:
        """Set the log level for this observation."""
        if self._span:
            self._span.update(level=level)

    def set_error(self, error: Exception) -> None:
        """Mark this observation as an error."""
        self._metadata["error"] = str(error)
        self._metadata["error_type"] = type(error).__name__
        if self._span:
            self._span.update(
                level=SpanLevel.ERROR,
                status_message=str(error),
                metadata=self._metadata,
            )

    def __enter__(self) -> ObservationContext:
        # Get manager
        if self._manager is None:
            from continuum.observability.tracing import TracingManager

            self._manager = TracingManager()

        # Create span
        current_trace = self._manager.get_current_trace()
        current_span = self._manager.get_current_span()

        parent = current_span or current_trace
        if parent:
            self._span = parent.span(
                self.name,
                input=self._input,
                metadata=self._metadata,
            )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span:
            if exc_type is not None:
                self._span.end(
                    output=self._output,
                    level=SpanLevel.ERROR,
                    status_message=str(exc_val),
                    metadata={**self._metadata, "error_type": exc_type.__name__},
                )
            else:
                self._span.end(output=self._output, metadata=self._metadata)
