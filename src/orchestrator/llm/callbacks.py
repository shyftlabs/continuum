"""
Callback handlers for LLM observability.

Provides Langfuse integration for logging and tracing LLM calls.
Uses the global Langfuse client from observability module.

IMPORTANT: This module now uses async-safe contextvars for trace context
propagation. The old global variables have been removed in favor of
orchestrator.observability.trace_context which works correctly with
concurrent async operations.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC
from typing import TYPE_CHECKING, Any

from orchestrator.config import settings
from orchestrator.logging import clear_log_context, get_log_context, get_logger, set_log_context

# Import async-safe trace context
from orchestrator.observability.trace_context import (
    TraceScope,
    get_current_session_id,
    get_current_span_id,
    get_current_trace_id,
    get_current_user_id,
    get_parent_observation_id,
)
from orchestrator.observability.trace_context import (
    clear_trace_context as _clear_trace_context,
)
from orchestrator.observability.trace_context import (
    set_trace_context as _set_trace_context,
)

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = get_logger(__name__)


def setup_langfuse() -> bool:
    """
    Initialize Langfuse observability.

    Traces are captured via the @observe decorator on LLMClient methods.
    Traces are captured via the @observe decorator on LLMClient methods.
    """
    from orchestrator.observability import (
        ObservabilityConfig,
        initialize_observability,
    )

    config = ObservabilityConfig()
    if not config.is_configured():
        logger.info("Observability not configured or disabled")
        return False

    manager = initialize_observability(config)
    if not manager.is_enabled:
        logger.info("Observability is disabled")
        return False

    from orchestrator.observability.providers.registry import get_provider

    langfuse_provider = get_provider("langfuse")
    if not langfuse_provider or not langfuse_provider.is_enabled:
        logger.info("Langfuse provider not available")
        return False

    client = langfuse_provider.client
    if not client or not client.is_enabled:
        logger.info("Langfuse client is disabled")
        return False

    logger.info("Langfuse observability initialized", extra={"host": settings.langfuse_host})
    return True


def get_langfuse_callback() -> Langfuse | None:
    """
    Get the current Langfuse client instance.

    Returns:
        The Langfuse client if initialized, None otherwise.
    """
    from orchestrator.observability.providers.registry import get_provider

    langfuse_provider = get_provider("langfuse")
    if langfuse_provider and langfuse_provider.is_enabled:
        return langfuse_provider.client.client if langfuse_provider.client else None
    return None


def set_trace_context(
    trace_id: str | None = None,
    span_id: str | None = None,
) -> None:
    """
    Set the current trace context for LLM calls.

    This allows LLM calls to be grouped under a specific trace.
    Also updates the logging context for correlation.

    NOTE: This is now async-safe using contextvars.

    Args:
        trace_id: The trace ID to associate LLM calls with.
        span_id: The parent span ID for nesting.
    """
    # Use async-safe context
    _set_trace_context(trace_id=trace_id, span_id=span_id)

    # Update logging context for correlation
    set_log_context(trace_id=trace_id, span_id=span_id)


def get_trace_context() -> tuple[str | None, str | None]:
    """
    Get the current trace context.

    NOTE: This is now async-safe using contextvars.

    Returns:
        Tuple of (trace_id, span_id).
    """
    return get_current_trace_id(), get_current_span_id()


def clear_trace_context() -> None:
    """
    Clear the current trace context.

    NOTE: This is now async-safe using contextvars.
    """
    _clear_trace_context()
    clear_log_context()


def get_langfuse_metadata(
    trace_id: str | None = None,
    span_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build metadata dict for Langfuse observability.

    This metadata is used to properly associate LLM calls with
    traces, sessions, and users.

    NOTE: Now uses async-safe contextvars for context propagation.

    Args:
        trace_id: Trace ID to group LLM calls under.
        span_id: Parent span ID for nesting.
        session_id: Session ID for grouping conversations.
        user_id: User identifier.
        tags: Tags to add to the trace.
        custom_metadata: Additional custom metadata.

    Returns:
        Metadata dictionary for Langfuse.
    """
    # CRITICAL: Always prioritize contextvars for trace context
    # This ensures LLM calls are always linked to the current trace
    effective_trace_id = trace_id or get_current_trace_id()
    effective_span_id = span_id or get_parent_observation_id()
    effective_session_id = session_id or get_current_session_id()
    effective_user_id = user_id or get_current_user_id()

    # Also check logging context as fallback (for compatibility)
    log_context = get_log_context()
    if not effective_trace_id:
        effective_trace_id = log_context.get("trace_id")
    if not effective_span_id:
        effective_span_id = log_context.get("span_id")

    # Log warning if no trace context found (helps diagnose issues)
    if not effective_trace_id:
        logger.debug(
            "LLM call made without trace context. "
            "Ensure trace is created before LLM calls (e.g., via AgentRunner._trace_run_start)."
        )

    metadata: dict[str, Any] = {}

    # Langfuse-specific metadata keys
    if effective_trace_id:
        metadata["trace_id"] = effective_trace_id
    if effective_span_id:
        metadata["parent_observation_id"] = effective_span_id
    if effective_session_id:
        metadata["session_id"] = effective_session_id
    if effective_user_id:
        metadata["trace_user_id"] = effective_user_id
    if tags:
        metadata["tags"] = tags

    # Add custom metadata
    if custom_metadata:
        metadata.update(custom_metadata)

    # Add environment info
    metadata["environment"] = settings.environment

    # Add SDK version for auditability
    from orchestrator import __version__

    metadata["sdk_version"] = __version__
    metadata["sdk_name"] = "orchestrator"

    # Add timestamp for audit trail
    from datetime import datetime

    metadata["timestamp"] = datetime.now(UTC).isoformat()

    return metadata



def flush_langfuse() -> None:
    """
    Flush any pending observability events.

    Call this before application shutdown to ensure all events are sent.
    """
    from orchestrator.observability.provider_manager import get_provider_manager

    get_provider_manager().flush()
    logger.debug("Langfuse events flushed")


def shutdown_langfuse() -> None:
    """
    Shutdown observability providers and cleanup resources.

    Call this on application shutdown.
    """
    from orchestrator.observability.provider_manager import get_provider_manager

    get_provider_manager().shutdown()
    logger.info("Observability providers shutdown complete")


@contextmanager
def trace_context(
    trace_id: str | None = None,
    span_id: str | None = None,
) -> Generator[None]:
    """
    Context manager for setting trace context.

    All LLM calls within this context will be associated with the given trace.

    Example:
        ```python
        with trace_context(trace_id="my-trace-id"):
            response = await client.chat(messages)
        ```
    """
    previous_trace_id, previous_span_id = get_trace_context()
    set_trace_context(trace_id, span_id)
    try:
        yield
    finally:
        set_trace_context(previous_trace_id, previous_span_id)


class LangfuseTraceContext:
    """
    Context manager for Langfuse tracing.

    Use this to create a trace that groups multiple LLM calls together.
    Now uses async-safe contextvars for context propagation.

    Example:
        ```python
        with LangfuseTraceContext(name="my-workflow") as trace:
            result1 = await client.chat(messages1)
            result2 = await client.chat(messages2)
        ```
    """

    def __init__(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        input: Any | None = None,
    ):
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.tags = tags or []
        self.input = input
        self.trace = None
        self._trace_scope: TraceScope | None = None
        self._client = None

    def _get_client(self):
        """Get the global Langfuse client."""
        if self._client is None:
            from orchestrator.observability.providers.registry import get_provider

            langfuse_provider = get_provider("langfuse")
            if langfuse_provider and langfuse_provider.is_enabled:
                self._client = langfuse_provider.client.client if langfuse_provider.client else None
            else:
                self._client = None
        return self._client

    def __enter__(self) -> LangfuseTraceContext:
        client = self._get_client()
        if client is not None and client.is_enabled:
            try:
                self.trace = client.trace(
                    name=self.name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    metadata=self.metadata,
                    tags=self.tags,
                    input=self.input,
                )
                if self.trace:
                    # Use async-safe TraceScope for context propagation
                    self._trace_scope = TraceScope(
                        trace_id=self.trace.id,
                        trace_client=self.trace,
                        user_id=self.user_id,
                        session_id=self.session_id,
                    )
                    self._trace_scope.__enter__()

                    # Update logging context for correlation
                    set_log_context(
                        trace_id=self.trace.id,
                        user_id=self.user_id,
                        session_id=self.session_id,
                    )
            except Exception as e:
                logger.error(f"Failed to create Langfuse trace: {e}")

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.trace is not None:
            try:
                if exc_type is not None:
                    # Report error to Langfuse
                    from orchestrator.observability.error_reporter import report_error

                    report_error(
                        exc_val,
                        context="trace",
                        trace_id=self.trace.id,
                        user_id=self.user_id,
                        session_id=self.session_id,
                    )
                    self.trace.update(
                        metadata={**self.metadata, "error": str(exc_val)},
                        level="ERROR",
                    )
            except Exception as e:
                logger.error(f"Failed to update Langfuse trace: {e}")

        # Restore previous context using TraceScope
        if self._trace_scope:
            self._trace_scope.__exit__(exc_type, exc_val, exc_tb)

    def get_trace_id(self) -> str | None:
        """Get the trace ID for correlation."""
        if self.trace is not None:
            return self.trace.id
        return None

    def update(
        self,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the trace with output or additional metadata."""
        if self.trace is not None:
            try:
                update_kwargs: dict[str, Any] = {}
                if output is not None:
                    update_kwargs["output"] = output
                if metadata:
                    update_kwargs["metadata"] = {**self.metadata, **metadata}
                    self.metadata.update(metadata)
                if update_kwargs:
                    self.trace.update(**update_kwargs)
            except Exception as e:
                logger.error(f"Failed to update Langfuse trace: {e}")

    def span(
        self,
        name: str,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a span within this trace."""
        if self.trace is not None:
            try:
                return self.trace.span(
                    name=name,
                    input=input,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to create Langfuse span: {e}")
        return None

    def event(
        self,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an event in this trace."""
        if self.trace is not None:
            try:
                self.trace.event(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to log Langfuse event: {e}")

    def score(
        self,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Add a score to this trace."""
        if self.trace is not None:
            try:
                self.trace.score(
                    name=name,
                    value=value,
                    comment=comment,
                )
            except Exception as e:
                logger.error(f"Failed to add Langfuse score: {e}")

    def get_trace_url(self) -> str | None:
        """Get the URL to view this trace in Langfuse UI."""
        if self.trace is not None:
            try:
                return self.trace.get_trace_url()
            except Exception:
                pass
        return None
