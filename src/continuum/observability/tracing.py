"""
Core tracing types and manager.

Provides the TracingManager singleton and trace/span abstractions.

NOTE: TracingManager now uses async-safe contextvars for state management.
This ensures correct behavior in concurrent async operations.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from continuum.logging import get_logger
from continuum.observability.trace_context import (
    TraceScope,
    get_current_span_client,
    get_current_span_id,
    get_current_trace_client,
    get_current_trace_id,
    restore_trace_context,
    set_trace_context,
)

if TYPE_CHECKING:
    from langfuse.client import StatefulGenerationClient, StatefulSpanClient, StatefulTraceClient

logger = get_logger(__name__)


class SpanLevel(str, Enum):
    """Log level for spans."""

    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SpanKind(str, Enum):
    """Type of span."""

    SPAN = "span"
    GENERATION = "generation"
    EVENT = "event"


class TraceData(BaseModel):
    """Data model for a trace."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    user_id: str | None = None
    session_id: str | None = None
    input: Any | None = None
    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    version: str | None = None
    release: str | None = None
    public: bool = False


class SpanData(BaseModel):
    """Data model for a span."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    input: Any | None = None
    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    level: SpanLevel = SpanLevel.DEFAULT
    status_message: str | None = None
    version: str | None = None


class GenerationData(BaseModel):
    """Data model for a generation (LLM call) span."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    model: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    input: Any | None = None
    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    level: SpanLevel = SpanLevel.DEFAULT
    status_message: str | None = None
    version: str | None = None
    # Usage metrics
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None
    # Cost tracking
    usage_input_cost: float | None = None
    usage_output_cost: float | None = None
    usage_total_cost: float | None = None


class Span:
    """
    Represents a span within a trace.

    A span is a unit of work within a trace, such as a tool invocation
    or a processing step.
    """

    def __init__(
        self,
        langfuse_span: StatefulSpanClient | None,
        data: SpanData,
    ):
        self._langfuse_span = langfuse_span
        self._data = data
        self._children: list[Span | GenerationSpan] = []

    @property
    def id(self) -> str:
        """Get the span ID."""
        return self._data.id

    @property
    def name(self) -> str:
        """Get the span name."""
        return self._data.name

    def update(
        self,
        *,
        name: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel | None = None,
        status_message: str | None = None,
    ) -> Span:
        """Update span data."""
        if name:
            self._data.name = name
        if input is not None:
            self._data.input = input
        if output is not None:
            self._data.output = output
        if metadata:
            self._data.metadata.update(metadata)
        if level:
            self._data.level = level
        if status_message:
            self._data.status_message = status_message

        if self._langfuse_span:
            try:
                self._langfuse_span.update(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level.value if level else None,
                    status_message=status_message,
                )
            except Exception as e:
                logger.warning("Failed to update Langfuse span: %s", e)

        return self

    def end(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel | None = None,
        status_message: str | None = None,
    ) -> None:
        """End the span."""
        self._data.end_time = datetime.now(UTC)

        if output is not None:
            self._data.output = output
        if metadata:
            self._data.metadata.update(metadata)
        if level:
            self._data.level = level
        if status_message:
            self._data.status_message = status_message

        if self._langfuse_span:
            try:
                self._langfuse_span.end(
                    output=output,
                    metadata=metadata,
                    level=level.value if level else None,
                    status_message=status_message,
                )
            except Exception as e:
                logger.warning("Failed to end Langfuse span: %s", e)

    def span(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel = SpanLevel.DEFAULT,
    ) -> Span:
        """Create a child span."""
        data = SpanData(
            name=name,
            input=input,
            metadata=metadata or {},
            level=level,
        )

        langfuse_span = None
        if self._langfuse_span:
            try:
                langfuse_span = self._langfuse_span.span(
                    id=data.id,
                    name=name,
                    input=input,
                    metadata=metadata,
                    level=level.value,
                )
            except Exception as e:
                logger.warning("Failed to create Langfuse child span: %s", e)

        child = Span(langfuse_span, data)
        self._children.append(child)
        return child

    def generation(
        self,
        name: str,
        *,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationSpan:
        """Create a generation (LLM call) span."""
        data = GenerationData(
            name=name,
            model=model,
            model_parameters=model_parameters or {},
            input=input,
            metadata=metadata or {},
        )

        langfuse_generation = None
        if self._langfuse_span:
            try:
                langfuse_generation = self._langfuse_span.generation(
                    id=data.id,
                    name=name,
                    model=model,
                    model_parameters=model_parameters,
                    input=input,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning("Failed to create Langfuse generation: %s", e)

        child = GenerationSpan(langfuse_generation, data)
        self._children.append(child)
        return child

    def event(
        self,
        name: str,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel = SpanLevel.DEFAULT,
    ) -> None:
        """Log an event within this span."""
        if self._langfuse_span:
            try:
                self._langfuse_span.event(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level.value,
                )
            except Exception as e:
                logger.warning("Failed to log Langfuse event: %s", e)

    def score(
        self,
        name: str,
        value: float,
        *,
        comment: str | None = None,
        data_type: str | None = None,
    ) -> None:
        """Add a score to this span."""
        if self._langfuse_span:
            try:
                self._langfuse_span.score(
                    name=name,
                    value=value,
                    comment=comment,
                    data_type=data_type,
                )
            except Exception as e:
                logger.warning("Failed to add Langfuse score: %s", e)


class GenerationSpan:
    """
    Represents an LLM generation span.

    A generation span is specifically for LLM calls and includes
    model, token usage, and cost information.
    """

    def __init__(
        self,
        langfuse_generation: StatefulGenerationClient | None,
        data: GenerationData,
    ):
        self._langfuse_generation = langfuse_generation
        self._data = data

    @property
    def id(self) -> str:
        """Get the generation ID."""
        return self._data.id

    @property
    def name(self) -> str:
        """Get the generation name."""
        return self._data.name

    def update(
        self,
        *,
        name: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel | None = None,
        status_message: str | None = None,
        usage_prompt_tokens: int | None = None,
        usage_completion_tokens: int | None = None,
        usage_total_tokens: int | None = None,
    ) -> GenerationSpan:
        """Update generation data."""
        if name:
            self._data.name = name
        if model:
            self._data.model = model
        if model_parameters:
            self._data.model_parameters.update(model_parameters)
        if input is not None:
            self._data.input = input
        if output is not None:
            self._data.output = output
        if metadata:
            self._data.metadata.update(metadata)
        if level:
            self._data.level = level
        if status_message:
            self._data.status_message = status_message
        if usage_prompt_tokens is not None:
            self._data.usage_prompt_tokens = usage_prompt_tokens
        if usage_completion_tokens is not None:
            self._data.usage_completion_tokens = usage_completion_tokens
        if usage_total_tokens is not None:
            self._data.usage_total_tokens = usage_total_tokens

        if self._langfuse_generation:
            try:
                usage = None
                if any([usage_prompt_tokens, usage_completion_tokens, usage_total_tokens]):
                    usage = {
                        "prompt_tokens": usage_prompt_tokens,
                        "completion_tokens": usage_completion_tokens,
                        "total_tokens": usage_total_tokens,
                    }
                    # Remove None values
                    usage = {k: v for k, v in usage.items() if v is not None}

                self._langfuse_generation.update(
                    name=name,
                    model=model,
                    model_parameters=model_parameters,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level.value if level else None,
                    status_message=status_message,
                    usage=usage,
                )
            except Exception as e:
                logger.warning("Failed to update Langfuse generation: %s", e)

        return self

    def end(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel | None = None,
        status_message: str | None = None,
        usage_prompt_tokens: int | None = None,
        usage_completion_tokens: int | None = None,
        usage_total_tokens: int | None = None,
    ) -> None:
        """End the generation."""
        self._data.end_time = datetime.now(UTC)

        if output is not None:
            self._data.output = output
        if metadata:
            self._data.metadata.update(metadata)
        if level:
            self._data.level = level
        if status_message:
            self._data.status_message = status_message
        if usage_prompt_tokens is not None:
            self._data.usage_prompt_tokens = usage_prompt_tokens
        if usage_completion_tokens is not None:
            self._data.usage_completion_tokens = usage_completion_tokens
        if usage_total_tokens is not None:
            self._data.usage_total_tokens = usage_total_tokens

        if self._langfuse_generation:
            try:
                usage = None
                if any([usage_prompt_tokens, usage_completion_tokens, usage_total_tokens]):
                    usage = {
                        "prompt_tokens": usage_prompt_tokens,
                        "completion_tokens": usage_completion_tokens,
                        "total_tokens": usage_total_tokens,
                    }
                    usage = {k: v for k, v in usage.items() if v is not None}

                self._langfuse_generation.end(
                    output=output,
                    metadata=metadata,
                    level=level.value if level else None,
                    status_message=status_message,
                    usage=usage,
                )
            except Exception as e:
                logger.warning("Failed to end Langfuse generation: %s", e)

    def score(
        self,
        name: str,
        value: float,
        *,
        comment: str | None = None,
        data_type: str | None = None,
    ) -> None:
        """Add a score to this generation."""
        if self._langfuse_generation:
            try:
                self._langfuse_generation.score(
                    name=name,
                    value=value,
                    comment=comment,
                    data_type=data_type,
                )
            except Exception as e:
                logger.warning("Failed to add Langfuse score: %s", e)


class Trace:
    """
    Represents a trace (one agent run / user query cycle).

    A trace groups together all spans and events for a single
    user interaction or agent execution.
    """

    def __init__(
        self,
        langfuse_trace: StatefulTraceClient | None,
        data: TraceData,
    ):
        self._langfuse_trace = langfuse_trace
        self._data = data
        self._spans: list[Span] = []
        self._generations: list[GenerationSpan] = []

    @property
    def id(self) -> str:
        """Get the trace ID."""
        return self._data.id

    @property
    def name(self) -> str:
        """Get the trace name."""
        return self._data.name

    @property
    def langfuse_trace(self) -> StatefulTraceClient | None:
        """Get the underlying Langfuse trace client."""
        return self._langfuse_trace

    def update(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        public: bool | None = None,
    ) -> Trace:
        """Update trace data."""
        if name:
            self._data.name = name
        if user_id:
            self._data.user_id = user_id
        if session_id:
            self._data.session_id = session_id
        if input is not None:
            self._data.input = input
        if output is not None:
            self._data.output = output
        if metadata:
            self._data.metadata.update(metadata)
        if tags:
            self._data.tags.extend(tags)
        if public is not None:
            self._data.public = public

        if self._langfuse_trace:
            try:
                self._langfuse_trace.update(
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    input=input,
                    output=output,
                    metadata=metadata,
                    tags=tags,
                    public=public,
                )
            except Exception as e:
                logger.warning("Failed to update Langfuse trace: %s", e)

        return self

    def span(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel = SpanLevel.DEFAULT,
    ) -> Span:
        """Create a span within this trace."""
        data = SpanData(
            name=name,
            input=input,
            metadata=metadata or {},
            level=level,
        )

        langfuse_span = None
        if self._langfuse_trace:
            try:
                langfuse_span = self._langfuse_trace.span(
                    id=data.id,
                    name=name,
                    input=input,
                    metadata=metadata,
                    level=level.value,
                )
            except Exception as e:
                logger.warning("Failed to create Langfuse span: %s", e)

        span = Span(langfuse_span, data)
        self._spans.append(span)
        return span

    def generation(
        self,
        name: str,
        *,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationSpan:
        """Create a generation (LLM call) span within this trace."""
        data = GenerationData(
            name=name,
            model=model,
            model_parameters=model_parameters or {},
            input=input,
            metadata=metadata or {},
        )

        langfuse_generation = None
        if self._langfuse_trace:
            try:
                langfuse_generation = self._langfuse_trace.generation(
                    id=data.id,
                    name=name,
                    model=model,
                    model_parameters=model_parameters,
                    input=input,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning("Failed to create Langfuse generation: %s", e)

        gen = GenerationSpan(langfuse_generation, data)
        self._generations.append(gen)
        return gen

    def event(
        self,
        name: str,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel = SpanLevel.DEFAULT,
    ) -> None:
        """Log an event in this trace."""
        if self._langfuse_trace:
            try:
                self._langfuse_trace.event(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    level=level.value,
                )
            except Exception as e:
                logger.warning("Failed to log Langfuse event: %s", e)

    def score(
        self,
        name: str,
        value: float,
        *,
        comment: str | None = None,
        data_type: str | None = None,
    ) -> None:
        """Add a score to this trace."""
        if self._langfuse_trace:
            try:
                self._langfuse_trace.score(
                    name=name,
                    value=value,
                    comment=comment,
                    data_type=data_type,
                )
            except Exception as e:
                logger.warning("Failed to add Langfuse score: %s", e)

    def get_trace_url(self) -> str | None:
        """Get the URL to view this trace in Langfuse UI."""
        if self._langfuse_trace:
            try:
                return self._langfuse_trace.get_trace_url()
            except Exception:
                pass
        return None


class TracingManager:
    """
    Manages tracing and observability.

    This is the main entry point for creating traces and managing
    the observability backend.

    Example:
        ```python
        from continuum.observability import TracingManager

        manager = TracingManager()

        # Create a trace for an agent run
        with manager.trace("agent-run", user_id="user-123") as trace:
            # Create a span for a tool call
            with manager.span("tool-call", input={"query": "..."}) as span:
                result = call_tool()
                span.end(output=result)

            # Create a generation for an LLM call
            gen = trace.generation("llm-call", model="gpt-4")
            response = await llm.chat(messages)
            gen.end(output=response, usage_total_tokens=100)
        ```
    """

    def __init__(self):
        """
        Initialize the tracing manager.

        NOTE: TracingManager now uses async-safe contextvars from trace_context.py
        for _current_trace and _current_span. This ensures correct behavior in
        concurrent async operations.
        """
        self._provider_manager = None
        # NOTE: _current_trace and _current_span are now async-safe via contextvars
        # Access via get_current_trace() and get_current_span() methods

    def _get_provider_manager(self):
        """Get or create ProviderManager instance."""
        if self._provider_manager is None:
            from continuum.observability.provider_manager import get_provider_manager

            self._provider_manager = get_provider_manager()
        return self._provider_manager

    @property
    def _current_trace(self) -> Trace | None:
        """Get current trace from async-safe context."""
        # Check if we have a Langfuse trace client in context
        trace_client = get_current_trace_client()
        trace_id = get_current_trace_id()
        if trace_client and trace_id:
            # Create a Trace wrapper for the client
            data = TraceData(id=trace_id, name="context-trace")
            return Trace(trace_client, data)
        return None

    @property
    def _current_span(self) -> Span | None:
        """Get current span from async-safe context."""
        span_client = get_current_span_client()
        span_id = get_current_span_id()
        if span_client and span_id:
            # Create a Span wrapper for the client
            data = SpanData(id=span_id, name="context-span")
            return Span(span_client, data)
        return None

    def create_trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        version: str | None = None,
        release: str | None = None,
        public: bool = False,
        force: bool = False,
    ) -> Trace:
        """
        Create a new trace.

        Args:
            name: Name of the trace (e.g., "agent-run", "chat-completion")
            user_id: Optional user identifier
            session_id: Optional session identifier for grouping traces
            input: Input data for the trace
            metadata: Additional metadata
            tags: Tags for filtering/grouping
            version: Version identifier
            release: Release identifier
            public: Whether the trace should be publicly accessible
            force: If True, create trace even if one exists in context (default: False)

        Returns:
            A Trace object for adding spans and events.
        """
        # Guard: Check if trace already exists in context (unless forced)
        if not force:
            from continuum.observability.trace_context import (
                get_current_session_id,
                get_current_trace_client,
                get_current_trace_id,
                get_current_user_id,
            )

            existing_trace_id = get_current_trace_id()
            if existing_trace_id:
                logger.warning(
                    "Attempted to create trace '%s' but trace %s already exists in context. Returning existing trace context. Use force=True to override.",
                    name,
                    existing_trace_id,
                )
                # Return a Trace wrapper for the existing trace
                existing_client = get_current_trace_client()
                if existing_client:
                    # Create Trace wrapper for existing trace
                    data = TraceData(
                        name=name,
                        user_id=user_id or get_current_user_id(),
                        session_id=session_id or get_current_session_id(),
                        input=input,
                        metadata=metadata or {},
                        tags=tags or [],
                        version=version,
                        release=release,
                        public=public,
                    )
                    return Trace(existing_client, data)
                # If no client but trace ID exists, return None
                return None

        data = TraceData(
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            metadata=metadata or {},
            tags=tags or [],
            version=version,
            release=release,
            public=public,
        )

        # Create trace via ProviderManager
        manager = self._get_provider_manager()
        langfuse_trace = None

        if manager and manager.is_enabled:
            try:
                langfuse_trace = manager.trace(
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    input=input,
                    metadata=metadata,
                    tags=tags,
                    version=version,
                    public=public,
                )
                logger.debug("Created trace '%s' with ID %s", name, data.id)
            except Exception as e:
                logger.warning("Failed to create trace via ProviderManager: %s", e)

        trace = Trace(langfuse_trace, data)

        # Set trace context globally (async-safe)
        set_trace_context(
            trace_id=data.id,
            trace_client=langfuse_trace,
            user_id=user_id,
            session_id=session_id,
        )

        return trace

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Generator[Trace]:
        """
        Context manager for creating a trace.

        NOTE: Now uses async-safe TraceScope from trace_context.py
        to properly propagate context in concurrent operations.

        Example:
            ```python
            with manager.trace("my-workflow", user_id="user-123") as trace:
                # Do work...
                trace.event("step-completed", output={"status": "success"})
            ```
        """
        trace = self.create_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            metadata=metadata,
            tags=tags,
        )

        # Use TraceScope for async-safe context management
        with TraceScope(
            trace_id=trace.id,
            trace_client=trace._langfuse_trace,
            user_id=user_id,
            session_id=session_id,
        ):
            try:
                yield trace
            except Exception as e:
                trace.update(
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )
                trace.event(
                    name="error",
                    output={"error": str(e), "type": type(e).__name__},
                    level=SpanLevel.ERROR,
                )
                raise

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: SpanLevel = SpanLevel.DEFAULT,
    ) -> Generator[Span]:
        """
        Context manager for creating a span.

        Creates a span in the current trace or current parent span.
        Uses async-safe contextvars for proper context propagation.

        Example:
            ```python
            with manager.span("process-data", input=data) as span:
                result = process(data)
                span.end(output=result)
            ```
        """
        parent = self._current_span or self._current_trace

        if parent is None:
            # Create a dummy span if no trace context
            data = SpanData(name=name, input=input, metadata=metadata or {}, level=level)
            span = Span(None, data)
            context_token = None
        elif isinstance(parent, Trace):
            span = parent.span(name, input=input, metadata=metadata, level=level)
            # Update context with new span using async-safe contextvars
            context_token = set_trace_context(
                span_id=span.id,
                span_client=span._langfuse_span,
            )
        else:
            span = parent.span(name, input=input, metadata=metadata, level=level)
            # Update context with new span using async-safe contextvars
            context_token = set_trace_context(
                span_id=span.id,
                span_client=span._langfuse_span,
            )

        try:
            yield span
        except Exception as e:
            span.end(
                metadata={"error": str(e), "error_type": type(e).__name__},
                level=SpanLevel.ERROR,
                status_message=str(e),
            )
            raise
        finally:
            if span._data.end_time is None:
                span.end()
            # Restore previous context using the token
            if context_token is not None:
                restore_trace_context(context_token)

    def get_current_trace(self) -> Trace | None:
        """Get the current trace context."""
        return self._current_trace

    def get_current_span(self) -> Span | None:
        """Get the current span context."""
        return self._current_span

    def flush(self) -> None:
        """Flush all pending events."""
        manager = self._get_provider_manager()
        if manager:
            try:
                manager.flush()
            except Exception as e:
                logger.warning("Failed to flush via ProviderManager: %s", e)

    def shutdown(self) -> None:
        """Shutdown the tracing manager and flush all events."""
        self.flush()

        manager = self._get_provider_manager()
        if manager:
            try:
                manager.shutdown()
            except Exception as e:
                logger.warning("Failed to shutdown via ProviderManager: %s", e)
