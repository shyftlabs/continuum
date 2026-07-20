"""
Error reporter for automatic observability error reporting.

Provides a centralized error reporting system that automatically sends
all errors to observability providers for full observability and auditing.

Features:
- Automatic error reporting to observability providers
- Error categorization and severity levels
- Context propagation (trace_id, span_id, user_id)
- Batch reporting for high-volume scenarios
- Graceful degradation when providers are unavailable
"""

from __future__ import annotations

import atexit
import threading
import traceback
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from continuum.config import settings
from continuum.utils.secrets import redact_dict, redact_sensitive_values

if TYPE_CHECKING:
    from continuum.exceptions import OrchestratorError


# Thread-safe error queue for batch reporting
_error_queue: deque[dict[str, Any]] = deque(maxlen=1000)
_queue_lock = threading.Lock()
_flush_timer: threading.Timer | None = None

# Global reporter state
_reporter_initialized = False
_reporter_enabled = True


class ErrorReporter:
    """
    Centralized error reporter for observability providers.

    Handles all error reporting to observability providers with support for:
    - Immediate reporting for critical errors
    - Batch reporting for non-critical errors
    - Automatic context enrichment
    - Graceful failure handling

    Example:
        ```python
        from continuum.observability.error_reporter import ErrorReporter

        reporter = ErrorReporter()

        # Report an error immediately
        reporter.report(error, trace_id="abc-123")

        # Report with additional context
        reporter.report(
            error,
            context="llm_call",
            metadata={"model": "gpt-4", "attempt": 2}
        )
        ```
    """

    def __init__(self, auto_flush_interval: float = 5.0):
        """
        Initialize the error reporter.

        Args:
            auto_flush_interval: Seconds between automatic flushes (0 to disable)
        """
        self._auto_flush_interval = auto_flush_interval
        self._initialized = False

    def _get_provider_manager(self) -> Any:
        """Get ProviderManager instance."""
        from continuum.observability.provider_manager import get_provider_manager

        return get_provider_manager()

    @property
    def is_enabled(self) -> bool:
        """Check if error reporting is enabled."""
        if not _reporter_enabled:
            return False
        manager = self._get_provider_manager()
        return manager is not None and manager.is_enabled

    def report(
        self,
        error: OrchestratorError | Exception,
        *,
        context: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        immediate: bool | None = None,
    ) -> None:
        """
        Report an error to observability providers.

        Args:
            error: The error to report
            context: Error context (e.g., "llm_call", "tool_execution", "tracing")
            trace_id: Associated trace ID
            span_id: Associated span ID
            user_id: User who encountered the error
            session_id: Session ID
            metadata: Additional metadata
            immediate: Force immediate reporting (default: True for critical errors)
        """
        if not _reporter_enabled:
            return

        # Build error data
        error_data = self._build_error_data(
            error=error,
            context=context,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

        # Determine if immediate reporting is needed
        if immediate is None:
            from continuum.exceptions import ErrorSeverity, OrchestratorError

            if isinstance(error, OrchestratorError):
                immediate = error.severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL)
            else:
                immediate = True  # Standard exceptions are reported immediately

        if immediate:
            self._report_immediate(error_data)
        else:
            self._queue_error(error_data)

    def _build_error_data(
        self,
        error: OrchestratorError | Exception,
        context: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build error data dictionary."""
        from continuum.exceptions import OrchestratorError
        from continuum.logging import get_log_context

        # Get current log context for trace correlation
        log_context = get_log_context()

        # Use provided values or fall back to log context
        trace_id = trace_id or log_context.get("trace_id")
        span_id = span_id or log_context.get("span_id")
        user_id = user_id or log_context.get("user_id")
        session_id = session_id or log_context.get("session_id")

        # Build base error data
        error_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "environment": settings.environment,
        }

        # Handle OrchestratorError
        if isinstance(error, OrchestratorError):
            error_data.update(
                {
                    "error_type": error.__class__.__name__,
                    "error_code": error.error_code,
                    "message": redact_sensitive_values(error.message),
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "context_data": redact_dict(error.context) if error.context else {},
                }
            )
            trace_id = trace_id or error.trace_id
            span_id = span_id or error.span_id
            if error.original_error:
                error_data["original_error"] = {
                    "type": type(error.original_error).__name__,
                    "message": redact_sensitive_values(str(error.original_error)),
                }
        else:
            # Handle standard exceptions
            error_data.update(
                {
                    "error_type": type(error).__name__,
                    "error_code": "UNKNOWN_ERROR",
                    "message": redact_sensitive_values(str(error)),
                    "category": "unknown",
                    "severity": "high",
                }
            )

        # Add traceback
        error_data["traceback"] = traceback.format_exc()

        # Add context
        if context:
            error_data["context"] = context
        if trace_id:
            error_data["trace_id"] = trace_id
        if span_id:
            error_data["span_id"] = span_id
        if user_id:
            error_data["user_id"] = user_id
        if session_id:
            error_data["session_id"] = session_id
        if metadata:
            error_data["metadata"] = metadata

        return error_data

    def _report_immediate(self, error_data: dict[str, Any]) -> None:
        """Report error immediately to observability providers."""
        manager = self._get_provider_manager()
        if manager and manager.is_enabled:
            try:
                self._report_via_provider_manager(manager, error_data)
            except Exception:
                # Never let error reporting crash the application
                pass

    def _report_via_provider_manager(self, manager: Any, error_data: dict[str, Any]) -> None:
        """Report error via ProviderManager."""
        trace_id = error_data.get("trace_id")

        # Build event name
        error_type = error_data.get("error_type", "Error")
        context = error_data.get("context", "")
        event_name = f"error.{context}.{error_type}" if context else f"error.{error_type}"

        # Build metadata
        metadata = {
            "error_code": error_data.get("error_code"),
            "category": error_data.get("category"),
            "severity": error_data.get("severity"),
            "environment": error_data.get("environment"),
            "timestamp": error_data.get("timestamp"),
        }

        if error_data.get("context_data"):
            metadata["context_data"] = error_data["context_data"]
        if error_data.get("original_error"):
            metadata["original_error"] = error_data["original_error"]
        if error_data.get("metadata"):
            metadata.update(redact_dict(error_data["metadata"]))

        # Input is the error message and traceback
        input_data = {
            "message": error_data.get("message"),
            "traceback": error_data.get("traceback"),
        }

        if trace_id:
            # Report as event on existing trace
            manager.event(
                trace_id=trace_id,
                name=event_name,
                input=input_data,
                metadata=metadata,
                level="ERROR",
            )
        else:
            # Create a new trace for the error
            error_trace = manager.trace(
                name=f"error-{error_data.get('error_code', 'unknown')}",
                user_id=error_data.get("user_id"),
                session_id=error_data.get("session_id"),
                input=input_data,
                metadata=metadata,
                tags=["error", error_data.get("category", "unknown")],
            )
            if error_trace and hasattr(error_trace, "event"):
                # Add the error event
                error_trace.event(
                    name=event_name,
                    input=input_data,
                    metadata=metadata,
                    level="ERROR",
                )

    def _queue_error(self, error_data: dict[str, Any]) -> None:
        """Queue error for batch reporting."""
        with _queue_lock:
            _error_queue.append(error_data)

    def flush(self) -> None:
        """Flush all queued errors to observability providers."""
        if not _reporter_enabled:
            return

        errors_to_report: list[dict[str, Any]] = []
        with _queue_lock:
            while _error_queue:
                errors_to_report.append(_error_queue.popleft())

        for error_data in errors_to_report:
            self._report_immediate(error_data)

        # Flush via ProviderManager
        manager = self._get_provider_manager()
        if manager:
            try:
                manager.flush()
            except Exception:
                pass


# Global reporter instance
_global_reporter: ErrorReporter | None = None


def get_error_reporter() -> ErrorReporter:
    """Get the global error reporter instance."""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ErrorReporter()
    return _global_reporter


def report_error(
    error: OrchestratorError | Exception,
    *,
    context: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    immediate: bool | None = None,
) -> None:
    """
    Report an error to observability providers.

    Convenience function that uses the global error reporter.

    Args:
        error: The error to report
        context: Error context (e.g., "llm_call", "tool_execution")
        trace_id: Associated trace ID
        span_id: Associated span ID
        user_id: User who encountered the error
        session_id: Session ID
        metadata: Additional metadata
        immediate: Force immediate reporting

    Example:
        ```python
        from continuum.observability.error_reporter import report_error

        try:
            await llm_call()
        except LLMError as e:
            report_error(e, context="llm_call", metadata={"model": "gpt-4"})
            raise
        ```
    """
    # Expected access-control denials (data-label / policy gates) are NOT errors —
    # they are the governance layer working as designed. Do not create a
    # high-severity error trace or escalate/alert for them (this is the single
    # chokepoint, so every call site is covered). They remain visible for audit
    # as "policy denied" events on the span/trace via the observe decorator.
    from continuum.exceptions import PolicyDeniedError

    if isinstance(error, PolicyDeniedError):
        return

    reporter = get_error_reporter()
    reporter.report(
        error=error,
        context=context,
        trace_id=trace_id,
        span_id=span_id,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        immediate=immediate,
    )


def report_exception(
    context: str,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Report the current exception to observability providers.

    Call this in an except block to report the current exception.

    Example:
        ```python
        try:
            risky_operation()
        except Exception:
            report_exception("risky_operation")
            raise
        ```
    """
    import sys

    exc_info = sys.exc_info()
    if exc_info[1] is not None:
        report_error(
            exc_info[1],
            context=context,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            metadata=metadata,
        )


def enable_error_reporting() -> None:
    """Enable error reporting to Langfuse."""
    global _reporter_enabled
    _reporter_enabled = True


def disable_error_reporting() -> None:
    """Disable error reporting to observability providers."""
    global _reporter_enabled
    _reporter_enabled = False


def flush_errors() -> None:
    """Flush all queued errors to Langfuse."""
    reporter = get_error_reporter()
    reporter.flush()


# Register flush on exit
_CLEANUP_FLUSH_TIMEOUT_SECONDS = 5.0


def _cleanup() -> None:
    """Bounded flush on interpreter exit.

    The provider flush chain can block indefinitely (Langfuse's ``flush()``
    ends in an unbounded ``queue.join()``; if the upload queue can't drain —
    no network, no reachable server — it never returns and the process hangs
    at exit). Run the flush in a daemon thread with a bounded join: a
    shutdown flush is best-effort and must never wedge the process.
    """
    try:
        flusher = threading.Thread(
            target=flush_errors, name="error-reporter-exit-flush", daemon=True
        )
        flusher.start()
        flusher.join(timeout=_CLEANUP_FLUSH_TIMEOUT_SECONDS)
        # If still alive, the daemon thread dies with the interpreter — exit
        # proceeds instead of hanging on telemetry.
    except Exception:
        pass


atexit.register(_cleanup)


class ErrorReportingContext:
    """
    Context manager for error reporting with automatic context.

    Automatically reports any exceptions that occur within the context
    to observability providers with the provided context information.

    Example:
        ```python
        with ErrorReportingContext("llm_call", trace_id=trace.id):
            response = await llm.chat(messages)
        ```
    """

    def __init__(
        self,
        context: str,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        reraise: bool = True,
    ):
        """
        Initialize error reporting context.

        Args:
            context: Error context name
            trace_id: Associated trace ID
            span_id: Associated span ID
            user_id: User ID
            session_id: Session ID
            metadata: Additional metadata
            reraise: Whether to re-raise exceptions after reporting
        """
        self.context = context
        self.trace_id = trace_id
        self.span_id = span_id
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.reraise = reraise

    def __enter__(self) -> ErrorReportingContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_val is not None:
            report_error(
                exc_val,
                context=self.context,
                trace_id=self.trace_id,
                span_id=self.span_id,
                user_id=self.user_id,
                session_id=self.session_id,
                metadata=self.metadata,
            )
        # Return True to suppress exception, False to propagate
        return not self.reraise if exc_val else False

    async def __aenter__(self) -> ErrorReportingContext:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)
