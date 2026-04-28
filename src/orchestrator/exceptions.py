"""
Generic exception types for the Orchestrator SDK.

Provides a hierarchy of exception classes for consistent error handling
across all SDK modules. All exceptions automatically report to Langfuse
when the error reporter is configured.

Exception Hierarchy:
    OrchestratorError (base)
    ├── ConfigurationError
    ├── ObservabilityError
    │   ├── LangfuseError
    │   └── TracingError
    ├── LLMError (defined in llm/exceptions.py, inherits from OrchestratorError)
    └── ValidationError
"""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from orchestrator.utils.secrets import redact_dict, redact_sensitive_values

# Module-level error reporter hook (set by observability module when initialized)
_error_reporter: Any = None


def set_error_reporter(reporter: Any) -> None:
    """Set the global error reporter function (called by observability init)."""
    global _error_reporter
    _error_reporter = reporter


def get_error_reporter() -> Any:
    """Get the current error reporter (for testing)."""
    return _error_reporter


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    LOW = "low"  # Minor issues, operation can continue
    MEDIUM = "medium"  # Degraded functionality
    HIGH = "high"  # Operation failed
    CRITICAL = "critical"  # System-level failure


class ErrorCategory(str, Enum):
    """Categories for error classification."""

    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    NETWORK = "network"
    PROVIDER = "provider"
    INTERNAL = "internal"
    OBSERVABILITY = "observability"
    UNKNOWN = "unknown"


class OrchestratorError(Exception):
    """
    Base exception class for all Orchestrator SDK errors.

    All exceptions in the SDK should inherit from this class to ensure
    consistent error handling, logging, and Langfuse reporting.

    Attributes:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        category: Error category for classification
        severity: Error severity level
        context: Additional context data
        original_error: The underlying exception if this wraps another error
        timestamp: When the error occurred
        trace_id: Associated trace ID if available
        span_id: Associated span ID if available

    Example:
        ```python
        try:
            result = some_operation()
        except OrchestratorError as e:
            print(f"Error [{e.error_code}]: {e.message}")
            print(f"Context: {e.context}")
        ```
    """

    default_message: str = "An error occurred in the Orchestrator SDK"
    default_error_code: str = "ORCHESTRATOR_ERROR"
    default_category: ErrorCategory = ErrorCategory.INTERNAL
    default_severity: ErrorSeverity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        should_report: bool = True,
    ):
        """
        Initialize an OrchestratorError.

        Args:
            message: Error message (uses default if not provided)
            error_code: Unique error code
            category: Error category
            severity: Error severity
            context: Additional context data
            original_error: Wrapped exception
            trace_id: Associated trace ID
            span_id: Associated span ID
            should_report: Whether to report this error to Langfuse
        """
        self.message = message or self.default_message
        self.error_code = error_code or self.default_error_code
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now(UTC)
        self.trace_id = trace_id
        self.span_id = span_id
        self.should_report = should_report

        super().__init__(self.message)

        if should_report and _error_reporter:
            try:
                _error_reporter(self)
            except Exception:
                pass

    def __str__(self) -> str:
        """Format error as string with sensitive data redacted."""
        parts = [f"[{self.error_code}] {redact_sensitive_values(self.message)}"]
        if self.context:
            safe_context = redact_dict(self.context)
            context_str = ", ".join(f"{k}={v}" for k, v in safe_context.items())
            parts.append(f"Context: {context_str}")
        if self.original_error:
            safe_msg = redact_sensitive_values(str(self.original_error))
            parts.append(f"Caused by: {type(self.original_error).__name__}: {safe_msg}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"category={self.category.value!r}, "
            f"severity={self.severity.value!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert error to dictionary for serialization.

        Sensitive data in context is automatically redacted.

        Returns:
            Dictionary representation of the error.
        """
        data = {
            "error_type": self.__class__.__name__,
            "message": redact_sensitive_values(self.message),
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": redact_dict(self.context) if self.context else {},
        }

        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        if self.original_error:
            data["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": redact_sensitive_values(str(self.original_error)),
            }

        return data

    def get_traceback(self) -> str | None:
        """Get the traceback string if available."""
        if self.original_error:
            return "".join(
                traceback.format_exception(
                    type(self.original_error),
                    self.original_error,
                    self.original_error.__traceback__,
                )
            )
        return None


class ConfigurationError(OrchestratorError):
    """
    Raised when there's a configuration issue.

    Examples:
        - Missing required environment variables
        - Invalid configuration values
        - Missing API keys
    """

    default_message = "Configuration error"
    default_error_code = "CONFIG_ERROR"
    default_category = ErrorCategory.CONFIGURATION
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str | None = None,
        *,
        config_key: str | None = None,
        expected_type: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if config_key:
            context["config_key"] = config_key
        if expected_type:
            context["expected_type"] = expected_type
        super().__init__(message, context=context, **kwargs)


class ValidationError(OrchestratorError):
    """
    Raised when input validation fails.

    Examples:
        - Invalid message format
        - Missing required fields
        - Type mismatches
    """

    default_message = "Validation error"
    default_error_code = "VALIDATION_ERROR"
    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.MEDIUM

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]  # Truncate long values
        if expected:
            context["expected"] = expected
        super().__init__(message, context=context, **kwargs)


class InputBlockedError(OrchestratorError):
    """
    Raised when an input scanner or injection-detection check blocks a message.

    Examples:
        - A custom input_scanner returns a blocked result
        - Injection detection fires in strict mode
    """

    default_message = "Input blocked by security policy"
    default_error_code = "INPUT_BLOCKED"
    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str | None = None,
        *,
        reason: str | None = None,
        scanner: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if reason:
            context["reason"] = reason
        if scanner:
            context["scanner"] = scanner
        super().__init__(message, context=context, **kwargs)


class ObservabilityError(OrchestratorError):
    """
    Base class for observability-related errors.

    Examples:
        - Langfuse connection failures
        - Tracing errors
        - Metrics collection failures
    """

    default_message = "Observability error"
    default_error_code = "OBSERVABILITY_ERROR"
    default_category = ErrorCategory.OBSERVABILITY
    default_severity = ErrorSeverity.MEDIUM  # Usually non-blocking


class LangfuseError(ObservabilityError):
    """
    Raised when Langfuse operations fail.

    Examples:
        - Connection errors
        - Authentication failures
        - API errors
    """

    default_message = "Langfuse error"
    default_error_code = "LANGFUSE_ERROR"

    def __init__(
        self,
        message: str | None = None,
        *,
        operation: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if operation:
            context["operation"] = operation
        # Don't report Langfuse errors to Langfuse (infinite loop)
        kwargs.setdefault("should_report", False)
        super().__init__(message, context=context, **kwargs)


class TracingError(ObservabilityError):
    """
    Raised when tracing operations fail.

    Examples:
        - Failed to create trace
        - Failed to end span
        - Context propagation issues
    """

    default_message = "Tracing error"
    default_error_code = "TRACING_ERROR"

    def __init__(
        self,
        message: str | None = None,
        *,
        trace_name: str | None = None,
        span_name: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if trace_name:
            context["trace_name"] = trace_name
        if span_name:
            context["span_name"] = span_name
        super().__init__(message, context=context, **kwargs)


class NetworkError(OrchestratorError):
    """
    Raised when network operations fail.

    Examples:
        - Connection timeouts
        - DNS resolution failures
        - SSL errors
    """

    default_message = "Network error"
    default_error_code = "NETWORK_ERROR"
    default_category = ErrorCategory.NETWORK
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str | None = None,
        *,
        url: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if url:
            context["url"] = url
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, context=context, **kwargs)


class ProviderError(OrchestratorError):
    """
    Raised when an external provider returns an error.

    Examples:
        - LLM provider errors
        - External API errors
    """

    default_message = "Provider error"
    default_error_code = "PROVIDER_ERROR"
    default_category = ErrorCategory.PROVIDER
    default_severity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        provider_error_code: str | None = None,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", {}) or {}
        if provider:
            context["provider"] = provider
        if provider_error_code:
            context["provider_error_code"] = provider_error_code
        super().__init__(message, context=context, **kwargs)


def wrap_exception(
    error: Exception,
    error_class: type[OrchestratorError] = OrchestratorError,
    message: str | None = None,
    **kwargs: Any,
) -> OrchestratorError:
    """
    Wrap a standard exception in an OrchestratorError.

    Utility function to convert any exception to an OrchestratorError
    while preserving the original error information.

    Args:
        error: The original exception
        error_class: The OrchestratorError subclass to use
        message: Optional custom message (uses original if not provided)
        **kwargs: Additional arguments passed to the error class

    Returns:
        An OrchestratorError wrapping the original exception.

    Example:
        ```python
        try:
            risky_operation()
        except ValueError as e:
            raise wrap_exception(e, ValidationError, field="input")
        ```
    """
    if isinstance(error, OrchestratorError):
        return error

    return error_class(
        message=message or str(error),
        original_error=error,
        **kwargs,
    )
