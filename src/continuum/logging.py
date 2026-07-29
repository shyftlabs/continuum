"""
Centralized logging configuration for the Orchestrator SDK.

Provides structured logging with support for:
- Development (human-readable) and Production (JSON) formats
- Automatic error reporting to Langfuse
- Configurable log levels per module
- Context propagation (trace_id, span_id, user_id)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from continuum.config import settings

if TYPE_CHECKING:
    from continuum.observability.providers.langfuse_client import LangfuseClient


# Context variables for trace correlation
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)

_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}")
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[a-z0-9._~+/=-]+")
_JWT_RE = re.compile(r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b")
_UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|"
    r"authorization|cookie)\s*[:=]\s*[^\s,;]+"
)
_SENSITIVE_KEYS = {
    "access_token",
    "authorization",
    "cookie",
    "email",
    "id_token",
    "name",
    "password",
    "refresh_token",
    "secret",
    "token",
}


def _sanitize_log_text(value: object) -> str:
    """Redact common PII and credential shapes before they leave the process."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = _EMAIL_RE.sub("[email]", text)
    text = _BEARER_RE.sub(r"\1 [credential]", text)
    text = _JWT_RE.sub("[credential]", text)
    text = _UUID_RE.sub("[identifier]", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1=[redacted]", text)


def _safe_log_value(value: Any, *, field_name: str = "", depth: int = 0) -> Any:
    """Sanitize structured extras without recursively expanding arbitrary data."""
    normalized_name = field_name.casefold()
    if any(marker in normalized_name for marker in _SENSITIVE_KEYS):
        return "[redacted]"
    if depth >= 4:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            str(key): _safe_log_value(
                item,
                field_name=str(key),
                depth=depth + 1,
            )
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_log_value(item, depth=depth + 1) for item in list(value)[:50]]
    if isinstance(value, str):
        return _sanitize_log_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_log_text(type(value).__name__)


def _identifier_alias(value: str) -> str:
    key = settings.log_pii_hmac_key
    if not key:
        # Development compatibility only. Production configuration refuses to
        # start without a secret HMAC key.
        key = "continuum-development-log-alias-key"
    digest = hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"hmac:{digest[:16]}"


def _safe_log_context() -> dict[str, str | None]:
    context = get_log_context()
    for key in ("user_id", "session_id"):
        if context.get(key):
            context[key] = _identifier_alias(context[key] or "")
    return context


class LogLevel(str, Enum):
    """Log levels supported by the SDK."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogContext:
    """
    Context manager for setting logging context.

    Automatically propagates trace_id, span_id, user_id to all log messages
    within the context.

    Example:
        ```python
        with LogContext(trace_id="abc-123", user_id="user-456"):
            logger.info("Processing request")  # Includes trace_id and user_id
        ```
    """

    def __init__(
        self,
        trace_id: str | None = None,
        span_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.user_id = user_id
        self.session_id = session_id
        self._tokens: list[Any] = []

    def __enter__(self) -> LogContext:
        if self.trace_id:
            self._tokens.append((_trace_id, _trace_id.set(self.trace_id)))
        if self.span_id:
            self._tokens.append((_span_id, _span_id.set(self.span_id)))
        if self.user_id:
            self._tokens.append((_user_id, _user_id.set(self.user_id)))
        if self.session_id:
            self._tokens.append((_session_id, _session_id.set(self.session_id)))
        return self

    def __exit__(self, *args: Any) -> None:
        for var, token in self._tokens:
            var.reset(token)


def get_log_context() -> dict[str, str | None]:
    """Get the current logging context."""
    return {
        "trace_id": _trace_id.get(),
        "span_id": _span_id.get(),
        "user_id": _user_id.get(),
        "session_id": _session_id.get(),
    }


def set_log_context(
    trace_id: str | None = None,
    span_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Set logging context variables."""
    if trace_id is not None:
        _trace_id.set(trace_id)
    if span_id is not None:
        _span_id.set(span_id)
    if user_id is not None:
        _user_id.set(user_id)
    if session_id is not None:
        _session_id.set(session_id)


def clear_log_context() -> None:
    """Clear all logging context variables."""
    _trace_id.set(None)
    _span_id.set(None)
    _user_id.set(None)
    _session_id.set(None)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for production logging.

    Outputs structured JSON logs with consistent fields for log aggregation
    and analysis tools (ELK, Datadog, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        production = settings.environment == "production"
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            # Production logs intentionally carry only a stable event marker.
            # Arbitrary model, tool, provider, and exception text can contain
            # names or business data that pattern-based redaction cannot prove
            # safe.
            "message": (
                "continuum_log_event" if production else _sanitize_log_text(record.getMessage())
            ),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context from context vars
        context = _safe_log_context()
        for key, value in context.items():
            if value:
                log_data[key] = value

        # Add extra fields from record
        if hasattr(record, "extra_data") and not production:
            log_data.update(_safe_log_value(record.extra_data))

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
            }
            if not production:
                log_data["exception"]["message"] = (
                    _sanitize_log_text(record.exc_info[1]) if record.exc_info[1] else None
                )
                log_data["exception"]["traceback"] = _sanitize_log_text(
                    self.formatException(record.exc_info)
                )

        # Add environment
        log_data["environment"] = settings.environment

        return json.dumps(log_data, default=str)


class DevelopmentFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Colorized output with clear structure for easy debugging.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build context string
        context = _safe_log_context()
        context_parts = []
        if context.get("trace_id"):
            context_parts.append(f"trace={context['trace_id'][:8]}")
        if context.get("user_id"):
            context_parts.append(f"user={context['user_id']}")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format the message
        message = (
            f"{timestamp} {color}{record.levelname:8}{reset} {record.name}"
            f"{context_str} - {_sanitize_log_text(record.getMessage())}"
        )

        # Add exception if present
        if record.exc_info:
            message += f"\n{_sanitize_log_text(self.formatException(record.exc_info))}"

        return message


class LangfuseHandler(logging.Handler):
    """
    Logging handler that sends ERROR and CRITICAL logs to Langfuse.

    Automatically creates events in Langfuse for error visibility
    and debugging in the observability dashboard.
    """

    def __init__(self, min_level: int = logging.ERROR):
        super().__init__(level=min_level)
        self._client: LangfuseClient | None = None

    def _get_client(self) -> LangfuseClient | None:
        """Lazy load the Langfuse client to avoid circular imports."""
        if self._client is None:
            try:
                from continuum.observability.providers.registry import get_provider

                langfuse_provider = get_provider("langfuse")
                if langfuse_provider and langfuse_provider.is_enabled:
                    self._client = langfuse_provider.client
            except Exception:
                pass
        return self._client

    def emit(self, record: logging.LogRecord) -> None:
        """Send log record to Langfuse."""
        client = self._get_client()
        if client is None or not client.is_enabled:
            return

        try:
            context = _safe_log_context()
            trace_id = context.get("trace_id")

            # Build metadata
            metadata = {
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "level": record.levelname,
                "environment": settings.environment,
            }

            # Add exception info
            if record.exc_info and record.exc_info[1]:
                metadata["exception_type"] = type(record.exc_info[1]).__name__
                if settings.environment != "production":
                    metadata["exception_message"] = _sanitize_log_text(record.exc_info[1])

            # Add context
            for key, value in context.items():
                if value:
                    metadata[key] = value

            # If we have a trace context, add event to that trace
            if trace_id and client.client:
                try:
                    client.client.event(
                        trace_id=trace_id,
                        name=f"log.{record.levelname.lower()}",
                        input={
                            "message": (
                                "continuum_log_event"
                                if settings.environment == "production"
                                else _sanitize_log_text(record.getMessage())
                            )
                        },
                        metadata=_safe_log_value(metadata),
                        level="ERROR" if record.levelno >= logging.ERROR else "WARNING",
                    )
                except Exception:
                    pass  # Don't fail on logging errors

        except Exception:
            # Never let logging handler errors propagate
            pass


class OrchestratorLogger(logging.Logger):
    """
    Enhanced logger with structured logging support.

    Provides additional methods for logging with extra context.
    """

    def _log_with_extra(
        self,
        level: int,
        msg: str,
        *args: Any,
        extra_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log with extra structured data."""
        if extra_data:
            kwargs.setdefault("extra", {})["extra_data"] = extra_data
        self.log(level, msg, *args, **kwargs)

    def debug_with_context(
        self, msg: str, *args: Any, extra: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Debug log with extra context."""
        self._log_with_extra(logging.DEBUG, msg, *args, extra_data=extra, **kwargs)

    def info_with_context(
        self, msg: str, *args: Any, extra: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Info log with extra context."""
        self._log_with_extra(logging.INFO, msg, *args, extra_data=extra, **kwargs)

    def warning_with_context(
        self, msg: str, *args: Any, extra: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Warning log with extra context."""
        self._log_with_extra(logging.WARNING, msg, *args, extra_data=extra, **kwargs)

    def error_with_context(
        self, msg: str, *args: Any, extra: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Error log with extra context."""
        self._log_with_extra(logging.ERROR, msg, *args, extra_data=extra, **kwargs)

    def critical_with_context(
        self, msg: str, *args: Any, extra: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Critical log with extra context."""
        self._log_with_extra(logging.CRITICAL, msg, *args, extra_data=extra, **kwargs)


# Register the custom logger class
logging.setLoggerClass(OrchestratorLogger)

# Module-level logger cache
_loggers: dict[str, OrchestratorLogger] = {}
_initialized = False


def setup_logging(
    level: str | LogLevel | None = None,
    json_format: bool | None = None,
    enable_langfuse_handler: bool = True,
) -> None:
    """
    Setup centralized logging for the Orchestrator SDK.

    Args:
        level: Log level (defaults to settings.log_level)
        json_format: Use JSON format (defaults to True in production)
        enable_langfuse_handler: Enable sending errors to Langfuse

    Example:
        ```python
        from continuum.logging import setup_logging

        # Development setup
        setup_logging(level="DEBUG", json_format=False)

        # Production setup (auto-detected from environment)
        setup_logging()
        ```
    """
    global _initialized

    # Determine log level
    if level is None:
        level = settings.log_level
    if isinstance(level, LogLevel):
        level = level.value
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Determine format based on environment
    if json_format is None:
        json_format = settings.environment in ("production", "staging")

    # Get root continuum logger
    root_logger = logging.getLogger("continuum")
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = DevelopmentFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Langfuse handler for errors
    if enable_langfuse_handler:
        langfuse_handler = LangfuseHandler(min_level=logging.ERROR)
        langfuse_handler.setFormatter(formatter)
        root_logger.addHandler(langfuse_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    _initialized = True


def get_logger(name: str | None = None) -> OrchestratorLogger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__). If None, returns root continuum logger.

    Returns:
        Configured OrchestratorLogger instance.

    Example:
        ```python
        from continuum.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.error("Something went wrong", exc_info=True)
        ```
    """
    global _initialized

    # Auto-initialize if not done
    if not _initialized:
        setup_logging()

    # Default to continuum namespace
    if name is None:
        name = "continuum"
    elif not name.startswith("continuum"):
        name = f"continuum.{name}"

    # Return cached logger or create new one
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger  # type: ignore

    return _loggers[name]  # type: ignore


def get_child_logger(parent_name: str, child_name: str) -> OrchestratorLogger:
    """
    Get a child logger.

    Args:
        parent_name: Parent logger name
        child_name: Child logger name

    Returns:
        Child logger instance.
    """
    full_name = f"{parent_name}.{child_name}"
    return get_logger(full_name)


# Convenience function for module-level usage
def logger_for_module(module_name: str) -> OrchestratorLogger:
    """
    Get a logger for a module.

    Convenience function to get a properly namespaced logger.

    Args:
        module_name: The module's __name__

    Returns:
        Configured logger for the module.

    Example:
        ```python
        from continuum.logging import logger_for_module

        logger = logger_for_module(__name__)
        ```
    """
    # Ensure the module name is under the 'continuum' namespace
    if module_name.startswith("continuum."):
        return get_logger(module_name)
    return get_logger(f"continuum.{module_name}")


def clear_loggers() -> int:
    """
    Clear the logger cache to free memory.

    This can be called periodically in long-running applications to prevent
    memory accumulation from dynamically created loggers.

    Returns:
        Number of loggers cleared.

    Example:
        ```python
        from continuum.logging import clear_loggers

        # Clear all cached loggers
        count = clear_loggers()
        print(f"Cleared {count} loggers")
        ```
    """
    global _loggers
    count = len(_loggers)
    _loggers.clear()
    return count


def get_logger_count() -> int:
    """
    Get the number of cached loggers.

    Useful for monitoring memory usage from loggers.

    Returns:
        Number of cached loggers.
    """
    return len(_loggers)
