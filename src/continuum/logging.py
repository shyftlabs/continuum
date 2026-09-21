"""
Centralized logging configuration for the Orchestrator SDK.

Provides structured logging with support for:
- Development (human-readable) and Production (JSON) formats
- Automatic error reporting to Langfuse
- Configurable log levels per module
- Context propagation (trace_id, span_id, user_id)
"""

from __future__ import annotations

import json
import logging
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


class _Content:
    """An argument the call site has declared to be user or model content.

    Redacted by ``PromptContentFilter`` whatever its length, which is the part a
    size threshold cannot do. ``__str__`` returns the *redacted* form so that a
    handler without the filter -- someone's own, or a bare ``basicConfig`` --
    fails closed rather than printing the value. The filter unwraps it to the
    real thing only when content logging is switched on.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"<{len(str(self.value))} chars>"

    __repr__ = __str__


def log_content(value: Any) -> _Content:
    """Mark a log argument as content, so the filter can withhold it.

    Use at any site that logs a prompt, a memory, a tool argument or result, or
    model output::

        logger.info("TOOL RESULT: %s -> %s", tool_name, log_content(result))

    Pass the whole value -- no ``[:200]`` slicing. Truncation at the call site
    both leaks a prefix and throws the rest away; here the operator gets nothing
    by default and everything when they set ``LOG_PROMPT_CONTENT=true``.
    """
    return _Content(value)


class PromptContentFilter(logging.Filter):
    """Keeps prompt and tool content out of the log unless asked for.

    Installed on the handlers rather than written into the call sites, so it
    covers every ``continuum.*`` logger -- including modules written after this,
    which is how ``tools/executor.py`` came to log a reviewer's free-text
    approval reason at INFO eleven days after the problem was first reported.

    The contract is one rule, and it is worth stating exactly because the
    coverage it gives is not automatic:

        an argument wrapped in log_content() is withheld; nothing else is.

    A length threshold sat here too, briefly, on the theory that a long argument
    is probably data. It was wrong in both directions and is gone: 46 characters
    of PHI passed it, while a 65-character pasteable `continuum mcp diff` command
    did not. Guessing from length damages diagnostics and protects nothing that
    can be relied on. ``tests/unit/test_log_canary.py`` replaces it -- a sentinel
    driven through the real paths, failing the build by name when a site forgets
    to declare its content, which is a net that says what is wrong instead of
    silently hiding the wrong things.

    ``logger.info("FINAL PROMPT [%s]\\n%s", name, prompt)`` keeps the format
    string and its values apart until the formatter runs, so the literal (the
    developer's structure) can be kept while the values (the data) go. A value is
    *replaced* rather than shortened -- the first 200 characters of a patient
    record are still a patient record.

    An f-string has already collapsed the two by the time the record exists, and
    nothing here can separate them again. Capping such a message by length was
    tried and removed: it cannot tell a prompt dump from a thorough operator
    message, and it destroyed the pasteable ``continuum mcp diff`` command in the
    tool-trust warnings. So a call site that logs content opts into protection by
    passing it as an argument -- ``log_content()`` when it knows, which also
    covers content too short to trip the length backstop.
    ``tests/unit/test_log_content_redaction.py`` pins every part of that
    contract, including the unprotected one.

    ``record.exc_info`` is left alone. The leak is the exception text
    interpolated into a message; the traceback is a separate field, carries no
    locals, and is the only thing that says where a failure happened.

    The record is mutated in place, so every handler downstream sees the redacted
    version -- which is the intent, not a side effect.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Withholding needs no work here: _Content renders as "<N chars>" by
        # itself, which is what makes an unfiltered handler fail closed. The
        # filter's only job is the opposite one -- unwrapping, when an operator
        # has asked for content.
        if not settings.log_prompt_content:
            return True

        # A mapping (for %(name)s interpolation) is not a shape this codebase
        # uses, and rewriting it would break the line.
        if isinstance(record.args, tuple) and record.args:
            record.args = tuple(
                arg.value if isinstance(arg, _Content) else arg for arg in record.args
            )

        return True


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for production logging.

    Outputs structured JSON logs with consistent fields for log aggregation
    and analysis tools (ELK, Datadog, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context from context vars
        context = get_log_context()
        for key, value in context.items():
            if value:
                log_data[key] = value

        # Add extra fields from record
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

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
        context = get_log_context()
        context_parts = []
        if context.get("trace_id"):
            context_parts.append(f"trace={context['trace_id'][:8]}")
        if context.get("user_id"):
            context_parts.append(f"user={context['user_id']}")
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format the message
        message = f"{timestamp} {color}{record.levelname:8}{reset} {record.name}{context_str} - {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

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
            context = get_log_context()
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
                metadata["exception_message"] = str(record.exc_info[1])

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
                        input={"message": record.getMessage()},
                        metadata=metadata,
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
    # On the handler, not the logger: every module logs through a child logger
    # (continuum.agent.execution.message_builder), whose records reach us by
    # propagation. Propagation runs the ancestors' *handlers* and skips their
    # filters, so a logger-level filter here would see almost nothing.
    console_handler.addFilter(PromptContentFilter())
    root_logger.addHandler(console_handler)

    # Langfuse handler for errors
    if enable_langfuse_handler:
        langfuse_handler = LangfuseHandler(min_level=logging.ERROR)
        langfuse_handler.setFormatter(formatter)
        langfuse_handler.addFilter(PromptContentFilter())
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
