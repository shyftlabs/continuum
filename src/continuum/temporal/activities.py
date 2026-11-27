"""
Temporal Activities for the Orchestrator SDK.

These activities bridge Temporal to the existing AgentRunner.
They are agent-agnostic: any registered BaseAgent can be executed.
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import timedelta
from typing import Any

try:
    from temporalio import activity
    from temporalio.exceptions import ApplicationError
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

from continuum.agent.exceptions import AgentConfigurationError, MaxTurnsExceededError
from continuum.agent.utils.circuit_breaker import CircuitBreaker, hold_failures
from continuum.exceptions import ConfigurationError, InputBlockedError, PolicyDeniedError
from continuum.llm.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)
from continuum.session.exceptions import SessionConnectionError, SessionError
from continuum.temporal.exceptions import AgentNotRegisteredError
from continuum.temporal.registry import get_agent_registry
from continuum.temporal.types import (
    AGENT_ERROR_MARKER,
    AgentActivityParams,
    AgentActivityResult,
    NotificationParams,
)

_HEARTBEAT_INTERVAL = 15  # seconds — must be less than heartbeat_timeout in the workflow

# Longest agent error text the activity sends to Temporal, in UTF-8 bytes. The
# failure and the last heartbeat each carry it, and a completion over gRPC's
# 4 MB limit is never delivered, so the attempt would wait out its timeout.
_MAX_ERROR_BYTES = 256 * 1024
_TRUNCATED = " ... [truncated]"

# Errors a retry cannot fix; with raise_on_error they fail the step at once.
_PERMANENT_ERRORS: tuple[type, ...] = (
    AgentNotRegisteredError,
    MaxTurnsExceededError,
    PolicyDeniedError,
    ConfigurationError,
    AgentConfigurationError,
    InputBlockedError,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMContentFilterError,
    SessionError,
)
# Subclasses of the above that a retry can fix: the session store may come back.
_TRANSIENT_ERRORS: tuple[type, ...] = (SessionConnectionError,)

# Longest wait before the next attempt that a rate limit's ``retry_after`` sets.
_MAX_RATE_LIMIT_DELAY = 300.0  # seconds


def _error_chain(error: BaseException, *, context: bool = False) -> list[Any]:
    """``error`` and every error it wraps (``original_error``, ``__cause__``), at any depth.

    With ``context``, the errors it was raised while handling (``__context__``)
    too. Each error appears once, so a cyclic chain ends.
    """
    chain: list[Any] = []
    todo: list[Any] = [error]
    while todo:
        e = todo.pop(0)
        if e is None or any(e is seen for seen in chain):
            continue
        chain.append(e)
        todo += [getattr(e, "original_error", None), getattr(e, "__cause__", None)]
        if context:
            todo.append(getattr(e, "__context__", None))
    return chain


def _is_permanent(error: BaseException) -> bool:
    """True for an agent error that a retry cannot fix.

    The runner and the LLM client wrap errors (``AgentExecutionError``,
    ``original_error``, ``raise ... from``), so the whole chain is checked.
    An ``ApplicationError(non_retryable=True)`` anywhere in it, for example
    from a tool, is honoured: its author asked for it not to run again.
    """
    if any(
        isinstance(e, ApplicationError) and e.non_retryable
        for e in _error_chain(error, context=True)
    ):
        return True
    return any(
        isinstance(e, _PERMANENT_ERRORS) and not isinstance(e, _TRANSIENT_ERRORS)
        for e in _error_chain(error)
    )


def _rate_limit_delay(error: BaseException) -> timedelta | None:
    """The wait a rate limit in ``error``'s chain asks for, capped; else ``None``."""
    for e in _error_chain(error):
        retry_after = getattr(e, "retry_after", None) if isinstance(e, LLMRateLimitError) else None
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except (TypeError, ValueError):
                continue
            if seconds == seconds:  # not NaN
                return timedelta(seconds=min(max(1.0, seconds), _MAX_RATE_LIMIT_DELAY))
    return None


def _is_last_attempt(params: AgentActivityParams) -> bool:
    """True on the last attempt the step allows.

    The activity info does not always carry the retry policy, so
    ``run_agent_step`` passes ``params.max_attempts``. With neither, or
    unlimited attempts, every attempt counts as the last.
    """
    info = activity.info()
    max_attempts = info.retry_policy.maximum_attempts if info.retry_policy else None
    max_attempts = max_attempts or params.max_attempts
    if not max_attempts:
        return True
    return info.attempt >= max_attempts


def _carried_agent_error() -> tuple[str, ...]:
    """The agent error an earlier attempt of this activity heartbeated, if any."""
    details = list(activity.info().heartbeat_details or ())
    if len(details) >= 3 and details[1] == AGENT_ERROR_MARKER:
        return (AGENT_ERROR_MARKER, str(details[2]))
    return ()


def _capped(message: str) -> str:
    """``message``, cut to ``_MAX_ERROR_BYTES`` and marked truncated if longer."""
    data = message.encode("utf-8", errors="replace")
    if len(data) <= _MAX_ERROR_BYTES:
        return message
    keep = _MAX_ERROR_BYTES - len(_TRUNCATED.encode())
    return data[:keep].decode("utf-8", errors="ignore") + _TRUNCATED


def _agent_error(message: str, **kwargs: Any) -> ApplicationError:
    """An ``ApplicationError`` the workflow can tell from the SDK's own errors.

    The marker goes in the error's details and, with the message, on a last
    heartbeat; later attempts carry that heartbeat on, so a last attempt that
    times out still reports the agent error (``steps.agent_error``). The
    message is capped at ``_MAX_ERROR_BYTES``.
    """
    message = _capped(message)
    activity.heartbeat("Agent failed", AGENT_ERROR_MARKER, message)
    return ApplicationError(message, AGENT_ERROR_MARKER, **kwargs)


@activity.defn
async def run_agent_activity(params: AgentActivityParams) -> AgentActivityResult:
    """Execute ANY registered agent via AgentRunner.

    Agent-agnostic: looks up agent by name from registry, calls runner.run().
    All existing SDK features (LLM, memory, session, tools, observability) are unchanged.

    By default an agent error comes back as a result with ``status="error"``.
    With ``params.raise_on_error`` (set by ``run_agent_step``) the activity
    raises ``ApplicationError`` instead, so the caller's retry policy applies:
    errors a retry cannot fix (``_PERMANENT_ERRORS``) are not retried, and an
    open circuit breaker is retried after its cooldown. A failed attempt
    counts toward the runner's circuit breaker only if it is the last one.
    """
    registry = get_agent_registry()
    carried = _carried_agent_error() if params.raise_on_error else ()

    activity.heartbeat(f"Running agent: {params.agent_name}", *carried)

    async def _heartbeat_loop() -> None:
        """Keep Temporal from timing out the activity during long LLM calls."""
        turn = 0
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            turn += 1
            activity.heartbeat(f"Agent {params.agent_name} running (turn ~{turn})", *carried)

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    held: list[CircuitBreaker] = []
    try:
        runner = registry.get_runner()
        agent = registry.get(params.agent_name)

        with hold_failures() if params.raise_on_error else nullcontext(held) as held:
            response = await runner.run(
                agent=agent,
                input=params.input,
                session_id=params.session_id,
                user_id=params.user_id,
                metadata=params.metadata if params.metadata else None,
                tags=params.tags if params.tags else None,
            )
        artifacts = getattr(response, "run_artifacts", None)
        retry_after = artifacts.get("retry_after_s") if isinstance(artifacts, dict) else None
        if params.raise_on_error and retry_after is not None:
            # The runner's circuit breaker is open: retry once it has cooled down.
            raise _agent_error(
                str(response.error),
                type="CircuitBreakerOpen",
                next_retry_delay=timedelta(seconds=max(1.0, float(retry_after))),
            )
        result = AgentActivityResult.from_agent_response(response)
        # A returned response is final (it is not retried), so failures held
        # during the run, e.g. by a workflow agent's sub-agents, count now.
        for breaker in held:
            breaker.record_failure()
        return result
    except Exception as e:
        if not params.raise_on_error:
            activity.logger.error(f"Agent '{params.agent_name}' failed: {e}")
            return AgentActivityResult(
                content="",
                status="error",
                error=str(e),
                agents_used=[params.agent_name],
            )
        if isinstance(e, ApplicationError) and AGENT_ERROR_MARKER in e.details:
            raise  # already shaped for Temporal (the open circuit breaker above)
        permanent = _is_permanent(e)
        if permanent or _is_last_attempt(params):
            for breaker in held:
                breaker.record_failure()
        attempt = activity.info().attempt
        activity.logger.error(f"Agent '{params.agent_name}' failed (attempt {attempt}): {e}")
        # ``from None``: Temporal serialises the cause chain (``__cause__``, or
        # ``__context__`` without it) into history and the worker log prints it,
        # and the wrapped errors hold the raw provider text that ``str(e)`` redacts.
        raise _agent_error(
            str(e),
            type=type(e).__name__,
            non_retryable=permanent,
            next_retry_delay=None if permanent else _rate_limit_delay(e),
        ) from None
    finally:
        heartbeat_task.cancel()


@activity.defn
async def send_notification_activity(params: NotificationParams) -> None:
    """Send notification that approval is needed.

    Delegates to user-configured notification handler (webhook, Slack, email, custom).
    """
    registry = get_agent_registry()
    handler = registry.get_notification_handler()
    if handler:
        await handler(params)
    else:
        activity.logger.warning("No notification handler configured; approval notification dropped")
