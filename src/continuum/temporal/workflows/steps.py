"""Running one agent step from a workflow.

``run_agent_step`` runs ``run_agent_activity`` with ``raise_on_error=True``, so
Temporal retries a failing agent with the step's retry policy. When the retries
run out, an agent error comes back as an ``AgentActivityResult`` with
``status="error"`` and the same ``error`` text as before, and the workflow
decides what to do with it. Failures that are not the agent's (the worker has
no such activity, a timeout) are raised as before.

The built-in workflows go through ``run_builtin_agent_step``. With default
inputs they run the agent as before: an agent error comes back as an error
result and is not retried, and no marker is recorded, so a 2.1 worker can
replay the run. Only a run that opts in (``retry_agent_errors=True`` or
``on_step_error="fail"``) takes the new path, behind the ``STEP_RETRIES_PATCH``
marker; a run started before the upgrade replays without the marker and keeps
the old path, so its history stays deterministic.
"""

from __future__ import annotations

from datetime import timedelta

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import (
        ActivityError,
        ApplicationError,
        FailureError,
        TimeoutError,
        is_cancelled_exception,
    )
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

with workflow.unsafe.imports_passed_through():
    from continuum.temporal.types import (
        AGENT_ERROR_MARKER,
        AgentActivityParams,
        AgentActivityResult,
    )

# workflow.patched() id for retried agent steps and on_step_error.
STEP_RETRIES_PATCH = "continuum-step-retries"


class StepFailedError(Exception):
    """An agent step failed after its retries; ends the workflow as ``failed``."""


def step_retries_enabled(retry_agent_errors: bool, on_step_error: str) -> bool:
    """True when this run takes the new step path.

    A run with default inputs never does, and never calls ``workflow.patched``,
    so its history has no marker and a 2.1 worker can replay it. An opted-in
    run records the marker; replaying a history from before the patch gives False.
    """
    if not retry_agent_errors and on_step_error != "fail":
        return False
    return workflow.patched(STEP_RETRIES_PATCH)


def check_on_step_error(value: str) -> None:
    """Fail the run at once on an ``on_step_error`` other than "continue" or "fail".

    The dataclass inputs type it as ``str``: a ``Literal`` there would make the
    worker fail to decode a typo and retry the workflow task forever.
    """
    if value not in ("continue", "fail"):
        raise ApplicationError(
            f'on_step_error must be "continue" or "fail", not {value!r}',
            type="InvalidWorkflowInput",
            non_retryable=True,
        )


def agent_error(error: ActivityError) -> str | None:
    """The message of the agent error behind ``error``, or None if it was not the agent's.

    ``run_agent_activity`` marks its own errors. When the last attempt timed
    out after an earlier attempt's agent error, that error rides on the
    timeout's heartbeat details (or its cause, the previous attempt's failure).
    """
    cause = error.cause
    if isinstance(cause, ApplicationError) and AGENT_ERROR_MARKER in cause.details:
        return cause.message
    if isinstance(cause, TimeoutError):
        details = list(cause.last_heartbeat_details or ())
        if len(details) >= 3 and details[1] == AGENT_ERROR_MARKER:
            return str(details[2])
        if isinstance(cause.cause, ApplicationError) and AGENT_ERROR_MARKER in cause.cause.details:
            return cause.cause.message
    return None


def failure_message(error: BaseException) -> str:
    """The underlying error of an ActivityError (its own text is generic)."""
    cause = getattr(error, "cause", None)
    while cause is not None and getattr(cause, "cause", None) is not None:
        cause = cause.cause
    return cause.message if isinstance(cause, FailureError) else str(cause or error)


def _retry_policy(retries: int) -> RetryPolicy:
    # Temporal reads maximum_attempts=0 as "retry forever"; 0 here means once.
    return RetryPolicy(
        maximum_attempts=max(1, retries),
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=60),
    )


async def _execute(params: AgentActivityParams, timeout: int, retries: int) -> AgentActivityResult:
    raw = await workflow.execute_activity(
        "run_agent_activity",
        params,
        start_to_close_timeout=timedelta(seconds=timeout),
        retry_policy=_retry_policy(retries),
        heartbeat_timeout=timedelta(seconds=60),
        result_type=AgentActivityResult,
    )
    return raw if isinstance(raw, AgentActivityResult) else AgentActivityResult.model_validate(raw)


async def run_agent_step(
    params: AgentActivityParams,
    *,
    timeout: int,
    retries: int,
    raise_timeouts: bool = False,
    retry_agent_errors: bool = True,
) -> AgentActivityResult:
    """Run ``run_agent_activity`` with retries; a final agent error comes back as an error result.

    ``retries`` is the number of attempts in all (0 counts as 1). Cancellation
    is re-raised so the workflow ends as cancelled, and so is an error the
    worker or SDK raised rather than the agent. With ``raise_timeouts`` a step
    that timed out on every attempt raises ``ActivityError`` instead of
    returning an error result; one that hit an agent error first returns it.
    With ``retry_agent_errors=False`` an agent error comes back on the first
    attempt, not retried; only timeouts are retried.
    """
    update: dict[str, object] = {"max_attempts": max(1, retries)}
    if retry_agent_errors:
        update["raise_on_error"] = True
    try:
        return await _execute(params.model_copy(update=update), timeout, retries)
    except ActivityError as e:
        if is_cancelled_exception(e):
            raise
        message = agent_error(e)
        if message is None:
            # Not the agent's error: a timeout (a result only when not
            # raise_timeouts) or one the worker or SDK raised (always raised).
            if raise_timeouts or not isinstance(e.cause, TimeoutError):
                raise
            message = failure_message(e)
        return AgentActivityResult(
            content="",
            status="error",
            error=message,
            agents_used=[params.agent_name],
        )


async def run_builtin_agent_step(
    params: AgentActivityParams,
    *,
    timeout: int,
    retries: int,
    retry_agent_errors: bool,
    on_step_error: str,
) -> AgentActivityResult:
    """``run_agent_step`` for the built-in workflows, behind ``STEP_RETRIES_PATCH``.

    With default inputs, or without the patch (a run from before it), the agent
    runs as on 2.1: an agent error comes back as an error result and is not
    retried, and a timeout on the last attempt raises.
    """
    if step_retries_enabled(retry_agent_errors, on_step_error):
        return await run_agent_step(
            params,
            timeout=timeout,
            retries=retries,
            # With "fail" a timed-out step is a failed result, not a raised error.
            raise_timeouts=on_step_error == "continue",
            retry_agent_errors=retry_agent_errors,
        )
    return await _execute(params, timeout, retries)
