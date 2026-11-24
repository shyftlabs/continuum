"""Running one agent step from a workflow.

``run_agent_step`` runs ``run_agent_activity`` with ``raise_on_error=True``, so
Temporal retries a failing agent with the step's retry policy. When the retries
run out, the failure comes back as an ``AgentActivityResult`` with
``status="error"`` and the workflow decides what to do with it.

The built-in workflows go through ``run_builtin_agent_step``, which takes this
path only behind the ``STEP_RETRIES_PATCH`` marker. A run started before the
upgrade replays without the marker and keeps the old path, so its history
stays deterministic.
"""

from __future__ import annotations

from datetime import timedelta

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError, ApplicationError, is_cancelled_exception
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

with workflow.unsafe.imports_passed_through():
    from continuum.temporal.types import AgentActivityParams, AgentActivityResult

# workflow.patched() id for retried agent steps and on_step_error.
STEP_RETRIES_PATCH = "continuum-step-retries"


class StepFailedError(Exception):
    """An agent step failed after its retries; ends the workflow as ``failed``."""


def step_retries_enabled() -> bool:
    """True unless this run is replaying a history from before the patch."""
    return workflow.patched(STEP_RETRIES_PATCH)


def failure_message(error: BaseException) -> str:
    """The underlying error of an ActivityError (its own text is generic)."""
    cause = getattr(error, "cause", None)
    while cause is not None and getattr(cause, "cause", None) is not None:
        cause = cause.cause
    return str(cause or error)


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
) -> AgentActivityResult:
    """Run ``run_agent_activity`` with retries; a final failure comes back as an error result.

    ``retries`` is the number of attempts in all (0 counts as 1). Cancellation
    is re-raised so the workflow ends as cancelled. With ``raise_timeouts`` a
    step that timed out on its last attempt raises ``ActivityError`` instead
    of returning an error result.
    """
    try:
        return await _execute(params.model_copy(update={"raise_on_error": True}), timeout, retries)
    except ActivityError as e:
        if is_cancelled_exception(e):
            raise
        if raise_timeouts and not isinstance(e.cause, ApplicationError):
            raise
        return AgentActivityResult(
            content="",
            status="error",
            error=failure_message(e),
            agents_used=[params.agent_name],
        )


async def run_builtin_agent_step(
    params: AgentActivityParams,
    *,
    timeout: int,
    retries: int,
    raise_timeouts: bool = True,
) -> AgentActivityResult:
    """``run_agent_step`` for the built-in workflows, behind ``STEP_RETRIES_PATCH``.

    Without the patch (a run from before it) the agent runs as it did then: an
    agent error comes back as an error result and is not retried, and a
    timeout on the last attempt raises.
    """
    if step_retries_enabled():
        return await run_agent_step(
            params, timeout=timeout, retries=retries, raise_timeouts=raise_timeouts
        )
    return await _execute(params, timeout, retries)
