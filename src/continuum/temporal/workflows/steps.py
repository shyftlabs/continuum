"""Running one agent step from a workflow.

``run_agent_activity`` raises when the agent fails, so Temporal retries it with
the step's retry policy. When the retries run out, ``run_agent_step`` turns the
failure into an ``AgentActivityResult`` with ``status="error"`` so the workflow
can stop with ``status="failed"`` and say which step failed.
"""

from __future__ import annotations

from datetime import timedelta

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

with workflow.unsafe.imports_passed_through():
    from continuum.temporal.types import AgentActivityParams, AgentActivityResult


class StepFailedError(Exception):
    """An agent step failed after its retries; ends the workflow as ``failed``."""


def failure_message(error: BaseException) -> str:
    """The underlying error of an ActivityError (its own text is generic)."""
    cause = getattr(error, "cause", None)
    while cause is not None and getattr(cause, "cause", None) is not None:
        cause = cause.cause
    return str(cause or error)


async def run_agent_step(
    params: AgentActivityParams,
    *,
    timeout: int,
    retries: int,
) -> AgentActivityResult:
    """Run ``run_agent_activity`` with retries; a final failure comes back as an error result."""
    try:
        raw = await workflow.execute_activity(
            "run_agent_activity",
            params,
            start_to_close_timeout=timedelta(seconds=timeout),
            retry_policy=RetryPolicy(
                maximum_attempts=retries,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=60),
            ),
            heartbeat_timeout=timedelta(seconds=60),
            result_type=AgentActivityResult,
        )
    except ActivityError as e:
        return AgentActivityResult(
            content="",
            status="error",
            error=failure_message(e),
            agents_used=[params.agent_name],
        )
    return raw if isinstance(raw, AgentActivityResult) else AgentActivityResult.model_validate(raw)
