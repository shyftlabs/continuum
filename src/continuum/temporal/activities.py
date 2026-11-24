"""
Temporal Activities for the Orchestrator SDK.

These activities bridge Temporal to the existing AgentRunner.
They are agent-agnostic: any registered BaseAgent can be executed.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

try:
    from temporalio import activity
    from temporalio.exceptions import ApplicationError
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

from continuum.agent.exceptions import AgentConfigurationError, MaxTurnsExceededError
from continuum.exceptions import ConfigurationError, PolicyDeniedError
from continuum.temporal.exceptions import AgentNotRegisteredError
from continuum.temporal.registry import get_agent_registry
from continuum.temporal.types import (
    AgentActivityParams,
    AgentActivityResult,
    NotificationParams,
)

_HEARTBEAT_INTERVAL = 15  # seconds — must be less than heartbeat_timeout in the workflow

# Errors a retry cannot fix; with raise_on_error they fail the step at once.
_PERMANENT_ERRORS: tuple[type, ...] = (
    AgentNotRegisteredError,
    MaxTurnsExceededError,
    PolicyDeniedError,
    ConfigurationError,
    AgentConfigurationError,
)


def _is_permanent(error: BaseException) -> bool:
    """True for an agent error that a retry cannot fix.

    The runner wraps unexpected errors in ``AgentExecutionError``, so the
    wrapped ``original_error`` is checked too.
    """
    for e in (error, getattr(error, "original_error", None)):
        if isinstance(e, _PERMANENT_ERRORS):
            return True
    return False


@activity.defn
async def run_agent_activity(params: AgentActivityParams) -> AgentActivityResult:
    """Execute ANY registered agent via AgentRunner.

    Agent-agnostic: looks up agent by name from registry, calls runner.run().
    All existing SDK features (LLM, memory, session, tools, observability) are unchanged.

    By default an agent error comes back as a result with ``status="error"``.
    With ``params.raise_on_error`` (set by ``run_agent_step``) the activity
    raises ``ApplicationError`` instead, so the caller's retry policy applies:
    an unregistered agent, max turns, a policy denial and a configuration error
    are not retried, and an open circuit breaker is retried after its cooldown.
    """
    registry = get_agent_registry()

    activity.heartbeat(f"Running agent: {params.agent_name}")

    async def _heartbeat_loop() -> None:
        """Keep Temporal from timing out the activity during long LLM calls."""
        turn = 0
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            turn += 1
            activity.heartbeat(f"Agent {params.agent_name} running (turn ~{turn})")

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    try:
        runner = registry.get_runner()
        agent = registry.get(params.agent_name)

        response = await runner.run(
            agent=agent,
            input=params.input,
            session_id=params.session_id,
            user_id=params.user_id,
            metadata=params.metadata if params.metadata else None,
            tags=params.tags if params.tags else None,
        )
        retry_after = (getattr(response, "run_artifacts", None) or {}).get("retry_after_s")
        if params.raise_on_error and retry_after is not None:
            # The runner's circuit breaker is open: retry once it has cooled down.
            raise ApplicationError(
                f"Agent '{params.agent_name}' failed: {response.error}",
                type="CircuitBreakerOpen",
                next_retry_delay=timedelta(seconds=max(1.0, float(retry_after))),
            )
        return AgentActivityResult.from_agent_response(response)
    except Exception as e:
        if params.raise_on_error and isinstance(e, ApplicationError):
            raise  # already shaped for Temporal (the open circuit breaker above)
        attempt = activity.info().attempt
        activity.logger.error(f"Agent '{params.agent_name}' failed (attempt {attempt}): {e}")
        if not params.raise_on_error:
            return AgentActivityResult(
                content="",
                status="error",
                error=str(e),
                agents_used=[params.agent_name],
            )
        raise ApplicationError(
            f"Agent '{params.agent_name}' failed: {e}",
            type=type(e).__name__,
            non_retryable=_is_permanent(e),
        ) from e
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
