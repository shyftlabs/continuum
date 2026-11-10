"""
Temporal Activities for the Orchestrator SDK.

These activities bridge Temporal to the existing AgentRunner.
They are agent-agnostic: any registered BaseAgent can be executed.
"""

from __future__ import annotations

import asyncio

try:
    from temporalio import activity
    from temporalio.exceptions import ApplicationError
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

from continuum.temporal.exceptions import AgentNotRegisteredError
from continuum.temporal.registry import get_agent_registry
from continuum.temporal.types import (
    AgentActivityParams,
    AgentActivityResult,
    NotificationParams,
)

_HEARTBEAT_INTERVAL = 15  # seconds — must be less than heartbeat_timeout in the workflow


@activity.defn
async def run_agent_activity(params: AgentActivityParams) -> AgentActivityResult:
    """Execute ANY registered agent via AgentRunner.

    Agent-agnostic: looks up agent by name from registry, calls runner.run().
    All existing SDK features (LLM, memory, session, tools, observability) are unchanged.

    Raises ApplicationError when the agent raises, so Temporal retries the step;
    an unregistered agent is not retried.
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
        return AgentActivityResult.from_agent_response(response)
    except AgentNotRegisteredError as e:
        # Retrying cannot register the agent; fail the step at once.
        raise ApplicationError(str(e), type="AgentNotRegisteredError", non_retryable=True) from e
    except Exception as e:
        # Raise instead of returning an error result, so Temporal retries the
        # step with the workflow's retry policy (a model timeout, a rate limit).
        attempt = activity.info().attempt
        activity.logger.error(f"Agent '{params.agent_name}' failed (attempt {attempt}): {e}")
        raise ApplicationError(
            f"Agent '{params.agent_name}' failed: {e}", type=type(e).__name__
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
