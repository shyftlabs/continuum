"""
Temporal Activities for the Orchestrator SDK.

These activities bridge Temporal to the existing AgentRunner.
They are agent-agnostic: any registered BaseAgent can be executed.
"""

from __future__ import annotations

try:
    from temporalio import activity
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. "
        "Install it with: pip install -e '.[temporal]'"
    ) from _err

from orchestrator.temporal.registry import get_agent_registry
from orchestrator.temporal.types import (
    AgentActivityParams,
    AgentActivityResult,
    NotificationParams,
)


@activity.defn
async def run_agent_activity(params: AgentActivityParams) -> AgentActivityResult:
    """Execute ANY registered agent via AgentRunner.

    Agent-agnostic: looks up agent by name from registry, calls runner.run().
    All existing SDK features (LLM, memory, session, tools, observability) are unchanged.
    """
    registry = get_agent_registry()

    activity.heartbeat(f"Running agent: {params.agent_name}")

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
    except Exception as e:
        activity.logger.error(f"Agent '{params.agent_name}' failed: {e}")
        return AgentActivityResult(
            content="",
            status="error",
            error=str(e),
            agents_used=[params.agent_name],
        )


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
        activity.logger.warning(
            "No notification handler configured; approval notification dropped"
        )
