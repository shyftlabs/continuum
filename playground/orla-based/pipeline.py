"""
Orchestration logic for the orla-based playground.

Provides three run modes:
1. direct(tier, message)  — routes via RouterAgent to free/premium agent
2. dag(message)           — runs the parallel DAG pipeline
3. summarize(message)     — runs the summarizer (low stage_priority demo)

Policy enforcement happens inside ToolExecutor via policy_store.
Data labels are stamped on RunContext at entry.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from orchestrator.agent.types import RunContext
from orchestrator.logging import get_logger

from agents import OrlaPlayground
from config import TIER_PRIORITY

logger = get_logger(__name__)


async def _get_session_id(app: OrlaPlayground, user_id: str, conversation_id: str) -> str | None:
    """Get or create a Redis session ID for short-term memory."""
    if not app.container:
        return None
    try:
        session_client = app.container.session_client
        if session_client and session_client.is_enabled:
            return await session_client.get_or_create_session(
                user_id=user_id,
                conversation_id=conversation_id,
            )
    except Exception as e:
        logger.warning(f"Session init failed: {e}")
    return None


async def run_direct(
    app: OrlaPlayground,
    message: str,
    tier: str = "free",
    data_labels: set[str] | None = None,
    user_id: str = "playground-user",
    conversation_id: str = "playground-conv",
) -> str:
    """Route message through RouterAgent to free or premium agent."""
    agent_name = "premium-agent" if tier == "premium" else "free-agent"
    agent = app.premium_agent if tier == "premium" else app.free_agent

    context = RunContext(
        run_id=f"run-{tier}-{abs(hash(message)) % 10000:04d}",
        priority=TIER_PRIORITY.get(tier, 5),
        data_labels=data_labels or set(),
        user_id=user_id,
    )

    # Stamp priority from router
    route = app.router.get_route(agent_name)
    if route:
        context.priority = route.dispatch_priority

    # Data labels: enforce restricted label blocks checkout for all tiers
    if "restricted" in context.data_labels:
        from orchestrator.security.policy import AccessPolicy
        app.config.policy_store.add_policy(AccessPolicy(
            name="_label_restricted",
            subjects=["*"],
            resources=["tool:checkout"],
            effect="deny",
            denial_message="'checkout' is blocked because this request involves restricted data. The operation cannot proceed regardless of tier.",
        ))
        logger.info(f"[data_labels] 'restricted' label active — checkout blocked for all tiers")

    logger.info(
        f"[pipeline] agent={agent_name} dispatch_priority={context.priority} "
        f"data_labels={sorted(context.data_labels)}"
    )
    print(
        f"  dispatch_priority={context.priority}  "
        f"data_labels={sorted(context.data_labels) or '(none)'}"
    )

    session_id = await _get_session_id(app, user_id, conversation_id)
    context.session_id = session_id

    try:
        response = await app.runner.run(
            agent=agent,
            input=message,
            context=context,
        )
        return response.content or ""
    except Exception as e:
        return f"[Error] {e}"
    finally:
        app.config.policy_store.remove_policy("_label_restricted")


async def run_dag(
    app: OrlaPlayground,
    message: str,
    data_labels: set[str] | None = None,
    user_id: str = "playground-user",
) -> str:
    """Run the parallel DAG pipeline: fetch+recommend → synthesize → reply."""
    context = RunContext(
        run_id=f"dag-{abs(hash(message)) % 10000:04d}",
        priority=5,
        data_labels=data_labels or set(),
        user_id=user_id,
    )

    try:
        response = await app.dag_agent.execute(
            input_text=message,
            runner=app.runner,
            context=context,
        )
        return response.content or ""
    except Exception as e:
        return f"[DAG Error] {e}"


async def run_summarize(app: OrlaPlayground, conversation: str) -> str:
    """Run the low-priority summarizer agent."""
    context = RunContext(
        run_id=f"sum-{abs(hash(conversation)) % 10000:04d}",
        priority=1,
    )
    try:
        response = await app.runner.run(
            agent=app.summarizer,
            input=conversation,
            context=context,
        )
        return response.content or ""
    except Exception as e:
        return f"[Summarizer Error] {e}"
