"""
Context utilities for agent execution.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from continuum.agent.types import RunContext
    from continuum.tools.types import ToolContextState


def publish_active_policy(
    method: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator for workflow ``execute()`` methods: publish the agent's policy
    context for the call's duration.

    Workflow agents run via ``execute()`` rather than ``AgentRunner.run``, so the
    run-level ambient publisher doesn't wrap them. Without this, an orchestrator's
    own coordination LLM calls (split/merge/synthesize/critique/route) would
    bypass the data-label model-routing and telemetry gates. The ``RunContext``
    is located among the call args (positional or ``context=`` keyword); if none
    is present the method runs unchanged.
    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        from continuum.agent.types import RunContext
        from continuum.security.policy_context import use_active_policy

        ctx = kwargs.get("context")
        if ctx is None:
            ctx = next((a for a in args if isinstance(a, RunContext)), None)

        if ctx is not None:
            with use_active_policy(getattr(self, "policy_store", None), self.name, ctx):
                return await method(self, *args, **kwargs)
        return await method(self, *args, **kwargs)

    wrapper.__publishes_active_policy__ = True  # type: ignore[attr-defined]
    return wrapper


def create_run_context(
    run_id: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
    max_turns: int = 25,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    data_labels: set[str] | None = None,
) -> RunContext:
    """
    Create a run context with default values.

    Args:
        run_id: Optional run ID (generated if not provided)
        session_id: Optional session ID
        conversation_id: Optional conversation ID (chat window ID from caller)
        user_id: Optional user ID
        trace_id: Optional trace ID
        max_turns: Maximum conversation turns
        metadata: Optional metadata
        tags: Optional tags
        data_labels: Optional initial data-sensitivity labels (run-level
            provenance) — e.g. the request arrived from a connector/endpoint
            already known to carry "pii". Taints the run from the start.

    Returns:
        RunContext instance
    """
    from continuum.agent.types import RunContext, generate_run_id

    return RunContext(
        run_id=run_id or generate_run_id(),
        session_id=session_id,
        conversation_id=conversation_id,
        user_id=user_id,
        trace_id=trace_id,
        max_turns=max_turns,
        metadata=metadata or {},
        tags=tags or [],
        data_labels=set(data_labels) if data_labels else set(),
    )


def inject_tool_context_to_prompt(
    context_state: ToolContextState,
) -> str | None:
    """
    Generate system prompt injection for tool context awareness.

    Args:
        context_state: Tool context state with captured variables

    Returns:
        Context string to inject into system prompt, or None if empty
    """
    if context_state.is_empty():
        return None

    base_context = context_state.to_prompt_context()

    # Check if we have a session_id - if so, tell LLM not to create a new one
    has_session_id = False
    for namespace in context_state.get_all_namespaces():
        if context_state.get(namespace, "session_id"):
            has_session_id = True
            break

    if has_session_id:
        return (
            f"{base_context}\n\n"
            "IMPORTANT: A session already exists. Do NOT call create_session again. "
            "Use the existing session_id for all tool calls that require it."
        )

    return base_context
