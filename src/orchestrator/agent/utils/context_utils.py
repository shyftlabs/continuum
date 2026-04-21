"""
Context utilities for agent execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.agent.types import RunContext
    from orchestrator.tools.types import ToolContextState


def create_run_context(
    run_id: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
    max_turns: int = 25,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
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

    Returns:
        RunContext instance
    """
    from orchestrator.agent.types import RunContext, generate_run_id

    return RunContext(
        run_id=run_id or generate_run_id(),
        session_id=session_id,
        conversation_id=conversation_id,
        user_id=user_id,
        trace_id=trace_id,
        max_turns=max_turns,
        metadata=metadata or {},
        tags=tags or [],
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
