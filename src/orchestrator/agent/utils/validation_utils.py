"""
Validation utilities for agent execution.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from orchestrator.agent.types import AgentResponse, ResponseStatus
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.types import RunContext
    from orchestrator.llm.types import ChatMessage

logger = get_logger(__name__)


async def validate_input(
    agent: BaseAgent,
    input: str | list[dict[str, Any]] | list[ChatMessage],
    context: RunContext,
) -> AgentResponse | None:
    """
    Validate input against agent's input_schema.

    Returns None if validation passes, or an error AgentResponse if validation fails.
    Handles failures gracefully by returning a structured error response.

    Args:
        agent: Agent to validate input for
        input: Input to validate
        context: Run context

    Returns:
        None if valid, AgentResponse with error if invalid
    """
    if agent.input_schema is None:
        return None

    try:
        # Extract content to validate
        if isinstance(input, str):
            content = input
        elif isinstance(input, list) and input:
            # Get content from last user message
            last_msg = input[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                content = last_msg.content or ""
            else:
                content = str(last_msg)
        else:
            content = ""

        # Try to parse as JSON for structured validation
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, try to validate as a simple string input
            data = {"input": content}

        # Validate against schema
        agent.input_schema.model_validate(data)

        logger.debug(f"Input validation passed for agent {agent.name}")
        return None  # Validation passed

    except ValidationError as e:
        logger.warning(
            f"Input validation failed for agent {agent.name}: {e}",
            extra={"errors": e.errors()},
        )

        # Return graceful error response (safe access to avoid KeyError if Pydantic structure changes)
        error_details = [
            f"- {err.get('loc', '?')}: {err.get('msg', 'unknown error')}"
            for err in e.errors()
        ]

        return AgentResponse(
            content="I couldn't process your request due to invalid input:\n"
            + "\n".join(error_details),
            run_id=context.run_id,
            agent_name=agent.name,
            status=ResponseStatus.ERROR,
            error=f"Input validation failed: {str(e)}",
            error_type="ValidationError",
            trace_id=context.trace_id,
        )

    except Exception as e:
        logger.error(f"Unexpected error during input validation: {e}")
        # Don't fail on validation errors, continue with execution
        return None
