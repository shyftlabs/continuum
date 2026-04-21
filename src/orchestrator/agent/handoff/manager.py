"""
Handoff Manager - Handles agent-to-agent transitions.

Manages handoffs with history summarization and state management.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from orchestrator.agent.exceptions import (
    HandoffCycleDetectedError,
    HandoffDepthExceededError,
    HandoffNotAllowedError,
)
from orchestrator.agent.handoff.history import (
    HistorySummarizer,
    flatten_nested_history,
)
from orchestrator.agent.types import (
    AgentEvent,
    EventType,
    Handoff,
    HandoffData,
    HandoffResult,
    HistorySummarizationMode,
    RunContext,
    generate_handoff_id,
)
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.llm import LLMClient
    from orchestrator.observability import TracingManager

logger = get_logger(__name__)


class HandoffManager:
    """
    Manages agent-to-agent handoffs.

    Handles:
    - History summarization for efficient context transfer
    - Handoff validation and authorization
    - Call stack management for return-to-parent
    - Langfuse tracing for full observability

    Example:
        ```python
        from orchestrator.agent.handoff import HandoffManager

        manager = HandoffManager()

        # Prepare handoff data
        handoff_data = await manager.prepare_handoff(
            from_agent=triage_agent,
            to_agent=specialist_agent,
            reason="User needs billing help",
            messages=conversation_history,
        )

        # Execute handoff
        result = await manager.execute_handoff(
            handoff_data=handoff_data,
            context=run_context,
            runner=agent_runner,
        )
        ```
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tracing_manager: TracingManager | None = None,
        max_depth: int = 10,
    ):
        """
        Initialize the handoff manager.

        Args:
            llm_client: LLM client for history summarization
            tracing_manager: Tracing manager for observability
            max_depth: Maximum handoff depth
        """
        self._llm_client = llm_client
        self._tracing_manager = tracing_manager
        self._max_depth = max_depth

    def validate_handoff(
        self,
        from_agent: BaseAgent,
        to_agent_name: str,
        current_depth: int = 0,
        agent_stack: list[str] | None = None,
    ) -> Handoff:
        """
        Validate that a handoff is allowed.

        Args:
            from_agent: Source agent
            to_agent_name: Target agent name
            current_depth: Current handoff depth
            agent_stack: Current stack of agent names in the handoff chain

        Returns:
            Handoff definition

        Raises:
            HandoffNotAllowedError: If handoff not defined
            HandoffDepthExceededError: If depth exceeded
            HandoffCycleDetectedError: If cycle detected in handoff chain
        """
        # Check depth
        if current_depth >= self._max_depth:
            raise HandoffDepthExceededError(
                current_depth=current_depth,
                max_depth=self._max_depth,
                agent_name=from_agent.name,
            )

        # Check for cycles in handoff chain
        if agent_stack is not None and to_agent_name in agent_stack:
            raise HandoffCycleDetectedError(
                from_agent=from_agent.name,
                to_agent=to_agent_name,
                agent_stack=agent_stack,
            )

        # Check if handoff is defined
        handoff = from_agent.get_handoff(to_agent_name)
        if handoff is None:
            raise HandoffNotAllowedError(
                from_agent=from_agent.name,
                to_agent=to_agent_name,
                reason="Handoff not defined in agent configuration",
            )

        return handoff

    def detect_cycle(
        self,
        agent_stack: list[str],
        target_agent: str,
    ) -> bool:
        """
        Check if adding target_agent to the stack would create a cycle.

        Args:
            agent_stack: Current stack of agent names
            target_agent: Name of agent to check

        Returns:
            True if a cycle would be created, False otherwise
        """
        return target_agent in agent_stack

    async def prepare_handoff(
        self,
        from_agent: BaseAgent,
        to_agent: BaseAgent,
        reason: str,
        messages: list[dict[str, Any]],
        context: str | None = None,
        handoff_config: Handoff | None = None,
        run_context: RunContext | None = None,
    ) -> HandoffData:
        """
        Prepare handoff data including history summarization.

        Args:
            from_agent: Source agent
            to_agent: Target agent
            reason: Reason for handoff
            messages: Conversation history
            context: Additional context
            handoff_config: Handoff configuration (if not using agent's)
            run_context: Current run context

        Returns:
            PreparedHandoffData ready for execution
        """
        # Get handoff config
        handoff = handoff_config or from_agent.get_handoff(to_agent.name)
        if handoff is None:
            handoff = Handoff(
                target_agent=to_agent.name,
                description="",
            )

        handoff_id = generate_handoff_id()

        # Prepare history
        history = []
        history_summary = None

        if handoff.transfer_history and messages:
            if handoff.summarize_history:
                # Summarize history
                summarizer = HistorySummarizer(
                    mode=handoff.summarization_mode,
                    recent_n=handoff.recent_messages,
                )

                summarized = await summarizer.summarize(
                    messages=messages,
                    llm_client=self._llm_client,
                    model=from_agent.model,
                )

                history = summarized

                # Also create text summary for logging
                if handoff.summarization_mode != HistorySummarizationMode.FULL:
                    text_summary = summarizer._text_summary(messages)
                    history_summary = text_summary.get("content", "")
            else:
                # Pass full history
                history = messages.copy()

        # Build metadata
        metadata = {
            "from_model": from_agent.model,
            "to_model": to_agent.model,
            "summarization_mode": handoff.summarization_mode.value
            if handoff.summarize_history
            else "none",
            "original_message_count": len(messages),
            "transferred_message_count": len(history),
        }

        if run_context:
            metadata["run_id"] = run_context.run_id
            metadata["session_id"] = run_context.session_id
            metadata["trace_id"] = run_context.trace_id
            metadata["handoff_depth"] = len(run_context.agent_stack)

        handoff_data = HandoffData(
            handoff_id=handoff_id,
            from_agent=from_agent.name,
            to_agent=to_agent.name,
            reason=reason,
            context=context,
            history=history,
            history_summary=history_summary,
            metadata=metadata,
        )

        logger.info(
            f"Prepared handoff: {from_agent.name} → {to_agent.name}",
            extra={
                "handoff_id": handoff_id,
                "from_agent": from_agent.name,
                "to_agent": to_agent.name,
                "reason": reason,
                "message_count": len(history),
            },
        )

        return handoff_data

    def build_handoff_messages(
        self,
        handoff_data: HandoffData,
        target_agent: BaseAgent,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the message list for the target agent.

        Args:
            handoff_data: Handoff data
            target_agent: Target agent

        Returns:
            Messages for target agent
        """
        messages = []

        # Add system message with agent instructions
        if target_agent.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": target_agent.instructions,
                }
            )

        # Add history (may be summarized)
        if handoff_data.history:
            # Flatten any nested histories first
            flattened = flatten_nested_history(handoff_data.history)
            # Strip system messages (source agent's instructions) and empty assistant
            # messages (in-progress turns) — target agent has its own system prompt
            flattened = [
                m for m in flattened
                if m.get("role") != "system"
                and not (m.get("role") == "assistant" and not m.get("content"))
            ]
            messages.extend(flattened)

        # Add handoff context as a system message
        context_parts = [
            f"You are receiving a handoff from {handoff_data.from_agent}.",
            f"Reason: {handoff_data.reason}",
        ]
        if handoff_data.context:
            context_parts.append(f"Context: {handoff_data.context}")
        # Always surface the session_id so the target agent can use it for tool calls
        if session_id:
            context_parts.append(f"session_id: {session_id}")

        messages.append(
            {
                "role": "system",
                "content": "\n".join(context_parts),
            }
        )

        return messages

    def create_handoff_event(
        self,
        event_type: EventType,
        handoff_data: HandoffData,
        run_id: str,
        additional_data: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """
        Create an agent event for handoff tracking.

        Args:
            event_type: Type of handoff event
            handoff_data: Handoff data
            run_id: Run ID
            additional_data: Additional data to include

        Returns:
            AgentEvent for the handoff
        """
        data = {
            "handoff_id": handoff_data.handoff_id,
            "from_agent": handoff_data.from_agent,
            "to_agent": handoff_data.to_agent,
            "reason": handoff_data.reason,
            "message_count": len(handoff_data.history),
        }

        if additional_data:
            data.update(additional_data)

        return AgentEvent(
            type=event_type,
            agent_name=handoff_data.from_agent
            if event_type == EventType.HANDOFF_START
            else handoff_data.to_agent,
            run_id=run_id,
            data=data,
            trace_id=handoff_data.metadata.get("trace_id"),
        )

    def should_return_to_parent(
        self,
        handoff: Handoff,
        agent_stack: list[str],
    ) -> bool:
        """
        Determine if control should return to parent agent.

        Args:
            handoff: Handoff configuration
            agent_stack: Current agent stack

        Returns:
            True if should return to parent
        """
        return handoff.return_to_parent and len(agent_stack) > 1

    async def trace_handoff(
        self,
        event_type: str,
        handoff_data: HandoffData,
        run_context: RunContext | None = None,
        result: HandoffResult | None = None,
    ) -> None:
        """
        Trace handoff event as an event under the current trace.

        CRITICAL: Creates an EVENT (not a trace) to ensure all handoffs
        appear under the single query execution trace.

        Args:
            event_type: Type of event (start, end, return)
            handoff_data: Handoff data
            run_context: Run context
            result: Handoff result (for end events)
        """
        try:
            from orchestrator.observability.provider_manager import get_provider_manager
            from orchestrator.observability.trace_context import get_current_trace_id

            manager = get_provider_manager()
            if not manager.is_enabled:
                return

            # Get trace ID from context (preferred) or run_context (fallback)
            trace_id = get_current_trace_id() or (run_context.trace_id if run_context else None)
            if not trace_id:
                logger.debug("Cannot trace handoff - no trace context available")
                return

            # Build event data
            event_data = {
                "handoff_id": handoff_data.handoff_id,
                "from_agent": handoff_data.from_agent,
                "to_agent": handoff_data.to_agent,
                "reason": handoff_data.reason,
                "context": handoff_data.context,
                "message_count": len(handoff_data.history),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if result:
                event_data["success"] = result.success
                event_data["returned_to_parent"] = result.returned_to_parent
                if result.error:
                    event_data["error"] = result.error

            # Create event name with agent names for clarity
            event_name = f"handoff.{event_type}"
            if event_type == "start":
                event_name = f"handoff.{handoff_data.from_agent}→{handoff_data.to_agent}"
            elif event_type == "end":
                event_name = f"handoff.complete.{handoff_data.to_agent}"

            # Use event() method via provider manager (creates event under trace)
            manager.event(
                trace_id=trace_id,
                name=event_name,
                metadata=event_data,
                level="DEFAULT" if (not result or result.success) else "ERROR",
            )
            logger.debug(f"Traced handoff event '{event_name}' under trace {trace_id}")

        except Exception as e:
            logger.warning(f"Failed to trace handoff: {e}")
