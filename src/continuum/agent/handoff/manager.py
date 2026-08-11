"""
Handoff Manager - Handles agent-to-agent transitions.

Manages handoffs with history summarization and state management.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from continuum.agent.exceptions import (
    HandoffCycleDetectedError,
    HandoffDepthExceededError,
    HandoffNotAllowedError,
)
from continuum.agent.handoff.history import HistorySummarizer
from continuum.agent.types import (
    AgentEvent,
    EventType,
    Handoff,
    HandoffData,
    HandoffResult,
    HistorySummarizationMode,
    RunContext,
    generate_handoff_id,
)
from continuum.llm.untrusted_content import fence_untrusted
from continuum.logging import get_logger

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.llm import LLMClient
    from continuum.observability import TracingManager

logger = get_logger(__name__)

# Envelope for the model-authored half of the handoff framing (see
# build_handoff_messages). Distinct from `handoff_summary` so the target agent can
# tell "what the previous agent said about this transfer" from "a summary of the
# transcript".
_HANDOFF_CONTEXT_TAG = "handoff_context"


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
        from continuum.agent.handoff import HandoffManager

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
                    recent_turns=handoff.recent_turns,
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
            # Only pass tool call/response pairs if the target agent has those tools.
            # Orphaned tool messages (result without preceding tool_calls) cause 400
            # errors from both OpenAI and Gemini.
            target_tool_names = {
                t.get("function", {}).get("name") for t in (target_agent.get_tools_for_llm() or [])
            }

            filtered = []
            i = 0
            while i < len(handoff_data.history):
                m = handoff_data.history[i]
                role = m.get("role")

                if role == "system":
                    i += 1
                    continue

                if role == "assistant" and m.get("tool_calls"):
                    called_tools = {tc.get("function", {}).get("name") for tc in m["tool_calls"]}
                    keep = called_tools.issubset(target_tool_names)
                    if keep:
                        filtered.append(m)
                    i += 1
                    # Grab all tool results for this assistant together
                    while i < len(handoff_data.history):
                        next_m = handoff_data.history[i]
                        if next_m.get("role") != "tool":
                            break
                        if keep:
                            filtered.append(next_m)
                        i += 1
                    continue

                if role == "tool":
                    # Stray tool result not consumed above — skip it
                    i += 1
                    continue

                if role == "assistant" and not m.get("content"):
                    i += 1
                    continue

                filtered.append(m)
                i += 1

            messages.extend(filtered)

        # Handoff framing, split by who wrote it (security finding F4).
        #
        # `reason` and `context` are parsed from the handoff tool call's arguments,
        # i.e. they are written by the SOURCE AGENT'S MODEL. Putting them in a
        # system message gave model-authored text the trust level of developer
        # instructions -- and on Anthropic it is worse than it looks: the provider
        # hoists every system message, wherever it sits in the list, and joins them
        # into the single top-level `system` parameter, so the text was
        # concatenated straight onto the agent's own system prompt.
        #
        # Only SDK-authored scaffolding stays in `system`. The model-authored parts
        # drop to `user` inside an untrusted envelope.
        system_parts = [f"You are receiving a handoff from {handoff_data.from_agent}."]
        # Always surface the session_id so the target agent can use it for tool
        # calls -- it comes from RunContext, not from the model, so it stays here.
        if session_id:
            system_parts.append(f"session_id: {session_id}")
        scaffolding = "\n".join(system_parts)

        # Fold into the leading system message (the target's instructions) rather
        # than appending a trailing one, so the untrusted block below can still see
        # the history's last message and merge with it. Mirrors
        # untrusted_content._ensure_system_instruction.
        if messages and messages[0].get("role") == "system":
            first = messages[0]
            messages[0] = {**first, "content": f"{first.get('content') or ''}\n\n{scaffolding}"}
        else:
            messages.insert(0, {"role": "system", "content": scaffolding})

        untrusted_parts = [f"Reason: {handoff_data.reason}"]
        if handoff_data.context:
            untrusted_parts.append(f"Context: {handoff_data.context}")
        block = fence_untrusted("\n".join(untrusted_parts), _HANDOFF_CONTEXT_TAG)

        # Merge into a trailing user message rather than appending a second one.
        # AnthropicProvider._split_messages appends user messages unconditionally
        # with no consecutive-user merging (contrast its tool_result branch, which
        # does merge), so two in a row would reach the provider as two user turns.
        # Copy-not-mutate: these dicts are shared with the caller's session history.
        if messages and messages[-1].get("role") == "user":
            prev = messages[-1]
            merged = f"{prev.get('content') or ''}\n\n{block}".lstrip()
            messages[-1] = {**prev, "content": merged}
        else:
            messages.append({"role": "user", "content": block})

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
            from continuum.observability.provider_manager import get_provider_manager
            from continuum.observability.trace_context import get_current_trace_id

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
