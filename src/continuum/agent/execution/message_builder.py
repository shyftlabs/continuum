"""
Message Builder - Prepares messages for agent execution.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from continuum.agent.interfaces.handler_interface import IMessageBuilder
from continuum.logging import get_logger
from continuum.observability.decorators import observe
from continuum.utils.sanitization import (
    detect_injection_patterns,
    sanitize_user_input,
)

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.services.memory_service import MemoryService
    from continuum.agent.services.session_service import SessionService
    from continuum.agent.types import RunContext
    from continuum.tools.types import ToolContextState

logger = get_logger(__name__)


_REACT_TEMPLATE_BASE = """
Before answering, call the 'think' tool to reason step by step.
Then give your final answer.
"""

_REACT_TEMPLATE_WITH_TOOLS = """
Before calling any tool or giving a final answer, call the 'think' tool to reason step by step.

Example flow:
1. think(thought="I need X, so I will call tool Y with Z")
2. Call the actual tool
3. think(thought="The result shows X, now I need to...") — if more steps needed
4. Give your final answer

Available tools (besides 'think'):
{tool_list}
"""


def _build_react_template(agent: Any) -> str:
    """Build the ReAct template, injecting tool names if the agent has tools."""
    tools = agent.get_tools_for_llm() if hasattr(agent, "get_tools_for_llm") else []
    if not tools:
        return _REACT_TEMPLATE_BASE
    tool_lines = []
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        if not name or name == "think":  # skip think — it's internal to ReAct
            continue
        desc = fn.get("description", "")
        tool_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    if not tool_lines:
        return _REACT_TEMPLATE_BASE
    return _REACT_TEMPLATE_WITH_TOOLS.format(tool_list="\n".join(tool_lines))


class MessageBuilder(IMessageBuilder):
    """
    Builder for preparing messages for agent execution.

    Handles:
    - System prompt injection
    - Tool context injection
    - Memory retrieval and injection
    - Session history loading
    - Context compression
    """

    def __init__(
        self,
        memory_service: MemoryService | None = None,
        session_service: SessionService | None = None,
    ):
        """
        Initialize message builder.

        Args:
            memory_service: Memory service for retrieving memories
            session_service: Session service for loading history
        """
        self._memory_service = memory_service
        self._session_service = session_service

    @observe(name="prepare_messages", capture_output=True)
    async def prepare_messages(
        self,
        agent: BaseAgent,
        input: str | list[dict[str, Any]] | list[Any],
        context: RunContext,
        tool_context_state: ToolContextState | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Prepare messages for agent execution.

        Args:
            agent: Agent to prepare messages for
            input: User input (string or messages)
            context: Run context
            tool_context_state: Optional tool context state

        Returns:
            Prepared message list
        """
        messages = []

        # Add system prompt
        if agent.system_prompt:
            messages.append({"role": "system", "content": agent.system_prompt})

        # Inject ReAct scaffold if enabled (must come before user messages)
        if agent.config and agent.config.react_mode:
            messages.append({"role": "system", "content": _build_react_template(agent)})

        # Inject tool context into system prompt for LLM awareness
        if tool_context_state and not tool_context_state.is_empty():
            # Validate tool context state before injection
            try:
                if not hasattr(tool_context_state, "to_prompt_context") or not hasattr(
                    tool_context_state, "get_all_namespaces"
                ):
                    logger.warning(
                        "Tool context state missing required methods (to_prompt_context, get_all_namespaces). "
                        "Skipping injection."
                    )
                else:
                    namespaces = tool_context_state.get_all_namespaces()
                    if not isinstance(namespaces, (list, set, tuple)):
                        logger.warning(
                            f"Tool context state returned invalid namespaces type: {type(namespaces)}. "
                            f"Skipping injection."
                        )
                    else:
                        context_prompt = self._inject_tool_context_to_prompt(tool_context_state)
                        if context_prompt:
                            messages.append({"role": "system", "content": context_prompt})
                            logger.info(
                                "📋 Injected tool context into system prompt (existing session_id found)"
                            )
            except Exception as e:
                logger.warning(
                    "Failed to validate/inject tool context state; agent=%s error_type=%s",
                    agent.name,
                    type(e).__name__,
                )

        # Inject memory facts early (user profile/background — stable context like instructions)
        if agent.memory_config and agent.memory_config.search_memories and self._memory_service:
            try:
                query = input if isinstance(input, str) else str(input)
                memories = await self._memory_service.retrieve_memories(agent, query, context)

                if memories:
                    memory_content = "User profile (long-term preferences and context):\n"
                    for m in memories:
                        memory_content += f"- {m.get('memory', str(m))}\n"

                    logger.info(
                        "Injecting memories into LLM context: agent=%s memory_count=%d",
                        agent.name,
                        len(memories),
                    )

                    messages.append({"role": "system", "content": memory_content})
            except Exception as e:
                logger.warning(
                    "Failed to retrieve memories: agent=%s error_type=%s",
                    agent.name,
                    type(e).__name__,
                )

        # Inject pipeline context from sequential/supervised/planner workflows
        # so sub-agents can see prior steps' outputs without loading Redis.
        pipeline_ctx = context.metadata.get("pipeline_context") if context.metadata else None
        if pipeline_ctx:
            messages.append({"role": "system", "content": pipeline_ctx})

        # Load session history if available.
        # Skip on handoff turns — the handoff messages passed as input already
        # carry the summarized context; loading Redis history would duplicate it.
        if context.session_id and self._session_service and not context.is_handoff:
            try:
                history_turns = (
                    agent.config.session_history_turns
                    if agent.config and agent.config.session_history_turns is not None
                    else 20
                )
                if history_turns == 0:
                    pass  # explicitly disabled — skip Redis call
                else:
                    history = await self._session_service.get_conversation_history(
                        context.session_id, limit=history_turns
                    )
                    if history:
                        logger.debug(
                            "Retrieved session history: agent=%s message_count=%d",
                            agent.name,
                            len(history),
                        )
                    messages.extend(history)
            except Exception as e:
                from continuum.session.exceptions import SessionNotFoundError

                if isinstance(e, SessionNotFoundError):
                    logger.warning(
                        "No history loaded because session does not exist: agent=%s",
                        agent.name,
                    )
                else:
                    logger.warning(
                        "Failed to load session history: agent=%s error_type=%s",
                        agent.name,
                        type(e).__name__,
                    )

        # Inject RAG context last (closest to current question for maximum recency effect)
        rag_context = agent.config.rag_context if agent.config else None
        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "--- PROVIDED CONTEXT (use for new factual/analytical questions; "
                        "for references to this conversation use the conversation history above) ---\n\n"
                        + rag_context
                        + "\n\n--- END CONTEXT ---"
                    ),
                }
            )

        # Sanitize user input if enabled via agent config
        should_sanitize = not agent.config or agent.config.input_sanitization
        should_detect = agent.config and agent.config.injection_detection
        if isinstance(input, str):
            if should_sanitize:
                input = sanitize_user_input(input)
            if should_detect:
                detected = detect_injection_patterns(input)
                if detected:
                    logger.warning(
                        "Potential prompt injection detected: agent=%s match_count=%d",
                        agent.name,
                        len(detected),
                    )

        # Run product input scanners (e.g. an LLM Guard PromptInjection/Gibberish scanner).
        # Any scanner that returns is_safe=False raises InputBlockedError — the calling router
        # catches this and returns a blocked response without invoking the LLM.
        if isinstance(input, str) and agent.config and agent.config.input_scanners:
            from continuum.exceptions import InputBlockedError

            for scanner in agent.config.input_scanners:
                try:
                    input, is_safe, reason = scanner(input)
                    if not is_safe:
                        logger.warning(
                            "Input scanner blocked request — agent=%s scanner=%s",
                            agent.name,
                            getattr(scanner, "__name__", type(scanner).__name__),
                        )
                        raise InputBlockedError(
                            f"Input blocked by scanner: {reason}",
                            scanner=reason or "",
                        )
                except InputBlockedError:
                    raise
                except Exception as e:
                    logger.warning(
                        "Input scanner failed (fail-open): agent=%s scanner=%s error_type=%s",
                        agent.name,
                        getattr(scanner, "__name__", type(scanner).__name__),
                        type(e).__name__,
                    )

        # Record the index where user input begins — used by save_messages to know
        # exactly which messages are new (avoids fragile initial_count - 1 arithmetic).
        user_message_index = len(messages)

        # Add user input
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            for item in input:
                msg = self._message_to_dict(item)
                if should_sanitize and msg.get("role") == "user" and msg.get("content"):
                    msg = msg.copy()
                    msg["content"] = sanitize_user_input(msg["content"])
                messages.append(msg)

        # Apply context management (proactive compression) if enabled
        try:
            from continuum.llm.context_management import (
                ContextManagementConfig,
                get_progressive_context_manager,
            )

            # Get agent-specific config or use global defaults
            context_config = None
            if agent.config and agent.config.context_management:
                context_config = agent.config.context_management
            else:
                context_config = ContextManagementConfig()

            if context_config.enabled:
                context_manager = get_progressive_context_manager(config=context_config)
                messages, compression_result = await context_manager.compress_if_needed(
                    messages=messages,
                    model=agent.model,
                    config=context_config,
                )

                if compression_result.was_compressed:
                    logger.info(
                        f"Agent {agent.name}: Context compressed proactively - "
                        f"{compression_result.original_token_count} → {compression_result.compressed_token_count} tokens "
                        f"({compression_result.compression_ratio:.1%} ratio, strategy: {compression_result.strategy_used})"
                    )
                    # Compression may have shortened the list, so find the user message
                    # by scanning backward from the end (it was the last message appended).
                    user_message_index = len(messages) - 1
                    while (
                        user_message_index > 0
                        and messages[user_message_index].get("role") != "user"
                    ):
                        user_message_index -= 1
        except Exception as e:
            logger.warning(
                "Context management failed; continuing without compression: agent=%s error_type=%s",
                agent.name,
                type(e).__name__,
            )

        # Run tool-attention routing: filters tools and produces Phase 1 summary.
        from continuum.tools.tool_attention.router import apply_tool_attention

        filtered_tools = (
            await apply_tool_attention(agent, messages, context) or agent.get_tools_for_llm()
        )
        if context.metadata is not None:
            context.metadata["_filtered_tools"] = filtered_tools

        # Include the Phase 1 tool summary's role in request-shape diagnostics without
        # logging its generated prompt content.
        _phase1 = context.metadata.get("tool_summary_message") if context.metadata else None
        if _phase1:
            _insert_at = 0
            for _i, _msg in enumerate(messages):
                if _msg.get("role") == "system":
                    _insert_at = _i + 1
                else:
                    break
            display_messages = messages[:_insert_at] + [_phase1] + messages[_insert_at:]
        else:
            display_messages = messages

        known_roles = {"system", "developer", "user", "assistant", "tool", "function"}
        role_sequence = [
            role if (role := message.get("role")) in known_roles else "unknown"
            for message in display_messages
        ]
        tool_names = [
            tool.get("function", {}).get("name", "unknown")
            if isinstance(tool, dict)
            else getattr(getattr(tool, "function", None), "name", "unknown")
            for tool in filtered_tools
        ]
        logger.info(
            "Prepared LLM request: agent=%s message_count=%d role_sequence=%s tool_names=%s",
            agent.name,
            len(display_messages),
            role_sequence,
            tool_names,
        )

        return messages, user_message_index

    def _inject_tool_context_to_prompt(
        self,
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

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert a message to dictionary format."""
        if isinstance(message, dict):
            return message
        if hasattr(message, "to_dict"):
            return message.to_dict()
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {"role": "user", "content": str(message)}
