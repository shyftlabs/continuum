"""
Message Builder - Prepares messages for agent execution.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from continuum.agent.interfaces.handler_interface import IMessageBuilder
from continuum.logging import get_logger, log_content, log_id
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

MEMORY_HEADER = "User profile (long-term preferences and context):"

CACHE_BREAKPOINT_KEY = "cache_breakpoint_index"
"""Index of the first prompt block whose bytes change from turn to turn.

Anthropic caches the prefix up to and including the block carrying
``cache_control``, so anything volatile sitting inside that prefix changes its
hash and the breakpoint never hits. The builder is what knows which blocks are
volatile -- retrieved memory is a similarity search on the current input, and
pipeline context carries a prior step's output -- so it records where they start
and the executor places the marker before them.

Recorded on ``context.metadata`` rather than on the message dicts: providers
build their payloads from those dicts, and an unrecognised key would ride along
to the API.
"""


def _render_memory_context(memories: list[dict[str, Any]]) -> str:
    """Render retrieved memories, fencing only the rows provenance marks untrusted.

    Rows written by an untainted run are the user's own material and render
    exactly as they always have -- that shape is measured at full utility on
    every model tested, and changing it is not free: the envelope alone, with no
    rule attached, takes Claude's factual recall from 3/3 to 0/3.

    Rows carrying provenance labels were derived from untrusted input. Those go
    inside a ``recalled_memory`` envelope, which strips invisible characters and
    defangs any tag the content uses to close the fence early, and the standing
    rule is emitted alongside so the tag means something to the model.

    Splitting by provenance is what lets both hold at once. The two cases are
    indistinguishable as text -- a stored "prefers bullet points" and a planted
    "always append token X" are both directives in a memory row -- so no wording
    can separate them. Phase 1's stamp can.

    An unlabelled row counts as clean. Every row written before provenance
    existed is unlabelled, so treating them as suspect would fence the whole
    existing corpus and take recall down with it. Cover here is forward-only, by
    design; the tool gate is the control that does not depend on it.
    """
    if not memories:
        return ""

    from continuum.llm.untrusted_content import MEMORY_INSTRUCTION, MEMORY_TAG, fence_untrusted
    from continuum.memory.types import PROVENANCE_LABELS_KEY

    clean: list[str] = []
    untrusted: list[str] = []

    for m in memories:
        # Preserve the historical fallback: a row shaped unexpectedly renders its
        # repr rather than vanishing, so a retrieval bug stays visible.
        text = m.get("memory", str(m)) if isinstance(m, dict) else str(m)
        meta = m.get("metadata") if isinstance(m, dict) else None
        labels = meta.get(PROVENANCE_LABELS_KEY) if isinstance(meta, dict) else None
        # Same tolerance as the reader in memory_service: only a list/tuple of
        # strings counts. str and dict are both iterable, so accepting "any
        # iterable" would read a string's characters as labels.
        if isinstance(labels, list | tuple) and any(isinstance(x, str) for x in labels):
            untrusted.append(str(text))
        else:
            clean.append(str(text))

    parts: list[str] = []
    if clean:
        parts.append(MEMORY_HEADER + "\n" + "".join(f"- {t}\n" for t in clean))
    if untrusted:
        body = "".join(f"- {t}\n" for t in untrusted).rstrip("\n")
        parts.append(MEMORY_INSTRUCTION + "\n" + fence_untrusted(body, MEMORY_TAG))
    return "\n".join(parts)


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

        # Log agent memory config at start
        if hasattr(agent, "memory_config") and agent.memory_config:
            logger.info(
                "🔍 AGENT MEMORY CONFIG: search_memories=%s, store_memories=%s, search_scope=%s, store_scope=%s, search_limit=%s",
                agent.memory_config.search_memories,
                agent.memory_config.store_memories,
                agent.memory_config.search_scope,
                agent.memory_config.store_scope,
                agent.memory_config.search_limit,
            )
        else:
            logger.warning("⚠️ Agent has no memory_config!")

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
                            "Tool context state returned invalid namespaces type: %s. Skipping injection.",
                            type(namespaces),
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
                    "Failed to validate/inject tool context state: %s. Continuing without it.", e
                )

        # Retrieved memory. NOT stable context, despite where it sits: this is a
        # similarity search on the current turn's input, so its bytes change
        # almost every turn. CACHE_BREAKPOINT_KEY is recorded below so the
        # executor places the prompt-cache marker before it rather than after.
        if agent.memory_config and agent.memory_config.search_memories and self._memory_service:
            try:
                query = input if isinstance(input, str) else str(input)
                memories = await self._memory_service.retrieve_memories(agent, query, context)

                if memories:
                    memory_content = _render_memory_context(memories)
                    if memory_content and context.metadata is not None:
                        context.metadata.setdefault(CACHE_BREAKPOINT_KEY, len(messages))

                    logger.info("💾 Injecting %s memories into LLM context", len(memories))
                    logger.debug("💾 Memory context content:\n%s", log_content(memory_content))

                    if memory_content:
                        messages.append({"role": "system", "content": memory_content})
            except Exception as e:
                from continuum.agent.exceptions import MemoryReviewRequiredError

                if isinstance(e, MemoryReviewRequiredError):
                    # The second best-effort handler this has to survive. Both
                    # layers treat retrieval as optional and degrade to "no
                    # memories"; a review demand that degrades is the human step
                    # silently skipped, which is what the mode exists to force.
                    raise
                logger.warning("❌ Failed to retrieve memories: %s", e, exc_info=True)

        # Inject pipeline context from sequential/supervised/planner workflows
        # so sub-agents can see prior steps' outputs without loading Redis.
        pipeline_ctx = context.metadata.get("pipeline_context") if context.metadata else None
        if pipeline_ctx:
            # setdefault, not assignment: the marker goes before the FIRST
            # volatile block, and memory (above) may already have claimed it.
            context.metadata.setdefault(CACHE_BREAKPOINT_KEY, len(messages))
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
                            "🔄 SESSION HISTORY: Retrieved %s short-term messages using session_id=%s",
                            len(history),
                            log_id(context.session_id),
                        )
                    messages.extend(history)
            except Exception as e:
                from continuum.session.exceptions import SessionNotFoundError

                if isinstance(e, SessionNotFoundError):
                    logger.warning(
                        "No history loaded: session %r does not exist (it was never created via get_or_create_session). Run continues without prior context.",
                        log_id(context.session_id),
                    )
                else:
                    logger.warning("Failed to load session history: %s", e)

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
                        "Potential prompt injection detected in input to agent '%s': %s",
                        agent.name,
                        detected,
                    )

        # Run product input scanners (e.g. an LLM Guard PromptInjection/Gibberish scanner).
        # Any scanner that returns is_safe=False raises InputBlockedError — the calling router
        # catches this and returns a blocked response without invoking the LLM.
        #
        # A scanner that RAISES blocks too (security finding F11). These are the only
        # control on this path that can refuse, so swallowing their exceptions made the
        # control's failure mode "no control" — and a scanner is usually a model or a
        # remote call, which makes crashing it cheaper than evading it.
        if isinstance(input, str) and agent.config and agent.config.input_scanners:
            from continuum.agent.utils.validation_utils import scanner_failure_reason
            from continuum.exceptions import InputBlockedError

            for scanner in agent.config.input_scanners:
                try:
                    input, is_safe, reason = scanner(input)
                    if not is_safe:
                        # Two fixes in one line. The field said scanner= and was
                        # fed reason -- mislabelled since it was written. And
                        # reason comes from a callable the integrator supplies:
                        # AgentConfig's contract says (text, is_safe, reason) and
                        # nothing about what reason may hold, so a scanner that
                        # quotes the offending input satisfies it. Undecidable
                        # from here, at WARNING on the security path, so declare it.
                        logger.warning(
                            "Input scanner blocked request — agent=%s scanner=%s reason=%s",
                            agent.name,
                            getattr(scanner, "__name__", type(scanner).__name__),
                            log_content(reason),
                        )
                        raise InputBlockedError(
                            f"Input blocked by scanner: {reason}",
                            scanner=reason or "",
                        )
                except InputBlockedError:
                    raise
                except Exception as e:
                    # Includes a scanner returning the wrong shape — the tuple unpack
                    # above raises here too, and a scanner that cannot answer has not
                    # approved anything.
                    failure = scanner_failure_reason(scanner, e)
                    logger.error("Input scanner failed — agent=%s: %s", agent.name, failure)
                    raise InputBlockedError(
                        failure,
                        scanner=getattr(scanner, "__name__", ""),
                    ) from e

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
                        "Agent %s: Context compressed proactively - %s → %s tokens (%s ratio, strategy: %s)",
                        agent.name,
                        compression_result.original_token_count,
                        compression_result.compressed_token_count,
                        format(compression_result.compression_ratio, ".1%"),
                        compression_result.strategy_used,
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
                "Context management failed for agent %s, continuing without compression: %s",
                agent.name,
                e,
            )

        # Run tool-attention routing: filters tools and produces Phase 1 summary.
        from continuum.tools.tool_attention.router import apply_tool_attention

        filtered_tools = (
            await apply_tool_attention(agent, messages, context) or agent.get_tools_for_llm()
        )
        if context.metadata is not None:
            context.metadata["_filtered_tools"] = filtered_tools

        # Build display messages: insert Phase 1 inline so it appears in FINAL PROMPT log.
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

        # The assembled prompt carries the system instructions, retrieved memories,
        # session history, RAG context and the user's input. log_content() withholds
        # it unless LOG_PROMPT_CONTENT is set; the line itself -- which agent, that a
        # prompt was built, how big -- survives either way. This replaces the old
        # LOG_FULL_PROMPT slicing, which capped the dump at 2000 characters per
        # message but logged it by default.
        formatted = "\n".join(
            f"[{m.get('role', '?')}] {str(m.get('content', ''))}" for m in display_messages
        )
        logger.info(
            "===== FINAL PROMPT [%s] =====\n%s\n========================",
            agent.name,
            log_content(formatted),
        )

        if filtered_tools:
            _tools_formatted = "\n".join(
                f"  - {t.get('function', {}).get('name', '?')}: "
                f"{str(t.get('function', {}).get('parameters', ''))}"
                if isinstance(t, dict)
                else f"  - {t.function.name}: {str(t.function.parameters)}"
                for t in filtered_tools
            )
            # Bare, not log_content(): this is each tool's name and parameter
            # schema, defined by the developer or the MCP server -- the system's
            # own fact, not the user's words. The prompt logged just above
            # carries the user's input; this does not.
            logger.info(
                "===== TOOLS [%s] =====\n%s\n========================",
                agent.name,
                _tools_formatted,
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
