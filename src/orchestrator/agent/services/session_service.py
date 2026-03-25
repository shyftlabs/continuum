"""
Session Service - Handles session integration for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.agent.interfaces.service_interface import ISessionService
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.tools.types import ToolContextState

logger = get_logger(__name__)


class SessionService(ISessionService):
    """
    Service for session integration.

    Handles saving messages, loading conversation history,
    and managing tool context state.
    """

    def __init__(
        self,
        session_client: Any | None = None,
    ):
        """
        Initialize session service.

        Args:
            session_client: Session client instance
        """
        self._session_client = session_client

    @property
    def session_client(self) -> Any | None:
        """Get session client."""
        return self._session_client

    @observe(name="save_messages", capture_output=False)
    async def save_messages(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        initial_count: int,
        session_id: str,
        trace_id: str | None = None,
        tool_execution_summary: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """
        Save new messages to session after run completion.

        IMPORTANT: This method filters out tool-related messages to:
        1. Eliminate cross-model tool_call_id compatibility issues (Gemini vs OpenAI)
        2. Reduce token usage in session history
        3. Keep conversation history clean (only user questions and final answers)

        Tool execution details are preserved in:
        - Langfuse spans (full debugging)
        - tool_execution_summary metadata on final assistant message

        Messages that ARE saved:
        - User messages with content
        - Final assistant messages (without tool_calls)

        Messages that are SKIPPED:
        - System messages (not conversation history)
        - Assistant messages with tool_calls (intermediate)
        - Tool result messages (role="tool")
        """
        if not self._session_client or not self._session_client.is_enabled:
            return

        try:
            from orchestrator.llm.types import ChatMessage

            # Get new messages (everything added during execution).
            # Offset by -1 so we include the final user message appended by
            # prepare_messages() — that message IS part of this run's conversation
            # and must be persisted.  max(0, ...) prevents negative index when
            # initial_count is 0 (i.e., no messages were prepared).
            start_index = max(0, initial_count - 1)
            new_messages = messages[start_index:]

            saved_count = 0
            skipped_count = 0

            # Pre-filter: find the LAST final assistant message index to avoid
            # saving duplicate assistant messages when consecutive responses occur
            last_final_assistant_idx = None
            for idx, msg_dict in enumerate(new_messages):
                if not isinstance(msg_dict, dict):
                    continue
                role = msg_dict.get("role")
                if role == "assistant" and not msg_dict.get("tool_calls") and msg_dict.get("content"):
                    last_final_assistant_idx = idx

            for idx, msg_dict in enumerate(new_messages):
                if not isinstance(msg_dict, dict):
                    continue

                role = msg_dict.get("role")
                content = msg_dict.get("content")

                # Skip system messages - they're not conversation history
                if role == "system":
                    skipped_count += 1
                    continue

                # Skip tool result messages - they have long tool_call_ids that cause
                # cross-model compatibility issues (Gemini IDs are 800+ chars, OpenAI limit is 40)
                # Tool execution details are in Langfuse and tool_execution_summary metadata
                if role == "tool":
                    skipped_count += 1
                    logger.debug("Skipping tool message (tool_call_id compatibility)")
                    continue

                # Skip assistant messages with tool_calls - these are intermediate messages
                # The final response (without tool_calls) will be saved
                if role == "assistant" and msg_dict.get("tool_calls"):
                    skipped_count += 1
                    logger.debug("Skipping intermediate assistant message with tool_calls")
                    continue

                # Skip non-final assistant messages (only save the last one to
                # prevent duplicates when consecutive assistant responses occur)
                if role == "assistant" and last_final_assistant_idx is not None and idx != last_final_assistant_idx:
                    skipped_count += 1
                    logger.debug("Skipping non-final assistant message (keeping only last)")
                    continue

                # Skip empty messages
                if not content and role != "assistant":
                    skipped_count += 1
                    continue

                # Create ChatMessage for session storage (without tool_calls/tool_call_id)
                # This ensures clean messages that work with any LLM provider
                msg = ChatMessage(
                    role=role,
                    content=content,
                    # Intentionally NOT including:
                    # - tool_calls (would require tool_call_ids which vary by provider)
                    # - tool_call_id (provider-specific, causes compatibility issues)
                )

                # Prepare metadata for the message
                msg_metadata: dict[str, Any] | None = None

                # For the final assistant message, attach tool execution summary
                if role == "assistant" and content and tool_execution_summary:
                    msg_metadata = {"tool_execution_summary": tool_execution_summary}
                    logger.debug(
                        f"Attaching tool summary to assistant message: "
                        f"{tool_execution_summary.get('tool_count', 0)} tools used"
                    )

                # Include session_id/run_id in metadata for RUN-scoped memory isolation
                # This allows filtering memories by session_id when agent scope is RUN
                # IMPORTANT: Use session_id (not run_id) because session_id is persistent
                # across multiple requests, while run_id is generated fresh each time
                if not msg_metadata:
                    msg_metadata = {}
                # Use session_id for RUN-scoped filtering (persistent across requests)
                # Fall back to run_id if session_id is not available
                filter_id = session_id or run_id
                if filter_id:
                    # Store in metadata for RUN-scoped memory filtering
                    # When agent scope is RUN, memories will be filtered by this identifier
                    msg_metadata["run_id"] = filter_id
                    if session_id:
                        msg_metadata["session_id"] = session_id

                # Save to session
                # Store in long-term memory for meaningful messages
                should_store_in_memory = bool(content)

                # Check if agent has memory storage enabled
                agent_store_enabled = (
                    hasattr(agent, "memory_config")
                    and agent.memory_config
                    and agent.memory_config.store_memories
                )
                session_client_available = self._session_client and self._session_client.is_enabled

                if should_store_in_memory and not agent_store_enabled:
                    logger.debug(
                        f"💾 Memory storage disabled in agent config: store_memories={agent_store_enabled}"
                    )
                if should_store_in_memory and not session_client_available:
                    logger.debug(
                        f"💾 Session client not available for memory storage: session_client={'available' if self._session_client else 'not available'}"
                    )

                if not should_store_in_memory:
                    logger.debug("Skipping memory storage (empty content)")

                # Extract memory hooks from agent config (product-level customization)
                _mem_cfg = getattr(agent, "memory_config", None)
                _extraction_prompt = getattr(_mem_cfg, "extraction_prompt", None)
                _pre_store_filter = getattr(_mem_cfg, "pre_store_filter", None)
                _on_stored = getattr(_mem_cfg, "on_stored", None)

                await self._session_client.add_message(
                    session_id=session_id,
                    message=msg,
                    store_in_memory=should_store_in_memory,
                    metadata=msg_metadata,
                    extraction_prompt=_extraction_prompt,
                    pre_store_filter=_pre_store_filter,
                    on_stored=_on_stored,
                )
                saved_count += 1

            logger.debug(
                f"Session {session_id}: saved {saved_count} messages, "
                f"skipped {skipped_count} (tool-related/system)"
            )

        except Exception as e:
            logger.warning(f"Failed to save messages to session: {e}")

    async def load_tool_context_state(
        self,
        session_id: str,
        trace_id: str | None = None,
    ) -> ToolContextState:
        """
        Load tool context state from session metadata.

        Context state stores captured variables (like session_id) from
        previous tool calls for injection into subsequent calls.

        Args:
            session_id: Session ID to load state from
            trace_id: Optional trace ID for observability

        Returns:
            ToolContextState loaded from session, or empty state if not found
        """
        from orchestrator.tools.types import ToolContextState

        if not self._session_client or not self._session_client.is_enabled:
            return ToolContextState()

        try:
            # Get session metadata
            metadata = await self._session_client.get_session_metadata(session_id)

            if metadata and metadata.custom.get("tool_context"):
                state = ToolContextState.from_dict(metadata.custom["tool_context"])
                logger.debug(
                    f"Loaded tool context state from session: "
                    f"{len(state.get_all_namespaces())} namespaces"
                )
                return state

            return ToolContextState()

        except Exception as e:
            logger.warning(f"Failed to load tool context state: {e}")
            return ToolContextState()

    async def save_tool_context_state(
        self,
        session_id: str,
        context_state: ToolContextState,
        trace_id: str | None = None,
    ) -> None:
        """
        Save tool context state to session metadata.

        Args:
            session_id: Session ID to save state to
            context_state: Tool context state to save
            trace_id: Optional trace ID for observability
        """
        if not self._session_client or not self._session_client.is_enabled:
            return

        if context_state.is_empty():
            return

        try:
            # Get current session metadata
            metadata = await self._session_client.get_session_metadata(session_id)

            if not metadata:
                logger.warning(f"Session metadata not found for {session_id}")
                return

            # Update custom metadata with tool context
            metadata.custom["tool_context"] = context_state.to_dict()

            # Save back using client method
            await self._session_client.update_session_metadata(session_id, metadata)

            logger.debug(
                f"Saved tool context state to session: "
                f"{len(context_state.get_all_namespaces())} namespaces"
            )

        except Exception as e:
            logger.warning(f"Failed to save tool context state: {e}")

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get conversation history from session.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        if not self._session_client or not self._session_client.is_enabled:
            return []

        try:
            history = await self._session_client.get_conversation_history(session_id, limit=limit)
            return [self._message_to_dict(msg) for msg in history]
        except Exception as e:
            logger.warning(f"Failed to load session history: {e}")
            return []

    def _message_to_dict(self, message: Any) -> dict[str, Any]:
        """Convert a message to dictionary format."""
        if isinstance(message, dict):
            return message
        if hasattr(message, "to_dict"):
            return message.to_dict()
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {"role": "user", "content": str(message)}
