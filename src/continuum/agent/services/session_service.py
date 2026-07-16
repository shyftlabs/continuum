"""
Session Service - Handles session integration for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from continuum.agent.interfaces.service_interface import ISessionService
from continuum.logging import get_logger
from continuum.observability.decorators import observe

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.tools.types import ToolContextState

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

    @staticmethod
    def _session_redaction_placeholder(
        agent: BaseAgent, data_labels: set[str] | None
    ) -> str | None:
        """Return a placeholder string when the agent's policy denies the run's
        data labels for the ``session`` (short-term memory) resource; else None.

        Uses the agent's ``policy_store`` and the run's final ``data_labels`` —
        explicit, so it does not depend on ambient-context lifetime during
        finalization. Open-default: no labels, no policy, or no matching deny → no
        redaction.
        """
        if not data_labels:
            return None
        policy_store = getattr(agent, "policy_store", None)
        if policy_store is None:
            return None
        decision = policy_store.check([agent.name, *sorted(data_labels)], "session")
        if decision.allowed:
            return None
        # This placeholder is replayed into later prompts, so the *model* reads it
        # as its own prior turn. Keep it plain: no internal jargon ("session" /
        # "short-term memory") and no policy name (which the model could parrot
        # back to the user). The technical detail goes to the debug log instead.
        logger.debug(
            "short-term gate: redacted assistant answer (policy=%s, labels=%s)",
            getattr(decision, "policy_name", None),
            sorted(data_labels),
        )
        return (
            "[Response omitted: it contained sensitive information and "
            "was not retained in this conversation.]"
        )

    @observe(name="save_messages", capture_output=False)
    async def save_messages(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        user_message_index: int,
        session_id: str,
        trace_id: str | None = None,
        tool_execution_summary: dict[str, Any] | None = None,
        run_id: str | None = None,
        disable_memory: bool = False,
        data_labels: set[str] | None = None,
    ) -> None:
        """
        Save new messages to session after run completion.

        Filters out tool-related messages to keep session history clean:
        - Saves: user messages, final assistant messages (without tool_calls)
        - Skips: system messages, assistant messages with tool_calls, tool result messages

        Short-term-memory data-label gate: ``data_labels`` are the run's taint at
        completion. If the agent's policy denies them for the ``session`` resource,
        the assistant answer is NOT persisted verbatim — a fixed placeholder is
        stored instead (the run may carry sensitive data and we cannot verify which
        parts, so we substitute the whole value, mirroring telemetry redaction).
        The user-facing response is untouched; only the persisted copy changes.
        """
        if not self._session_client or not self._session_client.is_enabled:
            return

        try:
            from continuum.llm.types import ChatMessage

            # Resolve the short-term (session) gate once for this turn.
            session_placeholder = self._session_redaction_placeholder(agent, data_labels)

            new_messages = messages[user_message_index:]
            saved_count = 0
            skipped_count = 0

            for msg_dict in new_messages:
                if not isinstance(msg_dict, dict):
                    continue

                role = msg_dict.get("role")
                content = msg_dict.get("content")

                if role == "system":
                    skipped_count += 1
                    continue

                if role == "tool":
                    skipped_count += 1
                    logger.debug("Skipping tool message (tool_call_id compatibility)")
                    continue

                if role == "assistant" and msg_dict.get("tool_calls"):
                    skipped_count += 1
                    logger.debug("Skipping intermediate assistant message with tool_calls")
                    continue

                if not content and role != "assistant":
                    skipped_count += 1
                    continue

                # Short-term gate: a denied (tainted) run's assistant answer is
                # persisted as a placeholder, not verbatim. The long-term write is
                # also suppressed for this message (a run too sensitive for the
                # session store should not be auto-extracted to long-term memory;
                # long-term is independently gated by `memory:*` as well).
                persisted_content = content
                gate_redacted = False
                if session_placeholder is not None and role == "assistant" and content:
                    persisted_content = session_placeholder
                    gate_redacted = True

                msg = ChatMessage(role=role, content=persisted_content)

                msg_metadata: dict[str, Any] = {}

                if role == "assistant" and content and tool_execution_summary:
                    msg_metadata["tool_execution_summary"] = tool_execution_summary

                filter_id = session_id or run_id
                if filter_id:
                    msg_metadata["run_id"] = filter_id
                    if session_id:
                        msg_metadata["session_id"] = session_id

                should_store = bool(
                    content
                    and not disable_memory
                    and not gate_redacted
                    and hasattr(agent, "memory_config")
                    and agent.memory_config
                    and agent.memory_config.store_memories
                )

                _mem_cfg = getattr(agent, "memory_config", None)
                _extraction_prompt = getattr(_mem_cfg, "extraction_prompt", None)
                _pre_store_filter = getattr(_mem_cfg, "pre_store_filter", None)
                _on_stored = getattr(_mem_cfg, "on_stored", None)

                await self._session_client.add_message(
                    session_id=session_id,
                    message=msg,
                    agent_id=agent.name,
                    store_in_memory=should_store,
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
            from continuum.session.exceptions import SessionNotFoundError

            if isinstance(e, SessionNotFoundError):
                logger.warning(
                    f"Messages and memory NOT persisted: session {session_id!r} does not "
                    f"exist (it was never created via get_or_create_session). The runner "
                    f"does not create sessions — create it first and pass the returned id "
                    f"into run()."
                )
            else:
                logger.warning(f"Failed to save messages to session: {e}")

    async def load_tool_context_state(
        self,
        session_id: str,
        trace_id: str | None = None,
        session_metadata: Any | None = None,
    ) -> ToolContextState:
        """
        Load tool context state from session metadata.

        Context state stores captured variables (like session_id) from
        previous tool calls for injection into subsequent calls.

        Args:
            session_id: Session ID to load state from
            trace_id: Optional trace ID for observability
            session_metadata: Pre-fetched session metadata to reuse. When the
                caller already loaded it (e.g. the runner's session preflight),
                pass it here to avoid a redundant session-store round-trip. When
                None, the metadata is fetched here.

        Returns:
            ToolContextState loaded from session, or empty state if not found
        """
        from continuum.tools.types import ToolContextState

        if not self._session_client or not self._session_client.is_enabled:
            return ToolContextState()

        try:
            # Reuse pre-fetched metadata when provided; otherwise fetch it.
            metadata = session_metadata
            if metadata is None:
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
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get conversation history from session.

        Args:
            session_id: Session ID
            limit: Number of complete turns (request+response pairs) to retrieve

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
