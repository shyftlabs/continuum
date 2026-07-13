"""
Memory Service - Handles memory integration for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any

from continuum.agent.interfaces.service_interface import IMemoryService
from continuum.logging import get_logger
from continuum.observability.decorators import observe

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.types import RunContext

logger = get_logger(__name__)


class MemoryService(IMemoryService):
    """
    Service for memory integration.

    Handles retrieving and storing memories for agents.
    """

    def __init__(
        self,
        memory_client: Any | None = None,
        session_client: Any | None = None,
        memory_client_resolver: Callable[[], Any | None] | None = None,
    ):
        """
        Initialize memory service.

        Args:
            memory_client: Memory client instance
            session_client: Session client for metadata access
            memory_client_resolver: Lazy resolver used when no client is injected
        """
        self._memory_client = memory_client
        self._session_client = session_client
        self._memory_client_resolver = memory_client_resolver
        self._memory_client_resolved = memory_client is not None or memory_client_resolver is None
        self._memory_client_lock = Lock()

    @property
    def memory_client(self) -> Any | None:
        """Get memory client."""
        return self._resolve_memory_client()

    def _resolve_memory_client(self) -> Any | None:
        """Resolve and cache the optional memory client on first use."""
        if self._memory_client_resolved:
            return self._memory_client

        with self._memory_client_lock:
            if not self._memory_client_resolved:
                resolver = self._memory_client_resolver
                self._memory_client = resolver() if resolver is not None else None
                self._memory_client_resolved = True

        return self._memory_client

    @observe(name="retrieve_memories", capture_output=True)
    async def retrieve_memories(
        self,
        agent: BaseAgent,
        query: str,
        context: RunContext,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories for the agent.

        Args:
            agent: Agent requesting memories
            query: Search query
            context: Run context

        Returns:
            List of memory dictionaries
        """
        if not agent.memory_config.search_memories:
            logger.debug("Skipping memory search because it is disabled for the agent")
            return []

        try:
            memory_client = self._resolve_memory_client()
            if memory_client is None:
                logger.debug("Skipping memory search because no memory client is available")
                return []

            # Determine search scope based on agent config
            search_scope = agent.memory_config.search_scope.value

            # Get memory client's isolation level to determine required identifiers
            memory_isolation = memory_client.config.memory_isolation

            # Mode-aware identifier selection
            user_id_for_memory = context.user_id if memory_isolation == "user" else None

            # CRITICAL: For agent isolation mode, get agent_id from session metadata if available.
            agent_id_for_memory = None
            if memory_isolation == "agent":
                # Try to get agent_id from session metadata first (most accurate)
                if context.session_id and self._session_client and self._session_client.is_enabled:
                    try:
                        session_metadata = await self._session_client.get_session_metadata(
                            context.session_id
                        )
                        if session_metadata and session_metadata.agent_id:
                            agent_id_for_memory = session_metadata.agent_id
                            logger.debug(
                                "Using agent identity from session metadata "
                                "(session_id_present=%s)",
                                bool(context.session_id),
                            )
                        else:
                            logger.warning(
                                "Session metadata has no agent identity; falling back to the configured "
                                "agent name. This may cause memory isolation issues."
                            )
                            agent_id_for_memory = agent.name
                    except Exception as e:
                        logger.debug(
                            "Could not get session metadata for agent identity (error_type=%s)",
                            type(e).__name__,
                        )
                        agent_id_for_memory = agent.name
                else:
                    # No session_id or session client not available - use agent.name
                    agent_id_for_memory = agent.name

                # Log final agent_id being used
                if agent_id_for_memory != agent.name:
                    logger.debug(
                        "Agent isolation mode is using a session-scoped identity "
                        "(differs_from_configured_agent=true)"
                    )

            conversation_id_for_memory = None
            if memory_isolation == "conversation":
                conversation_id_for_memory = context.conversation_id
                if not conversation_id_for_memory:
                    logger.warning(
                        "memory_isolation='conversation' but context.conversation_id is None — "
                        "memory search will be unscoped. Pass conversation_id when calling runner.run()."
                    )

            # Log memory search parameters at DEBUG level
            logger.debug(
                "Memory search: query_chars=%d scope=%s isolation=%s "
                "user_scoped=%s agent_scoped=%s conversation_scoped=%s",
                len(query),
                search_scope,
                memory_isolation,
                user_id_for_memory is not None,
                agent_id_for_memory is not None,
                conversation_id_for_memory is not None,
            )

            memories = await memory_client.search(
                query=query,
                user_id=user_id_for_memory,
                agent_id=agent_id_for_memory,
                conversation_id=conversation_id_for_memory,
                limit=agent.memory_config.search_limit,
            )

            # Log search results at DEBUG level
            logger.debug(
                f"💾 MEMORY SEARCH RESULT: found {len(memories.results)} memories "
                f"(total_results={memories.total_results if hasattr(memories, 'total_results') else 'N/A'})"
            )

            if not memories.results:
                logger.debug(
                    "Memory search returned no results (query_chars=%d scope=%s isolation=%s)",
                    len(query),
                    search_scope,
                    memory_isolation,
                )

            if memories.results:
                context.retrieved_memories = [m.to_dict() for m in memories.results]

                # Memory-scope provenance: reading data out of a scope declared
                # sensitive taints the run ("read = taint"). Only taints when data
                # actually flowed (results non-empty).
                scope_labels = agent.memory_config.scope_data_labels.get(search_scope)
                if scope_labels:
                    context.taint(*scope_labels)

                # Log memory search summary at DEBUG level
                logger.debug(
                    "Memory search completed: scope=%s isolation=%s found=%d",
                    search_scope,
                    memory_isolation,
                    len(memories.results),
                )

                # Keep diagnostics content-free. Memory text and scope identifiers can contain customer data.
                for idx, m in enumerate(memories.results, 1):
                    score_str = f"{m.score:.3f}" if m.score is not None else "N/A"
                    logger.debug(
                        "Memory result: rank=%d score=%s content_chars=%d",
                        idx,
                        score_str,
                        len(m.memory or ""),
                    )

                return context.retrieved_memories

            return []

        except Exception as e:
            logger.warning(
                "Failed to retrieve memories (error_type=%s)",
                type(e).__name__,
            )
            return []

    @observe(name="store_memories", capture_output=False)
    async def store_memories(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
    ) -> None:
        """
        Store memories from conversation.

        Note: Memory storage is handled by the session service when saving messages.
        This method is kept for interface compatibility but delegates to session.

        Args:
            agent: Agent storing memories
            messages: Conversation messages
            context: Run context
        """
        # Memory storage is handled by SessionService.save_messages()
        # This method exists for interface compatibility
        logger.debug("Memory storage is handled by session service during message save")
