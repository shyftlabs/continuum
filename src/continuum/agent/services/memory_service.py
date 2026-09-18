"""
Memory Service - Handles memory integration for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from continuum.agent.interfaces.service_interface import IMemoryService
from continuum.logging import get_logger
from continuum.observability.decorators import observe

if TYPE_CHECKING:
    from continuum.agent.base import BaseAgent
    from continuum.agent.types import RunContext

logger = get_logger(__name__)


def _row_provenance_labels(rows: list[Any]) -> set[str]:
    """Collect provenance labels stamped on retrieved memory rows.

    Tolerant by design. Row metadata is third-party data round-tripped through a
    vector store, so a malformed stamp is a data problem, not a reason to fail a
    read: one bad row must not take down every retrieval for that user. Anything
    that is not a list/tuple of strings is skipped.

    ``list``/``tuple`` only, deliberately -- a bare ``str`` and a ``dict`` are
    both iterable, so accepting "any iterable" would silently taint a run with
    the characters of a string or the keys of a dict.
    """
    from continuum.memory.types import PROVENANCE_LABELS_KEY

    labels: set[str] = set()
    for row in rows:
        raw = (getattr(row, "metadata", None) or {}).get(PROVENANCE_LABELS_KEY)
        if isinstance(raw, list | tuple):
            labels.update(x for x in raw if isinstance(x, str))
    return labels


class MemoryService(IMemoryService):
    """
    Service for memory integration.

    Handles retrieving and storing memories for agents.
    """

    def __init__(
        self,
        memory_client: Any | None = None,
        session_client: Any | None = None,
    ):
        """
        Initialize memory service.

        Args:
            memory_client: Memory client instance
            session_client: Session client for metadata access
        """
        self._memory_client = memory_client
        self._session_client = session_client

    @property
    def memory_client(self) -> Any | None:
        """Get memory client."""
        return self._memory_client

    def _warn_if_provenance_undeclared(self, agent: BaseAgent, context: RunContext) -> None:
        """Say once that the memory-poisoning defences are configured off.

        Every taint producer is gated on a declaration that defaults to empty --
        ``AgentConfig.tool_data_labels``, ``AgentMemoryConfig.scope_data_labels``,
        and the run-level seed. With none of them set nothing taints, so memory
        rows are stamped with nothing, the read path finds every row clean and
        fences nothing, and the tool gate never matches a label. The machinery is
        all present and does nothing.

        That is the intended default -- the SDK ships no detector and will not
        guess which tools return attacker-influenced data, because guessing wrong
        either gates benign work or gives false assurance. But it fails by doing
        nothing, which is the failure mode nobody notices, so it is reported the
        way a URL-derived MCP server name is: name the fields, state what is lost,
        then stay quiet.

        A tainted run counts as declared: run-level seeding is invisible in the
        agent config, so labels already present are proof the mechanism is live.

        Once per agent. This runs every turn, and a warning repeated each turn is
        one people filter out.
        """
        if getattr(agent, "_warned_provenance_undeclared", False):
            return

        mem_cfg = getattr(agent, "memory_config", None)
        if mem_cfg is None:
            return
        if not (
            getattr(mem_cfg, "search_memories", False) or getattr(mem_cfg, "store_memories", False)
        ):
            return  # memory off: nothing to protect, nothing to say

        if getattr(context, "data_labels", None):
            return  # the run is tainted, so provenance is declared somewhere
        if getattr(mem_cfg, "scope_data_labels", None):
            return
        if getattr(getattr(agent, "config", None), "tool_data_labels", None):
            return

        try:
            agent._warned_provenance_undeclared = True  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass  # a frozen/slotted agent just gets the warning again

        logger.warning(
            f"Agent '{getattr(agent, 'name', '?')}' has long-term memory enabled but declares no "
            "data provenance, so the memory-poisoning defences are inactive: rows are written "
            "without provenance, recalled content is never fenced, and no tool call can be gated "
            "on where its data came from. A fact laundered out of an injected tool result will "
            "come back indistinguishable from one the user stated. Declare which tools return "
            "attacker-influenced data via AgentConfig.tool_data_labels "
            '(e.g. {"fetch_page": {"external"}}), and/or which memory scopes are sensitive via '
            'AgentMemoryConfig.scope_data_labels (e.g. {"user": {"pii"}}), then add a PolicyStore '
            "rule denying those labels the actions they must not reach."
        )

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
        self._warn_if_provenance_undeclared(agent, context)

        if not agent.memory_config.search_memories or not self._memory_client:
            logger.debug(
                f"💾 Skipping memory search: search_memories={agent.memory_config.search_memories}, "
                f"memory_client={'available' if self._memory_client else 'not available'}"
            )
            return []

        try:
            # Determine search scope based on agent config
            search_scope = agent.memory_config.search_scope.value

            # Get memory client's isolation level to determine required identifiers
            memory_isolation = self._memory_client.config.memory_isolation

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
                                f"🔍 Using agent_id from session metadata: {agent_id_for_memory} "
                                f"(session_id={context.session_id}...)"
                            )
                        else:
                            logger.warning(
                                f"⚠️ Session {context.session_id}... exists but has no agent_id in metadata. "
                                f"Falling back to agent.name={agent.name}. This may cause memory isolation issues."
                            )
                            agent_id_for_memory = agent.name
                    except Exception as e:
                        logger.debug(f"Could not get session metadata for agent_id: {e}")
                        agent_id_for_memory = agent.name
                else:
                    # No session_id or session client not available - use agent.name
                    agent_id_for_memory = agent.name

                # Log final agent_id being used
                if agent_id_for_memory != agent.name:
                    logger.debug(
                        f"🔍 Agent isolation mode: Using agent_id={agent_id_for_memory} "
                        f"(agent.name={agent.name}, may differ when switching agents)"
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
                f"🔍 MEMORY SEARCH: query='{query[:100]}...', "
                f"scope={search_scope}, isolation={memory_isolation}, "
                f"user_id={user_id_for_memory if user_id_for_memory else 'none'}, "
                f"agent_id={agent_id_for_memory if agent_id_for_memory else 'none'}, "
                f"conversation_id={conversation_id_for_memory if conversation_id_for_memory else 'none'}"
            )

            memories = await self._memory_client.search(
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
                logger.warning(
                    f"⚠️ NO MEMORIES FOUND for query='{query[:100]}...' "
                    f"(isolation={memory_isolation}, user_id={user_id_for_memory if user_id_for_memory else 'none'}, "
                    f"agent_id={agent_id_for_memory if agent_id_for_memory else 'none'}, "
                    f"conversation_id={conversation_id_for_memory if conversation_id_for_memory else 'none'})"
                )

            if memories.results:
                # What to do with rows carrying provenance (F6). Recall is the
                # one taint source nobody asked for: the row arrives during
                # prompt assembly and was written in an earlier session, so by
                # the time the label is known the text would already be in the
                # prompt. "fence" is the default and the weakest -- it asks the
                # model not to obey. The other two keep it out of the prompt
                # entirely; "block" additionally forces a person to look.
                action = getattr(agent.memory_config, "on_labeled_recall", "fence")
                if action != "fence":
                    labeled = [m for m in memories.results if _row_provenance_labels([m])]
                    if labeled:
                        if action == "block":
                            from continuum.agent.exceptions import (
                                MemoryReviewRequiredError,
                            )

                            raise MemoryReviewRequiredError(
                                memory_ids=[str(getattr(m, "id", "")) for m in labeled],
                                labels=sorted(_row_provenance_labels(labeled)),
                            )
                        # drop: the rows never enter the prompt, so the run did
                        # not touch them and must not be tainted by them either.
                        kept = [m for m in memories.results if m not in labeled]
                        logger.info(
                            "🚫 Dropped %d recalled memory row(s) carrying provenance %s "
                            "(on_labeled_recall='drop')",
                            len(labeled),
                            sorted(_row_provenance_labels(labeled)),
                        )
                        memories.results = kept
                        if not kept:
                            return []

                context.retrieved_memories = [m.to_dict() for m in memories.results]

                # Memory-scope provenance: reading data out of a scope declared
                # sensitive taints the run ("read = taint"). Only taints when data
                # actually flowed (results non-empty).
                scope_labels = agent.memory_config.scope_data_labels.get(search_scope)
                if scope_labels:
                    context.taint(*scope_labels)

                # Row provenance (security finding F6): a row stamped by the run
                # that wrote it re-taints the run that reads it. Additive with the
                # scope labels above -- scope answers "is this store sensitive",
                # the row answers "was this particular fact derived from
                # untrusted input", and only the second can separate a genuine
                # user preference from a planted one sitting in the same scope.
                #
                # This is the half that does not depend on the model cooperating:
                # once the label is on the run, the tool gate denies the action
                # whatever the model was persuaded to believe.
                context.taint(*_row_provenance_labels(memories.results))

                # Log memory search summary at DEBUG level
                logger.debug(
                    f"💾 Memory search: scope={search_scope}, isolation={memory_isolation}, "
                    f"user_id={context.user_id if context.user_id else 'none'}, "
                    f"agent_id={agent_id_for_memory if agent_id_for_memory else 'N/A'}, "
                    f"conversation_id={conversation_id_for_memory if conversation_id_for_memory else 'none'}, "
                    f"found={len(memories.results)} memories"
                )

                # Log each memory with its metadata to verify isolation (INFO level)
                for idx, m in enumerate(memories.results, 1):
                    memory_metadata = getattr(m, "metadata", {}) or {}
                    memory_user_id = m.user_id or memory_metadata.get("_user_id") or "unknown"
                    score_str = f"{m.score:.3f}" if m.score is not None else "N/A"
                    logger.info(
                        f"📝 Memory #{idx}: '{m.memory[:100]}...' "
                        f"(score={score_str}, user_id={memory_user_id if memory_user_id != 'unknown' else 'unknown'})"
                    )

                return context.retrieved_memories

            return []

        except Exception as e:
            from continuum.agent.exceptions import (
                MemoryAccessDeniedError,
                MemoryReviewRequiredError,
            )

            if isinstance(e, MemoryReviewRequiredError):
                # Deliberately not swallowed. Everything else here is
                # best-effort and degrades to "no memories", but a review demand
                # that degrades is a human step silently skipped -- which is the
                # one thing this mode exists to prevent.
                raise
            if isinstance(e, MemoryAccessDeniedError):
                # Expected: a data-label policy blocked the read. That is the
                # gate working, not a fault, and the same call the write path
                # already makes (session/client.py). A traceback here tells an
                # operator something broke and sends them hunting a bug that is
                # not there. The turn continues without memory: for a read gate,
                # disclosing nothing and carrying on is the safe direction.
                logger.info(
                    "🛡️ Long-term memory read blocked by policy '%s' "
                    "(run carried restricted data labels)",
                    e.context.get("policy_name"),
                )
                return []
            logger.warning(f"❌ Failed to retrieve memories: {e}", exc_info=True)
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
        # Reported here too: a store-only agent never reaches the read path, but
        # its writes are the ones being stamped, so it has the same gap.
        self._warn_if_provenance_undeclared(agent, context)

        # Memory storage is handled by SessionService.save_messages()
        # This method exists for interface compatibility
        logger.debug("Memory storage is handled by session service during message save")
