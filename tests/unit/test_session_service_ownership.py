"""An ownership refusal must reach the caller, not be logged and swallowed.

``SessionService`` wraps every session-store call in a broad ``except Exception``
that degrades to a warning. That is the right posture for infrastructure: a
Redis blip should cost this run its history, not the whole request.

An authorization refusal is a different kind of event and must not share that
treatment. Swallowed, it produces a request that succeeds while quietly loading
no history and persisting nothing — silent data loss for a legitimate caller who
forgot to bind a principal, and for an attacker using someone else's session id,
a 200 that looks like it worked. Neither should be a warning in a log nobody is
reading.

The module already carves out one exception this way: ``SessionNotFoundError``
gets its own branch and message at ``save_messages``. This adds the same
treatment for ``SessionOwnershipError``, except that it propagates rather than
merely logging differently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.services.session_service import SessionService
from continuum.session.exceptions import (
    SessionConnectionError,
    SessionNotFoundError,
    SessionOwnershipError,
)
from continuum.session.types import SessionMetadata

OWNERSHIP = SessionOwnershipError("refused", session_id="sess-1")


def _service(failure: Exception | None = None) -> SessionService:
    """A SessionService whose store raises ``failure`` on every operation."""
    client = MagicMock()
    client.is_enabled = True
    for method in (
        "get_session_metadata",
        "update_session_metadata",
        "get_conversation_history",
        "add_message",
    ):
        setattr(client, method, AsyncMock(side_effect=failure) if failure else AsyncMock())
    return SessionService(session_client=client)


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.name = "a"
    agent.memory_config = MagicMock(
        store_memories=False, extraction_prompt=None, pre_store_filter=None, on_stored=None
    )
    return agent


def _save_args() -> dict:
    """Arguments for one completed turn, in save_messages' real shape."""
    return {
        "agent": _agent(),
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "user_message_index": 0,
        "session_id": "sess-1",
    }


@pytest.mark.asyncio
class TestOwnershipErrorsPropagate:
    async def test_load_tool_context_state_reraises(self):
        with pytest.raises(SessionOwnershipError):
            await _service(OWNERSHIP).load_tool_context_state("sess-1")

    async def test_save_tool_context_state_reraises(self):
        from continuum.tools.types import ToolContextState

        # Must carry something: an empty state returns before the store is
        # touched at all, so it could never exercise the refusal.
        state = ToolContextState()
        state.set("shop", "session_id", "cart-1")

        with pytest.raises(SessionOwnershipError):
            await _service(OWNERSHIP).save_tool_context_state("sess-1", state)

    async def test_get_conversation_history_reraises(self):
        with pytest.raises(SessionOwnershipError):
            await _service(OWNERSHIP).get_conversation_history("sess-1")

    async def test_save_messages_reraises(self):
        """The one that matters most: swallowed here, a refused write looks
        exactly like a successful one to the caller."""
        with pytest.raises(SessionOwnershipError):
            await _service(OWNERSHIP).save_messages(**_save_args())


@pytest.mark.asyncio
class TestInfrastructureFailuresStillDegrade:
    """The broad handler stays for everything else — this must not become a
    change of posture towards ordinary store failures."""

    async def test_a_connection_error_still_degrades_on_load(self):
        state = await _service(SessionConnectionError("redis down")).load_tool_context_state(
            "sess-1"
        )
        assert state.get_all_namespaces() == []

    async def test_a_connection_error_still_degrades_on_history(self):
        assert (
            await _service(SessionConnectionError("redis down")).get_conversation_history("sess-1")
            == []
        )

    async def test_a_connection_error_still_degrades_on_save(self):
        await _service(SessionConnectionError("redis down")).save_messages(
            **_save_args()
        )  # must not raise

    async def test_a_missing_session_still_degrades(self):
        """SessionNotFoundError keeps its existing guidance branch."""
        await _service(SessionNotFoundError("gone", session_id="sess-1")).save_messages(
            **_save_args()
        )  # must not raise

    async def test_a_missing_session_still_degrades_on_history(self):
        assert (
            await _service(
                SessionNotFoundError("gone", session_id="sess-1")
            ).get_conversation_history("sess-1")
            == []
        )


@pytest.mark.asyncio
class TestHappyPathUnaffected:
    async def test_history_is_returned_when_the_store_allows_it(self):
        from continuum.llm.types import ChatMessage

        svc = _service()
        svc._session_client.get_conversation_history = AsyncMock(
            return_value=[ChatMessage(role="user", content="hi")]
        )
        history = await svc.get_conversation_history("sess-1")
        assert [m["content"] for m in history] == ["hi"]

    async def test_tool_context_is_loaded_when_the_store_allows_it(self):
        svc = _service()
        now = datetime.now(UTC)
        svc._session_client.get_session_metadata = AsyncMock(
            return_value=SessionMetadata(
                session_id="sess-1",
                created_at=now,
                last_accessed_at=now,
                custom={
                    "tool_context": {
                        "variables": {"shop": {"session_id": "cart-1"}},
                        "metadata": {},
                    }
                },
            )
        )
        state = await svc.load_tool_context_state("sess-1")
        assert state.get("shop", "session_id") == "cart-1"
