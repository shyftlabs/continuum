"""
Runner-level user_id and conversation_id validation tests.

These tests go through AgentRunner._prepare_run() — the SDK boundary
where validate_user_id() and validate_conversation_id() are applied.

Contract (reject-with-error, not coerce):
  - None / whitespace-only / invisible-only  → None (anonymous), run proceeds.
  - Invisible unicode inside an otherwise-valid id → stripped, run proceeds.
  - A clean id (letters, digits, - _ . @)    → passes through unchanged.
  - Anything else (colons, spaces, quotes, over-length, ...) → the run is
    REJECTED: _prepare_run returns success=False with an ERROR AgentResponse.
    We reject rather than coerce because coercion can collapse two distinct
    identities into one scope key and leak data across tenants.

All tests use a real AgentRunner with a mocked LLM client so no
external services (Redis, Qdrant, LLM API) are needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import RunnerConfig
from continuum.agent.runner import AgentRunner
from continuum.agent.types import ResponseStatus
from continuum.agent.utils.context_utils import create_run_context
from continuum.llm.types import LLMResponse

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_llm_client(content: str = "ok") -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value=LLMResponse(model="gpt-4o-mini", content=content, role="assistant"))
    client.chat_stream = AsyncMock()
    return client


def _make_agent(name: str = "test-agent") -> BaseAgent:
    return BaseAgent(name=name, instructions="You are a test agent.", model="gpt-4o-mini")


def _make_runner(llm_client=None) -> AgentRunner:
    runner = AgentRunner(
        llm_client=llm_client or _make_llm_client(),
        config=RunnerConfig(persist_state=False),
    )
    return runner


# ---------------------------------------------------------------------------
# 1. user_id accepted / normalized at the runner boundary
# ---------------------------------------------------------------------------

class TestRunnerUserIdAccepted:
    """Inputs that are valid (possibly after stripping noise) proceed."""

    @pytest.mark.asyncio
    async def test_whitespace_only_user_id_becomes_none(self):
        """'   ' is normalized to None — runner treats it as anonymous."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="   ")
        assert result.success is True
        assert result.context.user_id is None

    @pytest.mark.asyncio
    async def test_tab_newline_user_id_becomes_none(self):
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="\t\n")
        assert result.success is True
        assert result.context.user_id is None

    @pytest.mark.asyncio
    async def test_none_user_id_stays_none(self):
        """Explicit None is intentional anonymous — must pass through unchanged."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id=None)
        assert result.success is True
        assert result.context.user_id is None

    @pytest.mark.asyncio
    async def test_invisible_unicode_stripped(self):
        """Zero-width space inside user_id is removed, leaving a valid id."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="ali​ce")
        assert result.success is True
        assert result.context.user_id == "alice"

    @pytest.mark.asyncio
    async def test_invisible_unicode_only_becomes_none(self):
        """A user_id made entirely of invisible chars becomes None."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="​‌")
        assert result.success is True
        assert result.context.user_id is None

    @pytest.mark.asyncio
    async def test_normal_user_id_passes_unchanged(self):
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="user-123")
        assert result.success is True
        assert result.context.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_email_style_user_id_passes(self):
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="user@example.com")
        assert result.success is True
        assert result.context.user_id == "user@example.com"


# ---------------------------------------------------------------------------
# 2. user_id rejected at the runner boundary
# ---------------------------------------------------------------------------

class TestRunnerUserIdRejected:
    """
    Malformed ids must NOT be coerced — the run is rejected with an
    ERROR AgentResponse so the caller fixes the id rather than silently
    sharing a scope with another tenant.
    """

    def _assert_rejected(self, result, field: str = "user_id"):
        assert result.success is False
        assert result.error_response is not None
        assert result.error_response.status == ResponseStatus.ERROR
        assert field in (result.error_response.error or "")

    @pytest.mark.asyncio
    async def test_colon_in_user_id_is_rejected(self):
        """'alice:bob' — colon is the Redis key delimiter → reject (no coercion)."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="alice:bob")
        self._assert_rejected(result)

    @pytest.mark.asyncio
    async def test_session_key_hijack_attempt_rejected(self):
        """'u:victim' — namespace-prefix collision attempt → reject."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="u:victim")
        self._assert_rejected(result)

    @pytest.mark.asyncio
    async def test_conv_key_hijack_attempt_rejected(self):
        """'c:conv1:u:victim' → reject."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="c:conv1:u:victim")
        self._assert_rejected(result)

    @pytest.mark.asyncio
    async def test_over_length_user_id_rejected(self):
        """200 chars exceeds the 128 limit → reject (truncating would collide)."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", user_id="a" * 200)
        self._assert_rejected(result)

    @pytest.mark.asyncio
    async def test_sql_injection_chars_rejected(self):
        """Spaces/quotes/semicolons are outside the allowlist → reject."""
        result = await _make_runner()._prepare_run(
            _make_agent(), "hello", user_id="'; DROP TABLE sessions;--"
        )
        self._assert_rejected(result)


# ---------------------------------------------------------------------------
# 3. conversation_id validation at the runner boundary
# ---------------------------------------------------------------------------

class TestRunnerConversationIdValidation:
    """
    conversation_id is also part of the Redis session key
    ("c:{conversation_id}:u:{user_id}") so the same rules apply.
    """

    @pytest.mark.asyncio
    async def test_whitespace_conversation_id_becomes_none(self):
        result = await _make_runner()._prepare_run(_make_agent(), "hello", conversation_id="   ")
        assert result.success is True
        assert result.context.conversation_id is None

    @pytest.mark.asyncio
    async def test_colon_in_conversation_id_is_rejected(self):
        """'chat:1' — colon is the Redis key delimiter → reject."""
        result = await _make_runner()._prepare_run(_make_agent(), "hello", conversation_id="chat:1")
        assert result.success is False
        assert "conversation_id" in (result.error_response.error or "")

    @pytest.mark.asyncio
    async def test_normal_conversation_id_passes_unchanged(self):
        result = await _make_runner()._prepare_run(_make_agent(), "hello", conversation_id="chat-abc-123")
        assert result.success is True
        assert result.context.conversation_id == "chat-abc-123"

    @pytest.mark.asyncio
    async def test_none_conversation_id_stays_none(self):
        result = await _make_runner()._prepare_run(_make_agent(), "hello", conversation_id=None)
        assert result.success is True
        assert result.context.conversation_id is None


# ---------------------------------------------------------------------------
# 4. Both together
# ---------------------------------------------------------------------------

class TestRunnerBothIdsTogether:

    @pytest.mark.asyncio
    async def test_both_clean_pass_through(self):
        result = await _make_runner()._prepare_run(
            _make_agent(), "hello", user_id="alice", conversation_id="chat-1"
        )
        assert result.context.user_id == "alice"
        assert result.context.conversation_id == "chat-1"

    @pytest.mark.asyncio
    async def test_both_with_colons_rejected(self):
        """Either malformed id rejects the whole run."""
        result = await _make_runner()._prepare_run(
            _make_agent(), "hello", user_id="alice:evil", conversation_id="chat:evil"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_whitespace_user_id_with_clean_conversation_id(self):
        result = await _make_runner()._prepare_run(
            _make_agent(), "hello", user_id="   ", conversation_id="chat-1"
        )
        assert result.success is True
        assert result.context.user_id is None
        assert result.context.conversation_id == "chat-1"


# ---------------------------------------------------------------------------
# 5. The bypass is closed — a caller-supplied RunContext is validated too
# ---------------------------------------------------------------------------

class TestRunnerContextBypassClosed:
    """
    Previously validation only ran when context was None, so a hand-built
    RunContext could smuggle raw ids straight into memory scoping. The runner
    now validates the ids carried on a caller-supplied context as well.
    """

    @pytest.mark.asyncio
    async def test_dirty_user_id_on_supplied_context_is_rejected(self):
        # create_run_context does no validation — it stores the raw value.
        ctx = create_run_context(user_id="u:victim", conversation_id="chat-1")
        result = await _make_runner()._prepare_run(_make_agent(), "hello", context=ctx)
        assert result.success is False
        assert "user_id" in (result.error_response.error or "")

    @pytest.mark.asyncio
    async def test_dirty_conversation_id_on_supplied_context_is_rejected(self):
        ctx = create_run_context(user_id="alice", conversation_id="chat:evil")
        result = await _make_runner()._prepare_run(_make_agent(), "hello", context=ctx)
        assert result.success is False
        assert "conversation_id" in (result.error_response.error or "")

    @pytest.mark.asyncio
    async def test_clean_supplied_context_passes_and_is_normalized(self):
        ctx = create_run_context(user_id="ali​ce", conversation_id="chat-1")
        result = await _make_runner()._prepare_run(_make_agent(), "hello", context=ctx)
        assert result.success is True
        assert result.context.user_id == "alice"  # invisible char stripped in place
        assert result.context.conversation_id == "chat-1"
