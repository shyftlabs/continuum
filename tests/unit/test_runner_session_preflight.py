"""
Runner-level session preflight tests.

These exercise AgentRunner._prepare_run()'s session guardrail — the check that
guides callers who pass a ``session_id`` without first creating the session.

Contract:
  - Stateless run (session_id=None), with or without user_id/conversation_id →
    no session-store lookup, no warning, no raise. Fully supported mode.
  - session_id passed but the session does NOT exist (caller forgot to create
    it) → WARN by default; RAISE SessionNotCreatedError when strict mode is on
    (require_session=True, or SessionConfig.strict_sessions=True). Per-call
    require_session overrides the config.
  - session exists but its user_id differs from the run's user_id → WARN
    (memory write/search scope mismatch).
  - Persistence degraded (Redis down in 'degrade' mode) → the "missing" signal
    is inconclusive, so never warn/raise about a missing session.

All tests use a real AgentRunner with a mocked LLM + a fake session client, so
no external services (Redis, Qdrant, LLM API) are needed. The runner's logger
sets propagate=False, so we patch it with a MagicMock to inspect warnings
rather than relying on caplog.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import RunnerConfig
from continuum.agent.runner import AgentRunner
from continuum.session.exceptions import SessionNotCreatedError
from continuum.session.types import SessionMetadata


def _make_llm_client(content: str = "ok") -> MagicMock:
    from continuum.llm.types import LLMResponse

    client = MagicMock()
    client.chat = AsyncMock(
        return_value=LLMResponse(model="gpt-4o-mini", content=content, role="assistant")
    )
    client.chat_stream = AsyncMock()
    return client


def _make_agent(name: str = "test-agent") -> BaseAgent:
    return BaseAgent(name=name, instructions="You are a test agent.", model="gpt-4o-mini")


def _make_metadata(user_id: str | None) -> SessionMetadata:
    now = datetime.now(UTC)
    return SessionMetadata(session_id="abc", user_id=user_id, created_at=now, last_accessed_at=now)


def _make_session_client(
    *,
    metadata: SessionMetadata | None,
    strict_sessions: bool = False,
    persistence_degraded: bool = False,
) -> MagicMock:
    """Fake SessionClient exposing only what the preflight/load paths touch."""
    client = MagicMock()
    client.is_enabled = True
    client.persistence_degraded = persistence_degraded
    client.config = SimpleNamespace(strict_sessions=strict_sessions)
    client.get_session_metadata = AsyncMock(return_value=metadata)
    client.get_conversation_history = AsyncMock(return_value=[])
    return client


def _make_runner(session_client) -> AgentRunner:
    # memory_client=disabled MagicMock so memory retrieval is a no-op (is_enabled
    # False short-circuits it) and doesn't add noise to the preflight assertions.
    mem = MagicMock()
    mem.is_enabled = False
    return AgentRunner(
        llm_client=_make_llm_client(),
        memory_client=mem,
        session_client=session_client,
        config=RunnerConfig(persist_state=False),
    )


def _patched_logger():
    """Patch the runner module logger; returns the patch context manager."""
    return patch("continuum.agent.runner.logger")


def _warning_text(mock_logger) -> str:
    return "\n".join(str(c.args[0]) for c in mock_logger.warning.call_args_list)


# ---------------------------------------------------------------------------
# 1. Stateless runs never trigger the guardrail
# ---------------------------------------------------------------------------


class TestStatelessRuns:
    @pytest.mark.asyncio
    async def test_no_session_id_skips_lookup(self):
        sc = _make_session_client(metadata=None, strict_sessions=True)
        result = await _make_runner(sc)._prepare_run(_make_agent(), "hello")
        assert result.success is True
        sc.get_session_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_id_only_is_stateless_no_raise(self):
        """user_id without session_id is a valid stateless pattern — even with
        strict mode on it must not raise or warn."""
        sc = _make_session_client(metadata=None, strict_sessions=True)
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(_make_agent(), "hello", user_id="user-123")
        assert result.success is True
        sc.get_session_metadata.assert_not_called()
        assert "does not create sessions" not in _warning_text(log)


# ---------------------------------------------------------------------------
# 2. session_id passed but never created
# ---------------------------------------------------------------------------


class TestSessionNotCreated:
    @pytest.mark.asyncio
    async def test_missing_session_warns_by_default(self):
        sc = _make_session_client(metadata=None, strict_sessions=False)
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(_make_agent(), "hello", session_id="abc")
        assert result.success is True  # non-breaking: run proceeds
        assert "get_or_create_session" in _warning_text(log)

    @pytest.mark.asyncio
    async def test_missing_session_raises_when_strict_config(self):
        sc = _make_session_client(metadata=None, strict_sessions=True)
        with pytest.raises(SessionNotCreatedError):
            await _make_runner(sc)._prepare_run(_make_agent(), "hello", session_id="abc")

    @pytest.mark.asyncio
    async def test_missing_session_raises_when_require_session_true(self):
        sc = _make_session_client(metadata=None, strict_sessions=False)
        with pytest.raises(SessionNotCreatedError):
            await _make_runner(sc)._prepare_run(
                _make_agent(), "hello", session_id="abc", require_session=True
            )

    @pytest.mark.asyncio
    async def test_require_session_false_overrides_strict_config(self):
        sc = _make_session_client(metadata=None, strict_sessions=True)
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(
                _make_agent(), "hello", session_id="abc", require_session=False
            )
        assert result.success is True
        assert "get_or_create_session" in _warning_text(log)


# ---------------------------------------------------------------------------
# 3. Existing session — user_id write/search alignment
# ---------------------------------------------------------------------------


class TestUserIdMismatch:
    @pytest.mark.asyncio
    async def test_mismatch_warns(self):
        sc = _make_session_client(metadata=_make_metadata("alice"))
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(
                _make_agent(), "hello", session_id="abc", user_id="bob"
            )
        assert result.success is True
        assert "user_id mismatch" in _warning_text(log)

    @pytest.mark.asyncio
    async def test_matching_user_id_no_warn(self):
        sc = _make_session_client(metadata=_make_metadata("alice"))
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(
                _make_agent(), "hello", session_id="abc", user_id="alice"
            )
        assert result.success is True
        assert "user_id mismatch" not in _warning_text(log)

    @pytest.mark.asyncio
    async def test_existing_session_no_created_warning(self):
        sc = _make_session_client(metadata=_make_metadata("alice"), strict_sessions=True)
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(
                _make_agent(), "hello", session_id="abc", user_id="alice"
            )
        assert result.success is True
        assert "does not create sessions" not in _warning_text(log)


# ---------------------------------------------------------------------------
# 4. Degraded persistence must not raise a false "missing session" alarm
# ---------------------------------------------------------------------------


class TestDegradedPersistence:
    @pytest.mark.asyncio
    async def test_degraded_missing_does_not_raise_even_strict(self):
        """Redis down in degrade mode → get_session_metadata returns None even
        for a real session. Must not warn/raise about a missing session."""
        sc = _make_session_client(metadata=None, strict_sessions=True, persistence_degraded=True)
        with _patched_logger() as log:
            result = await _make_runner(sc)._prepare_run(
                _make_agent(), "hello", session_id="abc", require_session=True
            )
        assert result.success is True  # no SessionNotCreatedError raised
        assert "does not create sessions" not in _warning_text(log)
