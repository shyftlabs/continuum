"""
Whitespace user_id — end-to-end journey test.

Starting point: the CLI in playground/gateway-local-shop/cli.py

    user_id = input("Enter user ID (or Enter for auto): ").strip() or None

Question: if someone types "   " (spaces only), does any layer in the
framework catch it before it reaches the memory store?

Each test in this file IS one checkpoint on the journey from CLI input
all the way down to MemoryScope.  They run in reading order so you can
follow the data as it moves through the stack.

No external services (Redis, Qdrant, LLM) are needed — all boundary
calls are mocked.
"""

from __future__ import annotations

import pytest

WHITESPACE = "   "   # three spaces — what a user might accidentally type
CLEAN_ID   = "alice" # control: a normal user_id

# =============================================================================
# Checkpoint 1 — CLI layer  (playground/gateway-local-shop/cli.py:52)
# =============================================================================

class TestCheckpoint1_CLI:
    """
    The CLI does:
        user_id = input("Enter user ID (or Enter for auto): ").strip() or None

    .strip() converts "   " → "".
    "" is falsy, so  `"" or None`  gives None.
    → The CLI itself is SAFE.  Whitespace never leaves this layer as whitespace.
    """

    def test_cli_strips_whitespace_to_none(self):
        """Typing spaces and pressing Enter produces None — same as pressing Enter alone."""
        raw_input = WHITESPACE
        user_id = raw_input.strip() or None
        assert user_id is None, (
            "CLI should convert whitespace-only input to None via .strip() or None"
        )

    def test_cli_preserves_real_user_id(self):
        """A proper user_id survives the strip."""
        raw_input = "  alice  "
        user_id = raw_input.strip() or None
        assert user_id == "alice"

    def test_cli_empty_enter_also_gives_none(self):
        """Just pressing Enter (empty string) also gives None."""
        raw_input = ""
        user_id = raw_input.strip() or None
        assert user_id is None


# =============================================================================
# Checkpoint 2 — Session ID computation  (session/providers/redis.py:194)
# =============================================================================

class TestCheckpoint2_SessionIdComputation:
    """
    _compute_session_id builds the Redis key that stores session metadata.
    It now runs every user_id/conversation_id through validate_user_id() /
    validate_conversation_id() before building the key, so this layer IS a
    guard:
      - whitespace/invisible-only → None → falls through to a random UUID
      - a malformed id (colon, space, ...) → InvalidIdentifierError

    Tests call the real RedisSessionProvider._compute_session_id (a pure method
    that needs no Redis connection) via auto_initialize=False.
    """

    def _provider(self):
        from continuum.session.config import SessionConfig
        from continuum.session.providers.redis import RedisSessionProvider
        return RedisSessionProvider(SessionConfig(), auto_initialize=False)

    def test_whitespace_user_id_falls_through_to_uuid(self):
        """
        FIXED — the guard now strips whitespace to None, so the key is a random
        UUID instead of the old "u:   ". Two callers typing "  " and " " no
        longer get distinct (silently-wrong) buckets; both become anonymous.
        """
        key = self._provider()._compute_session_id(None, WHITESPACE, None)
        assert not key.startswith("u:") and not key.startswith("c:")

    def test_colon_user_id_is_rejected(self):
        """A colon in user_id (Redis key delimiter) now raises rather than embeds."""
        from continuum.utils.sanitization import InvalidIdentifierError
        with pytest.raises(InvalidIdentifierError):
            self._provider()._compute_session_id(None, "u:victim", None)

    def test_none_user_id_falls_through_to_uuid(self):
        """None (what the CLI produces for whitespace input) skips the user_id branch."""
        key = self._provider()._compute_session_id(None, None, None)
        # Should be a UUID — not starting with "u:" or "c:"
        assert not key.startswith("u:") and not key.startswith("c:")

    def test_clean_user_id_produces_expected_key(self):
        """Control: a normal user_id produces a clean key."""
        key = self._provider()._compute_session_id(None, CLEAN_ID, None)
        assert key == f"u:{CLEAN_ID}"

    def test_explicit_session_id_is_trusted_as_is(self):
        """An explicit session_id (internal handoff calls) bypasses validation."""
        key = self._provider()._compute_session_id("internal:handoff:id", None, None)
        assert key == "internal:handoff:id"


# =============================================================================
# Checkpoint 3 — RunContext  (agent/utils/context_utils.py + agent/types.py)
# =============================================================================

class TestCheckpoint3_RunContext:
    """
    create_run_context(user_id="   ") → RunContext(user_id="   ")

    RunContext is a plain @dataclass — no Pydantic, no field validators.
    The whitespace string is stored verbatim and then forwarded to every
    downstream service (MemoryService, SessionService).
    → No guard at this layer.
    """

    def test_run_context_accepts_whitespace_user_id(self):
        """
        FINDING — plain @dataclass with no validation.
        Whitespace user_id is stored exactly as supplied.
        """
        from continuum.agent.utils.context_utils import create_run_context

        ctx = create_run_context(user_id=WHITESPACE)
        assert ctx.user_id == WHITESPACE, (
            "RunContext stored whitespace user_id verbatim — no validator present."
        )

    def test_run_context_type_is_dataclass_not_pydantic(self):
        """Confirm RunContext is a dataclass, not a Pydantic BaseModel."""
        import dataclasses

        from continuum.agent.types import RunContext

        assert dataclasses.is_dataclass(RunContext), "RunContext must be a dataclass"
        try:
            from pydantic import BaseModel
            assert not issubclass(RunContext, BaseModel), (
                "RunContext must NOT be a Pydantic model — it has no field-level validation"
            )
        except ImportError:
            pass  # pydantic not installed — dataclass confirmed above

    def test_run_context_none_user_id_is_also_accepted(self):
        """None is valid (anonymous / unauthenticated run)."""
        from continuum.agent.utils.context_utils import create_run_context

        ctx = create_run_context(user_id=None)
        assert ctx.user_id is None

    def test_run_context_clean_user_id_stored_as_is(self):
        """Control: a clean user_id is stored unchanged."""
        from continuum.agent.utils.context_utils import create_run_context

        ctx = create_run_context(user_id=CLEAN_ID)
        assert ctx.user_id == CLEAN_ID


# =============================================================================
# Checkpoint 4 — MemoryScope  (memory/scopes.py)
# =============================================================================

class TestCheckpoint4_MemoryScope:
    """
    MemoryScope.user(user_id) checks:

        if not user_id:
            raise ValueError(...)

    "   " is truthy  → the check passes → scope is built with user_id="   ".

    This is the deepest layer before the vector store write.
    → No guard here either.  The whitespace id enters the memory bucket.
    """

    def test_memory_scope_accepts_whitespace_user_id(self):
        """
        FINDING — 'if not user_id' is a truthy check.
        "   " is truthy → no error → MemoryScope built with whitespace id.
        """
        from continuum.memory.scopes import MemoryScope

        scope = MemoryScope.user(WHITESPACE)
        assert scope.user_id == WHITESPACE, (
            "MemoryScope.user() accepted whitespace user_id — "
            "'if not user_id' check does not catch non-empty whitespace strings."
        )

    def test_memory_scope_whitespace_reaches_to_identifiers(self):
        """
        The whitespace id also survives to_identifiers() — the dict that is
        passed as kwargs to the provider (Qdrant / mem0).
        """
        from continuum.memory.scopes import MemoryScope

        scope = MemoryScope.user(WHITESPACE)
        ids = scope.to_identifiers()
        assert ids == {"user_id": WHITESPACE}, (
            f"Whitespace user_id {WHITESPACE!r} reached provider identifiers: {ids}"
        )

    def test_memory_scope_rejects_empty_string(self):
        """
        Empty string IS caught — falsy check works for "".
        This is why the CLI's .strip() or None matters:
        "" → ValueError,  "   " → silent acceptance.
        """
        from continuum.memory.scopes import MemoryScope

        with pytest.raises(ValueError, match="user_id is required"):
            MemoryScope.user("")

    def test_from_isolation_mode_also_accepts_whitespace(self):
        """
        The higher-level factory used by MemoryClient._build_scope() has the
        same truthy check — whitespace passes through there too.
        """
        from continuum.memory.scopes import MemoryScope

        scope = MemoryScope.from_isolation_mode("user", user_id=WHITESPACE)
        assert scope.user_id == WHITESPACE

    def test_clean_user_id_accepted_normally(self):
        """Control: a clean user_id builds correctly."""
        from continuum.memory.scopes import MemoryScope

        scope = MemoryScope.user(CLEAN_ID)
        assert scope.user_id == CLEAN_ID
        assert scope.to_identifiers() == {"user_id": CLEAN_ID}


# =============================================================================
# Checkpoint 5 — Full journey summary (no mocking needed, pure logic)
# =============================================================================

class TestFullJourney:
    """
    Combines all checkpoints into one readable trace so you can see
    exactly where whitespace is blocked vs where it slips through.
    """

    def test_guards_now_live_at_the_framework_boundaries(self):
        """
        Whitespace is now caught at the framework BOUNDARIES (session key
        computation and the runner), not just the CLI. The low-level layers
        below the boundary (RunContext dataclass, MemoryScope) remain
        deliberately unguarded — they are protected transitively because every
        real entry path validates first.
        """
        # — Layer 1: CLI input handling (still the first line of defence) —
        assert WHITESPACE.strip() or None is None

        # — Layer 2: Session ID — NOW GUARDED. validate_user_id strips
        #   whitespace to None, so the provider falls through to a random UUID
        #   instead of embedding "u:   " in the Redis key.
        from continuum.session.config import SessionConfig
        from continuum.session.providers.redis import RedisSessionProvider
        provider = RedisSessionProvider(SessionConfig(), auto_initialize=False)
        session_key = provider._compute_session_id(None, WHITESPACE, None)
        assert not session_key.startswith("u:") and not session_key.startswith("c:"), (
            "Session layer now strips whitespace to None — no 'u:   ' key"
        )

        # — Layer 3: RunContext — still a plain dataclass with no validation.
        #   (Validation happens in the runner BEFORE it builds the context.)
        from continuum.agent.utils.context_utils import create_run_context
        ctx = create_run_context(user_id=WHITESPACE)
        assert ctx.user_id == WHITESPACE, "RunContext itself stores whitespace verbatim"

        # — Layer 4: MemoryScope — deliberately left as a trust-the-caller
        #   pass-through (see test_memory_adversarial.py). Protected because the
        #   runner validates ctx.user_id before memory_service ever uses it.
        from continuum.memory.scopes import MemoryScope
        scope = MemoryScope.user(WHITESPACE)
        assert scope.user_id == WHITESPACE, "MemoryScope stays a documented pass-through"

    def test_runner_boundary_catches_what_the_cli_misses(self):
        """
        validate_user_id is the guard that every runner.run() call passes
        through, so whitespace is normalized to None even if the CLI is bypassed.
        """
        from continuum.utils.sanitization import validate_user_id

        assert validate_user_id(WHITESPACE) is None, "whitespace → None at the boundary"
        assert validate_user_id(CLEAN_ID) == CLEAN_ID, "clean id survives validation"
