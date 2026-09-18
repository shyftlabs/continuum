"""
Phase 3 — data-label MEMORY-WRITE gate.

A run tainted with a label (e.g. "pii") can be denied persistence to a memory
scope via policy: `deny(subjects=["pii"], resources=["memory:*"])`. The gate
lives in MemoryClient.add(): labels are folded into the policy subjects, the
scope is the resource (`memory:write:<scope>`, plus the legacy
`memory:<scope>` for policies predating the read/write split), and a deny raises
MemoryAccessDeniedError before the provider writes.

Crucially, the write path doesn't thread RunContext, so the gate reads the
AMBIENT policy published by AgentRunner.run() (same mechanism as the model
gate) — no session-save plumbing required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.utils.context_utils import create_run_context
from continuum.memory.client import MemoryClient
from continuum.security.policy import PolicyDecision


def _client_with_provider() -> MemoryClient:
    mc = MemoryClient.__new__(MemoryClient)
    # __init__ is bypassed to isolate the policy gate, so the few attributes
    # add() reads outside that gate have to be supplied by hand.
    mc._config = SimpleNamespace(memory_isolation="user")
    mc._warned_shared_write = False
    mc._provider = MagicMock()
    mc._provider.add = AsyncMock(return_value=MagicMock())
    mc._ensure_enabled = lambda: None  # type: ignore[method-assign]
    scope = MagicMock()
    scope.to_identifiers.return_value = {}
    mc._build_scope = lambda *a, **k: scope  # type: ignore[method-assign,assignment]
    return mc


# ---------------------------------------------------------------------------
# Shared resolver — explicit args win, else fall back to the ambient policy.
# ---------------------------------------------------------------------------


class TestResolveActivePolicy:
    def test_explicit_values_returned_as_is(self):
        from continuum.security.policy_context import resolve_active_policy

        store = MagicMock()
        s, subj, labels = resolve_active_policy(store, "agent", {"pii"})
        assert s is store
        assert subj == "agent"
        assert labels == {"pii"}

    def test_falls_back_to_ambient_when_no_store(self):
        from continuum.security.policy_context import resolve_active_policy, use_active_policy

        ambient_store = MagicMock()
        ctx = create_run_context(data_labels={"phi"})
        with use_active_policy(ambient_store, "ambient-agent", ctx):
            s, subj, labels = resolve_active_policy(None, None, None)
        assert s is ambient_store
        assert subj == "ambient-agent"
        assert labels == {"phi"}

    def test_no_ambient_no_explicit_returns_none(self):
        from continuum.security.policy_context import resolve_active_policy

        assert resolve_active_policy(None, None, None) == (None, None, None)


# ---------------------------------------------------------------------------
# Memory-write gate behaviour.
# ---------------------------------------------------------------------------


class TestMemoryWritePolicy:
    async def test_denied_label_raises_before_provider_write(self):
        from continuum.agent.exceptions import MemoryAccessDeniedError

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(
            allowed=False, policy_name="no_pii_memory", reason="deny"
        )
        mc = _client_with_provider()

        with pytest.raises(MemoryAccessDeniedError):
            await mc.add(
                "remember this",
                user_id="u1",
                policy_store=ps,
                subject="agent",
                data_labels={"pii"},
            )
        mc._provider.add.assert_not_called()
        # The precise resource is checked first; a deny there short-circuits,
        # so the legacy form is never reached.
        ps.check.assert_called_once_with(["agent", "pii"], "memory:write:u1")

    async def test_allowed_label_proceeds_to_write(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        mc = _client_with_provider()

        await mc.add(
            "remember this",
            user_id="u1",
            policy_store=ps,
            subject="agent",
            data_labels={"pii"},
        )
        mc._provider.add.assert_awaited_once()
        # Allowed at the precise resource, so the legacy form is consulted too --
        # policies predating the read/write split are written against it.
        assert [c.args for c in ps.check.call_args_list] == [
            (["agent", "pii"], "memory:write:u1"),
            (["agent", "pii"], "memory:u1"),
        ]

    async def test_uses_ambient_policy_when_not_passed(self):
        # The session-save write path doesn't pass policy params; the gate must
        # pick up the ambient policy published by run().
        from continuum.agent.exceptions import MemoryAccessDeniedError
        from continuum.security.policy_context import use_active_policy

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=False, policy_name="p", reason="deny")
        mc = _client_with_provider()
        ctx = create_run_context(data_labels={"pii"})

        with use_active_policy(ps, "agent", ctx):
            with pytest.raises(MemoryAccessDeniedError):
                await mc.add("remember this", agent_id="a1")  # no policy args
        mc._provider.add.assert_not_called()
        ps.check.assert_called_once_with(["agent", "pii"], "memory:write:a1")

    async def test_no_policy_no_ambient_means_no_gate(self):
        mc = _client_with_provider()
        await mc.add("remember this", user_id="u1")  # nothing set anywhere
        mc._provider.add.assert_awaited_once()

    async def test_subject_only_when_no_labels(self):
        # Backward compatible: explicit subject, no labels → bare subject checked.
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        mc = _client_with_provider()

        await mc.add("x", user_id="u1", policy_store=ps, subject="agent")
        # Allowed at the precise resource, so the legacy form is checked too --
        # policies predating the read/write split are written against it.
        assert [c.args for c in ps.check.call_args_list] == [
            ("agent", "memory:write:u1"),
            ("agent", "memory:u1"),
        ]
