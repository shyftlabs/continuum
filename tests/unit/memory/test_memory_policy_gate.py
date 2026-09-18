"""One policy gate for both memory operations, and a read/write resource split.

Background
----------
``MemoryClient`` carried two copies of the same gate. ``add()`` resolved the
ambient run policy, so a tainted run's write was blocked. ``search()`` used the
raw arguments -- ``if policy_store is not None and subject is not None`` -- and
the only caller that matters, automatic retrieval in ``MemoryService``, passes
neither. So the read gate never ran: a run the policy said must not touch memory
could still read every row out of it.

Two copies of one rule is how one copy ends up wrong, and
``policy_context.resolve_active_policy`` says as much in its own docstring:
threading policy args through every call site is "fragile, and silently bypassed
by any call site that forgets". So the fix is a single shared enforcement point,
not a second patched copy.

The resource split is the other half. Both operations checked the same string,
``memory:<scope>``, which makes "block writes, allow reads" inexpressible -- and
means a rule written to stop persistence silently starts denying retrieval the
moment the read gate works. The clinic's own ``phi-never-persisted`` is exactly
that rule; its denial message says "must not be *written*".

Reads and writes now check ``memory:read:<scope>`` / ``memory:write:<scope>``,
plus the legacy ``memory:<scope>`` so existing policies keep working. fnmatch
makes ``memory:*`` cover both new forms, while ``memory:write:*`` covers only
writes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.exceptions import MemoryAccessDeniedError
from continuum.agent.services.memory_service import MemoryService
from continuum.agent.utils.context_utils import create_run_context
from continuum.memory.client import MemoryClient
from continuum.memory.config import MemoryConfig
from continuum.memory.types import MemoryEntry, MemorySearchResult
from continuum.security.policy import AccessPolicy, PolicyStore
from continuum.security.policy_context import use_active_policy


def _rows(labels=None):
    meta = {"_data_labels": list(labels)} if labels else {}
    return MemorySearchResult(
        query="q",
        limit=5,
        total_results=1,
        results=[MemoryEntry(id="m1", memory="a remembered fact", metadata=meta, score=0.9)],
    )


def _client(search_result=None):
    provider = MagicMock()
    provider.is_initialized = True
    provider.add = AsyncMock(return_value=MagicMock(results=[]))
    provider.search = AsyncMock(return_value=search_result or _rows())
    return MemoryClient(config=MemoryConfig(enabled=True), provider=provider), provider


def _deny(labels, resources, name="deny-rule"):
    store = PolicyStore()
    store.add_policy(
        AccessPolicy(name=name, subjects=list(labels), resources=list(resources), effect="deny")
    )
    return store


def _service(client):
    return MemoryService(memory_client=client, session_client=None)


def _agent():
    return BaseAgent(
        name="ag",
        instructions="x",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(search_memories=True, scope_data_labels={}),
    )


# ---------------------------------------------------------------------------
# The read gate has to fire on the path that actually retrieves
# ---------------------------------------------------------------------------


class TestAutomaticRetrievalIsGated:
    async def test_denied_read_never_reaches_the_provider(self):
        """The point of the gate: no query is issued, so nothing is disclosed."""
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with use_active_policy(_deny(["pii"], ["memory:read:*"]), "ag", ctx):
            out = await _service(client).retrieve_memories(_agent(), "q", ctx)

        assert out == []
        provider.search.assert_not_awaited()

    async def test_allowed_read_proceeds(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with use_active_policy(_deny(["phi"], ["memory:read:*"]), "ag", ctx):
            out = await _service(client).retrieve_memories(_agent(), "q", ctx)

        assert len(out) == 1
        provider.search.assert_awaited_once()

    async def test_untainted_run_is_unaffected(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1")

        with use_active_policy(_deny(["pii"], ["memory:read:*"]), "ag", ctx):
            out = await _service(client).retrieve_memories(_agent(), "q", ctx)

        assert len(out) == 1

    async def test_explicit_policy_args_still_win(self):
        """An explicit store overrides the ambient one, as on the write path."""
        client, _ = _client()
        ctx = create_run_context(user_id="u1")

        with (
            use_active_policy(PolicyStore(), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            await client.search(
                "q",
                user_id="u1",
                policy_store=_deny(["pii"], ["memory:read:*"]),
                subject="ag",
                data_labels={"pii"},
            )


# ---------------------------------------------------------------------------
# Read and write are separately expressible
# ---------------------------------------------------------------------------


class TestResourceSplit:
    async def test_write_only_rule_does_not_block_reads(self):
        """The regression this split exists to prevent. The clinic's
        phi-never-persisted means "never stored", not "never recalled"."""
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"phi"})

        with use_active_policy(_deny(["phi"], ["memory:write:*"]), "ag", ctx):
            out = await _service(client).retrieve_memories(_agent(), "q", ctx)

        assert len(out) == 1, "a write-only deny must leave retrieval alone"
        provider.search.assert_awaited_once()

    async def test_write_only_rule_still_blocks_writes(self):
        client, _ = _client()
        ctx = create_run_context(user_id="u1", data_labels={"phi"})

        with (
            use_active_policy(_deny(["phi"], ["memory:write:*"]), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            await client.add("a fact", user_id="u1")

    async def test_read_only_rule_does_not_block_writes(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with use_active_policy(_deny(["pii"], ["memory:read:*"]), "ag", ctx):
            await client.add("a fact", user_id="u1")

        provider.add.assert_awaited_once()


class TestLegacyRulesKeepWorking:
    """``memory:*`` and an exact ``memory:<scope>`` predate the split and are in
    shipped policies. fnmatch covers the glob; the bare form is checked
    explicitly alongside the operation-specific one."""

    async def test_glob_rule_blocks_reads(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with use_active_policy(_deny(["pii"], ["memory:*"]), "ag", ctx):
            out = await _service(client).retrieve_memories(_agent(), "q", ctx)

        assert out == []
        provider.search.assert_not_awaited()

    async def test_glob_rule_blocks_writes(self):
        client, _ = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with (
            use_active_policy(_deny(["pii"], ["memory:*"]), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            await client.add("a fact", user_id="u1")

    async def test_exact_legacy_resource_still_blocks_reads(self):
        """``resources=["memory:u1"]`` matches neither new form by glob, so the
        legacy string is checked too or such a policy would silently stop."""
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with use_active_policy(_deny(["pii"], ["memory:u1"]), "ag", ctx):
            out = await _service(client).retrieve_memories(_agent(), "q", ctx)

        assert out == []
        provider.search.assert_not_awaited()

    async def test_exact_legacy_resource_still_blocks_writes(self):
        client, _ = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with (
            use_active_policy(_deny(["pii"], ["memory:u1"]), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            await client.add("a fact", user_id="u1")


# ---------------------------------------------------------------------------
# A denial is the gate working, not a crash
# ---------------------------------------------------------------------------


class TestDenialIsReportedQuietly:
    async def test_denied_read_is_reported_at_info_naming_the_policy(self, monkeypatch):
        """The write path already ruled on this: an expected policy denial is
        logged quietly and never escalated. A stack trace tells an operator
        something broke and sends them hunting a bug that is not there."""
        from continuum.agent.services import memory_service as mod

        client, _ = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})
        seen: list[tuple[str, tuple, dict]] = []
        for level in ("info", "warning", "error"):
            monkeypatch.setattr(
                mod.logger, level, lambda *a, _l=level, **k: seen.append((_l, a, k))
            )

        with use_active_policy(_deny(["pii"], ["memory:read:*"], name="pii-no-read"), "ag", ctx):
            await _service(client).retrieve_memories(_agent(), "q", ctx)

        infos = [a for lvl, a, _ in seen if lvl == "info"]
        assert any("pii-no-read" in str(a) for a in infos), (
            f"the denial should be reported at INFO naming the policy; got {infos}"
        )
        assert not [(a, k) for lvl, a, k in seen if lvl == "warning" and k.get("exc_info")], (
            "a denial must not be logged with a traceback"
        )
        assert not [a for lvl, a, _ in seen if lvl == "error"], "a denial is not an error"


# ---------------------------------------------------------------------------
# The split also settles the read-taints-read interaction
# ---------------------------------------------------------------------------


class TestReadTaintDoesNotSelfBlockUnderAWriteRule:
    async def test_second_read_still_works_when_only_writes_are_denied(self):
        """Row provenance taints the reading run, so with one shared resource a
        deny meant for writes would make the first read disable the second. With
        the split that only happens for a rule that asks for it."""
        client, provider = _client(search_result=_rows(labels=["external"]))
        ctx = create_run_context(user_id="u1")
        svc, agent = _service(client), _agent()

        with use_active_policy(_deny(["external"], ["memory:write:*"]), "ag", ctx):
            first = await svc.retrieve_memories(agent, "q1", ctx)
            assert "external" in ctx.data_labels, "the read must taint the run"
            second = await svc.retrieve_memories(agent, "q2", ctx)

        assert len(first) == 1
        assert len(second) == 1, "a write-only deny must not disable further reads"
        assert provider.search.await_count == 2
