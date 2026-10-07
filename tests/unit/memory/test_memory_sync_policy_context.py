"""The ``*_sync`` memory methods must carry the run's policy with them (F12).

Background
----------
The memory gate finds the run's policy through a ``ContextVar`` published by the
runner (``security/policy_context.py``). A ContextVar belongs to the current
context, and a plain ``ThreadPoolExecutor.submit`` starts its callable with an
empty one. ``MemoryClient._run_sync`` did exactly that whenever an event loop was
already running -- which is every call from a sync tool during an agent run -- so
``resolve_active_policy`` found nothing and the gate allowed the operation. No
error, no log: a run whose policy denied memory could read and write it anyway,
as long as it went through ``search_sync`` / ``add_sync``.

The fix copies the caller's context into the worker. These tests call the sync
methods from inside a running loop, the shape that lost the policy.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.exceptions import MemoryAccessDeniedError
from continuum.agent.utils.context_utils import create_run_context
from continuum.memory.client import MemoryClient
from continuum.memory.config import MemoryConfig
from continuum.memory.types import MemorySearchResult
from continuum.security.policy import AccessPolicy, PolicyStore
from continuum.security.policy_context import get_active_policy, use_active_policy


def _client():
    provider = MagicMock()
    provider.is_initialized = True
    provider.add = AsyncMock(return_value=MagicMock(results=[]))
    provider.search = AsyncMock(
        return_value=MemorySearchResult(query="q", limit=5, total_results=0, results=[])
    )
    return MemoryClient(config=MemoryConfig(enabled=True), provider=provider), provider


def _deny(labels, resources):
    store = PolicyStore()
    store.add_policy(
        AccessPolicy(
            name="deny-rule", subjects=list(labels), resources=list(resources), effect="deny"
        )
    )
    return store


# ---------------------------------------------------------------------------
# Called from inside a running loop -- a sync tool during an agent run
# ---------------------------------------------------------------------------


class TestSyncMethodsInsideARun:
    async def test_denied_search_sync_is_blocked(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with (
            use_active_policy(_deny(["pii"], ["memory:read:*"]), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            client.search_sync("q", user_id="u1")

        provider.search.assert_not_awaited()

    async def test_denied_add_sync_is_blocked(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with (
            use_active_policy(_deny(["pii"], ["memory:write:*"]), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            client.add_sync("a fact", user_id="u1")

        provider.add.assert_not_awaited()

    async def test_allowed_sync_calls_still_proceed(self):
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with use_active_policy(_deny(["phi"], ["memory:*"]), "ag", ctx):
            client.add_sync("a fact", user_id="u1")
            client.search_sync("q", user_id="u1")

        provider.add.assert_awaited_once()
        provider.search.assert_awaited_once()

    async def test_the_worker_sees_the_same_policy(self):
        """The mechanism itself, independent of any one gate."""
        client, _ = _client()
        store = PolicyStore()
        ctx = create_run_context(user_id="u1")

        async def read_policy():
            return get_active_policy()

        with use_active_policy(store, "ag", ctx):
            seen = client._run_sync(read_policy())

        assert seen is not None and seen.policy_store is store


# ---------------------------------------------------------------------------
# What must not change
# ---------------------------------------------------------------------------


class TestWhatMustNotChange:
    def test_sync_caller_without_a_loop_is_still_gated(self):
        """The ``asyncio.run`` branch already inherited the context; keep it so."""
        client, provider = _client()
        ctx = create_run_context(user_id="u1", data_labels={"pii"})

        with (
            use_active_policy(_deny(["pii"], ["memory:read:*"]), "ag", ctx),
            pytest.raises(MemoryAccessDeniedError),
        ):
            client.search_sync("q", user_id="u1")

        provider.search.assert_not_awaited()

    async def test_no_policy_means_no_gate(self):
        """Outside a run there is no policy, and nothing is denied -- F12 is about
        the policy getting lost, not about what happens when there is none."""
        client, provider = _client()

        client.search_sync("q", user_id="u1")

        provider.search.assert_awaited_once()

    async def test_the_worker_cannot_leak_a_policy_back(self):
        """A copy, not a share: a value set in the worker stays in the worker."""
        client, _ = _client()

        async def publish():
            use_active_policy(PolicyStore(), "leak", None).__enter__()

        client._run_sync(publish())

        assert get_active_policy() is None


# ---------------------------------------------------------------------------
# No new bare thread hand-off in the memory client
# ---------------------------------------------------------------------------


def test_memory_client_submits_nothing_without_the_context():
    """Every ``.submit(...)`` in the memory client must pass ``copy_context().run``.

    A bare ``submit(fn, ...)`` is how the policy was lost; this keeps a future
    edit from reintroducing it. ``asyncio.to_thread`` copies the context itself
    and needs no such care.
    """
    import continuum.memory.client as module

    tree = ast.parse(Path(module.__file__).read_text())
    bare = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "submit"
        ):
            # `asyncio.run` also ends in `.run`, so match the call that makes
            # the copy, not just the attribute name.
            first = node.args[0] if node.args else None
            copies = (
                isinstance(first, ast.Attribute)
                and first.attr == "run"
                and isinstance(first.value, ast.Call)
                and getattr(first.value.func, "attr", getattr(first.value.func, "id", None))
                == "copy_context"
            )
            if not copies:
                bare.append(node.lineno)

    assert not bare, f"memory/client.py submits without copying the context at lines {bare}"
