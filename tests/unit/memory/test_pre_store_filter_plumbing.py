"""The filter has to reach the gate, and the report has to match what was stored.

The gate in ``filtered_memory`` only fires if the filter is in scope when mem0
runs, which means threading it from ``AgentMemoryConfig`` down through
``SessionClient._store_in_memory`` → ``MemoryClient.add`` → ``Mem0Provider.add``
and into the ContextVar around the ``to_thread`` call. Every one of those hops
is somewhere the filter can be dropped silently -- the write still succeeds, it
just is not gated -- so each is asserted here.

The second half is the reporting. mem0's ADD branch appends
``{"id": None, "memory": text}`` whatever ``_create_memory`` returns, so a
suppressed fact still comes back in the result. Passing that to ``on_stored``
would claim a fact was stored that never was: the same defect as reporting a
failed delete as a success, pointing the other way.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Mem0Provider — puts the filter in scope around the call into mem0.
# ---------------------------------------------------------------------------


def _provider():
    from continuum.memory.providers.mem0 import Mem0Provider

    p = Mem0Provider.__new__(Mem0Provider)
    p._sync_memory = MagicMock()
    # A MagicMock invents attributes on access, so the "no filter set" state has
    # to be explicit or every read looks like a configured filter.
    p._sync_memory._continuum_active_filter = None
    p._sync_memory._continuum_filter_lock = None
    p._initialized = True
    p._ensure_initialized = lambda: None  # type: ignore[method-assign]
    return p


class TestTheProviderScopesTheFilter:
    async def test_the_filter_is_active_while_mem0_runs(self):
        """Asserted from inside the call, because a filter set after mem0 has
        already written is indistinguishable from no filter at all."""
        from continuum.memory.providers.filtered_memory import _FILTER_ATTR

        seen: list[object] = []

        def fake_add(**_kwargs):
            seen.append(getattr(p._sync_memory, _FILTER_ATTR, None))
            return {"results": []}

        p = _provider()
        p._sync_memory.add = fake_add
        my_filter = lambda facts: facts  # noqa: E731

        await p.add("hello", user_id="u1", pre_store_filter=my_filter)

        assert seen and seen[0] is not None, "the filter was not in scope"
        assert seen[0][0] is my_filter

    async def test_the_scope_is_released_afterwards(self):
        from continuum.memory.providers.filtered_memory import _FILTER_ATTR

        p = _provider()
        p._sync_memory.add = lambda **_k: {"results": []}
        await p.add("hello", user_id="u1", pre_store_filter=lambda f: f)
        assert getattr(p._sync_memory, _FILTER_ATTR, None) is None

    async def test_suppressed_texts_come_back_to_the_caller(self):
        """Otherwise the caller cannot tell a suppressed fact from a stored one."""
        from continuum.memory.providers.filtered_memory import _FILTER_ATTR

        def fake_add(**_kwargs):
            _filter, suppressed = getattr(p._sync_memory, _FILTER_ATTR)
            suppressed.append("SSN is 123-45-6789")
            return {"results": [{"id": None, "memory": "SSN is 123-45-6789", "event": "ADD"}]}

        p = _provider()
        p._sync_memory.add = fake_add
        result = await p.add("x", user_id="u1", pre_store_filter=lambda f: [])
        assert result.suppressed == ["SSN is 123-45-6789"]

    async def test_no_filter_leaves_no_scope(self):
        from continuum.memory.providers.filtered_memory import _FILTER_ATTR

        seen = []
        p = _provider()
        p._sync_memory.add = lambda **_k: (
            seen.append(getattr(p._sync_memory, _FILTER_ATTR, None)),
            {"results": []},
        )[1]
        await p.add("hello", user_id="u1")
        assert seen == [None]


# ---------------------------------------------------------------------------
# MemoryClient — passes it through without deciding anything.
# ---------------------------------------------------------------------------


class TestTheClientPassesItThrough:
    async def test_add_forwards_the_filter_to_the_provider(self):
        from continuum.memory.client import MemoryClient

        seen: dict = {}

        class GatingProvider:
            """Declares the parameter, which is how a provider opts in."""

            async def add(self, messages, *, pre_store_filter=None, **kwargs):
                seen["filter"] = pre_store_filter
                return SimpleNamespace(results=[], suppressed=[])

        mc = MemoryClient.__new__(MemoryClient)
        mc._config = SimpleNamespace(memory_isolation="user")
        mc._warned_shared_write = False
        mc._provider = GatingProvider()
        mc._ensure_enabled = lambda: None  # type: ignore[method-assign]
        scope = MagicMock()
        scope.to_identifiers.return_value = {}
        mc._build_scope = lambda *a, **k: scope  # type: ignore[method-assign,assignment]

        f = lambda facts: facts  # noqa: E731
        await mc.add("remember this", user_id="u1", pre_store_filter=f)

        assert seen["filter"] is f

    async def test_the_real_provider_declares_the_parameter(self):
        """The forwarding above is only meaningful if the shipped provider opts
        in; a rename there would degrade every install to delete-after-write."""
        import inspect

        from continuum.memory.providers.mem0 import Mem0Provider

        assert "pre_store_filter" in inspect.signature(Mem0Provider.add).parameters


# ---------------------------------------------------------------------------
# SessionClient — the report must match what is in the store.
# ---------------------------------------------------------------------------


def _session_client(add_result):
    from continuum.session.client import SessionClient

    sc = SessionClient.__new__(SessionClient)
    # `memory_client` is a read-only property resolving from the container, so
    # the stub goes on the resolver it delegates to.
    mem = MagicMock()
    mem.add = AsyncMock(return_value=add_result)
    mem.delete = AsyncMock(return_value=True)
    sc._resolve_memory_client = lambda: mem  # type: ignore[method-assign]
    sc._call = AsyncMock(
        return_value=SimpleNamespace(user_id="u1", conversation_id="c1", agent_id=None)
    )
    return sc, mem


class TestOnStoredMatchesTheStore:
    async def test_a_suppressed_fact_is_not_reported_as_stored(self):
        from continuum.session.types import ChatMessage

        result = MagicMock()
        result.results = [
            {"id": None, "memory": "SSN is 123-45-6789", "event": "ADD"},
            {"id": "keep-1", "memory": "Prefers morning appointments", "event": "ADD"},
        ]
        result.suppressed = ["SSN is 123-45-6789"]

        stored: list[list[str]] = []
        sc, mem = _session_client(result)
        await sc._store_in_memory(
            session_id="s1",
            message=ChatMessage(role="user", content="..."),
            agent_id=None,
            metadata=None,
            extraction_prompt=None,
            pre_store_filter=lambda f: f,
            on_stored=stored.append,
        )
        assert stored == [["Prefers morning appointments"]]

    async def test_nothing_is_deleted_when_the_gate_suppressed_it(self):
        """The delete path is the fallback. If it runs for a fact the gate
        already stopped, the gate is not doing its job and the old race is back.
        """
        from continuum.session.types import ChatMessage

        result = MagicMock()
        result.results = [{"id": None, "memory": "SSN is 123-45-6789", "event": "ADD"}]
        result.suppressed = ["SSN is 123-45-6789"]

        sc, mem = _session_client(result)
        await sc._store_in_memory(
            session_id="s1",
            message=ChatMessage(role="user", content="..."),
            agent_id=None,
            metadata=None,
            extraction_prompt=None,
            pre_store_filter=lambda f: [],
            on_stored=None,
        )
        mem.delete.assert_not_awaited()

    async def test_a_fact_that_slips_past_the_gate_is_still_deleted(self):
        """Belt and braces: the gate is new and depends on mem0 internals, so
        the delete path stays for anything that reaches the store anyway."""
        from continuum.session.types import ChatMessage

        result = MagicMock()
        result.results = [{"id": "leaked-1", "memory": "SSN is 123-45-6789", "event": "ADD"}]
        result.suppressed = []  # the gate did not catch it

        sc, mem = _session_client(result)
        await sc._store_in_memory(
            session_id="s1",
            message=ChatMessage(role="user", content="..."),
            agent_id=None,
            metadata=None,
            extraction_prompt=None,
            pre_store_filter=lambda facts: [f for f in facts if "SSN" not in f],
            on_stored=None,
        )
        mem.delete.assert_awaited_once_with("leaked-1")

    async def test_the_filter_reaches_the_client(self):
        from continuum.session.types import ChatMessage

        result = MagicMock()
        result.results = []
        result.suppressed = []
        sc, mem = _session_client(result)
        f = lambda facts: facts  # noqa: E731
        await sc._store_in_memory(
            session_id="s1",
            message=ChatMessage(role="user", content="..."),
            agent_id=None,
            metadata=None,
            extraction_prompt=None,
            pre_store_filter=f,
            on_stored=None,
        )
        assert mem.add.await_args.kwargs["pre_store_filter"] is f


class TestBackwardCompatibility:
    async def test_a_provider_result_without_suppressed_still_works(self):
        """MemoryAddResult is constructed in more than one place, and a result
        built by older code has no `suppressed` attribute."""
        from continuum.session.types import ChatMessage

        result = SimpleNamespace(results=[{"id": "a", "memory": "kept"}])  # no `suppressed`
        stored: list[list[str]] = []
        sc, mem = _session_client(result)
        await sc._store_in_memory(
            session_id="s1",
            message=ChatMessage(role="user", content="..."),
            agent_id=None,
            metadata=None,
            extraction_prompt=None,
            pre_store_filter=None,
            on_stored=stored.append,
        )
        assert stored == [["kept"]]


class TestProvidersThatDoNotSupportTheGate:
    """`BaseMemoryProvider` is a public interface. A provider written before the
    gate existed does not accept `pre_store_filter`, and passing it would raise
    TypeError at the write -- breaking memory entirely for an integration that
    was working."""

    async def test_a_provider_without_the_parameter_still_writes(self):
        from continuum.memory.client import MemoryClient

        seen: dict = {}

        class LegacyProvider:
            async def add(self, messages, **kwargs):  # no pre_store_filter
                seen.update(kwargs)
                return SimpleNamespace(results=[], suppressed=[])

        mc = MemoryClient.__new__(MemoryClient)
        mc._config = SimpleNamespace(memory_isolation="user")
        mc._warned_shared_write = False
        mc._provider = LegacyProvider()
        mc._ensure_enabled = lambda: None  # type: ignore[method-assign]
        scope = MagicMock()
        scope.to_identifiers.return_value = {}
        mc._build_scope = lambda *a, **k: scope  # type: ignore[method-assign,assignment]

        await mc.add("x", user_id="u1", pre_store_filter=lambda f: f)
        assert "pre_store_filter" not in seen, "the legacy provider must not be handed it"

    async def test_the_degradation_is_announced_once(self):
        """Silently falling back to delete-after-write is the failure mode this
        whole path exists to avoid, so it is said out loud -- once, since this
        runs on every write.

        A handler is attached directly rather than using ``caplog``: Continuum's
        ``get_logger`` returns an ``OrchestratorLogger`` that pytest's capture
        fixture does not see, so a caplog-based assertion here passes with zero
        records whether or not the warning is ever emitted.
        """
        import logging

        from continuum.memory.client import MemoryClient

        class LegacyProvider:
            async def add(self, messages, **kwargs):
                return SimpleNamespace(results=[], suppressed=[])

        mc = MemoryClient.__new__(MemoryClient)
        mc._config = SimpleNamespace(memory_isolation="user")
        mc._warned_shared_write = False
        mc._provider = LegacyProvider()
        mc._ensure_enabled = lambda: None  # type: ignore[method-assign]
        scope = MagicMock()
        scope.to_identifiers.return_value = {}
        mc._build_scope = lambda *a, **k: scope  # type: ignore[method-assign,assignment]

        seen_records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                seen_records.append(record)

        lg = logging.getLogger("continuum.memory.client")
        handler = Capture(level=logging.WARNING)
        lg.addHandler(handler)
        try:
            await mc.add("x", user_id="u1", pre_store_filter=lambda f: f)
            await mc.add("y", user_id="u1", pre_store_filter=lambda f: f)
        finally:
            lg.removeHandler(handler)

        hits = [r for r in seen_records if "pre_store_filter" in r.getMessage()]
        assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
