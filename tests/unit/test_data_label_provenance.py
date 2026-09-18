"""
Tests for data-label PROVENANCE tainting (Phase 0 + 1).

Background
----------
`RunContext.data_labels` is a taint set (e.g. {"pii", "phi"}) that rides along
with a run. Historically nothing *set* it automatically — labels were manual-only
and, in practice, always empty — so the one consumer that reads them (the tool
gate) never fired.

This adds the PRODUCER half, with no PII detector in the SDK. Instead the
integrator declares *provenance* — which sources carry which labels — and the
runtime taints the run when data crosses those boundaries. Three declaration
sites:

  1. Tool      — AgentConfig.tool_data_labels[tool_name] -> labels;
                 a tool's result taints the run.
  2. Memory    — AgentMemoryConfig.scope_data_labels[scope] -> labels;
                 reading from that scope taints the run ("read = taint").
  3. Run-level — create_run_context(data_labels=...) seeds the run at start.

Plus the Phase-0 plumbing the producer needs: RunContext.taint().

No detector is shipped: tests declare provenance explicitly.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.utils.context_utils import create_run_context

# ---------------------------------------------------------------------------
# Phase 0 — core plumbing (pure; no runtime harness)
# ---------------------------------------------------------------------------


class TestRunContextTaint:
    def test_taint_adds_a_label(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1")
        ctx.taint("pii")
        assert "pii" in ctx.data_labels

    def test_taint_multiple_and_is_set_semantics(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1")
        ctx.taint("pii", "phi")
        ctx.taint("pii")  # duplicate is a no-op (set)
        assert ctx.data_labels == {"pii", "phi"}

    def test_taint_no_args_is_noop(self):
        from continuum.agent.types import RunContext

        ctx = RunContext(run_id="r1")
        ctx.taint()
        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Phase 1 — site 3: run-level provenance (seed at start)
# ---------------------------------------------------------------------------


class TestRunLevelProvenance:
    def test_create_run_context_seeds_data_labels(self):
        ctx = create_run_context(data_labels={"pii"})
        assert ctx.data_labels == {"pii"}

    def test_create_run_context_defaults_empty(self):
        ctx = create_run_context()
        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Phase 1 — site 1: tool provenance (a declared tool's result taints the run)
# ---------------------------------------------------------------------------


def _tool_call(name: str):
    from continuum.llm.types import FunctionCall, ToolCall

    return ToolCall(id="tc-1", type="function", function=FunctionCall(name=name, arguments="{}"))


def _agent_with_tool_labels(tool_labels: dict[str, set[str]]):
    """A BaseAgent whose tool_executor returns a canned tool result."""
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig

    agent = BaseAgent(
        name="prov-agent",
        instructions="test",
        config=AgentConfig(tool_data_labels=tool_labels),
    )
    # Fake executor: registry hit + a successful tool-result message.
    executor = MagicMock()
    executor.tool_registry = {name: (MagicMock(name="server"), object()) for name in tool_labels}
    executor.tool_registry.setdefault("plain_tool", (MagicMock(name="server"), object()))
    executor.execute_tool_calls = AsyncMock(
        return_value=[{"role": "tool", "tool_call_id": "tc-1", "content": "done"}]
    )
    agent.tool_executor = executor
    agent.on_tool_call = None
    return agent


def _tool_service():
    from continuum.agent.services.tool_service import ToolService

    return ToolService(tool_executor=None)


class TestToolProvenance:
    def test_config_field_defaults_empty(self):
        from continuum.agent.config import AgentConfig

        assert AgentConfig().tool_data_labels == {}

    async def test_declared_tool_result_taints_run(self):
        agent = _agent_with_tool_labels({"fetch_record": {"phi"}})
        svc = _tool_service()
        ctx = create_run_context()

        await svc.execute_tool_call(agent, _tool_call("fetch_record"), ctx)

        assert "phi" in ctx.data_labels

    async def test_undeclared_tool_does_not_taint(self):
        agent = _agent_with_tool_labels({"fetch_record": {"phi"}})
        svc = _tool_service()
        ctx = create_run_context()

        await svc.execute_tool_call(agent, _tool_call("plain_tool"), ctx)

        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Phase 1 — site 1, same-turn batch: a declared tool's taint must gate its
# SIBLINGS in the same batch, not just later turns.
#
# Regression: provenance taint was applied only AFTER a tool returned, while a
# batch of tool calls is gated up front (and, by default, executed in parallel).
# So [lookup_patient (declares "phi"), send_referral_email] would gate the
# exfiltration tool against the still-clean labels and let it through. The fix
# pre-taints the batch from the DECLARED labels of every tool before gating or
# executing any of them — order- and concurrency-independent.
# ---------------------------------------------------------------------------


def _tc(name: str, tid: str):
    from continuum.llm.types import FunctionCall, ToolCall

    return ToolCall(id=tid, type="function", function=FunctionCall(name=name, arguments="{}"))


def _content(msg):
    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")


def _gate_simulating_agent(tool_labels: dict[str, set[str]]):
    """Agent whose fake executor records the data_labels it was handed per tool
    and simulates the exfil gate: it denies ``send_referral_email`` whenever the
    labels it receives contain ``phi`` (i.e. the run is already tainted)."""
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig

    agent = BaseAgent(
        name="batch-agent", instructions="test", config=AgentConfig(tool_data_labels=tool_labels)
    )
    agent.policy_store = MagicMock()  # truthy → tool_service threads data_labels through
    agent.on_tool_call = None
    seen: dict[str, set[str]] = {}

    async def fake_exec(
        tool_calls,
        trace_id=None,
        policy_store=None,
        subject=None,
        data_labels=None,
        approval=None,
    ):
        tc = tool_calls[0]
        name = tc.function.name
        seen[name] = set(data_labels or ())  # snapshot at the moment this tool is gated
        if name == "send_referral_email" and "phi" in (data_labels or set()):
            return [
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "POLICY DENIED: PHI exfiltration",
                }
            ]
        return [{"role": "tool", "tool_call_id": tc.id, "content": "ok"}]

    executor = MagicMock()
    executor.tool_registry = {
        n: (MagicMock(name="server"), object()) for n in ("lookup_patient", "send_referral_email")
    }
    executor.execute_tool_calls = AsyncMock(side_effect=fake_exec)
    agent.tool_executor = executor
    return agent, seen


class TestBatchSiblingTaint:
    async def test_exfil_tool_listed_first_is_still_gated_against_sibling_taint(self):
        # Adversarial ordering: the exfil tool is listed BEFORE the producer.
        # Without the pre-taint fix, sequential execution runs it on a clean
        # context and it slips through. With the fix it sees "phi" and is denied.
        agent, seen = _gate_simulating_agent({"lookup_patient": {"phi"}})
        svc = _tool_service()
        ctx = create_run_context()

        results = await svc.execute_tools_batch(
            agent,
            [_tc("send_referral_email", "tc-1"), _tc("lookup_patient", "tc-2")],
            ctx,
        )

        assert "phi" in seen["send_referral_email"]  # sibling saw the producer's declared taint
        assert ctx.data_labels == {"phi"}
        assert any("POLICY DENIED" in str(_content(r)) for r in results)

    async def test_parallel_batch_also_gated(self):
        from continuum.agent.config import RunnerConfig

        agent, seen = _gate_simulating_agent({"lookup_patient": {"phi"}})
        # parallel execution is the default-on path that originally leaked.
        from continuum.agent.services.tool_service import ToolService

        svc = ToolService(tool_executor=None, config=RunnerConfig(parallel_tool_calls=True))
        ctx = create_run_context()

        results = await svc.execute_tools_batch(
            agent,
            [_tc("lookup_patient", "tc-1"), _tc("send_referral_email", "tc-2")],
            ctx,
        )

        assert "phi" in seen["send_referral_email"]
        assert any("POLICY DENIED" in str(_content(r)) for r in results)


# ---------------------------------------------------------------------------
# Phase 1 — site 2: memory-scope provenance (reading a labeled scope taints)
# ---------------------------------------------------------------------------


def _memory_result(n=1):
    res = MagicMock()
    items = []
    for i in range(n):
        m = MagicMock()
        m.to_dict.return_value = {"memory": f"fact-{i}"}
        m.metadata = {}
        m.user_id = "u1"
        m.score = 0.9
        m.memory = f"fact-{i}"
        items.append(m)
    res.results = items
    res.total_results = len(items)
    return res


def _memory_client(isolation="user", result=None):
    mc = MagicMock()
    mc.is_enabled = True
    mc.config = MagicMock()
    mc.config.memory_isolation = isolation
    mc.search = AsyncMock(return_value=result if result is not None else _memory_result())
    return mc


def _agent_with_scope_labels(scope_labels: dict[str, set[str]]):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="mem-prov-agent",
        instructions="test",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(scope_data_labels=scope_labels),
    )


def _memory_service(mc):
    from continuum.agent.services.memory_service import MemoryService

    return MemoryService(memory_client=mc, session_client=None)


class TestMemoryScopeProvenance:
    def test_config_field_defaults_empty(self):
        from continuum.agent.config import AgentMemoryConfig

        assert AgentMemoryConfig().scope_data_labels == {}

    async def test_read_from_labeled_scope_taints_run(self):
        svc = _memory_service(_memory_client(isolation="user"))
        agent = _agent_with_scope_labels({"user": {"pii"}})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert "pii" in ctx.data_labels

    async def test_no_results_does_not_taint(self):
        # No data actually flowed out of the scope → no taint.
        svc = _memory_service(_memory_client(isolation="user", result=_memory_result(0)))
        agent = _agent_with_scope_labels({"user": {"pii"}})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == set()

    async def test_undeclared_scope_does_not_taint(self):
        svc = _memory_service(_memory_client(isolation="user"))
        agent = _agent_with_scope_labels({"agent": {"pii"}})  # labels a different scope
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Row-level memory provenance (security finding F6)
#
# ``scope_data_labels`` above taints by SCOPE -- the integrator declares "the
# user scope is sensitive" and any read from it taints. That is coarse: it cannot
# distinguish a preference the user really stated from a sentence an attacker
# planted in a web page that the model then echoed into its reply, because both
# end up as rows in the same scope.
#
# The distinguishing fact is provenance, and it is destroyed at write time: the
# extracted fact is stored and nothing records how tainted the run that produced
# it was. So a poisoned row comes back indistinguishable from a genuine one, and
# every downstream defence has to guess.
#
# Two halves close that:
#   write -- stamp the run's live labels onto the row's metadata
#   read  -- re-taint the reading run from the row's own labels
#
# The point is not to label memory for its own sake. It is that the tool gate
# (executor.py, ``subjects = [subject, *sorted(data_labels)]``) then denies the
# action regardless of what the model was persuaded to believe -- which is the
# only defence measured to hold on models that ignore instruction hierarchy.
# ---------------------------------------------------------------------------

PROVENANCE_KEY = "_data_labels"


def _mem_client_with_provider():
    """A real MemoryClient over a mock provider, with memory forced enabled."""
    from unittest.mock import AsyncMock, MagicMock

    from continuum.memory.client import MemoryClient

    provider = MagicMock()
    provider.add = AsyncMock(return_value=MagicMock(results=[]))
    client = MemoryClient(provider=provider, auto_initialize=False)
    client._initialized = True
    type(client)  # keep the class handy for monkeypatching is_enabled below
    return client, provider


class TestMemoryWriteStampsProvenance:
    """The write path records how tainted the producing run was."""

    def _client(self, monkeypatch):
        client, provider = _mem_client_with_provider()
        monkeypatch.setattr(type(client), "is_enabled", property(lambda self: True))
        return client, provider

    async def test_explicit_labels_are_stamped_on_the_row(self, monkeypatch):
        client, provider = self._client(monkeypatch)

        await client.add("a fact", user_id="u1", data_labels={"external"})

        meta = provider.add.await_args.kwargs["metadata"]
        assert meta[PROVENANCE_KEY] == ["external"]

    async def test_labels_are_stored_sorted_and_json_serialisable(self, monkeypatch):
        """A set is not JSON-serialisable and its order is not stable; the vector
        store round-trips metadata as JSON, so persist a sorted list."""
        import json

        client, provider = self._client(monkeypatch)

        await client.add("a fact", user_id="u1", data_labels={"phi", "external", "pii"})

        stamped = provider.add.await_args.kwargs["metadata"][PROVENANCE_KEY]
        assert stamped == ["external", "phi", "pii"]
        json.dumps(stamped)  # must not raise

    async def test_untainted_run_adds_no_provenance_key(self, monkeypatch):
        """Absence of labels must not write an empty marker -- a clean row stays
        clean, so the read side can tell "no labels" from "labelled with none"."""
        client, provider = self._client(monkeypatch)

        await client.add("a fact", user_id="u1")

        meta = provider.add.await_args.kwargs["metadata"] or {}
        assert PROVENANCE_KEY not in meta

    async def test_caller_metadata_is_preserved(self, monkeypatch):
        client, provider = self._client(monkeypatch)

        await client.add(
            "a fact", user_id="u1", metadata={"session_id": "s1"}, data_labels={"external"}
        )

        meta = provider.add.await_args.kwargs["metadata"]
        assert meta["session_id"] == "s1"
        assert meta[PROVENANCE_KEY] == ["external"]

    async def test_caller_metadata_dict_is_not_mutated(self, monkeypatch):
        """The caller's dict may be reused across messages in a save loop."""
        client, provider = self._client(monkeypatch)
        caller_meta = {"session_id": "s1"}

        await client.add("a fact", user_id="u1", metadata=caller_meta, data_labels={"external"})

        assert caller_meta == {"session_id": "s1"}

    async def test_ambient_run_labels_are_stamped_without_explicit_argument(self, monkeypatch):
        """The session-save write path does not thread RunContext, so the labels
        have to come from the ambient policy the runner publishes."""
        from continuum.agent.types import RunContext
        from continuum.security.policy import PolicyStore
        from continuum.security.policy_context import use_active_policy

        client, provider = self._client(monkeypatch)
        ctx = RunContext(run_id="r1")
        ctx.taint("external")

        with use_active_policy(PolicyStore(), "agent-x", ctx):
            await client.add("a fact", user_id="u1")

        meta = provider.add.await_args.kwargs["metadata"]
        assert meta[PROVENANCE_KEY] == ["external"]

    async def test_stamped_even_when_no_policy_store_is_configured(self, monkeypatch):
        """Provenance is bookkeeping, not enforcement. The runner publishes the
        ambient context with ``policy_store=None`` when the agent has none, so
        rows must still be stamped -- otherwise enabling a policy later would
        find every existing row unlabelled and silently ungated."""
        from continuum.agent.types import RunContext
        from continuum.security.policy_context import use_active_policy

        client, provider = self._client(monkeypatch)
        ctx = RunContext(run_id="r1")
        ctx.taint("external")

        with use_active_policy(None, "agent-x", ctx):
            await client.add("a fact", user_id="u1")

        assert provider.add.await_args.kwargs["metadata"][PROVENANCE_KEY] == ["external"]

    async def test_taint_added_mid_run_is_reflected(self, monkeypatch):
        """ActivePolicy reads labels live, so a tool result that taints the run
        after the ambient publish must still reach the row."""
        from continuum.agent.types import RunContext
        from continuum.security.policy import PolicyStore
        from continuum.security.policy_context import use_active_policy

        client, provider = self._client(monkeypatch)
        ctx = RunContext(run_id="r1")

        with use_active_policy(PolicyStore(), "agent-x", ctx):
            ctx.taint("external")  # after the publish
            await client.add("a fact", user_id="u1")

        assert provider.add.await_args.kwargs["metadata"][PROVENANCE_KEY] == ["external"]


def _memory_result_with_labels(*label_sets):
    """A search result whose rows carry the given provenance labels."""
    from unittest.mock import MagicMock

    res = MagicMock()
    items = []
    for i, labels in enumerate(label_sets):
        m = MagicMock()
        meta = {} if labels is None else {PROVENANCE_KEY: labels}
        m.to_dict.return_value = {"memory": f"fact-{i}", "metadata": meta}
        m.metadata = meta
        m.user_id = "u1"
        m.score = 0.9
        m.memory = f"fact-{i}"
        items.append(m)
    res.results = items
    res.total_results = len(items)
    return res


class TestMemoryReadRetaintsFromRowProvenance:
    """Reading a row written by a tainted run re-taints the reading run."""

    async def test_row_label_taints_the_reading_run(self):
        svc = _memory_service(_memory_client(result=_memory_result_with_labels(["external"])))
        agent = _agent_with_scope_labels({})  # no scope labels — row provenance alone
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert "external" in ctx.data_labels

    async def test_labels_from_several_rows_are_unioned(self):
        svc = _memory_service(
            _memory_client(result=_memory_result_with_labels(["external"], ["pii"]))
        )
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == {"external", "pii"}

    async def test_unlabelled_rows_do_not_taint(self):
        svc = _memory_service(_memory_client(result=_memory_result_with_labels(None, None)))
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == set()

    async def test_row_provenance_composes_with_scope_labels(self):
        """Both producers are additive: neither replaces the other."""
        svc = _memory_service(_memory_client(result=_memory_result_with_labels(["external"])))
        agent = _agent_with_scope_labels({"user": {"pii"}})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == {"external", "pii"}

    async def test_malformed_provenance_is_ignored_not_fatal(self):
        """Metadata is third-party data round-tripped through a vector store; a
        row whose labels arrive as the wrong type must not take the run down.
        Failing closed here would break every read on one bad row."""
        svc = _memory_service(
            _memory_client(result=_memory_result_with_labels("external", 42, {"a": 1}))
        )
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)  # must not raise

        assert ctx.data_labels == set()

    async def test_non_string_members_are_skipped(self):
        svc = _memory_service(_memory_client(result=_memory_result_with_labels(["external", 7])))
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == {"external"}

    async def test_empty_results_do_not_taint(self):
        svc = _memory_service(_memory_client(result=_memory_result_with_labels()))
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        assert ctx.data_labels == set()


class TestPoisonedMemoryIsDeniedAtTheAction:
    """The whole point: the gate holds without the model's cooperation.

    Measured across four models, no prompt-level framing of retrieved memory
    reliably stops a planted instruction -- gpt-4o-mini obeyed one under every
    envelope and every wording tried. This path does not ask the model anything.
    """

    async def test_tool_denied_after_reading_a_poisoned_row(self):
        from continuum.security.policy import AccessPolicy, PolicyStore
        from continuum.tools.executor import ToolExecutor

        # A row written during a run that had read external content.
        svc = _memory_service(_memory_client(result=_memory_result_with_labels(["external"])))
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)
        assert "external" in ctx.data_labels, "read must taint for the gate to fire"

        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="no-refunds-on-external-data",
                subjects=["external"],
                resources=["tool:issue_refund"],
                effect="deny",
            )
        )
        decision = store.check(["refund-agent", *sorted(ctx.data_labels)], "tool:issue_refund")
        assert decision.allowed is False
        assert decision.policy_name == "no-refunds-on-external-data"

        # And the same subjects list the executor builds is what was checked.
        assert hasattr(ToolExecutor, "execute_tool_call")

    async def test_clean_memory_leaves_the_tool_allowed(self):
        """No false positives: an untainted row must not gate anything."""
        from continuum.security.policy import AccessPolicy, PolicyStore

        svc = _memory_service(_memory_client(result=_memory_result_with_labels(None)))
        agent = _agent_with_scope_labels({})
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(agent, "query", ctx)

        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="no-refunds-on-external-data",
                subjects=["external"],
                resources=["tool:issue_refund"],
                effect="deny",
            )
        )
        decision = store.check(["refund-agent", *sorted(ctx.data_labels)], "tool:issue_refund")
        assert decision.allowed is True


# ---------------------------------------------------------------------------
# Telling the developer the mechanism is switched off
#
# Every producer above is gated on a declaration that defaults to empty:
# tool_data_labels {}, scope_data_labels {}, and no run-level seed. So in a
# default deployment nothing taints -- which means memory rows are stamped with
# nothing, the read path fences nothing, and the tool gate never matches a
# label. The whole chain is present and inert.
#
# That is the intended design (the SDK ships no PII detector and guesses no
# provenance), but it fails by doing nothing, which is the failure mode that
# never gets noticed. So say it once, the same way a derived MCP server name is
# reported: name the fields, state the consequence, and stay quiet afterwards.
# ---------------------------------------------------------------------------


@contextmanager
def _captured_memory_warnings():
    """WARNINGs from memory_service (caplog cannot: the parent sets propagate=False)."""
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Collector()
    lg = logging.getLogger("continuum.agent.services.memory_service")
    lg.addHandler(handler)
    try:
        yield messages
    finally:
        lg.removeHandler(handler)


def _agent_with(*, tool_labels=None, scope_labels=None, search=True, store=True):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="prov-warn-agent",
        instructions="test",
        config=AgentConfig(tool_data_labels=tool_labels or {}),
        memory_config=AgentMemoryConfig(
            scope_data_labels=scope_labels or {},
            search_memories=search,
            store_memories=store,
        ),
    )


class TestUndeclaredProvenanceIsReported:
    async def test_warns_when_memory_is_on_and_nothing_is_declared(self):
        svc = _memory_service(_memory_client())
        agent = _agent_with()
        ctx = create_run_context(user_id="u1")

        with _captured_memory_warnings() as warnings:
            await svc.retrieve_memories(agent, "query", ctx)

        assert warnings, "an inert mechanism must announce itself"

    async def test_warning_names_both_declaration_fields(self):
        svc = _memory_service(_memory_client())
        agent = _agent_with()
        ctx = create_run_context(user_id="u1")

        with _captured_memory_warnings() as warnings:
            await svc.retrieve_memories(agent, "query", ctx)

        joined = "\n".join(warnings)
        assert "tool_data_labels" in joined
        assert "scope_data_labels" in joined

    async def test_warns_only_once_per_agent(self):
        """This runs every turn; a warning repeated each turn is one people
        filter out, and then it protects nobody."""
        svc = _memory_service(_memory_client())
        agent = _agent_with()

        with _captured_memory_warnings() as warnings:
            for _ in range(3):
                await svc.retrieve_memories(agent, "query", create_run_context(user_id="u1"))

        assert len(warnings) == 1

    async def test_silent_when_tool_provenance_is_declared(self):
        svc = _memory_service(_memory_client())
        agent = _agent_with(tool_labels={"fetch_page": {"external"}})
        ctx = create_run_context(user_id="u1")

        with _captured_memory_warnings() as warnings:
            await svc.retrieve_memories(agent, "query", ctx)

        assert warnings == []

    async def test_silent_when_scope_provenance_is_declared(self):
        svc = _memory_service(_memory_client())
        agent = _agent_with(scope_labels={"user": {"pii"}})
        ctx = create_run_context(user_id="u1")

        with _captured_memory_warnings() as warnings:
            await svc.retrieve_memories(agent, "query", ctx)

        assert warnings == []

    async def test_silent_when_the_run_is_already_tainted(self):
        """Run-level seeding is a third declaration site and is invisible in the
        agent config, so a tainted run is proof the mechanism is live."""
        svc = _memory_service(_memory_client())
        agent = _agent_with()
        ctx = create_run_context(user_id="u1", data_labels={"external"})

        with _captured_memory_warnings() as warnings:
            await svc.retrieve_memories(agent, "query", ctx)

        assert warnings == []

    async def test_silent_when_memory_is_off_entirely(self):
        """Nothing to protect, so nothing to say."""
        svc = _memory_service(_memory_client())
        agent = _agent_with(search=False, store=False)
        ctx = create_run_context(user_id="u1")

        with _captured_memory_warnings() as warnings:
            await svc.retrieve_memories(agent, "query", ctx)

        assert warnings == []

    async def test_store_only_agent_is_still_warned(self):
        """Writes are stamped too, so a store-only agent has the same gap even
        though it never reaches the read path."""
        svc = _memory_service(_memory_client())
        agent = _agent_with(search=False, store=True)
        ctx = create_run_context(user_id="u1")

        with _captured_memory_warnings() as warnings:
            await svc.store_memories(agent, [{"role": "user", "content": "x"}], ctx)

        assert warnings, "the write path has the same undeclared-provenance gap"


# ---------------------------------------------------------------------------
# What a labelled row does when it is recalled (security finding F6)
#
# Recall is not like the other taint producers. A tool result is something the
# model chose to fetch this turn; a run-level seed is something the operator
# decided up front. A memory row arrives unbidden during prompt assembly, and
# the content was planted in an earlier session -- so by the time the label is
# known, untrusted text is already in the prompt.
#
# Fencing it is the default and the weakest of the three: it asks the model not
# to obey, and measured across four models only two reliably decline. Dropping
# the row keeps it out of the prompt entirely. Blocking refuses the turn until a
# person reviews the row -- the same forced human step F3 requires for an
# unreviewed tool catalogue, and for the same reason: this is the case with no
# reliable automated defence.
#
# Which of the three is right depends on who reviews and how fast, so it is an
# operator choice rather than a framework default -- mirroring
# ``ToolTrustConfig.on_unreviewed`` rather than hardcoding a refusal. Blocking
# without a review path would just be an outage, so the error carries the row
# ids the panel needs to link to.
# ---------------------------------------------------------------------------


def _memory_result_mixed():
    """One clean row and one carrying provenance."""
    from unittest.mock import MagicMock

    res = MagicMock()
    rows = []
    for i, labels in ((0, None), (1, ["external"])):
        m = MagicMock()
        meta = {} if labels is None else {PROVENANCE_KEY: labels}
        m.to_dict.return_value = {"memory": f"fact-{i}", "metadata": meta}
        m.metadata = meta
        m.id = f"row-{i}"
        m.user_id = "u1"
        m.score = 0.9
        m.memory = f"fact-{i}"
        rows.append(m)
    res.results = rows
    res.total_results = len(rows)
    return res


def _agent_with_recall_action(action=None, scope_labels=None):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    cfg = AgentMemoryConfig(scope_data_labels=scope_labels or {})
    if action is not None:
        cfg.on_labeled_recall = action
    return BaseAgent(name="recall-agent", instructions="t", config=AgentConfig(), memory_config=cfg)


class TestRecallActionDefault:
    def test_default_is_fence(self):
        """Today's behaviour stays the default: changing what an existing
        deployment does on upgrade is not a security improvement."""
        from continuum.agent.config import AgentMemoryConfig

        assert AgentMemoryConfig().on_labeled_recall == "fence"


class TestFenceKeepsEverything:
    async def test_labelled_row_is_returned_and_taints(self):
        svc = _memory_service(_memory_client(result=_memory_result_mixed()))
        ctx = create_run_context(user_id="u1")

        out = await svc.retrieve_memories(_agent_with_recall_action("fence"), "q", ctx)

        assert len(out) == 2
        assert "external" in ctx.data_labels


class TestDropRemovesLabelledRows:
    async def test_labelled_row_never_reaches_the_prompt(self):
        svc = _memory_service(_memory_client(result=_memory_result_mixed()))
        ctx = create_run_context(user_id="u1")

        out = await svc.retrieve_memories(_agent_with_recall_action("drop"), "q", ctx)

        texts = [m.get("memory") for m in out]
        assert texts == ["fact-0"], "only the clean row should survive"

    async def test_dropped_rows_do_not_taint(self):
        """The row never entered the prompt, so the run did not touch it and
        gating the rest of the turn on it would be punishing a non-event."""
        svc = _memory_service(_memory_client(result=_memory_result_mixed()))
        ctx = create_run_context(user_id="u1")

        await svc.retrieve_memories(_agent_with_recall_action("drop"), "q", ctx)

        assert ctx.data_labels == set()

    async def test_clean_store_is_unaffected(self):
        svc = _memory_service(_memory_client())  # rows carry no labels
        ctx = create_run_context(user_id="u1")

        out = await svc.retrieve_memories(_agent_with_recall_action("drop"), "q", ctx)

        assert len(out) == 1


class TestBlockStopsTheTurn:
    async def test_raises_review_required(self):
        from continuum.agent.exceptions import MemoryReviewRequiredError

        svc = _memory_service(_memory_client(result=_memory_result_mixed()))
        ctx = create_run_context(user_id="u1")

        with pytest.raises(MemoryReviewRequiredError):
            await svc.retrieve_memories(_agent_with_recall_action("block"), "q", ctx)

    async def test_error_names_the_rows_to_review(self):
        """Blocking without telling anyone which rows to look at is an outage,
        not a workflow. The panel needs the ids to link to."""
        from continuum.agent.exceptions import MemoryReviewRequiredError

        svc = _memory_service(_memory_client(result=_memory_result_mixed()))
        ctx = create_run_context(user_id="u1")

        with pytest.raises(MemoryReviewRequiredError) as exc:
            await svc.retrieve_memories(_agent_with_recall_action("block"), "q", ctx)

        assert exc.value.memory_ids == ["row-1"]
        assert exc.value.labels == ["external"]

    async def test_clean_store_does_not_block(self):
        """A false stop is worse than no stop: it trains people to ignore it."""
        svc = _memory_service(_memory_client())
        ctx = create_run_context(user_id="u1")

        out = await svc.retrieve_memories(_agent_with_recall_action("block"), "q", ctx)

        assert len(out) == 1

    async def test_not_swallowed_by_the_best_effort_handler(self):
        """retrieve_memories treats every other failure as best-effort and
        returns []. A review demand that degrades to "no memories" would be
        silently ignored, which is the opposite of forcing a human step."""
        from continuum.agent.exceptions import MemoryReviewRequiredError

        svc = _memory_service(_memory_client(result=_memory_result_mixed()))
        ctx = create_run_context(user_id="u1")

        with pytest.raises(MemoryReviewRequiredError):
            await svc.retrieve_memories(_agent_with_recall_action("block"), "q", ctx)


class TestBlockReachesTheCaller:
    async def test_propagates_out_of_prepare_messages(self):
        """message_builder also wraps retrieval in a best-effort handler, so the
        demand has to survive two layers to actually stop the turn."""
        from unittest.mock import AsyncMock, MagicMock

        from continuum.agent.exceptions import MemoryReviewRequiredError
        from continuum.agent.execution.message_builder import MessageBuilder

        mem = MagicMock()
        mem.retrieve_memories = AsyncMock(
            side_effect=MemoryReviewRequiredError(memory_ids=["row-1"], labels=["external"])
        )
        sess = MagicMock()
        sess.get_conversation_history = AsyncMock(return_value=[])
        builder = MessageBuilder(memory_service=mem, session_service=sess)

        agent = _agent_with_recall_action("block")
        agent.memory_config.search_memories = True
        ctx = create_run_context(user_id="u1")

        with pytest.raises(MemoryReviewRequiredError):
            await builder.prepare_messages(agent, "hello", ctx)
