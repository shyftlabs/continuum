"""Provenance labels must survive tool namespacing.

``AgentConfig.tool_data_labels`` is keyed by tool name, but once a server is
given a ``name=`` every tool reaches the runtime as ``<server>__<tool>``. An
exact-match lookup then misses, no taint is applied, and every downstream
protection keyed on that taint quietly stops applying -- the model keeps its
cloud route, exfiltration tools stay allowed, telemetry stays unredacted.

Observed live in the data-label clinic: ``lookup_patient`` returned an SSN and
a diagnosis while the run reported ``clean`` and the "PHI may not use
exfiltration tools" rule never fired, so a summary was emailed to an external
address with no gate tripped. Both taint sites were dead -- including the
batch pre-taint written specifically to stop that exact sequence.

Resolution accepts either spelling, because neither alone is right:

  * the **raw** name is what a user knows (it is in their server's source) and
    is the only usable form when no ``name=`` was given, since the derived one
    is ``streamable_http_http_localhost_8911_mcp__lookup_patient`` and changes
    with the URL;
  * the **namespaced** name is needed when two servers expose the same tool.

This mirrors ToolContextConfig, which already compares raw names for the same
stated reason: a namespaced name "would break whenever namespace_tools changed".
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.utils.context_utils import create_run_context

pytestmark = pytest.mark.unit


def _tool_call(name: str, call_id: str = "tc-1"):
    from continuum.llm.types import FunctionCall, ToolCall

    return ToolCall(id=call_id, type="function", function=FunctionCall(name=name, arguments="{}"))


def _agent(tool_labels: dict[str, set[str]], registry_names: list[str]):
    """An agent whose registry uses `registry_names` (i.e. namespaced keys)."""
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig

    agent = BaseAgent(
        name="prov-agent", instructions="test", config=AgentConfig(tool_data_labels=tool_labels)
    )
    executor = MagicMock()
    executor.tool_registry = {n: (MagicMock(name="server"), object()) for n in registry_names}
    executor.execute_tool_calls = AsyncMock(
        return_value=[{"role": "tool", "tool_call_id": "tc-1", "content": "done"}]
    )
    agent.tool_executor = executor
    agent.on_tool_call = None
    return agent


def _service():
    from continuum.agent.services.tool_service import ToolService

    return ToolService(tool_executor=None)


@contextmanager
def _captured_logs():
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector()
    logger = logging.getLogger("continuum.agent.services.tool_service")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _warnings(records) -> list[str]:
    return [r.getMessage() for r in records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestRawNameStillTaints:
    """The regression: a raw key against a namespaced call."""

    async def test_single_call(self):
        agent = _agent({"lookup_patient": {"phi"}}, ["clinic__lookup_patient"])
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("clinic__lookup_patient"), ctx)

        assert "phi" in ctx.data_labels

    async def test_batch_pre_taint(self):
        """The pre-taint path, which the batch case depends on."""
        agent = _agent(
            {"lookup_patient": {"phi"}}, ["clinic__lookup_patient", "clinic__send_email"]
        )
        ctx = create_run_context()

        await _service().execute_tools_batch(
            agent,
            [_tool_call("clinic__lookup_patient", "a"), _tool_call("clinic__send_email", "b")],
            ctx,
        )

        assert "phi" in ctx.data_labels

    async def test_survives_a_server_rename(self):
        """The reason resolution beats telling users to write namespaced keys.

        A config holding `clinic__lookup_patient` works until someone renames
        the server, then silently stops tainting -- the same failure, restored.
        """
        agent = _agent({"lookup_patient": {"phi"}}, ["clinic-prod__lookup_patient"])
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("clinic-prod__lookup_patient"), ctx)

        assert "phi" in ctx.data_labels


class TestNamespacedNameAlsoWorks:
    async def test_exact_namespaced_key(self):
        agent = _agent({"clinic__lookup_patient": {"phi"}}, ["clinic__lookup_patient"])
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("clinic__lookup_patient"), ctx)

        assert "phi" in ctx.data_labels

    async def test_namespaced_wins_over_raw(self):
        """Precision must be available when two servers share a tool name."""
        agent = _agent(
            {"lookup_patient": {"public"}, "clinic__lookup_patient": {"phi"}},
            ["clinic__lookup_patient"],
        )
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("clinic__lookup_patient"), ctx)

        assert ctx.data_labels == {"phi"}

    async def test_an_unnamespaced_registry_still_works(self):
        """namespace_tools=False leaves registry keys raw."""
        agent = _agent({"lookup_patient": {"phi"}}, ["lookup_patient"])
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("lookup_patient"), ctx)

        assert "phi" in ctx.data_labels


class TestUndeclaredToolsAreUnaffected:
    async def test_a_tool_with_no_label_does_not_taint(self):
        agent = _agent({"lookup_patient": {"phi"}}, ["clinic__send_email"])
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("clinic__send_email"), ctx)

        assert ctx.data_labels == set()

    async def test_a_suffix_match_is_not_enough(self):
        """`__` is the separator, so only a whole trailing segment counts.

        Matching any suffix would let `clinic__bulk_lookup_patient` inherit
        `lookup_patient`'s labels -- over-tainting is safe, but silently
        labelling the wrong tool makes the declaration untrustworthy.
        """
        agent = _agent({"lookup_patient": {"phi"}}, ["clinic__bulk_lookup_patient"])
        ctx = create_run_context()

        await _service().execute_tool_call(agent, _tool_call("clinic__bulk_lookup_patient"), ctx)

        assert ctx.data_labels == set()


# ---------------------------------------------------------------------------
# Warnings: the silent no-op is the whole hazard
# ---------------------------------------------------------------------------


class TestUnmatchedEntriesAreReported:
    """A label naming no tool protects nothing, and says nothing.

    Precedent: _warn_on_unmatched_context_tool_names exists for capture_from /
    inject_into because "a typo or a stale name is a pure no-op ... no error,
    no log". tool_data_labels has the same shape and guards more.
    """

    async def test_a_name_matching_nothing_warns(self):
        agent = _agent({"lookup_patinet": {"phi"}}, ["clinic__lookup_patient"])
        ctx = create_run_context()

        with _captured_logs() as records:
            await _service().execute_tools_batch(agent, [_tool_call("clinic__lookup_patient")], ctx)

        assert any("lookup_patinet" in w for w in _warnings(records)), _warnings(records)

    async def test_a_matching_name_is_silent(self):
        agent = _agent({"lookup_patient": {"phi"}}, ["clinic__lookup_patient"])
        ctx = create_run_context()

        with _captured_logs() as records:
            await _service().execute_tools_batch(agent, [_tool_call("clinic__lookup_patient")], ctx)

        assert _warnings(records) == []

    async def test_reported_once_not_every_batch(self):
        agent = _agent({"nope": {"phi"}}, ["clinic__lookup_patient"])
        ctx = create_run_context()
        svc = _service()
        await svc.execute_tools_batch(agent, [_tool_call("clinic__lookup_patient")], ctx)

        with _captured_logs() as records:
            await svc.execute_tools_batch(agent, [_tool_call("clinic__lookup_patient")], ctx)

        assert _warnings(records) == []


class TestAmbiguousRawNamesAreReported:
    """One raw name, two servers: taint both, but say so.

    Over-tainting fails closed, so it is safe -- but labelling an unrelated
    server's tool PHI blocks benign work, and the natural response to that is
    to loosen the policy. Better to name the servers and let the user
    disambiguate.
    """

    async def test_warns_naming_both_matches(self):
        agent = _agent(
            {"lookup_patient": {"phi"}}, ["clinic__lookup_patient", "hr__lookup_patient"]
        )
        ctx = create_run_context()

        with _captured_logs() as records:
            await _service().execute_tools_batch(agent, [_tool_call("clinic__lookup_patient")], ctx)

        warning = next((w for w in _warnings(records) if "lookup_patient" in w), "")
        assert "clinic__lookup_patient" in warning and "hr__lookup_patient" in warning, warning

    async def test_still_taints_rather_than_giving_up(self):
        agent = _agent(
            {"lookup_patient": {"phi"}}, ["clinic__lookup_patient", "hr__lookup_patient"]
        )
        ctx = create_run_context()

        with _captured_logs():
            await _service().execute_tool_call(agent, _tool_call("clinic__lookup_patient"), ctx)

        assert "phi" in ctx.data_labels, "ambiguity must fail closed, not silently unlabel"
