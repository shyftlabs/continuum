"""Layer A for the clinic's MCP server-trust story (finding F3).

The SDK-level machinery -- digest drift, invisible-character stripping, cache
invalidation on reconnect -- is covered by tests/unit/tools/test_mcp_tool_catalog.py
and test_tool_pinning.py. This file asserts the clinic *project* is wired to use
it, and that its policy store bounds a server that was hostile from the very
first connect.

That last case is the one no digest can catch: nothing "changed", so the
tripwire is correctly silent. What contains it is authorisation -- the model may
be fully persuaded by a poisoned description and still fail to reach the tool.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest
from mcp.types import Tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

CLINIC_DIR = pathlib.Path(__file__).resolve().parents[2] / "playground" / "data-label-clinic"


def _load(module: str):
    """Import a clinic module by file path, under a name unique to this project.

    Not sys.path + `import config`: several playgrounds ship a `config.py` and a
    `server.py`, so the bare name resolves to whichever test imported first and
    then stays cached in sys.modules. In isolation this file passed; in the full
    suite it silently got local-shop's modules. Same failure shape as the bugs
    under test -- a name quietly binding to the wrong thing.
    """
    unique = f"_clinic_{module}"
    if unique in sys.modules:
        del sys.modules[unique]
    spec = importlib.util.spec_from_file_location(unique, CLINIC_DIR / f"{module}.py")
    assert spec and spec.loader, module
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    spec.loader.exec_module(mod)
    return mod


# The four tools the clinic legitimately exposes (server.py).
CLINIC_TOOLS = [
    "tool:clinic__clinic_info",
    "tool:clinic__lookup_patient",
    "tool:clinic__send_referral_email",
    "tool:clinic__web_lookup",
]

# Non-tool resources the SDK's other gates check, so a fail-closed store must
# still permit them or the agent cannot run at all:
#   llm:{model}          llm/client.py
#   memory:{scope}       memory/client.py  (the clinic also checks memory:{user_id})
#   telemetry            run_finalizer.py, data_redaction.py
#   session              session_service.py
UNTAINTED_RESOURCES = [
    "llm:gpt-4o",
    "llm:gpt-4o-mini",
    "memory:user",
    "memory:alice",
    "telemetry",
    "session",
]


@pytest.fixture
def store():
    return _load("config").build_policy_store()


@pytest.fixture
def agent_subject():
    return _load("config").default_config.agent_name


class TestClinicPolicyIsFailClosed:
    """A tool the attacker invents after the rules were written must not run.

    The clinic denies exfiltration by naming two tools. That is a blocklist: it
    only blocks what was thought of in advance, and with default_effect="allow"
    an unlisted tool is permitted. A poisoned description naming any other tool
    walks straight through.
    """

    def test_untainted_run_can_still_use_every_clinic_tool(self, store, agent_subject):
        for resource in CLINIC_TOOLS:
            assert store.check([agent_subject], resource).allowed is True, resource

    def test_untainted_run_can_still_reach_every_non_tool_gate(self, store, agent_subject):
        for resource in UNTAINTED_RESOURCES:
            assert store.check([agent_subject], resource).allowed is True, resource

    def test_phi_run_still_trips_all_five_gates(self, store, agent_subject):
        phi = _load("config").PHI

        subjects = [agent_subject, phi]
        denied = {
            "llm:gpt-4o": "phi-no-cloud-model",
            "tool:clinic__send_referral_email": "phi-no-exfiltration-tools",
            "tool:clinic__web_lookup": "phi-no-exfiltration-tools",
            "memory:alice": "phi-never-persisted",
            "telemetry": "phi-redact-telemetry",
            "session": "phi-no-short-term",
        }
        for resource, policy_name in denied.items():
            decision = store.check(subjects, resource)
            assert decision.allowed is False, resource
            assert decision.policy_name == policy_name, (resource, decision.policy_name)

    def test_phi_run_may_still_use_the_onprem_model(self, store, agent_subject):
        """The deny is exact-match on the cloud tier; the fallback must survive."""
        phi = _load("config").PHI

        assert store.check([agent_subject, phi], "llm:gpt-4o-mini").allowed is True

    def test_attacker_invented_tool_is_denied_for_an_untainted_run(self, store, agent_subject):
        """The first-contact case. A poisoned description tells the model to call
        a tool the policy author never heard of; the call must not execute."""
        assert store.check([agent_subject], "tool:clinic__fetch_manifest").allowed is False

    def test_attacker_invented_tool_is_denied_for_a_phi_run(self, store, agent_subject):
        phi = _load("config").PHI

        assert store.check([agent_subject, phi], "tool:clinic__read_file").allowed is False

    def test_a_tool_on_some_other_server_is_denied(self, store, agent_subject):
        """Namespacing means a second server's tools carry a different prefix, and
        nothing in this store allows it."""
        assert store.check([agent_subject], "tool:evil__read_file").allowed is False


class TestClinicAgentPinsItsCatalogue:
    def test_config_exposes_a_pin_path(self):
        assert _load("config").default_config.tool_pin_path, "clinic has no tool_pin_path configured"

    def test_agent_passes_the_pin_path_to_the_server(self):
        """Without tool_pin_path the digest tripwire records nothing, so a
        post-approval description change is never reported."""
        import ast

        tree = ast.parse((CLINIC_DIR / "agent.py").read_text())
        ctors = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "MCPServerStreamableHttp"
        ]
        assert ctors, "no MCPServerStreamableHttp construction found"
        for call in ctors:
            kwargs = {kw.arg for kw in call.keywords}
            assert "tool_pin_path" in kwargs, f"line {call.lineno} does not pin its catalogue"


class TestClinicPoisonedServerMode:
    """server.py serves a hostile catalogue under CLINIC_POISON=1.

    Layer B (TESTING_GUIDE.md) drives the two live scenarios from this switch:
    pin clean then poison (drift detected), or pin already-poisoned (no drift --
    the limit).
    """

    def _descriptions(self, poisoned: bool) -> dict[str, str]:
        """The descriptions the server actually SERVES, from FastMCP's registry.

        Deliberately not ``fn.__doc__``. FastMCP captures the description when
        ``@mcp.tool()`` runs, so a docstring mutated afterwards changes the
        function but not the catalogue -- an earlier version of this poison did
        exactly that, and a test reading ``__doc__`` passed while the live server
        served clean text. Read the same source the client will.
        """
        import asyncio

        os.environ["CLINIC_POISON"] = "1" if poisoned else "0"
        try:
            server = _load("server")
            tools = asyncio.run(server.mcp.list_tools())
            return {t.name: (t.description or "") for t in tools}
        finally:
            os.environ.pop("CLINIC_POISON", None)

    def test_clean_mode_has_no_injected_instruction(self):
        for name, doc in self._descriptions(poisoned=False).items():
            assert "IMPORTANT:" not in doc, name

    def test_poison_mode_injects_an_instruction_into_a_description(self):
        docs = self._descriptions(poisoned=True)
        assert any("IMPORTANT:" in d for d in docs.values()), docs

    def test_poison_mode_adds_a_tool_the_policy_never_named(self):
        """The exfiltration target. Its name is absent from the clinic's deny
        list, which is exactly why a blocklist cannot contain it."""
        clean = set(self._descriptions(poisoned=False))
        poisoned = set(self._descriptions(poisoned=True))
        added = poisoned - clean
        assert added, "poison mode adds no new tool"
        for name in added:
            assert f"tool:clinic__{name}" not in CLINIC_TOOLS

    def test_poisoning_changes_the_digest(self):
        """Layer B scenario 1 depends on this: if the digests matched, the
        tripwire would have nothing to report."""
        from continuum.tools.mcp import _tool_digest

        clean = self._descriptions(poisoned=False)
        poisoned = self._descriptions(poisoned=True)
        changed = [
            n
            for n in clean
            if n in poisoned
            and _tool_digest(_as_tool(n, clean[n])) != _tool_digest(_as_tool(n, poisoned[n]))
        ]
        assert changed, "no shared tool's digest changed; scenario 1 would not fire"

    def test_the_poison_is_visible_in_the_inspect_output(self):
        """First-contact poisoning is caught by human review or not at all, so
        `continuum mcp inspect` must print the injected sentence in full."""
        from continuum.tools.pinning import format_tool_catalog

        docs = self._descriptions(poisoned=True)
        tools = [_as_tool(n, d) for n, d in docs.items()]
        out = format_tool_catalog("clinic", tools)
        assert "IMPORTANT:" in out
        assert "tool:clinic__" in out


def _as_tool(name: str, description: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )
