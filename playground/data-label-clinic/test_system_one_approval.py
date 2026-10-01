"""System One (Jev) tool approval in the clinic, behind a header toggle.

Like gateway-multi-agent-shop's System One modes: available only when .env names
a backend in SYSTEM_ONE_BACKEND, off by default, and with it off the clinic is
exactly as before (approval per CLINIC_APPROVAL). With it on, a gated tool call
goes to system_one_approval_handler first: rule 1 (the tool is eligible) and
rule 2 (the run carries no data label) are checked before anything is sent, and
only a call Jev scores below P(risky) 0.1 is auto-approved. Everything else goes
to the person CLINIC_APPROVAL names -- the browser prompt when that is "off".

Two things the clinic needed that the shop did not:

* Its policy store is fail-closed (default_deny). Every System One call is
  checked against ``system_one:<egress>:<backend>:<model>``, which the baseline
  did not allow, so Jev could never be reached: every gated call would escalate
  with "the risk classifier could not answer". The exact resource is now allowed
  (no glob, the clinic's rule), and PHI runs are denied every system_one
  resource -- a second line behind rule 2, so PHI never reaches Jev even if the
  label rule were switched off.
* Only one tool was gated, and a harmless one. With the toggle on,
  send_referral_email is gated too, so a risky call (mail to an outside address)
  can be seen going to a person.

Run by path:  pytest playground/data-label-clinic/test_system_one_approval.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

CLINIC_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CLINIC_DIR.parents[1] / "src"))
sys.path.insert(0, str(CLINIC_DIR.parents[1] / "tests" / "unit" / "system_one"))

JEV = "openrouter:typesafe/jev-1.13"
JEV_RESOURCE = "system_one:remote:openrouter:typesafe/jev-1.13"
AGENT = "clinic-intake-assistant"
GATED = {"pharmacy__check_interactions", "clinic__send_referral_email"}


def _load(module: str):
    """Import a clinic module by path under a unique name (see test_server_trust)."""
    unique = f"_clinic_s1_{module}"
    sys.modules.pop(unique, None)
    spec = importlib.util.spec_from_file_location(unique, CLINIC_DIR / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    sys.path.insert(0, str(CLINIC_DIR))
    shared = ("config", "server", "pharmacy_server", "approval_ui", "agent")
    preexisting = {n for n in shared if n in sys.modules}
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(CLINIC_DIR))
        for n in shared:
            if n in sys.modules and n not in preexisting:
                del sys.modules[n]
    return mod


@pytest.fixture
def backend(monkeypatch):
    """SYSTEM_ONE_BACKEND as .env sets it. Returns a setter; None means unset."""
    from continuum.config import settings
    from continuum.core.container import reset_container
    from continuum.system_one.registry import clear_classifier_cache

    def set_backend(spec):
        monkeypatch.setattr(settings, "system_one_backend", spec)
        monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
        reset_container()
        clear_classifier_cache()

    monkeypatch.setattr(settings, "system_one_disabled", False)
    monkeypatch.delenv("CLINIC_APPROVAL", raising=False)
    set_backend(None)
    yield set_backend
    clear_classifier_cache()
    reset_container()


def _jev(p_risky):
    """A scripted backend in the container, scoring every call ``p_risky``."""
    from conftest import FakeClassifier

    from continuum.core.container import get_container

    fake = FakeClassifier(answer=lambda qid, q: {"true": p_risky, "false": 1 - p_risky})
    get_container().set_system_one_classifier(fake)
    return fake


def _request(tool="pharmacy__check_interactions", labels=()):
    from continuum.agent.approval import ToolApprovalRequest

    return ToolApprovalRequest(
        tool_name=tool,
        arguments={"medications": ["warfarin", "ibuprofen"]},
        agent_name=AGENT,
        run_id="r",
        data_labels=frozenset(labels),
    )


class _Logs:
    def __init__(self, name):
        self.name, self.records = name, []

    def __enter__(self):
        outer = self

        class H(logging.Handler):
            def emit(self, record):
                outer.records.append(record.getMessage())

        self._h = H()
        logging.getLogger(self.name).addHandler(self._h)
        return self

    def __exit__(self, *exc):
        logging.getLogger(self.name).removeHandler(self._h)


# =============================================================================
# Availability
# =============================================================================


class TestAvailability:
    def test_without_a_backend_it_is_not_offered(self, backend):
        status = _load("config").system_one_status()
        assert status["configured"] is False

    def test_with_a_backend_it_is(self, backend):
        backend(JEV)
        status = _load("config").system_one_status()
        assert status["configured"] is True and status["backend"] == JEV

    def test_a_backend_without_its_key_is_not_ready(self, backend, monkeypatch):
        """A backend named in .env but missing its API key cannot answer: every
        gated call would escalate, while the UI claimed System One was deciding."""
        from continuum.config import settings

        backend(JEV)
        monkeypatch.setattr(settings, "openrouter_api_key", None)
        status = _load("config").system_one_status()
        assert status["ready"] is False
        assert "OPENROUTER_API_KEY" in status["problem"]

    def test_a_backend_with_its_key_is_ready(self, backend):
        backend(JEV)
        status = _load("config").system_one_status()
        assert status["ready"] is True and status["problem"] is None

    def test_without_a_backend_it_is_not_ready(self, backend):
        status = _load("config").system_one_status()
        assert status["ready"] is False
        assert "SYSTEM_ONE_BACKEND" in status["problem"]

    def test_the_handler_cannot_be_built_without_a_backend(self, backend):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError):
            _load("config").build_system_one_approval_handler()


# =============================================================================
# Policy: the fail-closed store must let Jev be reached, never for PHI
# =============================================================================


class TestThePolicyLetsJevBeReached:
    def test_the_exact_backend_resource_is_allowed_for_a_clean_run(self, backend):
        backend(JEV)
        store = _load("config").build_policy_store()
        assert store.check([AGENT], JEV_RESOURCE).allowed is True

    def test_without_a_backend_nothing_system_one_is_allowed(self, backend):
        store = _load("config").build_policy_store()
        assert store.check([AGENT], JEV_RESOURCE).allowed is False

    def test_no_other_system_one_resource_is_allowed(self, backend):
        """The clinic names resources exactly, never by glob."""
        backend(JEV)
        store = _load("config").build_policy_store()
        assert store.check([AGENT], "system_one:remote:jev:jev-latest").allowed is False

    def test_a_phi_run_is_denied_every_system_one_resource(self, backend):
        """The second line behind rule 2."""
        backend(JEV)
        config = _load("config")
        store = config.build_policy_store()
        assert store.check([AGENT, config.PHI], JEV_RESOURCE).allowed is False


# =============================================================================
# The handler: rules first, Jev only for clean calls, people for the rest
# =============================================================================


class TestTheApprovalHandler:
    def _handler(self, person):
        return _load("config").build_system_one_approval_handler(escalate_to=person)

    def test_a_low_risk_clean_call_is_auto_approved_by_jev(self, backend):
        backend(JEV)
        _jev(0.04)
        person = AsyncMock()
        decision = asyncio.run(self._handler(person)(_request()))
        assert decision.approved is True
        assert decision.reviewer.startswith("system_one:")
        person.assert_not_awaited()

    def test_a_risky_call_goes_to_the_person(self, backend):
        from continuum.agent.approval import ToolApprovalDecision

        backend(JEV)
        _jev(0.9)
        person = AsyncMock(return_value=ToolApprovalDecision(approved=False, reviewer="ui"))
        decision = asyncio.run(self._handler(person)(_request("clinic__send_referral_email")))
        person.assert_awaited_once()
        assert decision.reviewer == "ui"

    def test_a_phi_run_goes_to_the_person_and_jev_is_never_asked(self, backend):
        from continuum.agent.approval import ToolApprovalDecision

        backend(JEV)
        fake = _jev(0.01)
        person = AsyncMock(return_value=ToolApprovalDecision(approved=True, reviewer="ui"))
        config = _load("config")
        asyncio.run(self._handler(person)(_request(labels={config.PHI})))
        person.assert_awaited_once()
        assert fake.calls == [], "PHI must not reach Jev"

    def test_both_gated_tools_are_eligible(self, backend):
        backend(JEV)
        _jev(0.04)
        for tool in GATED:
            decision = asyncio.run(self._handler(AsyncMock())(_request(tool)))
            assert decision.approved is True, tool

    def test_the_person_is_the_one_clinic_approval_names(self, backend, monkeypatch):
        backend(JEV)
        monkeypatch.setenv("CLINIC_APPROVAL", "deny")
        config = _load("config")
        assert config.system_one_escalation_target() is config.build_approval_handler()

    def test_with_clinic_approval_off_the_person_is_the_browser_prompt(self, backend):
        backend(JEV)
        config = _load("config")
        target = config.system_one_escalation_target()
        assert target.__name__ == "ui_approval_handler"


# =============================================================================
# The agent: the toggle applies per request, and restores
# =============================================================================


def _agent():
    agent_mod = _load("agent")
    clinic = agent_mod.ClinicAgent.__new__(agent_mod.ClinicAgent)
    clinic.config = _load("config").default_config
    clinic._agent = MagicMock()
    clinic._agent.config.tool_approval = {"pharmacy__check_interactions"}
    clinic._agent.config.approval_handler = "the CLINIC_APPROVAL handler"
    clinic._agent.config.output_scanners = []
    clinic._system_one_handler = "the system one handler"
    return clinic


class TestTheToggleOnTheAgent:
    def test_on_gates_both_tools_through_system_one(self, backend):
        backend(JEV)
        clinic = _agent()
        clinic._apply_system_one(True)
        assert clinic._agent.config.tool_approval >= GATED
        assert clinic._agent.config.approval_handler == "the system one handler"

    def test_off_restores_exactly_what_clinic_approval_set(self, backend):
        backend(JEV)
        clinic = _agent()
        clinic._apply_system_one(True)
        clinic._apply_system_one(False)
        assert clinic._agent.config.tool_approval == {"pharmacy__check_interactions"}
        assert clinic._agent.config.approval_handler == "the CLINIC_APPROVAL handler"

    def test_the_turns_decisions_reach_the_gate_panel(self, backend):
        """Each auto-approval or escalation the handler logs shows as a gate event."""
        backend(JEV)
        clinic = _agent()
        lines = clinic._system_one_events(
            [
                "Tool 'pharmacy__check_interactions' auto-approved by "
                "system_one:openrouter:typesafe/jev-1.13 (P(risky)=0.040 < 0.1)",
                "Tool 'pharmacy__check_interactions' escalated: the run carries data label(s) ['phi']",
                "unrelated line",
            ]
        )
        assert len(lines) == 2
        assert all(line.startswith("🤖 SYSTEM ONE") for line in lines)
        assert "auto-approved" in lines[0] and "data label(s) ['phi']" in lines[1]

    def test_the_kill_switch_is_said(self, backend, monkeypatch):
        from continuum.config import settings

        backend(JEV)
        monkeypatch.setattr(settings, "system_one_disabled", True)
        lines = _agent()._system_one_events([])
        assert any("SYSTEM_ONE_DISABLED" in line for line in lines)


# =============================================================================
# The web UI
# =============================================================================


class TestTheWebUI:
    def _web(self):
        return _load("web")

    def test_the_request_carries_the_toggle_off_by_default(self, backend):
        assert self._web().ChatRequest(message="hi").system_one is False

    def test_the_toggle_is_offered_with_the_backend_named(self, backend):
        backend(JEV)
        page = asyncio.run(self._web().index())
        assert 'id="s1-toggle"' in page and JEV in page
        assert 'id="s1-toggle" disabled' not in page

    def test_without_a_backend_it_is_shown_disabled(self, backend):
        page = asyncio.run(self._web().index())
        assert 'id="s1-toggle" disabled' in page
        assert "SYSTEM_ONE_BACKEND" in page

    def test_without_the_key_it_is_shown_disabled_naming_the_key(self, backend, monkeypatch):
        from continuum.config import settings

        backend(JEV)
        monkeypatch.setattr(settings, "openrouter_api_key", None)
        page = asyncio.run(self._web().index())
        assert 'id="s1-toggle" disabled' in page
        assert "OPENROUTER_API_KEY" in page

    def test_chat_refuses_it_without_the_key(self, backend, monkeypatch):
        from continuum.config import settings

        backend(JEV)
        monkeypatch.setattr(settings, "openrouter_api_key", None)
        web = self._web()
        web._agent = MagicMock()
        web._agent.chat = AsyncMock(side_effect=AssertionError("must not run"))
        out = asyncio.run(web.chat(web.ChatRequest(message="hi", system_one=True)))
        assert "OPENROUTER_API_KEY" in out["response"]

    def test_there_is_a_chip_for_a_risky_clean_call(self, backend):
        """Way 2: a clean run (no patient lookup) sending mail outside the clinic,
        which Jev escalates while the harmless interaction check is auto-approved."""
        page = asyncio.run(self._web().index())
        assert "Email our clinic hours to my.friend@gmail.com" in page

    def test_chat_refuses_it_without_a_backend(self, backend):
        web = self._web()
        web._agent = MagicMock()
        web._agent.chat = AsyncMock(side_effect=AssertionError("must not run"))
        out = asyncio.run(web.chat(web.ChatRequest(message="hi", system_one=True)))
        assert "SYSTEM_ONE_BACKEND" in out["response"]

    def test_status_reports_it(self, backend):
        backend(JEV)
        web = self._web()
        web._agent = None
        assert asyncio.run(web.status())["system_one"]["configured"] is True
