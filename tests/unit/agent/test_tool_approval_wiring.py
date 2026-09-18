"""The approval gate has to be reachable from the ordinary run path (F7).

``agent/approval.py`` holds the primitive and ``tools/executor.py`` enforces it,
but until the runner builds ``ToolApprovalSettings`` from the agent's config and
passes it down, the gate is inert by construction: declared tools run
unapproved and nothing says so.

That is the failure mode this whole finding is about -- a control that exists,
reads as configured, and is wired to nothing -- so the wiring is asserted rather
than assumed. There are TWO call sites in ``ToolService``; one covered and one
missed is indistinguishable from working for whichever path a given deployment
happens not to take.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Most of this file is pure wiring and needs no optional extra. One class reaches
# into continuum.temporal, which raises ImportError without `.[temporal]` — a
# developer venv has it and CI does not, so an unguarded class passes locally and
# fails in CI. Skipped at class level rather than module level: the wiring
# assertions must keep running everywhere, since they are the ones guarding
# against a gate that is configured but connected to nothing.
_HAS_TEMPORAL = importlib.util.find_spec("temporalio") is not None


def _agent(*, tools=None, handler=None, timeout=30.0, name="clinic"):
    from continuum.agent.config import AgentConfig

    cfg = AgentConfig()
    cfg.tool_approval = set(tools or ())
    cfg.approval_handler = handler
    cfg.approval_timeout = timeout
    return SimpleNamespace(
        name=name,
        config=cfg,
        policy_store=None,
        tool_executor=None,
        on_tool_call=None,
        on_tool_result=None,
    )


class TestSettingsAreBuiltFromConfig:
    def test_a_configured_agent_produces_settings(self):
        from continuum.agent.approval import build_approval_settings

        async def handler(_req): ...

        settings = build_approval_settings(
            _agent(tools={"send_referral_email"}, handler=handler, timeout=12.0)
        )
        assert settings is not None
        assert settings.tools == frozenset({"send_referral_email"})
        assert settings.handler is handler
        assert settings.timeout == 12.0
        assert settings.agent_name == "clinic"

    def test_an_agent_with_nothing_declared_produces_none(self):
        """Passing settings that gate nothing would mean every call pays a
        lookup for a feature nobody enabled."""
        from continuum.agent.approval import build_approval_settings

        assert build_approval_settings(_agent()) is None

    def test_an_agent_without_a_config_produces_none(self):
        from continuum.agent.approval import build_approval_settings

        assert build_approval_settings(SimpleNamespace(name="bare", config=None)) is None

    def test_declared_without_a_handler_still_produces_settings(self):
        """It must reach the gate precisely BECAUSE it is misconfigured: the gate
        refuses, loudly. Returning None here would turn 'declared but unwired'
        into 'silently ungated', which is the worse of the two."""
        from continuum.agent.approval import build_approval_settings

        settings = build_approval_settings(_agent(tools={"send_referral_email"}))
        assert settings is not None
        assert settings.handler is None


class TestTheRunnerPassesThemDown:
    async def test_the_run_path_passes_approval(self):
        from continuum.agent.services.tool_service import ToolService

        captured = await _run_tool_service(ToolService, streaming=False)
        assert captured.get("approval") is not None, (
            "the non-streaming path does not pass approval settings — declared "
            "tools would run unapproved"
        )
        assert captured["approval"].tools == frozenset({"send_referral_email"})

    async def test_both_call_sites_pass_approval(self):
        """Two sites call execute_tool_calls. One wired and one not is
        indistinguishable from working, for whichever path a deployment does not
        exercise."""
        import inspect

        from continuum.agent.services import tool_service

        src = inspect.getsource(tool_service)
        calls = src.count("execute_tool_calls(")
        passes = src.count("approval=")
        assert passes >= calls, (
            f"{calls} call sites but only {passes} pass approval — "
            "a gate missed at one site is a gate that does not exist there"
        )


async def _run_tool_service(service_cls, *, streaming: bool):
    """Drive ToolService far enough to capture what reaches execute_tool_calls."""
    captured: dict = {}

    async def fake_execute(**kwargs):
        captured.update(kwargs)
        return []

    async def handler(_req): ...

    agent = _agent(tools={"send_referral_email"}, handler=handler)
    executor = MagicMock()
    executor.execute_tool_calls = fake_execute
    executor.tool_registry = {}
    agent.tool_executor = executor

    svc = service_cls.__new__(service_cls)
    svc._tool_executor = executor
    svc._message_to_dict = lambda m: {"content": ""}

    context = SimpleNamespace(
        trace_id="t1", data_labels=set(), metadata={}, run_id="r1", taint=lambda *a: None
    )
    tool_call = {"id": "c1", "function": {"name": "send_referral_email", "arguments": "{}"}}
    await svc.execute_tool_call(agent, tool_call, context)
    return captured


@pytest.mark.skipif(not _HAS_TEMPORAL, reason="the temporal extra is not installed")
class TestTheTemporalRouteIsNowAvailable:
    """This class used to assert the opposite, and the change is the point.

    It held two facts: HumanInLoopManager has no ask-and-wait API, and nothing
    outside the workflow can register an approval -- ``_pending_approvals`` was
    appended to in exactly one place, inside ``_run_approval_step``. Both were
    true, and together they meant the durable route could not be built.

    The second is no longer true: ``request_tool_approval`` is a signal an
    activity can send. The old test failed the moment that landed, which is what
    a tripwire is for -- it named the decision instead of letting a stale
    assertion quietly pass. It is replaced here rather than deleted so the
    constraint that replaced it is guarded in turn.
    """

    def test_a_workflow_can_be_asked_to_register_an_approval(self):
        from continuum.temporal.workflows.agent_workflow import AgentWorkflow

        assert hasattr(AgentWorkflow, "request_tool_approval")
        assert hasattr(AgentWorkflow, "get_approval_decision")

    def test_one_signal_still_serves_both_kinds_of_approval(self):
        """A tool approval must not become a second door into the workflow. The
        reviewer's side -- submit_approval, the allow-list check -- stays shared,
        or the two paths drift and one of them ends up weaker."""
        import inspect

        from continuum.temporal.workflows import agent_workflow

        src = inspect.getsource(agent_workflow)
        assert src.count("async def submit_approval") == 1, (
            "a second decision-submission entry point appeared; the allow-list "
            "check can now be bypassed by signalling the other one"
        )
        assert "is_authorized(step, decision)" in src, (
            "the shared authorization rule is no longer applied"
        )

    def test_an_unreachable_workflow_defers_rather_than_denying(self):
        """The adapter's honesty property, asserted from the wiring side too:
        Temporal being down is not a reviewer saying no."""
        import inspect

        from continuum.temporal import approval_adapter

        src = inspect.getsource(approval_adapter)
        assert "deferred=True" in src
        assert "approved=True," not in src.split("def handler")[0], (
            "a failure path returns approval"
        )


def _request(tool: str = "send_referral_email", **kwargs):
    from continuum.agent.approval import ToolApprovalRequest

    return ToolApprovalRequest(
        tool_name=tool,
        arguments=kwargs or {"to": "dr@example.com"},
        agent_name="clinic",
        run_id="r1",
        data_labels=frozenset(),
    )
