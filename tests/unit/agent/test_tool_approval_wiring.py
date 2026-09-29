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
        """The agent's own executor -- the path most deployments take.

        This used to read the harness's last call, and with an agent executor
        the global fallback runs after it (the fake returns no result), so it
        was checking the SECOND site. Dropping approval at this site passed
        every test here, the source count included (it counts the text
        ``approval=``, not the value).
        """
        from continuum.agent.services.tool_service import ToolService

        captured = await _run_tool_service(ToolService, streaming=False)
        first = captured["_calls"][0]
        assert first.get("approval") is not None, (
            "the agent-executor call site does not pass approval settings — "
            "declared tools would run unapproved"
        )
        assert first["approval"].tools == frozenset({"send_referral_email"})

    async def test_the_global_fallback_passes_approval(self):
        """The second site: an agent with no executor of its own."""
        from continuum.agent.services.tool_service import ToolService

        captured = await _run_tool_service(ToolService, streaming=False, agent_executor=False)
        assert len(captured["_calls"]) == 1, "only the global executor should run"
        only = captured["_calls"][0]
        assert only.get("approval") is not None, (
            "the global-executor call site does not pass approval settings — "
            "declared tools would run unapproved on the fallback path"
        )
        assert only["approval"].tools == frozenset({"send_referral_email"})

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


class TestTheRunTaintReachesTheApprovalRequest:
    """The run's data labels must reach the approval gate with or without a
    policy store.

    They used to be passed only ``if agent_policy_store``. That was right when
    the policy check was their only reader (c25c1b5), and became wrong when the
    approval gate (5a3e13f) started building ``ToolApprovalRequest.data_labels``
    from the same argument: an app with approval but no policy store handed every
    reviewer ``data_labels=frozenset()`` -- a tainted run described as clean.
    """

    async def test_the_agent_executor_site_passes_labels_without_a_policy_store(self):
        from continuum.agent.services.tool_service import ToolService

        captured = await _run_tool_service(ToolService, streaming=False, labels={"untrusted"})
        # The fake agent executor returns no result, so ToolService then falls
        # back to the global executor as well: the FIRST call is the agent's site.
        first = captured["_calls"][0]
        assert first["policy_store"] is None
        assert first["data_labels"] == {"untrusted"}

    async def test_the_global_executor_site_passes_labels_without_a_policy_store(self):
        from continuum.agent.services.tool_service import ToolService

        captured = await _run_tool_service(
            ToolService, streaming=False, labels={"untrusted"}, agent_executor=False
        )
        assert len(captured["_calls"]) == 1, "only the global executor should run"
        assert captured["_calls"][0]["data_labels"] == {"untrusted"}

    async def test_the_policy_subject_still_needs_a_policy_store(self):
        """Only the labels change. With no store there is no policy check, so
        there is still no subject to check it for."""
        from continuum.agent.services.tool_service import ToolService

        captured = await _run_tool_service(ToolService, streaming=False, labels={"untrusted"})
        assert captured.get("policy_store") is None
        assert captured.get("subject") is None

    async def test_a_reviewer_sees_the_taint_through_the_real_gate(self, monkeypatch):
        """End to end: ToolService -> a real ToolExecutor -> the handler. Each
        half was individually correct; the reviewer still saw a clean run."""
        from unittest.mock import AsyncMock

        from continuum.agent.approval import ToolApprovalDecision
        from continuum.agent.services.tool_service import ToolService
        from continuum.tools.executor import ToolExecutor

        seen: list = []

        async def reviewer(req):
            seen.append(req)
            return ToolApprovalDecision(approved=True, reviewer="alice")

        ex = ToolExecutor.__new__(ToolExecutor)
        server, tool = MagicMock(), MagicMock()
        server.name = "clinic"
        tool.name = "send_referral_email"
        ex.tool_registry = {"send_referral_email": (server, tool)}
        ex._rate_limiter = MagicMock()
        ex._rate_limiter.acquire = AsyncMock()
        ex._semaphore = MagicMock()
        ex._semaphore.__aenter__ = AsyncMock()
        ex._semaphore.__aexit__ = AsyncMock(return_value=False)
        ex._inject_context_variables = lambda _s, _t, args: args
        ex._config = SimpleNamespace(timeout_seconds=30)
        ex._run_artifacts = MagicMock()
        ex._context_state = MagicMock()
        ex._on_tool_result = lambda *a, **k: None
        ex._capture_context_variables = lambda *a, **k: None
        monkeypatch.setattr(
            "continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact",
            AsyncMock(return_value=("sent", None)),
        )

        agent = _agent(tools={"send_referral_email"}, handler=reviewer)
        agent.tool_executor = ex
        svc = ToolService.__new__(ToolService)
        svc._tool_executor = None
        svc._message_to_dict = lambda m: {"content": ""}
        context = SimpleNamespace(
            trace_id="t1",
            data_labels={"untrusted"},
            metadata={},
            run_id="r1",
            taint=lambda *a: None,
        )
        tool_call = {"id": "c1", "function": {"name": "send_referral_email", "arguments": "{}"}}
        await svc.execute_tool_call(agent, tool_call, context)

        assert seen, "the reviewer was never asked"
        assert seen[0].data_labels == frozenset({"untrusted"})


async def _run_tool_service(
    service_cls, *, streaming: bool, labels: set[str] | None = None, agent_executor: bool = True
):
    """Drive ToolService far enough to capture what reaches execute_tool_calls.

    ``captured`` holds the LAST call's kwargs (what the older tests read) and
    ``captured["_calls"]`` every call in order. The fake returns no results, so
    with an agent executor BOTH sites run -- the agent's, then the global
    fallback -- and a test about the first site must read ``_calls[0]``.
    ``agent_executor=False`` takes only the second site.
    """
    captured: dict = {"_calls": []}

    async def fake_execute(**kwargs):
        captured["_calls"].append(dict(kwargs))
        captured.update(kwargs)
        return []

    async def handler(_req): ...

    agent = _agent(tools={"send_referral_email"}, handler=handler)
    executor = MagicMock()
    executor.execute_tool_calls = fake_execute
    executor.tool_registry = {}
    agent.tool_executor = executor if agent_executor else None

    svc = service_cls.__new__(service_cls)
    svc._tool_executor = executor
    svc._message_to_dict = lambda m: {"content": ""}

    context = SimpleNamespace(
        trace_id="t1",
        data_labels=set(labels or ()),
        metadata={},
        run_id="r1",
        taint=lambda *a: None,
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
