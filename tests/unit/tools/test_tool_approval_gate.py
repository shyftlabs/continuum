"""The approval gate where a tool call actually happens (security finding F7).

``agent/approval.py`` holds the primitive; this is the part that decides whether
a call proceeds. It sits immediately after the policy check and before argument
injection, so the two gates read in the order they mean: policy asks *may this
run*, approval asks *should this call*.

It has to sit OUTSIDE the ``asyncio.wait_for`` that bounds tool execution. That
timeout exists to stop a hung MCP server; a reviewer thinking for twenty seconds
is not a hung server, and sharing one budget would make every approval look like
a tool failure.

A denial returns a tool result rather than raising out of the run, matching how
``POLICY DENIED`` already behaves: the model is told, tells the user, and the
turn ends normally instead of the whole run exploding over a designed outcome.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _executor(*, declared=None, handler=None, timeout=30.0, tool_name="send_referral_email"):
    """A ToolExecutor with one registered tool and the approval knobs set."""
    from continuum.tools.executor import ToolExecutor

    ex = ToolExecutor.__new__(ToolExecutor)
    server, tool = MagicMock(), MagicMock()
    tool.name = tool_name
    ex.tool_registry = {tool_name: (server, tool)}
    ex._rate_limiter = MagicMock()
    ex._rate_limiter.acquire = AsyncMock()
    ex._semaphore = MagicMock()
    ex._semaphore.__aenter__ = AsyncMock()
    ex._semaphore.__aexit__ = AsyncMock(return_value=False)
    ex._inject_context_variables = lambda _s, _t, args: args
    ex._config = SimpleNamespace(timeout_seconds=30)
    # Attributes the success path touches after the gate has let a call through.
    ex._run_artifacts = MagicMock()
    ex._context_state = MagicMock()
    ex._on_tool_result = lambda *a, **k: None
    ex._capture_context_variables = lambda *a, **k: None
    return ex


def _settings(declared=None, handler=None, timeout=30.0):
    from continuum.agent.approval import ToolApprovalSettings

    return ToolApprovalSettings(
        tools=frozenset(declared or ()),
        handler=handler,
        timeout=timeout,
        agent_name="clinic",
    )


def _call(name="send_referral_email", arguments='{"to": "dr@example.com"}'):
    return SimpleNamespace(id="call-1", function=SimpleNamespace(name=name, arguments=arguments))


class TestTheGateFires:
    async def test_a_declared_tool_is_approved_before_running(self, monkeypatch):
        from continuum.agent.approval import ToolApprovalDecision

        asked: list = []

        async def handler(req):
            asked.append(req)
            return ToolApprovalDecision(approved=True, reviewer="alice")

        ex = _executor()
        settings = _settings({"send_referral_email"}, handler)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        await ex.execute_tool_call(_call(), approval=settings)

        assert asked, "the gate did not ask"
        assert invoked.await_count == 1, "the tool did not run after approval"

    async def test_the_handler_receives_the_arguments(self, monkeypatch):
        """The whole reason this gate exists: neither tool-trust nor the policy
        gate can see them, so neither can judge an amount."""
        from continuum.agent.approval import ToolApprovalDecision

        seen: list = []

        async def handler(req):
            seen.append(req)
            return ToolApprovalDecision(approved=True)

        ex = _executor(tool_name="transfer_funds")
        settings = _settings({"*"}, handler)
        monkeypatch.setattr(
            "continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact",
            AsyncMock(return_value=("ok", None)),
        )

        await ex.execute_tool_call(
            _call("transfer_funds", '{"amount": 5000000}'),
            data_labels={"external"},
            approval=settings,
        )

        assert seen[0].arguments == {"amount": 5000000}
        assert seen[0].data_labels == frozenset({"external"})
        assert seen[0].tool_name == "transfer_funds"

    async def test_an_undeclared_tool_is_not_gated(self, monkeypatch):
        async def handler(_req):  # pragma: no cover - must never run
            raise AssertionError("an undeclared tool was sent for approval")

        ex = _executor(tool_name="clinic_info")
        settings = _settings({"send_referral_email"}, handler)
        invoked = AsyncMock(return_value=("hours are 9-5", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        await ex.execute_tool_call(_call("clinic_info", "{}"), approval=settings)
        assert invoked.await_count == 1

    async def test_nothing_declared_means_no_gate(self, monkeypatch):
        """The shipped default. An app that knows nothing of this is unchanged."""
        ex = _executor()
        settings = _settings()
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)
        await ex.execute_tool_call(_call(), approval=settings)
        assert invoked.await_count == 1


class TestDenial:
    async def test_a_refused_call_never_runs(self, monkeypatch):
        from continuum.agent.approval import ToolApprovalDecision

        async def refuse(_req):
            return ToolApprovalDecision(approved=False, reviewer="bob", reason="wrong recipient")

        ex = _executor()
        settings = _settings({"send_referral_email"}, refuse)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        from continuum.agent.exceptions import ToolApprovalDeniedError

        with pytest.raises(ToolApprovalDeniedError) as exc:
            await ex.execute_tool_call(_call(), approval=settings)

        assert invoked.await_count == 0, "the tool ran despite being refused"
        assert "wrong recipient" in str(exc.value)

    async def test_a_declared_tool_with_no_handler_is_refused(self, monkeypatch):
        """Fail closed. Declaring a tool for approval and wiring nobody must not
        quietly mean 'approved' -- that is the configuration most likely to be
        mistaken for protection."""
        from continuum.agent.exceptions import ToolApprovalDeniedError

        ex = _executor()
        settings = _settings({"send_referral_email"}, None)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        with pytest.raises(ToolApprovalDeniedError):
            await ex.execute_tool_call(_call(), approval=settings)
        assert invoked.await_count == 0

    async def test_a_timeout_refuses(self, monkeypatch):
        import asyncio

        from continuum.agent.exceptions import ToolApprovalDeniedError

        async def never(_req):
            await asyncio.sleep(10)

        ex = _executor()
        settings = _settings({"send_referral_email"}, never, timeout=0.05)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        with pytest.raises(ToolApprovalDeniedError):
            await ex.execute_tool_call(_call(), approval=settings)
        assert invoked.await_count == 0


class TestWhatTheModelIsTold:
    async def test_a_refusal_becomes_a_tool_result_not_a_crash(self):
        """Matching POLICY DENIED: the model is told and can tell the user, and
        the turn ends normally rather than the run exploding over a designed
        outcome."""
        from continuum.agent.exceptions import ToolApprovalDeniedError
        from continuum.tools.executor import ToolExecutor

        ex = ToolExecutor.__new__(ToolExecutor)
        err = ToolApprovalDeniedError(
            tool_name="send_referral_email", reviewer="bob", reason="wrong recipient"
        )
        messages = ex._process_tool_results([err], [_call()])

        assert len(messages) == 1
        assert "APPROVAL DENIED" in messages[0].content
        assert "wrong recipient" in messages[0].content

    async def test_the_reason_is_relayed_when_there_is_one(self):
        from continuum.agent.exceptions import ToolApprovalDeniedError
        from continuum.tools.executor import ToolExecutor

        ex = ToolExecutor.__new__(ToolExecutor)
        err = ToolApprovalDeniedError(tool_name="t", reviewer=None, reason=None)
        messages = ex._process_tool_results([err], [_call("t")])
        assert "APPROVAL DENIED" in messages[0].content


class TestTheTimeoutBoundary:
    async def test_approval_time_is_not_charged_to_the_tool_timeout(self, monkeypatch):
        """The tool timeout bounds a hung MCP server. A reviewer thinking is not
        a hung server, and sharing one budget would make every slow approval look
        like a tool failure."""
        import asyncio

        from continuum.agent.approval import ToolApprovalDecision

        async def slow_reviewer(_req):
            await asyncio.sleep(0.15)
            return ToolApprovalDecision(approved=True)

        ex = _executor()
        ex._config = SimpleNamespace(timeout_seconds=0.10)  # shorter than the approval
        settings = _settings({"send_referral_email"}, slow_reviewer, timeout=5)

        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        await ex.execute_tool_call(_call(), approval=settings)
        assert invoked.await_count == 1, (
            "the approval wait was charged against the tool execution timeout"
        )


class TestDeferralAtTheGate:
    """A deferred call must reach the model as pending, not as refused."""

    async def test_a_deferred_call_does_not_run(self, monkeypatch):
        from continuum.agent.approval import ToolApprovalDecision
        from continuum.agent.exceptions import ToolApprovalDeniedError

        async def defer(_req):
            return ToolApprovalDecision(
                approved=False, deferred=True, reason="sent to the on-call clinician"
            )

        ex = _executor()
        settings = _settings({"send_referral_email"}, defer)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        with pytest.raises(ToolApprovalDeniedError) as exc:
            await ex.execute_tool_call(_call(), approval=settings)

        assert invoked.await_count == 0
        assert exc.value.context.get("deferred") is True

    async def test_the_model_is_told_pending_not_denied(self):
        """The one thing deferral has to communicate. 'DENIED' would have the
        model report a refusal for an action that is merely waiting, and a user
        told their request was refused does not go looking for an approver."""
        from continuum.agent.exceptions import ToolApprovalDeniedError
        from continuum.tools.executor import ToolExecutor

        ex = ToolExecutor.__new__(ToolExecutor)
        err = ToolApprovalDeniedError(
            tool_name="send_referral_email",
            reason="sent to the on-call clinician",
            deferred=True,
        )
        messages = ex._process_tool_results([err], [_call()])

        content = messages[0].content
        assert "APPROVAL PENDING" in content
        assert "APPROVAL DENIED" not in content
        assert "sent to the on-call clinician" in content

    async def test_a_plain_refusal_still_says_denied(self):
        from continuum.agent.exceptions import ToolApprovalDeniedError
        from continuum.tools.executor import ToolExecutor

        ex = ToolExecutor.__new__(ToolExecutor)
        err = ToolApprovalDeniedError(tool_name="t", reviewer="bob", reason="no")
        content = ex._process_tool_results([err], [_call("t")])[0].content
        assert "APPROVAL DENIED" in content
        assert "PENDING" not in content
