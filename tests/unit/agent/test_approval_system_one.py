"""A System One approval handler: three outcomes, not LangChain's block-only.

Plugged in as ``AgentConfig(approval_handler=system_one_approval_handler(...))``.
It asks "is this call risky?" and:

* auto-approves only when P(risky) is below a LOW threshold (default 0.1);
* sends everything else to a human (``escalate_to``), whose decision --
  approve, refuse, or defer -- is returned as-is;
* with no human configured, refuses.

The default is chosen so a mis-scaled backend errs toward the safe side: a
backend that under-scores risk sends more calls to a person, it does not
auto-approve dangerous ones. A classifier that cannot answer, and the kill
switch, both go to the human (or refuse): the approval gate fails closed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.system_one.conftest import FakeClassifier


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from continuum.config import settings
    from continuum.core.container import reset_container
    from continuum.system_one.registry import clear_classifier_cache

    monkeypatch.setattr(settings, "system_one_backend", None)
    monkeypatch.setattr(settings, "system_one_disabled", False)
    reset_container()
    clear_classifier_cache()
    yield
    clear_classifier_cache()
    reset_container()


def _use(p_risky: float, **kwargs):
    from continuum.core.container import get_container

    backend = FakeClassifier(
        answer=lambda qid, q: {"true": p_risky, "false": 1 - p_risky}, **kwargs
    )
    get_container().set_system_one_classifier(backend)
    return backend


def _request(**overrides):
    from continuum.agent.approval import ToolApprovalRequest

    fields = {
        "tool_name": "transfer_funds",
        "arguments": {"amount": 5_000_000, "to": "acct-9"},
        "agent_name": "banker",
        "run_id": "r1",
        "data_labels": frozenset({"external"}),
    }
    fields.update(overrides)
    return ToolApprovalRequest(**fields)


def _human(approved=True, deferred=False, reviewer="alice"):
    from continuum.agent.approval import ToolApprovalDecision

    return AsyncMock(
        return_value=ToolApprovalDecision(approved=approved, reviewer=reviewer, deferred=deferred)
    )


def _handler(**kwargs):
    from continuum.agent.system_one_approval import system_one_approval_handler

    return system_one_approval_handler(**kwargs)


class TestConstruction:
    def test_no_backend_is_an_error(self):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError):
            _handler()

    def test_its_own_backend_spec_satisfies_it(self):
        _handler(backend="local:cross-encoder/nli-deberta-v3-small")

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.0, 1.5])
    def test_the_threshold_must_be_strictly_between_zero_and_one(self, bad):
        _use(0.0)
        with pytest.raises(ValueError):
            _handler(auto_approve_below=bad)

    def test_the_default_threshold_is_low(self):
        """0.1, not LangChain's 0.5: the safe side of a wrong threshold is 'ask
        a person', never 'approve'."""
        import inspect

        from continuum.agent.system_one_approval import system_one_approval_handler

        default = inspect.signature(system_one_approval_handler).parameters["auto_approve_below"]
        assert default.default == 0.1


class TestLowRiskIsApproved:
    async def test_below_the_threshold_it_approves_without_a_human(self):
        _use(0.03)
        human = _human()
        decision = await _handler(escalate_to=human)(_request())

        assert decision.approved is True
        assert decision.reviewer == "system_one:fake:fake-1"
        assert "0.03" in (decision.reason or "")
        human.assert_not_called()

    async def test_the_classifier_sees_the_call_its_arguments_and_the_run_taint(self):
        backend = _use(0.03)
        await _handler()(_request())

        state, questions = backend.calls[0]
        assert state["tool_call"] == {
            "name": "transfer_funds",
            "arguments": {"amount": 5_000_000, "to": "acct-9"},
        }
        assert state["run_data_labels"] == ["external"]
        (question,) = questions.values()
        assert question.kind == "binary"
        assert question.true_criteria, "local NLI backends need a statement to test"


class TestEverythingElseGoesToAPerson:
    async def test_at_the_threshold_it_escalates(self):
        _use(0.1)
        human = _human()
        request = _request()
        decision = await _handler(escalate_to=human)(request)

        human.assert_awaited_once_with(request)
        assert decision.reviewer == "alice"

    async def test_the_humans_refusal_is_returned(self):
        _use(0.9)
        decision = await _handler(escalate_to=_human(approved=False))(_request())
        assert decision.approved is False

    async def test_the_humans_deferral_is_returned(self):
        _use(0.9)
        decision = await _handler(escalate_to=_human(approved=False, deferred=True))(_request())
        assert decision.deferred is True

    async def test_with_no_human_a_risky_call_is_refused(self):
        _use(0.6)
        decision = await _handler()(_request())
        assert decision.approved is False
        assert decision.deferred is False
        assert "0.60" in (decision.reason or "")


class TestItFailsClosed:
    class _Down(FakeClassifier):
        async def classify(self, state, questions):
            from continuum.system_one import SystemOneBackendError

            raise SystemOneBackendError("down", backend="fake")

    def _use_down(self):
        from continuum.core.container import get_container

        get_container().set_system_one_classifier(self._Down())

    async def test_a_failing_classifier_goes_to_the_human(self):
        self._use_down()
        human = _human()
        decision = await _handler(escalate_to=human)(_request())
        human.assert_awaited_once()
        assert decision.reviewer == "alice"

    async def test_a_failing_classifier_with_no_human_refuses(self):
        self._use_down()
        decision = await _handler()(_request())
        assert decision.approved is False

    async def test_an_egress_denial_goes_to_the_human(self):
        from continuum.security.policy import AccessPolicy, PolicyStore
        from continuum.security.policy_context import use_active_policy

        backend = _use(0.0, egress="remote")
        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="phi-local", subjects=["phi"], resources=["system_one:remote:*"], effect="deny"
            )
        )
        human = _human()

        class Ctx:
            data_labels = {"phi"}

        with use_active_policy(store, "banker", Ctx()):
            await _handler(escalate_to=human)(_request())
        human.assert_awaited_once()
        assert backend.calls == []

    async def test_the_kill_switch_goes_to_the_human(self, monkeypatch):
        """'Previous behaviour' at this gate is a person deciding."""
        from continuum.config import settings

        backend = _use(0.0)
        monkeypatch.setattr(settings, "system_one_disabled", True)
        human = _human()

        await _handler(escalate_to=human)(_request())
        human.assert_awaited_once()
        assert backend.calls == []

    async def test_the_kill_switch_with_no_human_refuses(self, monkeypatch):
        from continuum.config import settings

        _use(0.0)
        monkeypatch.setattr(settings, "system_one_disabled", True)
        decision = await _handler()(_request())
        assert decision.approved is False


class TestThroughTheRealGate:
    """The handler as the executor actually calls it, via request_approval."""

    def _executor(self, tool_name="transfer_funds"):
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
        ex._run_artifacts = MagicMock()
        ex._context_state = MagicMock()
        ex._on_tool_result = lambda *a, **k: None
        ex._capture_context_variables = lambda *a, **k: None
        return ex

    def _settings(self, handler):
        from continuum.agent.approval import ToolApprovalSettings

        return ToolApprovalSettings(
            tools=frozenset({"transfer_*"}), handler=handler, timeout=5.0, agent_name="banker"
        )

    def _call(self):
        return SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="transfer_funds", arguments='{"amount": 5}'),
        )

    async def test_a_low_risk_call_runs(self, monkeypatch):
        _use(0.02)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        await self._executor().execute_tool_call(self._call(), approval=self._settings(_handler()))
        assert invoked.await_count == 1

    async def test_a_risky_call_with_no_human_does_not_run(self, monkeypatch):
        """Refused at the gate: the executor raises ToolApprovalDeniedError (the
        tool service turns it into a tool result) and the tool never runs."""
        from continuum.agent.exceptions import ToolApprovalDeniedError

        _use(0.8)
        invoked = AsyncMock(return_value=("sent", None))
        monkeypatch.setattr("continuum.tools.util.MCPUtil.invoke_mcp_tool_with_artifact", invoked)

        with pytest.raises(ToolApprovalDeniedError) as exc:
            await self._executor().execute_tool_call(
                self._call(), approval=self._settings(_handler())
            )
        assert "P(risky)=0.80" in str(exc.value)
        assert invoked.await_count == 0
