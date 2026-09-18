"""The handler that reaches a Temporal reviewer from inside a tool call (F7).

The gate calls an async handler and waits. Under Temporal the tool call runs
inside an ACTIVITY, which cannot touch workflow state directly -- so this signals
the workflow to register the request, then polls a query until someone answers,
heartbeating so Temporal does not kill the activity mid-wait.

WHAT IT BUYS, STATED PRECISELY

Work the turn did before the gate is not repeated, and the user does not have to
ask again -- the activity blocks in place and resumes. It does NOT survive a
worker restart: a retried activity starts from the beginning. For that,
``deferred`` (the queue path) is the honest answer, which is why an unreachable
workflow degrades to deferred rather than to denied or, worse, approved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("temporalio", reason="the temporal extra is not installed")


def _request(tool: str = "pharmacy__check_interactions", **kwargs):
    from continuum.agent.approval import ToolApprovalRequest

    return ToolApprovalRequest(
        tool_name=tool,
        arguments=kwargs or {"medications": ["metformin", "lisinopril"]},
        agent_name="clinic",
        run_id="r1",
        data_labels=frozenset(),
    )


def _handle(*, decisions, approvers=None):
    """A handler wired to a fake workflow handle returning `decisions` in turn."""
    from continuum.temporal.approval_adapter import temporal_approval_handler

    handle = MagicMock()
    handle.signal = AsyncMock()
    handle.query = AsyncMock(side_effect=list(decisions))
    return (
        temporal_approval_handler(
            handle, approvers=approvers or [], poll_interval=0.01, heartbeat=lambda _m: None
        ),
        handle,
    )


class TestAsking:
    async def test_the_request_is_signalled_to_the_workflow(self):
        handler, handle = _handle(decisions=[{"status": "approved", "decided_by": "alice"}])
        await handler(_request())

        handle.signal.assert_awaited_once()
        name, payload = handle.signal.await_args.args
        assert name == "request_tool_approval"
        assert payload["description"] == "pharmacy__check_interactions"

    async def test_the_arguments_travel_to_the_reviewer(self):
        """The capability this gate adds. A reviewer shown only a tool name is
        approving the name."""
        handler, handle = _handle(decisions=[{"status": "approved", "decided_by": "a"}])
        await handler(_request("transfer_funds", amount=5_000_000))

        payload = handle.signal.await_args.args[1]
        assert "5000000" in payload["context"]

    async def test_an_approval_lets_the_call_through(self):
        handler, _ = _handle(decisions=[{"status": "approved", "decided_by": "alice"}])
        decision = await handler(_request())
        assert decision.approved
        assert decision.reviewer == "alice"

    async def test_a_rejection_is_a_refusal(self):
        handler, _ = _handle(
            decisions=[{"status": "rejected", "decided_by": "bob", "reason": "wrong drug"}]
        )
        decision = await handler(_request())
        assert not decision.approved
        assert not decision.deferred
        assert "wrong drug" in (decision.reason or "")

    async def test_it_polls_until_answered(self):
        """The reviewer is not expected to be instant; that is the whole point."""
        handler, handle = _handle(
            decisions=[
                {"status": "pending"},
                {"status": "pending"},
                {"status": "approved", "decided_by": "alice"},
            ]
        )
        decision = await handler(_request())
        assert decision.approved
        assert handle.query.await_count == 3

    async def test_the_activity_is_kept_alive_while_waiting(self):
        """Without a heartbeat Temporal kills the activity mid-wait, and the
        reviewer's answer arrives to nobody."""
        from continuum.temporal.approval_adapter import temporal_approval_handler

        beats: list[str] = []
        handle = MagicMock()
        handle.signal = AsyncMock()
        handle.query = AsyncMock(
            side_effect=[{"status": "pending"}, {"status": "approved", "decided_by": "a"}]
        )
        handler = temporal_approval_handler(handle, poll_interval=0.01, heartbeat=beats.append)
        await handler(_request())
        assert beats, "the activity was never heartbeated while waiting"


class TestDegradingHonestly:
    async def test_an_unreachable_workflow_defers_rather_than_denying(self):
        """Temporal being down is not a reviewer saying no. Deferring keeps the
        request honest -- the action did not happen and can be asked again --
        where a denial would tell the user it was refused."""
        from continuum.temporal.approval_adapter import temporal_approval_handler

        handle = MagicMock()
        handle.signal = AsyncMock(side_effect=ConnectionError("temporal is down"))
        handler = temporal_approval_handler(handle, poll_interval=0.01, heartbeat=lambda _m: None)

        decision = await handler(_request())
        assert not decision.approved
        assert decision.deferred
        assert "temporal is down" in (decision.reason or "")

    async def test_a_query_failure_mid_wait_defers(self):
        from continuum.temporal.approval_adapter import temporal_approval_handler

        handle = MagicMock()
        handle.signal = AsyncMock()
        handle.query = AsyncMock(side_effect=ConnectionError("lost the server"))
        handler = temporal_approval_handler(handle, poll_interval=0.01, heartbeat=lambda _m: None)

        decision = await handler(_request())
        assert decision.deferred

    async def test_an_unknown_status_is_never_read_as_approval(self):
        """The workflow returns `unknown` for an id it never registered -- a
        signal that got lost. Silence must not read as permission."""
        from continuum.temporal.approval_adapter import temporal_approval_handler

        handle = MagicMock()
        handle.signal = AsyncMock()
        handle.query = AsyncMock(return_value={"status": "unknown"})
        handler = temporal_approval_handler(handle, poll_interval=0.01, heartbeat=lambda _m: None)

        decision = await handler(_request())
        assert not decision.approved

    async def test_the_sdk_timeout_still_bounds_the_wait(self):
        """This handler polls forever by design -- the durable case is a reviewer
        at lunch. The bound comes from the SDK's approval_timeout, which cancels
        it and denies; the adapter must not swallow that cancellation."""
        from continuum.agent.approval import request_approval
        from continuum.temporal.approval_adapter import temporal_approval_handler

        handle = MagicMock()
        handle.signal = AsyncMock()
        handle.query = AsyncMock(return_value={"status": "pending"})
        handler = temporal_approval_handler(handle, poll_interval=0.01, heartbeat=lambda _m: None)

        decision = await request_approval(_request(), handler, timeout=0.05)
        assert not decision.approved
        assert "timed out" in (decision.reason or "").lower()


class TestApprovers:
    async def test_the_allow_list_is_sent_with_the_request(self):
        """Enforced workflow-side by the same is_authorized the planned approval
        step uses, so a tool approval is not a weaker door."""
        handler, handle = _handle(
            decisions=[{"status": "approved", "decided_by": "alice"}], approvers=["alice", "bob"]
        )
        await handler(_request())
        assert handle.signal.await_args.args[1]["approvers"] == ["alice", "bob"]

    async def test_each_call_gets_its_own_request_id(self):
        """Two calls awaiting approval must not have one answer resolve both."""
        handler, handle = _handle(
            decisions=[
                {"status": "approved", "decided_by": "a"},
                {"status": "approved", "decided_by": "a"},
            ]
        )
        await handler(_request())
        await handler(_request("other_tool"))

        ids = [c.args[1]["request_id"] for c in handle.signal.await_args_list]
        assert len(set(ids)) == 2


class TestOutsideAnActivity:
    async def test_building_without_a_handle_is_refused_loudly(self):
        """A handler that silently does nothing is the failure this whole
        finding is about."""
        from continuum.temporal.approval_adapter import temporal_approval_handler

        with pytest.raises(ValueError, match="workflow handle"):
            temporal_approval_handler(None)  # type: ignore[arg-type]


class TestTheSelfResolvingHandler:
    """An agent is built long before any workflow exists, so a handler cannot be
    handed a workflow handle at construction time. This one finds its own at
    call time: the id from ``activity.info()``, the handle from the global
    client."""

    async def test_it_finds_the_workflow_it_is_running_under(self, monkeypatch):
        from continuum.temporal import approval_adapter as mod

        handle = MagicMock()
        handle.signal = AsyncMock()
        handle.query = AsyncMock(return_value={"status": "approved", "decided_by": "alice"})

        client = MagicMock()
        client.get_workflow_handle = AsyncMock(return_value=handle)
        monkeypatch.setattr(mod, "_activity_workflow_id", lambda: "wf-42")
        monkeypatch.setattr(mod, "_temporal_client", lambda: client)

        decision = await mod.temporal_tool_approval(poll_interval=0.01)(_request())

        assert decision.approved
        client.get_workflow_handle.assert_awaited_once_with("wf-42")

    async def test_outside_an_activity_it_defers_rather_than_approving(self, monkeypatch):
        """Configured for Temporal but running outside one -- a local script, a
        test, an HTTP server. The action must not proceed unreviewed, and it was
        not refused by anyone either."""
        from continuum.temporal import approval_adapter as mod

        monkeypatch.setattr(mod, "_activity_workflow_id", lambda: None)

        decision = await mod.temporal_tool_approval(poll_interval=0.01)(_request())
        assert not decision.approved
        assert decision.deferred
        assert "activity" in (decision.reason or "").lower()

    async def test_an_unreachable_client_defers(self, monkeypatch):
        from continuum.temporal import approval_adapter as mod

        monkeypatch.setattr(mod, "_activity_workflow_id", lambda: "wf-42")

        def boom():
            raise ConnectionError("no temporal client")

        monkeypatch.setattr(mod, "_temporal_client", boom)

        decision = await mod.temporal_tool_approval(poll_interval=0.01)(_request())
        assert decision.deferred


class TestTheGlobalClientMustBeConnected:
    """Found live: the handler resolves its handle through the GLOBAL client,
    which is not the worker's own. A worker connects its client to run
    activities; that does nothing for this one, so an otherwise correct setup
    defers every approval with 'Not connected to Temporal server'.

    It fails safe, which is why the live run produced a deferral rather than an
    unreviewed call. But an operator seeing every approval defer needs the
    message to name the cause rather than read as a network problem.
    """

    async def test_an_unconnected_global_client_says_what_to_do(self, monkeypatch):
        from continuum.temporal import approval_adapter as mod
        from continuum.temporal.exceptions import TemporalConnectionError

        class Unconnected:
            async def get_workflow_handle(self, _id):
                raise TemporalConnectionError("Not connected to Temporal server")

        monkeypatch.setattr(mod, "_activity_workflow_id", lambda: "wf-1")
        monkeypatch.setattr(mod, "_temporal_client", Unconnected)

        decision = await mod.temporal_tool_approval(poll_interval=0.01)(_request())
        assert decision.deferred
        reason = (decision.reason or "").lower()
        assert "connect" in reason, "the message must name the cause, not just the symptom"
