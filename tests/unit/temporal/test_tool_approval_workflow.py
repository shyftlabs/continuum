"""Registering an ad-hoc tool approval on a running workflow (F7, option 3).

The approval gate fires inside a tool call, which under Temporal runs inside an
ACTIVITY. Today nothing outside the workflow can register an approval request --
``_pending_approvals`` is only ever appended to by ``_run_approval_step``, which
serves approvals the workflow itself planned as a step. So the gate had no way to
reach a durable reviewer, and fell back to the in-process handler.

Two additions close that: a signal so an activity can raise a request, and a
query so it can read the answer back. ``submit_approval`` is deliberately reused
for the answer, so the reviewer's side -- the UI, ``HumanInLoopManager``,
``get_pending_approvals`` -- is unchanged and one approval flow serves both.

WHAT THIS BUYS, AND WHAT IT DOES NOT

The whole agent turn is ONE activity (``run_agent_activity`` calls
``runner.run()``), so the workflow cannot pause *between* the agent's own steps.
The activity blocks in place while the reviewer decides, which means work done
before the gate is not repeated and the user does not have to ask again. It does
NOT survive a worker restart: the activity would retry from the beginning. For
that, ``CLINIC_APPROVAL=queue`` (the deferred path) is the honest answer.

These tests drive the workflow object directly rather than through a Temporal
worker. The signal and query handlers are ordinary methods; what needs asserting
is the bookkeeping -- that a request appears, that a decision is matched to the
right request, that an unauthorized answer is refused -- none of which needs a
running server. The live round trip is a separate, manual step.
"""

from __future__ import annotations

import pytest

pytest.importorskip("temporalio", reason="the temporal extra is not installed")


def _workflow():
    """A workflow instance with its state initialised, no Temporal runtime."""
    from continuum.temporal.workflows.agent_workflow import AgentWorkflow

    wf = AgentWorkflow.__new__(AgentWorkflow)
    wf._approval_decisions = []
    wf._pending_approvals = []
    wf._tool_approvals = {}
    wf._pending_decision = None
    wf._cancelled = False
    wf._status = "running"
    return wf


def _decision(request_id: str, decision: str = "approved", by: str = "alice"):
    from continuum.temporal.types import ApprovalDecision

    return ApprovalDecision(request_id=request_id, decision=decision, decided_by=by)


class TestRegisteringAnAdHocApproval:
    async def test_a_request_becomes_visible_to_reviewers(self):
        """It goes on the same list the planned approval step uses, so the
        existing reviewer UI and get_pending_approvals see it without change."""
        wf = _workflow()
        await wf.request_tool_approval(
            {
                "request_id": "tool-1",
                "description": "pharmacy__check_interactions",
                "context": '{"medications": ["metformin", "lisinopril"]}',
                "approvers": ["alice"],
            }
        )
        pending = wf.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0]["request_id"] == "tool-1"
        assert "check_interactions" in pending[0]["description"]

    async def test_the_arguments_travel_with_it(self):
        """A reviewer shown only a tool name is approving the name. The whole
        point of this gate is that it sees the arguments."""
        wf = _workflow()
        await wf.request_tool_approval(
            {
                "request_id": "tool-1",
                "description": "transfer_funds",
                "context": '{"amount": 5000000}',
            }
        )
        assert "5000000" in wf.get_pending_approvals()[0]["context"]

    async def test_two_requests_coexist(self):
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "a", "description": "t1"})
        await wf.request_tool_approval({"request_id": "b", "description": "t2"})
        assert {p["request_id"] for p in wf.get_pending_approvals()} == {"a", "b"}

    async def test_a_duplicate_id_does_not_double_register(self):
        """A retried signal must not leave two prompts for one call."""
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "a", "description": "t1"})
        await wf.request_tool_approval({"request_id": "a", "description": "t1"})
        assert len(wf.get_pending_approvals()) == 1


class TestReadingTheDecisionBack:
    async def test_unanswered_reads_as_pending(self):
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "tool-1", "description": "t"})
        assert wf.get_approval_decision("tool-1") == {"status": "pending"}

    async def test_an_approval_is_readable(self):
        wf = _workflow()
        await wf.request_tool_approval(
            {"request_id": "tool-1", "description": "t", "approvers": ["alice"]}
        )
        await wf.submit_approval(_decision("tool-1", "approved", "alice"))

        got = wf.get_approval_decision("tool-1")
        assert got["status"] == "approved"
        assert got["decided_by"] == "alice"

    async def test_a_rejection_is_readable(self):
        wf = _workflow()
        await wf.request_tool_approval(
            {"request_id": "tool-1", "description": "t", "approvers": ["alice"]}
        )
        await wf.submit_approval(_decision("tool-1", "rejected", "alice"))
        assert wf.get_approval_decision("tool-1")["status"] == "rejected"

    async def test_an_unknown_id_reads_as_unknown_not_approved(self):
        """A caller polling for an id the workflow never saw must not read that
        silence as permission."""
        wf = _workflow()
        assert wf.get_approval_decision("never-registered") == {"status": "unknown"}

    async def test_an_answered_request_leaves_the_pending_list(self):
        """Otherwise a reviewer keeps seeing a prompt they already answered."""
        wf = _workflow()
        await wf.request_tool_approval(
            {"request_id": "tool-1", "description": "t", "approvers": ["alice"]}
        )
        await wf.submit_approval(_decision("tool-1", "approved", "alice"))
        assert wf.get_pending_approvals() == []


class TestTheAnswerIsValidated:
    async def test_an_unauthorized_approver_is_refused(self):
        """Same allow-list rule as the planned approval step, via the shared
        is_authorized. A tool approval is not a weaker door into the same
        workflow."""
        wf = _workflow()
        await wf.request_tool_approval(
            {"request_id": "tool-1", "description": "t", "approvers": ["alice"]}
        )
        await wf.submit_approval(_decision("tool-1", "approved", "mallory"))

        assert wf.get_approval_decision("tool-1") == {"status": "pending"}
        assert wf.get_pending_approvals(), "an unauthorized answer must not resolve the prompt"

    async def test_an_empty_approver_list_accepts_anyone(self):
        """Backward compatible with workflows that never named approvers."""
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "tool-1", "description": "t"})
        await wf.submit_approval(_decision("tool-1", "approved", "anyone"))
        assert wf.get_approval_decision("tool-1")["status"] == "approved"

    async def test_a_decision_for_another_request_is_ignored(self):
        """Two calls awaiting approval must not have one answer resolve both."""
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "a", "description": "t1"})
        await wf.request_tool_approval({"request_id": "b", "description": "t2"})
        await wf.submit_approval(_decision("a", "approved", "alice"))

        assert wf.get_approval_decision("a")["status"] == "approved"
        assert wf.get_approval_decision("b") == {"status": "pending"}

    async def test_the_planned_approval_step_still_receives_its_decision(self):
        """`submit_approval` serves both paths. A decision for an id this
        mechanism never registered has to stay available to _run_approval_step,
        which is waiting on _pending_decision -- otherwise adding tool approvals
        breaks the workflow approvals that already worked."""
        wf = _workflow()
        await wf.submit_approval(_decision("planned-step-1", "approved", "alice"))
        assert wf._pending_decision is not None
        assert wf._pending_decision.request_id == "planned-step-1"


class TestAMalformedSignalCannotWedgeTheWorkflow:
    """Found live, from the Temporal UI.

    A signal handler that raises does not just lose that signal -- it fails the
    workflow ACTIVATION, and Temporal retries an activation forever. So one
    empty Send-a-Signal form, from anyone who can reach the UI, parks the
    workflow permanently: the reviewer's real answer can never land afterwards
    because the activation never completes.

        TypeError: AgentWorkflow.submit_approval() missing 1 required
        positional argument: 'decision'

    A reviewer sending a bad payload is a mistake to ignore loudly, not a reason
    to take the run down. Note what is NOT softened: a decision that deserializes
    fine but names an unknown request still falls through to _pending_decision,
    and an unauthorized one is still refused. Only the unusable payload is
    dropped.
    """

    async def test_a_signal_with_no_payload_is_ignored_not_raised(self):
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "tool-1", "description": "t"})

        await wf.submit_approval()  # the empty Data field, verbatim

        assert wf.get_approval_decision("tool-1") == {"status": "pending"}, (
            "the prompt must stay answerable after a junk signal"
        )

    async def test_a_null_payload_is_ignored(self):
        wf = _workflow()
        await wf.submit_approval(None)
        assert wf._pending_decision is None

    async def test_a_payload_missing_fields_is_ignored(self):
        """Belt and braces: the pydantic converter rejects most of these before
        the handler sees them, but the handler must not assume that."""
        wf = _workflow()
        await wf.submit_approval({"decided_by": "tom"})  # type: ignore[arg-type]
        assert wf._pending_decision is None

    async def test_a_real_decision_still_lands_after_a_junk_one(self):
        """The property that actually matters: the run is still answerable."""
        wf = _workflow()
        await wf.request_tool_approval(
            {"request_id": "tool-1", "description": "t", "approvers": ["alice"]}
        )
        await wf.submit_approval()
        await wf.submit_approval(_decision("tool-1", "approved", "alice"))

        assert wf.get_approval_decision("tool-1")["status"] == "approved"

    async def test_a_wrong_request_id_is_still_forwarded_not_swallowed(self):
        """The mistake a reviewer actually makes is pasting the WORKFLOW id
        where the tool request id goes. That deserializes fine, so it must keep
        its existing behaviour -- fall through to the planned approval step --
        rather than being caught by the new guard."""
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "tool-1", "description": "t"})
        await wf.submit_approval(_decision("f7-10acdb7f", "approved", "tom"))

        assert wf._pending_decision is not None
        assert wf.get_approval_decision("tool-1") == {"status": "pending"}


class TestTheDecisionRecordIsKept:
    async def test_decisions_are_recorded_for_the_workflow_result(self):
        """approval_decisions is part of WorkflowResult: what was approved, by
        whom, is the audit trail."""
        wf = _workflow()
        await wf.request_tool_approval({"request_id": "tool-1", "description": "t"})
        await wf.submit_approval(_decision("tool-1", "approved", "alice"))
        assert any(d.request_id == "tool-1" for d in wf._approval_decisions)
