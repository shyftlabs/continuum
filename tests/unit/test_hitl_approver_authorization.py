"""
Tests for HITL approver authorization (enforce ``decided_by in approvers``).

Context
-------
An ApprovalStep carries an ``approvers`` allow-list, but historically the
workflow never checked that the person who submitted a decision
(``ApprovalDecision.decided_by``) was actually on that list. Anyone who knew
the workflow id + request id could approve a paused step.

These tests pin down the fix:

1. ``is_authorized`` — a pure predicate (no Temporal runtime) holding the rule:
     * empty ``approvers``     -> unrestricted (anyone may decide; back-compat)
     * ``escalated`` decisions -> always allowed (escalation re-targets approval)
     * otherwise               -> ``decided_by`` must be in ``approvers``
2. Source-level enforcement — ``_run_approval_step`` must call the predicate
   and keep waiting (loop) instead of accepting an unauthorized signal.
3. Behavioral — a full workflow run where an unauthorized signal is ignored
   and a later authorized signal approves.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# temporalio is an optional extra. The pure-predicate tests below must run
# without it; the workflow/source tests skip cleanly when it is absent.
# ---------------------------------------------------------------------------
try:
    import temporalio  # noqa: F401

    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False

requires_temporal = pytest.mark.skipif(
    not HAS_TEMPORAL, reason="temporalio not installed (pip install -e '.[temporal]')"
)


# ---------------------------------------------------------------------------
# 1. Pure predicate — the authorization rule itself (no Temporal runtime).
# ---------------------------------------------------------------------------


class TestIsAuthorizedPredicate:
    """The standalone rule that decides whether a decision may be accepted."""

    def _step(self, approvers):
        from continuum.temporal.types import ApprovalStep

        return ApprovalStep(description="approve the thing", approvers=approvers)

    def _decision(self, decided_by, decision="approved"):
        from continuum.temporal.types import ApprovalDecision

        return ApprovalDecision(
            request_id="approval-123", decision=decision, decided_by=decided_by
        )

    def test_listed_approver_is_authorized(self):
        from continuum.temporal.types import is_authorized

        step = self._step(["alice", "bob"])
        assert is_authorized(step, self._decision("alice")) is True

    def test_non_approver_is_denied(self):
        from continuum.temporal.types import is_authorized

        step = self._step(["alice", "bob"])
        # This is the vulnerability: 'mallory' is not on the list.
        assert is_authorized(step, self._decision("mallory")) is False

    def test_empty_approvers_allows_anyone(self):
        from continuum.temporal.types import is_authorized

        # Back-compat: an unset approvers list means "unrestricted".
        step = self._step([])
        assert is_authorized(step, self._decision("anyone")) is True

    def test_non_approver_cannot_reject_either(self):
        from continuum.temporal.types import is_authorized

        # Authorization gates rejection too, not just approval.
        step = self._step(["alice"])
        assert is_authorized(step, self._decision("mallory", decision="rejected")) is False

    def test_listed_approver_may_reject(self):
        from continuum.temporal.types import is_authorized

        step = self._step(["alice"])
        assert is_authorized(step, self._decision("alice", decision="rejected")) is True

    def test_escalation_to_unlisted_person_is_allowed(self):
        from continuum.temporal.types import is_authorized

        # escalate() targets a NEW person who by definition is not on the
        # original approvers list; escalation re-routes approval, it is not
        # itself an approval, so it must not be blocked by the membership rule.
        step = self._step(["alice"])
        assert is_authorized(step, self._decision("carol", decision="escalated")) is True


# ---------------------------------------------------------------------------
# 2. Source-level enforcement — the workflow must USE the predicate and loop.
#    Mirrors the repo idiom in test_issue04_temporal_eval.TestRetryPolicyExplicit.
# ---------------------------------------------------------------------------


@requires_temporal
class TestApprovalStepEnforcesAuthorization:
    def _source(self):
        import inspect

        from continuum.temporal.workflows.agent_workflow import AgentWorkflow

        return inspect.getsource(AgentWorkflow._run_approval_step)

    def test_calls_authorization_predicate(self):
        assert "is_authorized" in self._source(), (
            "_run_approval_step must call is_authorized() to enforce the approvers list"
        )

    def test_unauthorized_decision_does_not_fall_through(self):
        src = self._source()
        # An unauthorized signal must be discarded and the step keep waiting,
        # which requires a wait LOOP rather than a single wait_condition.
        assert "while" in src, (
            "_run_approval_step must loop so an unauthorized signal keeps the "
            "step pending instead of being accepted or terminating it"
        )


# ---------------------------------------------------------------------------
# 3. Behavioral — full workflow run proving ignore-then-accept.
# ---------------------------------------------------------------------------


@requires_temporal
@pytest.mark.temporal
class TestApprovalAuthorizationWorkflow:
    async def _run_scenario(self, approvers, attempts):
        """Start an approval-only workflow and replay a list of (who, decision)
        attempts against it. Returns the final WorkflowResult.

        ``attempts`` is a list of dicts: {"decided_by": str, "decision": str,
        "expect_pending_after": bool}. After each attempt that is expected to
        leave the step pending, we assert the workflow has NOT finished.
        """
        import uuid

        from temporalio import activity
        from temporalio.testing import WorkflowEnvironment
        from temporalio.worker import Worker

        from continuum.temporal.types import (
            ApprovalDecision,
            ApprovalStep,
            WorkflowInput,
        )
        from continuum.temporal.workflows.agent_workflow import AgentWorkflow

        @activity.defn(name="send_notification_activity")
        async def fake_send_notification(params) -> None:  # noqa: ANN001
            return None

        @activity.defn(name="run_agent_activity")
        async def fake_run_agent(params):  # noqa: ANN001 - unused in approval-only flow
            from continuum.temporal.types import AgentActivityResult

            return AgentActivityResult(content="", status="success")

        step = ApprovalStep(description="approve", approvers=approvers, timeout=3600)
        wf_input = WorkflowInput(steps=[step.model_dump()], initial_input="payload")

        async with await WorkflowEnvironment.start_time_skipping() as env:
            task_queue = "hitl-auth-test"
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[AgentWorkflow],
                activities=[fake_send_notification, fake_run_agent],
            ):
                handle = await env.client.start_workflow(
                    AgentWorkflow.run,
                    wf_input,
                    id=f"hitl-auth-{uuid.uuid4().hex[:8]}",
                    task_queue=task_queue,
                )

                # Wait for the approval to register, then grab its real request_id.
                request_id = None
                for _ in range(50):
                    pending = await handle.query(AgentWorkflow.get_pending_approvals)
                    if pending:
                        request_id = pending[0]["request_id"]
                        break
                assert request_id is not None, "approval step never became pending"

                for attempt in attempts:
                    await handle.signal(
                        AgentWorkflow.submit_approval,
                        ApprovalDecision(
                            request_id=request_id,
                            decision=attempt["decision"],
                            decided_by=attempt["decided_by"],
                        ),
                    )
                    if attempt.get("expect_pending_after"):
                        status = await handle.query(AgentWorkflow.get_status)
                        assert status["status"] == "waiting_for_approval", (
                            f"unauthorized decision by {attempt['decided_by']} should leave "
                            f"the step pending, got status={status['status']}"
                        )

                return await handle.result()

    async def test_unauthorized_then_authorized(self):
        """An unauthorized signal is ignored; a later authorized one approves."""
        result = await self._run_scenario(
            approvers=["alice"],
            attempts=[
                {"decided_by": "mallory", "decision": "approved", "expect_pending_after": True},
                {"decided_by": "alice", "decision": "approved", "expect_pending_after": False},
            ],
        )
        assert result.status == "completed"
        # Exactly one decision was recorded — the authorized one.
        assert [d.decided_by for d in result.approval_decisions] == ["alice"]

    async def test_authorized_rejection_is_honored(self):
        result = await self._run_scenario(
            approvers=["alice"],
            attempts=[
                {"decided_by": "alice", "decision": "rejected", "expect_pending_after": False},
            ],
        )
        assert result.status == "rejected"

    async def test_empty_approvers_accepts_anyone(self):
        result = await self._run_scenario(
            approvers=[],
            attempts=[
                {"decided_by": "anyone", "decision": "approved", "expect_pending_after": False},
            ],
        )
        assert result.status == "completed"
