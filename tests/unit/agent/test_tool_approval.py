"""A human-in-the-loop gate before a declared tool call (security finding F7).

An approval workflow existed but fired only for a Temporal workflow step of type
"approval". The default ``AgentRunner.run()`` path -- the one every quick-start
uses -- had no gate at all: zero references to it under ``agent/``.

Neither existing control covers this. Tool-trust asks "is this server serving the
catalogue I vetted?" and is answered once, offline, per server. The policy gate
asks "is this run allowed this resource?" and checks ``tool:{name}``. So neither
can tell ``transfer(amount=5)`` from ``transfer(amount=5_000_000)``: the gate
here is the only one that sees the ARGUMENTS, which is the whole point of asking
a person.

No default list of risky tool names. A name-matcher both misses ``wire_funds``
and blocks ``send_receipt``; the operator declares, the runtime enforces.
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# The primitive.
# ---------------------------------------------------------------------------


class TestWhichToolsNeedApproval:
    def test_an_exact_name_matches(self):
        from continuum.agent.approval import needs_approval

        assert needs_approval("send_referral_email", {"send_referral_email"})

    def test_an_undeclared_tool_does_not(self):
        from continuum.agent.approval import needs_approval

        assert not needs_approval("clinic_info", {"send_referral_email"})

    def test_globs_match_like_policy_resources(self):
        """Declared the same way as policy `resources`, so an operator writes one
        kind of pattern rather than two."""
        from continuum.agent.approval import needs_approval

        assert needs_approval("pharmacy__check_interactions", {"pharmacy__*"})
        assert needs_approval("clinic__send_referral_email", {"*send_*"})
        assert not needs_approval("clinic__lookup_patient", {"*send_*"})

    def test_nothing_declared_means_no_gate(self):
        from continuum.agent.approval import needs_approval

        assert not needs_approval("anything", set())


class TestTheRequestCarriesWhatAReviewerNeeds:
    def test_arguments_are_present(self):
        """The reason this gate exists. Neither the policy gate nor tool-trust
        can see them, so neither can judge an amount."""
        from continuum.agent.approval import ToolApprovalRequest

        req = ToolApprovalRequest(
            tool_name="transfer_funds",
            arguments={"amount": 5_000_000},
            agent_name="banker",
            run_id="r1",
            data_labels=frozenset(),
        )
        assert req.arguments["amount"] == 5_000_000

    def test_the_run_taint_is_present(self):
        """'This run has read the public web' is exactly what a reviewer wants
        to know before approving an outbound action."""
        from continuum.agent.approval import ToolApprovalRequest

        req = ToolApprovalRequest(
            tool_name="send_referral_email",
            arguments={},
            agent_name="clinic",
            run_id="r1",
            data_labels=frozenset({"external"}),
        )
        assert "external" in req.data_labels

    def test_a_request_is_immutable(self):
        """It is handed to app code that may keep it; a handler must not be able
        to edit the arguments the gate is about to execute."""
        from continuum.agent.approval import ToolApprovalRequest

        req = ToolApprovalRequest(
            tool_name="t", arguments={}, agent_name="a", run_id=None, data_labels=frozenset()
        )
        with pytest.raises(Exception):
            req.tool_name = "something_else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Asking. Fails closed, because a gate that cannot answer has not approved.
# ---------------------------------------------------------------------------


def _request(tool: str = "send_referral_email", **kwargs):
    from continuum.agent.approval import ToolApprovalRequest

    return ToolApprovalRequest(
        tool_name=tool,
        arguments=kwargs or {"to": "dr@example.com"},
        agent_name="clinic",
        run_id="r1",
        data_labels=frozenset(),
    )


class TestAsking:
    async def test_an_approval_lets_the_call_through(self):
        from continuum.agent.approval import ToolApprovalDecision, request_approval

        async def yes(_req):
            return ToolApprovalDecision(approved=True, reviewer="alice")

        decision = await request_approval(_request(), yes, timeout=5)
        assert decision.approved
        assert decision.reviewer == "alice"

    async def test_a_refusal_is_reported_with_its_reason(self):
        from continuum.agent.approval import ToolApprovalDecision, request_approval

        async def no(_req):
            return ToolApprovalDecision(approved=False, reviewer="bob", reason="wrong recipient")

        decision = await request_approval(_request(), no, timeout=5)
        assert not decision.approved
        assert decision.reason == "wrong recipient"

    async def test_a_handler_that_raises_denies(self):
        """Fail closed. A handler that crashed has said nothing about the call,
        and proceeding would make the gate an illusion at exactly the moment it
        stopped working."""
        from continuum.agent.approval import request_approval

        async def broken(_req):
            raise RuntimeError("approval service unreachable")

        decision = await request_approval(_request(), broken, timeout=5)
        assert not decision.approved
        assert "approval service unreachable" in (decision.reason or "")

    async def test_a_timeout_denies(self):
        from continuum.agent.approval import request_approval

        async def slow(_req):
            await asyncio.sleep(10)

        decision = await request_approval(_request(), slow, timeout=0.05)
        assert not decision.approved
        assert "timed out" in (decision.reason or "").lower()

    async def test_a_handler_returning_junk_denies(self):
        """App code can return anything. A bare True, or None from a function
        that forgot to return, must not read as approval."""
        from continuum.agent.approval import request_approval

        for junk in (None, True, "yes", {"approved": True}):

            async def bad(_req, _j=junk):
                return _j

            decision = await request_approval(_request(), bad, timeout=5)
            assert not decision.approved, f"{junk!r} was treated as an approval"

    async def test_the_handler_sees_the_request(self):
        from continuum.agent.approval import ToolApprovalDecision, request_approval

        seen = []

        async def record(req):
            seen.append(req)
            return ToolApprovalDecision(approved=True)

        await request_approval(_request(amount=5_000_000), record, timeout=5)
        assert seen[0].arguments == {"amount": 5_000_000}


class TestSerialisation:
    async def test_two_approvals_in_one_turn_do_not_overlap(self):
        """Tool calls run through asyncio.gather, so without a lock two handlers
        fire at once -- two prompts racing for one terminal, or two modals with
        no indication they belong to the same turn."""
        from continuum.agent.approval import ToolApprovalDecision, request_approval

        concurrent = 0
        peak = 0

        async def slow_handler(_req):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.02)
            concurrent -= 1
            return ToolApprovalDecision(approved=True)

        await asyncio.gather(
            request_approval(_request("a"), slow_handler, timeout=5),
            request_approval(_request("b"), slow_handler, timeout=5),
            request_approval(_request("c"), slow_handler, timeout=5),
        )
        assert peak == 1, f"{peak} handlers ran at once; approvals must serialise"

    async def test_a_timeout_releases_the_lock(self):
        """Otherwise one unanswered prompt wedges every later approval."""
        from continuum.agent.approval import ToolApprovalDecision, request_approval

        async def slow(_req):
            await asyncio.sleep(10)

        async def quick(_req):
            return ToolApprovalDecision(approved=True)

        first, second = await asyncio.gather(
            request_approval(_request("a"), slow, timeout=0.05),
            request_approval(_request("b"), quick, timeout=5),
        )
        assert not first.approved
        assert second.approved, "the lock was not released after a timeout"


class TestTheInertMechanismWarning:
    def test_declaring_tools_without_a_handler_is_reported(self):
        """The F6 lesson: a control that is configured but wired to nothing fails
        by doing nothing, which is the failure nobody notices."""
        import logging

        from continuum.agent.approval import warn_if_approval_unwired

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        lg = logging.getLogger("continuum.agent.approval")
        handler = Capture(level=logging.WARNING)
        lg.addHandler(handler)
        try:
            warn_if_approval_unwired("clinic", {"send_referral_email"}, None)
            warn_if_approval_unwired("clinic", {"send_referral_email"}, None)
        finally:
            lg.removeHandler(handler)

        hits = [r for r in records if "approval" in r.getMessage().lower()]
        assert len(hits) == 1, f"expected one warning per agent, got {len(hits)}"
        assert "send_referral_email" in hits[0].getMessage()

    def test_no_warning_when_properly_wired(self):
        import logging

        from continuum.agent.approval import ToolApprovalDecision, warn_if_approval_unwired

        async def handler(_req):
            return ToolApprovalDecision(approved=True)

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        lg = logging.getLogger("continuum.agent.approval")
        h = Capture(level=logging.WARNING)
        lg.addHandler(h)
        try:
            warn_if_approval_unwired("wired-agent", {"send_referral_email"}, handler)
        finally:
            lg.removeHandler(h)
        assert not records

    def test_no_warning_when_nothing_is_declared(self):
        """The shipped default. Nothing declared is not a misconfiguration."""
        import logging

        from continuum.agent.approval import warn_if_approval_unwired

        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        lg = logging.getLogger("continuum.agent.approval")
        h = Capture(level=logging.WARNING)
        lg.addHandler(h)
        try:
            warn_if_approval_unwired("plain-agent", set(), None)
        finally:
            lg.removeHandler(h)
        assert not records


# ---------------------------------------------------------------------------
# The config surface.
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_the_defaults_gate_nothing(self):
        """Backward compatible: an app that knows nothing of this behaves exactly
        as before."""
        from continuum.agent.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.tool_approval == set()
        assert cfg.approval_handler is None

    def test_the_timeout_default_is_http_safe(self):
        """A run blocked on a human holds an HTTP request open. The default has
        to sit inside ordinary proxy limits (~60s) or approvals look like hangs.
        """
        from continuum.agent.config import AgentConfig

        assert 0 < AgentConfig().approval_timeout <= 60


class TestDeferral:
    """A deferred call is not a refused one, and the difference has to be
    representable or the model tells the user the wrong thing.

    Without this, a handler that parks a request for a human to answer later has
    only ``approved=False`` to return, so the model is told APPROVAL DENIED and
    reports a refusal for an action that is merely waiting. That is the whole of
    the refuse-and-resume pattern: the turn ends, someone answers out of band,
    the user asks again.
    """

    async def test_a_decision_can_say_deferred(self):
        from continuum.agent.approval import ToolApprovalDecision

        d = ToolApprovalDecision(approved=False, deferred=True, reason="queued for review")
        assert d.deferred
        assert not d.approved

    async def test_deferred_defaults_off_so_a_refusal_stays_a_refusal(self):
        from continuum.agent.approval import ToolApprovalDecision

        assert not ToolApprovalDecision(approved=False).deferred
        assert not ToolApprovalDecision(approved=True).deferred

    async def test_an_approved_decision_cannot_also_be_deferred(self):
        """Approved-and-deferred has no meaning: either the call proceeds now or
        it does not. Allowing both would leave the executor guessing."""
        from continuum.agent.approval import ToolApprovalDecision

        with pytest.raises(ValueError, match="deferred"):
            ToolApprovalDecision(approved=True, deferred=True)

    async def test_a_deferring_handler_round_trips(self):
        from continuum.agent.approval import ToolApprovalDecision, request_approval

        parked: list = []

        async def defer(req):
            parked.append(req)
            return ToolApprovalDecision(
                approved=False, deferred=True, reason="sent to the on-call clinician"
            )

        decision = await request_approval(_request(), defer, timeout=5)
        assert decision.deferred
        assert len(parked) == 1

    async def test_a_timeout_is_a_refusal_not_a_deferral(self):
        """Nobody answered inside the window, so the action did not happen and
        is not queued. Reporting it as pending would promise a resumption that
        nothing is going to deliver."""
        from continuum.agent.approval import request_approval

        async def never(_req):
            await asyncio.sleep(10)

        decision = await request_approval(_request(), never, timeout=0.05)
        assert not decision.approved
        assert not decision.deferred

    async def test_a_raising_handler_is_a_refusal_not_a_deferral(self):
        from continuum.agent.approval import request_approval

        async def broken(_req):
            raise RuntimeError("queue unreachable")

        decision = await request_approval(_request(), broken, timeout=5)
        assert not decision.approved
        assert not decision.deferred
