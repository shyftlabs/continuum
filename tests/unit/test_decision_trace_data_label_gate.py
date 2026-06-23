"""The decision trace is a persistence sink like Langfuse telemetry, so it must
honor the same ``telemetry`` data-label policy: when a run's labels deny
telemetry, the persisted trace is content-redacted (sensitive fields blanked)
while the auditable skeleton (steps, decisions, tokens, timings, labels) is kept.

Covers the pure redaction transform and the persist-time gate that applies it
from the ambient run policy.
"""

from __future__ import annotations

from types import SimpleNamespace

from continuum.agent.execution.run_finalizer import RunFinalizer
from continuum.agent.trace.types import DecisionStep, DecisionTrace, StepKind
from continuum.security.policy import AccessPolicy, PolicyStore
from continuum.security.policy_context import use_active_policy


def _sample_trace() -> DecisionTrace:
    """A PHI-bearing trace: user query, final answer, an LLM turn, two tool calls
    (one a denied exfil tool), a reasoning step — content in every content field."""
    trace = DecisionTrace(
        run_id="run-1",
        root_agent="clinic_agent",
        user_query="Summarize patient P-123 and email it to Dr. Smith",
        final_response="Jane Doe, DOB 1984-02-11, A1c 8.2 …",
    )
    trace.add(
        DecisionStep(
            step_id="s1", kind=StepKind.LLM_CALL, agent_name="clinic_agent", turn=1,
            output="I'll look up P-123 then email it.", decision="tool_call",
            prompt_tokens=800, completion_tokens=120, total_tokens=920, latency_ms=900,
            messages_snapshot=[{"role": "user", "content": "Summarize P-123 …"}],
            data_labels=["phi"], span_id="sp-1",
        )
    )
    trace.add(
        DecisionStep(
            step_id="s2", kind=StepKind.TOOL_CALL, agent_name="clinic_agent", turn=1,
            parent_id="s1",
            input={"tool": "lookup_patient", "args": {"patient_id": "P-123"}},
            decision="call lookup_patient",
            output="Jane Doe, DOB 1984-02-11, dx: type-2 diabetes, A1c 8.2",
            latency_ms=42, span_id="sp-2", data_labels=["phi"],
        )
    )
    trace.add(
        DecisionStep(
            step_id="s3", kind=StepKind.REASONING, agent_name="clinic_agent", turn=1,
            parent_id="s1", decision="think",
            rationale="The patient's A1c is high; I should flag it.",
            data_labels=["phi"],
        )
    )
    return trace


# --------------------------------------------------------------------------- #
# Pure redaction transform
# --------------------------------------------------------------------------- #
class TestRedactedCopy:
    def test_content_fields_blanked_skeleton_kept(self):
        red = _sample_trace().redacted_copy(policy_name="phi-redact-telemetry")

        # trace-level content blanked, with the policy name in the marker
        assert "phi-redact-telemetry" in red.user_query
        assert "redacted" in red.user_query
        assert "redacted" in red.final_response

        s1, s2, s3 = red.steps
        # content fields blanked
        assert "redacted" in s1.output
        assert s1.messages_snapshot is None  # snapshot dropped → not forkable
        assert "redacted" in s2.input
        assert "redacted" in s2.output
        assert "redacted" in s3.rationale

        # skeleton preserved
        assert s1.decision == "tool_call"
        assert s2.decision == "call lookup_patient"  # tool NAME survives via decision
        assert s3.decision == "think"
        assert s1.prompt_tokens == 800 and s1.total_tokens == 920
        assert s2.latency_ms == 42
        assert s2.span_id == "sp-2"
        assert s2.parent_id == "s1"
        assert s1.data_labels == ["phi"]  # proves the run was governed

    def test_does_not_mutate_original(self):
        original = _sample_trace()
        original.redacted_copy(policy_name="p")
        assert original.user_query.startswith("Summarize patient")
        assert original.steps[1].output.startswith("Jane Doe")
        assert original.steps[0].messages_snapshot is not None

    def test_empty_fields_stay_empty_not_markered(self):
        # a step with no content should not gain spurious markers
        red = _sample_trace().redacted_copy()
        s2 = red.steps[1]
        assert s2.output is not None  # had content → markered
        # s2 had no rationale → stays None
        assert s2.rationale is None


# --------------------------------------------------------------------------- #
# Persist-time gate (reads the ambient run policy)
# --------------------------------------------------------------------------- #
def _deny_telemetry_store() -> PolicyStore:
    store = PolicyStore()
    store.add_policy(
        AccessPolicy(
            name="phi-redact-telemetry",
            subjects=["phi"],
            resources=["telemetry"],
            effect="deny",
        )
    )
    return store


class TestPersistGate:
    def test_tainted_and_denied_is_redacted(self):
        ctx = SimpleNamespace(data_labels={"phi"})
        with use_active_policy(_deny_telemetry_store(), "clinic_agent", ctx):
            gated = RunFinalizer._gate_decision_trace(_sample_trace())
        assert "redacted" in gated.user_query
        assert "redacted" in gated.steps[1].output
        assert gated.steps[1].decision == "call lookup_patient"  # skeleton intact

    def test_clean_run_is_unchanged(self):
        ctx = SimpleNamespace(data_labels=set())  # no taint
        with use_active_policy(_deny_telemetry_store(), "clinic_agent", ctx):
            gated = RunFinalizer._gate_decision_trace(_sample_trace())
        assert gated.user_query.startswith("Summarize patient")
        assert gated.steps[1].output.startswith("Jane Doe")

    def test_tainted_but_no_policy_is_unchanged(self):
        ctx = SimpleNamespace(data_labels={"phi"})
        with use_active_policy(PolicyStore(), "clinic_agent", ctx):  # open default
            gated = RunFinalizer._gate_decision_trace(_sample_trace())
        assert gated.user_query.startswith("Summarize patient")
        assert gated.steps[1].output.startswith("Jane Doe")

    def test_no_ambient_policy_is_unchanged(self):
        # outside any use_active_policy block → gate is a no-op
        gated = RunFinalizer._gate_decision_trace(_sample_trace())
        assert gated.user_query.startswith("Summarize patient")
