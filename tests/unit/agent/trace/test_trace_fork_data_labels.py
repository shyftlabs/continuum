"""
Fork preserves data-label taint (per-step variant).

A run tainted with data labels (e.g. {"pii"}) records those labels on each
LLM-call step. fork() seeds the resumed run's RunContext from the forked step's
labels, so a replayed run is gated the same way the original was — closing the
"fork starts with empty labels" under-enforcement gap.

Covers: DecisionStep serialization round-trip (+ back-compat for old traces with
no field), the recorder storing labels, and fork seeding the resumed context.
"""

from __future__ import annotations

from continuum.agent.trace.types import SCHEMA_VERSION, DecisionStep, DecisionTrace, StepKind


class TestDecisionStepDataLabels:
    def test_round_trips(self):
        step = DecisionStep(
            step_id="s1", kind=StepKind.LLM_CALL, agent_name="a", data_labels=["phi", "pii"]
        )
        restored = DecisionStep.from_dict(step.to_dict())
        assert restored.data_labels == ["phi", "pii"]

    def test_old_trace_without_field_defaults_empty(self):
        # A trace serialized before this field existed has no "data_labels" key.
        legacy = {"step_id": "s1", "kind": "llm_call", "agent_name": "a"}
        restored = DecisionStep.from_dict(legacy)
        assert restored.data_labels == []

    def test_schema_version_bumped(self):
        # The persisted format changed (new field), so the version must advance.
        assert SCHEMA_VERSION >= 2


class TestRecorderRecordsLabels:
    def test_record_llm_call_stores_sorted_labels(self):
        from continuum.agent.trace.recorder import TraceRecorder

        rec = TraceRecorder("run-1", "agent-a", "q", checkpoint=True)
        rec.record_llm_call("agent-a", 1, output="hi", data_labels={"pii", "financial"})
        step = rec.trace.steps[-1]
        assert step.data_labels == ["financial", "pii"]  # sorted


class TestForkSeedsLabels:
    async def test_fork_seeds_context_data_labels_from_step(self, monkeypatch):
        from continuum.agent.base import BaseAgent
        from continuum.agent.runner import AgentRunner
        from continuum.agent.trace import config as trace_config
        from continuum.agent.types import AgentResponse, ResponseStatus
        from continuum.config import settings

        monkeypatch.setattr(settings, "decision_trace_store", "memory")
        monkeypatch.setattr(settings, "decision_trace_enabled", True)
        trace_config.get_trace_store.cache_clear()
        store = trace_config.get_trace_store()

        # Parent run: an LLM-call step (the fork resume point) tainted {"pii"}.
        parent = DecisionTrace(run_id="p1", root_agent="a")
        parent.steps.append(
            DecisionStep(
                step_id="s1",
                kind=StepKind.LLM_CALL,
                agent_name="a",
                messages_snapshot=[{"role": "user", "content": "hi"}],
                data_labels=["pii"],
            )
        )
        await store.save(parent)

        runner = AgentRunner()
        agent = BaseAgent(name="a", instructions="t")

        captured: dict = {}

        async def fake_loop(agent, messages, context, run_state):
            captured["labels"] = set(context.data_labels)
            return AgentResponse(content="ok", agent_name=agent.name, status=ResponseStatus.SUCCESS)

        monkeypatch.setattr(runner._executor, "execute_loop", fake_loop)

        await runner.fork("p1", "s1", agent=agent)

        # The resumed run started with the forked step's taint — not empty.
        assert captured["labels"] == {"pii"}

        trace_config.get_trace_store.cache_clear()
