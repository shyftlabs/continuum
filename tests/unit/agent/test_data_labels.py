"""
Tests for data_labels propagation through handoffs (RunContext.data_labels).
"""

from __future__ import annotations

from orchestrator.agent.types import RunContext


class TestRunContextDataLabels:
    def test_default_is_empty_set(self):
        ctx = RunContext(run_id="r1")
        assert ctx.data_labels == set()
        assert isinstance(ctx.data_labels, set)

    def test_set_on_construction(self):
        ctx = RunContext(run_id="r1", data_labels={"pii", "financial"})
        assert "pii" in ctx.data_labels
        assert "financial" in ctx.data_labels

    def test_labels_in_to_dict(self):
        ctx = RunContext(run_id="r1", data_labels={"pii"})
        d = ctx.to_dict()
        assert "data_labels" in d
        assert "pii" in d["data_labels"]

    def test_to_dict_labels_are_sorted(self):
        ctx = RunContext(run_id="r1", data_labels={"z_label", "a_label", "m_label"})
        d = ctx.to_dict()
        assert d["data_labels"] == sorted(["z_label", "a_label", "m_label"])

    def test_copy_preserves_labels(self):
        ctx = RunContext(run_id="r1", data_labels={"pii", "confidential"})
        copied = ctx.data_labels.copy()
        assert copied == {"pii", "confidential"}
        # Mutation of copy doesn't affect original
        copied.add("new_label")
        assert "new_label" not in ctx.data_labels

    def test_handoff_propagates_labels(self):
        """Simulates what handoff_executor does: copy data_labels to new context."""
        source_ctx = RunContext(run_id="r1", data_labels={"pii", "financial"})
        # Simulate handoff_executor creating new context with copied labels
        target_ctx = RunContext(run_id="r2", data_labels=source_ctx.data_labels.copy())
        assert target_ctx.data_labels == {"pii", "financial"}

    def test_handoff_copy_is_independent(self):
        """Mutations to target context don't affect source."""
        source_ctx = RunContext(run_id="r1", data_labels={"pii"})
        target_ctx = RunContext(run_id="r2", data_labels=source_ctx.data_labels.copy())
        target_ctx.data_labels.add("new_label")
        assert "new_label" not in source_ctx.data_labels

    def test_empty_labels_propagate_correctly(self):
        source_ctx = RunContext(run_id="r1", data_labels=set())
        target_ctx = RunContext(run_id="r2", data_labels=source_ctx.data_labels.copy())
        assert target_ctx.data_labels == set()

    def test_labels_mutable_after_creation(self):
        ctx = RunContext(run_id="r1")
        ctx.data_labels.add("pii")
        assert "pii" in ctx.data_labels
