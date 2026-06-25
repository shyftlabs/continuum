"""
Decision-trace data model.

A run is an ordered list of :class:`DecisionStep` records that form a tree via
``parent_id`` (turns nest under nothing; a handoff's sub-steps nest under the
handoff step). Everything here is JSON-serializable and round-trips exactly so a
trace can be attached to a response, persisted durably, and read back later.

The model is intentionally framework-agnostic: it records *what was decided* and
*why*, not Langfuse/Redis specifics. Each step carries a ``span_id`` so a trace
links back to its Langfuse span without depending on Langfuse being enabled.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any

SCHEMA_VERSION = 2


class StepKind(str, Enum):
    """The kind of decision a step represents."""

    LLM_CALL = "llm_call"  # a model turn that produced content/tool calls
    REASONING = "reasoning"  # a ReAct "think" step or two-pass reasoning output
    TOOL_CALL = "tool_call"  # a tool was invoked
    HANDOFF = "handoff"  # control transferred to another agent
    MEMORY_RETRIEVAL = "memory_retrieval"  # long-term memories were pulled in
    MEMORY_WRITE = "memory_write"  # facts were written to long-term memory
    ROUTING = "routing"  # smart-layer model-tier / route decision
    GUARDRAIL = "guardrail"  # an input/output scanner ran (PII, injection, …)
    WORKFLOW_STEP = "workflow_step"  # a workflow stage / iteration boundary


class TraceDetail(str, Enum):
    """How much of the trace is attached to the response.

    The full trace is always persisted; this only controls the in-response copy.
    """

    OFF = "off"  # capture + persist, but attach nothing to the response
    FULL = "full"  # everything inlined (prompts, outputs)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def redaction_marker(policy_name: str | None) -> str:
    """The placeholder written in place of redacted trace content."""
    marker = "[redacted by data-label policy"
    if policy_name:
        marker += f" '{policy_name}'"
    return marker + "]"


@dataclass
class DecisionStep:
    """One decision made during a run.

    ``input``/``output`` hold the data entering and leaving the step; ``decision``
    is the choice made (e.g. which tool, which handoff target) and ``rationale``
    is the *why* (the ReAct thought or reasoning text), when available. ``input``
    is what makes deterministic replay / fork possible later: it is the resume
    point for this step.
    """

    step_id: str
    kind: StepKind
    agent_name: str
    turn: int = 0
    parent_id: str | None = None

    # The handoff stack active when this step ran (root → … → current agent),
    # e.g. ["triage", "refund-officer", "fraud-review"]. Lets fork() resume the
    # step's agent with the correct depth/cycle state in a multi-agent run.
    agent_stack: list[str] = field(default_factory=list)

    input: Any = None
    decision: Any = None
    rationale: str | None = None
    output: Any = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

    status: str = "ok"  # "ok" | "error"
    error: str | None = None

    # Link back to the Langfuse span for this decision (one source of truth).
    span_id: str | None = None
    started_at: datetime = field(default_factory=_utcnow)

    # Resume point for fork: the exact message array sent to the LLM at this
    # step. Only populated on LLM_CALL steps when DECISION_TRACE_CHECKPOINT is on
    # (it is heavy); kept in the persisted trace so fork() can replay from here.
    messages_snapshot: list[dict[str, Any]] | None = None

    # Data-sensitivity labels (sorted) active going into this step — the run's
    # accumulated taint at the resume point. fork() seeds the resumed context
    # from these so a replayed run is gated like the original (taint isn't lost
    # for the steps fork replays-from-checkpoint rather than re-executes).
    data_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["started_at"] = self.started_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionStep:
        data = copy.deepcopy(data)
        kind = data.get("kind")
        if isinstance(kind, str):
            data["kind"] = StepKind(kind)
        started = data.get("started_at")
        if isinstance(started, str):
            data["started_at"] = datetime.fromisoformat(started)
        return cls(**data)

    def redacted_copy(self, marker: str) -> DecisionStep:
        """Return a copy with sensitive *content* blanked, skeleton preserved.

        Content fields (``input``/``output``/``rationale``/``messages_snapshot``)
        may carry user/tool data, so they are replaced with ``marker`` (snapshot
        dropped — a redacted trace is intentionally not forkable). Everything that
        describes *what happened without exposing data* is kept: ``decision`` (a
        tool name / category / handoff target — not user data), token counts,
        timings, status, ids, ``agent_stack`` and ``data_labels``.
        """
        return replace(
            self,
            input=marker if self.input is not None else None,
            output=marker if self.output is not None else None,
            rationale=marker if self.rationale is not None else None,
            messages_snapshot=None,
        )


@dataclass
class DecisionTrace:
    """The ordered, tree-structured record of a single run."""

    run_id: str
    root_agent: str
    schema_version: int = SCHEMA_VERSION
    user_query: str = ""
    final_response: str = ""
    status: str = "success"  # mirrors ResponseStatus value
    steps: list[DecisionStep] = field(default_factory=list)
    handoff_chain: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None

    # Fork lineage. A forked run records where it came from; the parent is never
    # mutated. None for original (non-forked) runs.
    parent_run_id: str | None = None
    forked_from_step: str | None = None
    edit: dict[str, Any] | None = None

    def add(self, step: DecisionStep) -> DecisionStep:
        """Append a step and return it (so callers can keep its id)."""
        self.steps.append(step)
        return step

    def redacted_copy(self, policy_name: str | None = None) -> DecisionTrace:
        """Return a copy with sensitive content blanked for persistence.

        The decision trace is a persistence sink like Langfuse telemetry; when a
        run's data labels deny the ``telemetry`` resource, its content must not be
        stored. This keeps the auditable skeleton (steps, decisions, tokens,
        timings, lineage, ``data_labels``) while replacing ``user_query``,
        ``final_response`` and every step's content with a policy marker.
        """
        marker = redaction_marker(policy_name)
        return replace(
            self,
            user_query=marker if self.user_query else "",
            final_response=marker if self.final_response else "",
            steps=[s.redacted_copy(marker) for s in self.steps],
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "step_count": len(self.steps),
            "turn_count": max((s.turn for s in self.steps), default=0),
            "total_tokens": sum(s.total_tokens for s in self.steps),
            "total_latency_ms": sum(s.latency_ms for s in self.steps),
            "error_count": sum(1 for s in self.steps if s.status == "error"),
            "agents": sorted({s.agent_name for s in self.steps if s.agent_name}),
        }

    def to_dict(self, detail: TraceDetail = TraceDetail.FULL) -> dict[str, Any]:
        # `detail` is retained for API compatibility. OFF is handled by the caller
        # (run_finalizer skips attaching the trace), so it always serializes in full.
        steps = self.steps
        return {
            "run_id": self.run_id,
            "root_agent": self.root_agent,
            "schema_version": self.schema_version,
            "user_query": self.user_query,
            "final_response": self.final_response,
            "status": self.status,
            "handoff_chain": self.handoff_chain,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parent_run_id": self.parent_run_id,
            "forked_from_step": self.forked_from_step,
            "edit": self.edit,
            "steps": [s.to_dict() for s in steps],
            "metrics": self.metrics(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTrace:
        data = copy.deepcopy(data)
        data.pop("metrics", None)  # derived, not a field
        steps = [DecisionStep.from_dict(s) for s in data.pop("steps", [])]
        created = data.get("created_at")
        if isinstance(created, str):
            data["created_at"] = datetime.fromisoformat(created)
        completed = data.get("completed_at")
        if isinstance(completed, str):
            data["completed_at"] = datetime.fromisoformat(completed)
        return cls(steps=steps, **data)
