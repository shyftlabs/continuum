"""
Decision-trace recorder.

A :class:`TraceRecorder` lives on the :class:`RunContext` for a run. The executor
(and, later, handoff/workflow code) calls ``record_*`` at the same points where
spans are already opened; each recorded step is stamped with the current Langfuse
``span_id`` so the trace and the observability view stay in sync — one source of
truth, two projections.

The recorder is intentionally dumb and synchronous: appending a step is a cheap
in-memory operation that never does I/O and never raises into the run. When
tracing is disabled, no recorder is created and the executor's ``if recorder``
guards skip everything.
"""

from __future__ import annotations

from typing import Any

from continuum.agent.trace.types import DecisionStep, DecisionTrace, StepKind
from continuum.observability.trace_context import get_current_span_id


class TraceRecorder:
    """Collects :class:`DecisionStep` records and assembles a :class:`DecisionTrace`."""

    def __init__(
        self, run_id: str, root_agent: str, user_query: str = "", *, checkpoint: bool = False
    ) -> None:
        self._trace = DecisionTrace(run_id=run_id, root_agent=root_agent, user_query=user_query)
        self._counter = 0
        self.checkpoint = checkpoint

    # -- recording --------------------------------------------------------- #
    def record(
        self,
        kind: StepKind,
        agent_name: str,
        *,
        turn: int = 0,
        parent_id: str | None = None,
        agent_stack: list[str] | None = None,
        input: Any = None,
        decision: Any = None,
        rationale: str | None = None,
        output: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: int = 0,
        status: str = "ok",
        error: str | None = None,
        messages_snapshot: list[Any] | None = None,
        data_labels: set[str] | list[str] | None = None,
    ) -> str:
        """Append a step and return its id (so callers can nest children under it).

        ``agent_stack`` is the caller's handoff stack (root → … → current agent)
        at the moment of recording. It is passed in per call — never read from a
        shared field — so concurrent agents (Parallel/Scatter) can't clobber each
        other's stack.
        """
        self._counter += 1
        step = DecisionStep(
            step_id=f"s{self._counter}",
            kind=kind,
            agent_name=agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=list(agent_stack) if agent_stack else [],
            input=input,
            decision=decision,
            rationale=rationale,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error,
            span_id=get_current_span_id(),
            messages_snapshot=messages_snapshot,
            data_labels=sorted(data_labels) if data_labels else [],
        )
        self._trace.add(step)
        return step.step_id

    # -- convenience wrappers (the executor's vocabulary) ------------------ #
    def record_llm_call(
        self,
        agent_name: str,
        turn: int,
        *,
        output: str = "",
        parent_id: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: int = 0,
        decision: Any = None,
        messages_snapshot: list[Any] | None = None,
        agent_stack: list[str] | None = None,
        data_labels: set[str] | list[str] | None = None,
    ) -> str:
        return self.record(
            StepKind.LLM_CALL,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            output=output,
            decision=decision,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            messages_snapshot=messages_snapshot if self.checkpoint else None,
            data_labels=data_labels,
        )

    def record_reasoning(
        self,
        agent_name: str,
        turn: int,
        thought: str,
        *,
        parent_id: str | None = None,
        agent_stack: list[str] | None = None,
    ) -> str:
        return self.record(
            StepKind.REASONING,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            decision="think",
            rationale=thought,
        )

    def record_tool_call(
        self,
        agent_name: str,
        turn: int,
        tool_name: str,
        args: Any,
        output: Any,
        *,
        parent_id: str | None = None,
        latency_ms: int = 0,
        status: str = "ok",
        error: str | None = None,
        agent_stack: list[str] | None = None,
    ) -> str:
        return self.record(
            StepKind.TOOL_CALL,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            input={"tool": tool_name, "args": args},
            decision=f"call {tool_name}",
            output=output,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        turn: int,
        reason: str = "",
        *,
        parent_id: str | None = None,
        agent_stack: list[str] | None = None,
        return_to_parent: bool = False,
    ) -> str:
        self._trace.handoff_chain.append(to_agent)
        return self.record(
            StepKind.HANDOFF,
            from_agent,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            # return_to_parent is recorded so fork() can refuse to resume a step
            # inside a return-to-parent child (it can't reconstruct the parent's
            # synthesis) instead of silently returning the child's partial answer.
            decision={"handoff_to": to_agent, "return_to_parent": return_to_parent},
            rationale=reason,
        )

    def record_memory_write(
        self,
        agent_name: str,
        facts: Any,
        *,
        turn: int = 0,
        parent_id: str | None = None,
        agent_stack: list[str] | None = None,
    ) -> str:
        """Record that facts were written to long-term memory (the write side of
        MEMORY_RETRIEVAL)."""
        return self.record(
            StepKind.MEMORY_WRITE,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            decision="memory_write",
            output=facts,
        )

    def record_guardrail(
        self,
        agent_name: str,
        scanner: str,
        *,
        turn: int = 0,
        parent_id: str | None = None,
        blocked: bool = False,
        modified: bool = False,
        detail: Any = None,
        agent_stack: list[str] | None = None,
    ) -> str:
        """Record that an input/output guardrail/scanner ran (PII, injection, …)
        and whether it blocked or modified the content."""
        return self.record(
            StepKind.GUARDRAIL,
            agent_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            decision={"scanner": scanner, "blocked": blocked, "modified": modified},
            output=detail,
            status="error" if blocked else "ok",
        )

    def record_workflow_step(
        self,
        workflow_name: str,
        *,
        stage: int,
        label: str = "",
        turn: int = 0,
        parent_id: str | None = None,
        agent_stack: list[str] | None = None,
    ) -> str:
        """Record a workflow stage / iteration boundary (e.g. Sequential stage N,
        Loop iteration N). Purely structural — marks where a stage begins."""
        return self.record(
            StepKind.WORKFLOW_STEP,
            workflow_name,
            turn=turn,
            parent_id=parent_id,
            agent_stack=agent_stack,
            decision={"stage": stage, "label": label},
        )

    def absorb(
        self,
        steps: list[Any],
        *,
        stage: int,
        label: str,
        orchestrator_name: str,
    ) -> None:
        """Merge a concurrent branch's recorded steps into this trace as one
        ordered, contiguous segment (Phase 6 — ordered capture).

        Concurrent branches (Parallel/Scatter/Debate) each record into their own
        isolated recorder so their steps don't interleave. After they finish, the
        orchestrator calls this once per branch, in a deterministic order: it
        records a ``WORKFLOW_STEP`` marker carrying the branch's ``stage`` index,
        then appends the branch's steps with fresh ids (remapping ``parent_id``)
        and ``orchestrator_name`` prepended to each step's ``agent_stack``. The
        result is a stable, segmentable trace regardless of real wall-clock
        interleaving.
        """
        self.record_workflow_step(
            orchestrator_name, stage=stage, label=label, agent_stack=[orchestrator_name]
        )
        id_map: dict[str, str] = {}
        for st in steps:
            old_id = st.step_id
            self._counter += 1
            st.step_id = f"s{self._counter}"
            id_map[old_id] = st.step_id
            if st.parent_id is not None and st.parent_id in id_map:
                st.parent_id = id_map[st.parent_id]
            if orchestrator_name not in (st.agent_stack or []):
                st.agent_stack = [orchestrator_name, *(st.agent_stack or [])]
            self._trace.add(st)

    # -- assembly ---------------------------------------------------------- #
    def build_trace(
        self,
        *,
        final_response: str = "",
        status: str = "success",
        completed_at: Any = None,
    ) -> DecisionTrace:
        self._trace.final_response = final_response
        self._trace.status = status
        self._trace.completed_at = completed_at
        return self._trace

    @property
    def trace(self) -> DecisionTrace:
        return self._trace

    def last_step_id(self) -> str | None:
        return self._trace.steps[-1].step_id if self._trace.steps else None
