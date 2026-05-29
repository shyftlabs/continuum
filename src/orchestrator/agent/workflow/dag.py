"""
DAGAgent — dependency-aware parallel workflow.

Stages with no dependencies start immediately.  A stage whose dependencies
are all satisfied starts as soon as the last one finishes.  Independent
branches run in parallel automatically.

Example::

    from orchestrator.agent.workflow import DAGAgent, create_dag_agent

    dag = create_dag_agent(
        name="research-pipeline",
        stages=[
            ("fetch_a",    fetch_a_agent,    []),
            ("fetch_b",    fetch_b_agent,    []),
            ("synthesize", synthesize_agent, ["fetch_a", "fetch_b"]),
            ("format",     format_agent,     ["synthesize"]),
        ],
    )

    result = await runner.run(dag, "Research topic X")

Or build incrementally::

    dag = DAGAgent(name="my-dag")
    dag.add_stage("a", agent_a)
    dag.add_stage("b", agent_b)
    dag.add_stage("c", agent_c, depends_on=["a", "b"])
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.exceptions import WorkflowError
from orchestrator.agent.types import (
    AgentResponse,
    FailStrategy,
    MergeStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)
from orchestrator.logging import get_logger
from orchestrator.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DAGError(WorkflowError):
    """Base error for DAGAgent failures."""


class DAGCycleError(DAGError):
    """Raised when a cycle is detected in the dependency graph."""

    def __init__(self, cycle_path: list[str], **kwargs: Any) -> None:
        path_str = " → ".join(cycle_path)
        super().__init__(f"Cycle detected in DAG: {path_str}", **kwargs)
        self.cycle_path = cycle_path


class DAGStageError(DAGError):
    """Raised when a stage fails and fail_strategy is FAIL_FAST."""

    def __init__(self, stage_id: str, cause: Exception, **kwargs: Any) -> None:
        super().__init__(
            f"Stage '{stage_id}' failed: {cause}",
            original_error=cause,
            **kwargs,
        )
        self.stage_id = stage_id
        self.context["failed_agent"] = stage_id


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------


@dataclass
class DAGStage:
    """One node in the DAG.

    Attributes:
        stage_id:   Unique identifier used in ``depends_on`` lists.
        agent:      The agent to run for this stage.
        depends_on: IDs of stages that must complete before this one starts.
    """

    stage_id: str
    agent: BaseAgent
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DAGAgent
# ---------------------------------------------------------------------------


@dataclass
class DAGAgent(BaseAgent):
    """Executes agents in dependency order with automatic parallelism.

    Stages whose dependencies are all satisfied run concurrently.
    Each stage receives the output of its predecessors as input
    (or the original input if it has no dependencies).

    Merge behaviour when a stage has multiple predecessors:
    - ``MergeStrategy.CONCATENATE`` (default): predecessor outputs joined with
      double newlines.
    - ``MergeStrategy.STRUCTURED``: JSON dict mapping predecessor stage IDs
      to their outputs.

    The final result is the output of the terminal stage(s) (those with no
    successors), merged the same way if there are multiple.
    """

    # Stages are added via add_stage(), not passed to __init__.
    # Using a private dict so the dataclass repr stays clean.
    _stages: dict[str, DAGStage] = field(default_factory=dict, init=False, repr=False)

    # How to combine outputs when a stage has multiple predecessors,
    # and how to combine outputs of multiple terminal stages.
    merge_strategy: MergeStrategy = MergeStrategy.CONCATENATE

    # Whether to stop immediately when any stage raises an exception.
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST

    def __post_init__(self) -> None:
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError
            raise AgentConfigurationError("DAGAgent name is required")

    # ------------------------------------------------------------------
    # Public build API
    # ------------------------------------------------------------------

    def add_stage(
        self,
        stage_id: str,
        agent: BaseAgent,
        depends_on: list[str] | None = None,
    ) -> DAGAgent:
        """Register a stage.

        Args:
            stage_id:   Unique ID for this stage (used in other stages'
                        ``depends_on`` lists).
            agent:      Agent to run.
            depends_on: Stage IDs that must finish before this one starts.

        Returns:
            ``self`` so calls can be chained.

        Raises:
            ValueError: If ``stage_id`` is already registered, or if any ID
                        in ``depends_on`` has not been registered yet.
        """
        if stage_id in self._stages:
            raise ValueError(f"DAGAgent '{self.name}': stage '{stage_id}' already registered")

        deps = depends_on or []
        for dep in deps:
            if dep not in self._stages:
                raise ValueError(
                    f"DAGAgent '{self.name}': stage '{stage_id}' depends on "
                    f"'{dep}' which has not been registered yet"
                )

        self._stages[stage_id] = DAGStage(stage_id=stage_id, agent=agent, depends_on=deps)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
        **_: Any,
    ) -> AgentResponse:
        if not self._stages:
            from orchestrator.agent.exceptions import AgentConfigurationError
            raise AgentConfigurationError(f"DAGAgent '{self.name}' has no stages")

        self._validate_no_cycles()

        async with SpanScope(
            f"workflow.dag.{self.name}",
            input={
                "input_preview": input_text[:500] if input_text else None,
                "stage_count": len(self._stages),
                "stages": list(self._stages.keys()),
            },
            metadata={"workflow_type": "dag"},
        ) as workflow_span:

            context.suppress_session_log = True
            results = await self._run_dag(input_text, runner, context)

            # Collect successful results and any errors
            successful: dict[str, AgentResponse] = {}
            errors: dict[str, str] = {}
            for sid, res in results.items():
                if isinstance(res, AgentResponse):
                    successful[sid] = res
                else:
                    errors[sid] = str(res)

            if not successful:
                return AgentResponse(
                    content="All DAG stages failed",
                    agent_name=self.name,
                    status=ResponseStatus.ERROR,
                    error="; ".join(f"{k}: {v}" for k, v in errors.items()),
                )

            # Terminal stages: those with no successors
            all_successor_deps: set[str] = set()
            for s in self._stages.values():
                all_successor_deps.update(s.depends_on)
            terminal_ids = [
                sid for sid in self._stages
                if sid not in all_successor_deps and sid in successful
            ]

            if len(terminal_ids) == 1:
                content = successful[terminal_ids[0]].content or ""
            else:
                terminal_results = {sid: successful[sid] for sid in terminal_ids if sid in successful}
                content = self._merge({sid: r.content or "" for sid, r in terminal_results.items()})

            total_usage = TokenUsage()
            for resp in successful.values():
                total_usage = total_usage.add(resp.usage)

            workflow_span.set_output({
                "success": True,
                "stages_executed": list(successful.keys()),
                "stages_failed": list(errors.keys()),
                "total_tokens": total_usage.total_tokens if total_usage else 0,
            })

            if context.session_id:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=content,
                    agent=None,
                )

            return AgentResponse(
                content=content,
                agent_name=self.name,
                status=ResponseStatus.SUCCESS,
                usage=total_usage,
                agents_used=list(successful.keys()),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_dag(
        self,
        original_input: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> dict[str, AgentResponse | Exception]:
        """Execute all stages with asyncio.Event-based dependency gating.

        Every stage task starts immediately but awaits an event for each of
        its dependencies before calling the agent.  A stage sets its own
        event when it finishes, unblocking its dependants.
        """
        results: dict[str, AgentResponse | Exception] = {}
        events: dict[str, asyncio.Event] = {sid: asyncio.Event() for sid in self._stages}
        abort = asyncio.Event()  # set on FAIL_FAST to stop waiting stages early

        async def run_stage(stage: DAGStage) -> None:
            # Wait for all dependency events (or abort signal)
            for dep in stage.depends_on:
                # Wait until the dependency finishes OR abort is triggered
                await _wait_any(events[dep], abort)
                if abort.is_set():
                    return

            # Gather predecessor outputs as this stage's input
            if not stage.depends_on:
                stage_input = original_input
            else:
                pred_outputs: dict[str, str] = {}
                for dep in stage.depends_on:
                    r = results.get(dep)
                    pred_outputs[dep] = (r.content or "") if isinstance(r, AgentResponse) else ""
                stage_input = self._merge(pred_outputs)

            async with SpanScope(
                f"workflow.dag.stage.{stage.stage_id}",
                input={"stage_id": stage.stage_id, "depends_on": stage.depends_on},
            ):
                try:
                    response = await runner.run(
                        agent=stage.agent,
                        input=stage_input,
                        context=context,
                    )
                    results[stage.stage_id] = response
                except Exception as e:
                    logger.error(f"DAGAgent '{self.name}': stage '{stage.stage_id}' failed: {e}")
                    results[stage.stage_id] = e
                    if self.fail_strategy == FailStrategy.FAIL_FAST:
                        abort.set()
                finally:
                    events[stage.stage_id].set()  # always unblock dependants

        await asyncio.gather(*[run_stage(s) for s in self._stages.values()])

        # If FAIL_FAST and there was an error, raise
        if self.fail_strategy == FailStrategy.FAIL_FAST:
            for sid, res in results.items():
                if isinstance(res, Exception):
                    raise DAGStageError(sid, res)

        return results

    def _merge(self, outputs: dict[str, str]) -> str:
        """Combine multiple predecessor outputs into one input string."""
        if not outputs:
            return ""
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        if self.merge_strategy == MergeStrategy.STRUCTURED:
            return json.dumps(outputs, indent=2)
        # Default: CONCATENATE
        return "\n\n".join(f"## {sid}\n{text}" for sid, text in outputs.items())

    def _validate_no_cycles(self) -> None:
        """DFS cycle detection. Raises DAGCycleError if a cycle exists."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in self._stages}
        path: list[str] = []

        def dfs(sid: str) -> None:
            color[sid] = GRAY
            path.append(sid)
            for dep in self._stages[sid].depends_on:
                if color[dep] == GRAY:
                    cycle_start = path.index(dep)
                    raise DAGCycleError(path[cycle_start:] + [dep])
                if color[dep] == WHITE:
                    dfs(dep)
            path.pop()
            color[sid] = BLACK

        for sid in self._stages:
            if color[sid] == WHITE:
                dfs(sid)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "stages": [
                {"stage_id": s.stage_id, "agent": s.agent.name, "depends_on": s.depends_on}
                for s in self._stages.values()
            ],
            "merge_strategy": self.merge_strategy.value,
            "fail_strategy": self.fail_strategy.value,
            "workflow_type": "dag",
        })
        return base


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


async def _wait_any(*events: asyncio.Event) -> None:
    """Return as soon as any of the given events is set."""
    tasks = [asyncio.create_task(e.wait()) for e in events]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    except Exception:
        for t in tasks:
            t.cancel()
        raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_dag_agent(
    name: str,
    stages: list[tuple[str, BaseAgent, list[str]]],
    *,
    merge_strategy: MergeStrategy = MergeStrategy.CONCATENATE,
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST,
) -> DAGAgent:
    """Create a DAGAgent from a flat list of ``(stage_id, agent, depends_on)`` tuples.

    Stages are registered in list order, so each stage's dependencies must
    appear earlier in the list.

    Example::

        dag = create_dag_agent(
            name="pipeline",
            stages=[
                ("fetch_a",    agent_a, []),
                ("fetch_b",    agent_b, []),
                ("synthesize", agent_c, ["fetch_a", "fetch_b"]),
                ("format",     agent_d, ["synthesize"]),
            ],
        )
    """
    dag = DAGAgent(name=name, merge_strategy=merge_strategy, fail_strategy=fail_strategy)
    for stage_id, agent, depends_on in stages:
        dag.add_stage(stage_id, agent, depends_on)
    return dag
