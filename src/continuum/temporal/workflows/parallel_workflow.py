"""
Parallel Agent Workflow -- convenience wrapper.

Takes a list of agent names, runs them all concurrently, merges results.
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from temporalio import workflow
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

with workflow.unsafe.imports_passed_through():
    from continuum.temporal.types import (
        AgentActivityParams,
        AgentActivityResult,
        OnStepError,
        WorkflowResult,
    )
    from continuum.temporal.workflows.steps import run_builtin_agent_step, step_retries_enabled


from dataclasses import dataclass, field


@dataclass
class ParallelWorkflowInput:
    """Input for the parallel workflow."""

    agent_names: list[str]
    initial_input: str
    session_id: str | None = None
    user_id: str | None = None
    merge_strategy: str = "concatenate"
    timeout_per_agent: int = 300
    # "continue" completes with whatever merged (the 2.x behaviour); "fail"
    # returns status="failed" naming the failed agents, with the rest merged.
    on_step_error: OnStepError = "continue"
    # Agents that run once, never retried: agents with side effects.
    single_attempt_agents: list[str] = field(default_factory=list)


@workflow.defn(sandboxed=False)
class ParallelAgentWorkflow:
    """Runs registered agents concurrently and merges results."""

    def __init__(self) -> None:
        self._status = "running"
        self._cancelled = False

    @workflow.signal
    async def cancel_workflow(self) -> None:
        self._cancelled = True

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        return {"status": self._status}

    @workflow.run
    async def run(self, input: ParallelWorkflowInput) -> WorkflowResult:
        results: list[AgentActivityResult] = list(
            await asyncio.gather(
                *(
                    run_builtin_agent_step(
                        AgentActivityParams(
                            agent_name=agent_name,
                            input=input.initial_input,
                            session_id=input.session_id,
                            user_id=input.user_id,
                        ),
                        timeout=input.timeout_per_agent,
                        retries=1 if agent_name in input.single_attempt_agents else 3,
                        # With "fail" a timed-out agent is a failed result and the others merge.
                        raise_timeouts=input.on_step_error == "continue",
                    )
                    for agent_name in input.agent_names
                )
            )
        )
        failed = [
            f"{r.agents_used[0] if r.agents_used else '?'}: {r.error}"
            for r in results
            if r.status == "error"
        ]

        # Merge
        if input.merge_strategy == "first_success":
            content = ""
            for r in results:
                if r.status != "error":
                    content = r.content
                    break
        elif input.merge_strategy == "structured":
            parts = {}
            for i, r in enumerate(results):
                key = input.agent_names[i] if i < len(input.agent_names) else f"agent-{i}"
                parts[key] = r.content
            content = str(parts)
        else:
            content = "\n\n".join(r.content for r in results if r.content)

        if failed and input.on_step_error == "fail" and step_retries_enabled():
            self._status = "failed"
            return WorkflowResult(
                status="failed",
                content=content,
                step_results=results,
                error="; ".join(failed),
            )

        self._status = "completed"
        return WorkflowResult(
            status="completed",
            content=content,
            step_results=results,
        )
