"""
Parallel Agent Workflow -- convenience wrapper.

Takes a list of agent names, runs them all concurrently, merges results.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. Install it with: pip install -e '.[temporal]'"
    ) from _err

with workflow.unsafe.imports_passed_through():
    from orchestrator.temporal.types import (
        AgentActivityParams,
        AgentActivityResult,
        WorkflowResult,
    )


from dataclasses import dataclass


@dataclass
class ParallelWorkflowInput:
    """Input for the parallel workflow."""

    agent_names: list[str]
    initial_input: str
    session_id: str | None = None
    user_id: str | None = None
    merge_strategy: str = "concatenate"
    timeout_per_agent: int = 300


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
        handles = []
        for agent_name in input.agent_names:
            handle = workflow.start_activity(
                "run_agent_activity",
                AgentActivityParams(
                    agent_name=agent_name,
                    input=input.initial_input,
                    session_id=input.session_id,
                    user_id=input.user_id,
                ),
                start_to_close_timeout=timedelta(seconds=input.timeout_per_agent),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=60),
                ),
                heartbeat_timeout=timedelta(seconds=60),
                result_type=AgentActivityResult,
            )
            handles.append(handle)

        raw_results = await asyncio.gather(*handles)
        results: list[AgentActivityResult] = [
            r if isinstance(r, AgentActivityResult) else AgentActivityResult.model_validate(r)
            for r in raw_results
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

        self._status = "completed"
        return WorkflowResult(
            status="completed",
            content=content,
            step_results=results,
        )
