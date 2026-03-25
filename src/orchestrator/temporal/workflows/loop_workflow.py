"""
Loop Agent Workflow -- convenience wrapper.

Runs a single agent repeatedly until a termination condition is met.
Supports max iterations and optional approval per iteration.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from orchestrator.temporal.types import (
        AgentActivityParams,
        AgentActivityResult,
        ApprovalDecision,
        NotificationParams,
        WorkflowResult,
    )


from dataclasses import dataclass


@dataclass
class LoopWorkflowInput:
    """Input for the loop workflow."""

    agent_name: str
    initial_input: str
    max_iterations: int = 10
    termination_phrase: str = "COMPLETE"
    session_id: str | None = None
    user_id: str | None = None
    approval_per_iteration: bool = False
    approval_timeout: int = 86400


@workflow.defn(sandboxed=False)
class LoopAgentWorkflow:
    """Runs a single agent repeatedly until termination condition."""

    def __init__(self) -> None:
        self._status = "running"
        self._step_results: list[AgentActivityResult] = []
        self._approval_decisions: list[ApprovalDecision] = []
        self._pending_approvals: list[dict[str, Any]] = []
        self._cancelled = False
        self._pending_decision: ApprovalDecision | None = None
        self._iteration = 0

    @workflow.signal
    async def submit_approval(self, decision: ApprovalDecision) -> None:
        self._pending_decision = decision

    @workflow.signal
    async def cancel_workflow(self) -> None:
        self._cancelled = True

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "iteration": self._iteration,
            "completed_steps": len(self._step_results),
        }

    @workflow.query
    def get_pending_approvals(self) -> list[dict[str, Any]]:
        return list(self._pending_approvals)

    @workflow.run
    async def run(self, input: LoopWorkflowInput) -> WorkflowResult:
        last_output = input.initial_input

        for iteration in range(input.max_iterations):
            if self._cancelled:
                return WorkflowResult(
                    status="cancelled",
                    content=last_output,
                    step_results=self._step_results,
                )

            self._iteration = iteration + 1

            # Optional approval gate
            if input.approval_per_iteration and iteration > 0:
                approved = await self._wait_for_approval(
                    step_index=iteration,
                    description=f"Approve iteration {iteration + 1} of {input.agent_name}",
                    context=last_output,
                    timeout=input.approval_timeout,
                )
                if not approved:
                    return WorkflowResult(
                        status="rejected",
                        content=last_output,
                        step_results=self._step_results,
                        approval_decisions=self._approval_decisions,
                    )

            raw = await workflow.execute_activity(
                "run_agent_activity",
                AgentActivityParams(
                    agent_name=input.agent_name,
                    input=last_output,
                    session_id=input.session_id,
                    user_id=input.user_id,
                ),
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=60),
                ),
                heartbeat_timeout=timedelta(seconds=60),
                result_type=AgentActivityResult,
            )
            result = raw if isinstance(raw, AgentActivityResult) else AgentActivityResult.model_validate(raw)
            self._step_results.append(result)

            if result.content:
                last_output = result.content

            # Check termination
            if input.termination_phrase.lower() in (result.content or "").lower():
                break

        self._status = "completed"
        return WorkflowResult(
            status="completed",
            content=last_output,
            step_results=self._step_results,
            approval_decisions=self._approval_decisions,
        )

    async def _wait_for_approval(
        self,
        step_index: int,
        description: str,
        context: str,
        timeout: int,
    ) -> bool:
        request_id = f"loop-approval-{uuid.uuid4().hex[:12]}"
        info: dict[str, Any] = {
            "request_id": request_id,
            "workflow_id": workflow.info().workflow_id,
            "step_index": step_index,
            "description": description,
            "context": context,
            "timeout": timeout,
        }
        self._pending_approvals.append(info)
        self._status = "waiting_for_approval"
        self._pending_decision = None

        try:
            await workflow.execute_activity(
                "send_notification_activity",
                NotificationParams(type="approval_required", payload=info),
                start_to_close_timeout=timedelta(seconds=30),
            )
        except Exception as e:
            workflow.logger.warning(
                f"Notification activity failed for approval {request_id}: {e}. "
                "Workflow continues without notification."
            )

        try:
            await workflow.wait_condition(
                lambda: self._pending_decision is not None or self._cancelled,
                timeout=timedelta(seconds=timeout),
            )
        except asyncio.TimeoutError:
            self._pending_approvals = [
                a for a in self._pending_approvals if a["request_id"] != request_id
            ]
            return False

        if self._cancelled:
            return False

        decision = self._pending_decision
        self._pending_decision = None
        self._pending_approvals = [
            a for a in self._pending_approvals if a["request_id"] != request_id
        ]

        if decision:
            self._approval_decisions.append(decision)
            self._status = "running"
            return decision.decision == "approved"

        return False
