"""
Sequential Agent Workflow -- convenience wrapper.

Takes a list of agent names, runs them in order.
Optionally inserts approval gates between steps.
"""

from __future__ import annotations

import asyncio
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

import uuid
from dataclasses import dataclass, field


@dataclass
class SequentialWorkflowInput:
    """Input for the sequential workflow."""

    agent_names: list[str]
    initial_input: str
    session_id: str | None = None
    user_id: str | None = None
    approval_between_steps: bool = False
    approval_timeout: int = 86400


@workflow.defn(sandboxed=False)
class SequentialAgentWorkflow:
    """Runs registered agents sequentially, passing output between them."""

    def __init__(self) -> None:
        self._status = "running"
        self._step_results: list[AgentActivityResult] = []
        self._approval_decisions: list[ApprovalDecision] = []
        self._pending_approvals: list[dict[str, Any]] = []
        self._cancelled = False
        self._pending_decision: ApprovalDecision | None = None

    @workflow.signal
    async def submit_approval(self, decision: ApprovalDecision) -> None:
        self._pending_decision = decision

    @workflow.signal
    async def cancel_workflow(self) -> None:
        self._cancelled = True

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        return {"status": self._status, "completed_steps": len(self._step_results)}

    @workflow.query
    def get_pending_approvals(self) -> list[dict[str, Any]]:
        return list(self._pending_approvals)

    @workflow.run
    async def run(self, input: SequentialWorkflowInput) -> WorkflowResult:
        last_output = input.initial_input

        for i, agent_name in enumerate(input.agent_names):
            if self._cancelled:
                return WorkflowResult(
                    status="cancelled",
                    content=last_output,
                    step_results=self._step_results,
                )

            # Optional approval gate
            if input.approval_between_steps and i > 0:
                approved = await self._wait_for_approval(
                    step_index=i,
                    description=f"Approve before running {agent_name}",
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
                    agent_name=agent_name,
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
        request_id = f"seq-approval-{uuid.uuid4().hex[:12]}"
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
