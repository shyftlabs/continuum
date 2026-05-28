"""
Human-in-the-Loop Manager for Temporal workflows.

Provides a high-level API for managing human approvals that works
with any workflow implementing the approval signal pattern.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from orchestrator.temporal.client import TemporalClient
from orchestrator.temporal.types import ApprovalDecision, ApprovalRequest
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow


@dataclass
class ApprovalNotificationConfig:
    """Configuration for approval notifications."""

    handler: Callable[[ApprovalRequest], Awaitable[None]] | None = None
    webhook_url: str | None = None
    timeout_seconds: int = 86400
    escalation_timeout: int = 7200
    escalation_handler: Callable[[ApprovalRequest], Awaitable[None]] | None = None
    auto_approve_conditions: list[Callable[[ApprovalRequest], bool]] = field(default_factory=list)


class HumanInLoopManager:
    """High-level API for human-in-the-loop approvals.

    Works with any workflow that has a submit_approval signal handler
    and get_pending_approvals query handler.
    """

    def __init__(
        self,
        client: TemporalClient,
        notification_config: ApprovalNotificationConfig | None = None,
    ) -> None:
        self._client = client
        self._notification_config = notification_config or ApprovalNotificationConfig()

    async def approve(
        self,
        workflow_id: str,
        request_id: str,
        decided_by: str,
        reason: str = "",
    ) -> None:
        """Approve a pending request."""
        decision = ApprovalDecision(
            request_id=request_id,
            decision="approved",
            decided_by=decided_by,
            reason=reason or None,
            decided_at=datetime.now(UTC).isoformat(),
        )
        await self.submit_decision(workflow_id, decision)

    async def reject(
        self,
        workflow_id: str,
        request_id: str,
        decided_by: str,
        reason: str = "",
    ) -> None:
        """Reject a pending request."""
        decision = ApprovalDecision(
            request_id=request_id,
            decision="rejected",
            decided_by=decided_by,
            reason=reason or None,
            decided_at=datetime.now(UTC).isoformat(),
        )
        await self.submit_decision(workflow_id, decision)

    async def submit_decision(self, workflow_id: str, decision: ApprovalDecision) -> None:
        """Submit any approval decision."""
        handle = await self._client.get_workflow_handle(workflow_id)
        await handle.signal(AgentWorkflow.submit_approval, decision)

    async def get_pending_approvals(self, workflow_id: str) -> list[dict[str, Any]]:
        """Get pending approvals for a workflow."""
        handle = await self._client.get_workflow_handle(workflow_id)
        return await handle.query(AgentWorkflow.get_pending_approvals)

    async def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Get current workflow status."""
        handle = await self._client.get_workflow_handle(workflow_id)
        return await handle.query(AgentWorkflow.get_status)

    async def escalate(
        self,
        workflow_id: str,
        request_id: str,
        escalate_to: str,
    ) -> None:
        """Escalate a pending approval."""
        decision = ApprovalDecision(
            request_id=request_id,
            decision="escalated",
            decided_by=escalate_to,
            reason=f"Escalated to {escalate_to}",
            decided_at=datetime.now(UTC).isoformat(),
        )

        if self._notification_config.escalation_handler:
            pending = await self.get_pending_approvals(workflow_id)
            matched = False
            for req_dict in pending:
                if req_dict.get("request_id") == request_id:
                    req = ApprovalRequest.model_validate(req_dict)
                    await self._notification_config.escalation_handler(req)
                    matched = True
                    break
            if not matched:
                import logging

                logging.getLogger(__name__).warning(
                    f"Escalation: request_id '{request_id}' not found in pending "
                    f"approvals for workflow '{workflow_id}'. "
                    f"Available request_ids: {[r.get('request_id') for r in pending]}"
                )

        await self.submit_decision(workflow_id, decision)
