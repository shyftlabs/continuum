"""
Generic AgentWorkflow -- the core Temporal workflow.

Interprets a declarative list of steps and executes any user-defined agents.
Supports: agent, approval, parallel, conditional, wait step types.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
except ImportError as _err:
    raise ImportError(
        "temporalio is required for Temporal support. "
        "Install it with: pip install -e '.[temporal]'"
    ) from _err

with workflow.unsafe.imports_passed_through():
    from orchestrator.temporal.types import (
        AgentActivityParams,
        AgentActivityResult,
        AgentStep,
        ApprovalDecision,
        ApprovalStep,
        ConditionalStep,
        NotificationParams,
        ParallelStep,
        WaitStep,
        WorkflowInput,
        WorkflowResult,
        parse_step,
    )


@workflow.defn(sandboxed=False)
class AgentWorkflow:
    """Generic workflow that interprets a declarative step list.

    The workflow knows nothing about what agents do internally. Agents are
    looked up by name from the registry at activity execution time.
    """

    def __init__(self) -> None:
        self._status = "running"
        self._current_step_index = 0
        self._step_results: list[AgentActivityResult] = []
        self._approval_decisions: list[ApprovalDecision] = []
        self._pending_approvals: list[dict[str, Any]] = []
        self._cancelled = False
        self._pending_decision: ApprovalDecision | None = None
        self._injected_input: dict[str, Any] | None = None
        self._last_output: str = ""

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    @workflow.signal
    async def submit_approval(self, decision: ApprovalDecision) -> None:
        """Human submits approval/rejection."""
        self._pending_decision = decision

    @workflow.signal
    async def cancel_workflow(self) -> None:
        """Cancel the workflow."""
        self._cancelled = True

    @workflow.signal
    async def inject_input(self, data: dict[str, Any]) -> None:
        """Inject data mid-workflow."""
        self._injected_input = data

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        """Current step, overall status, completed steps."""
        return {
            "status": self._status,
            "current_step_index": self._current_step_index,
            "total_steps": getattr(self, "_total_steps", 0),
            "completed_steps": len(self._step_results),
            "cancelled": self._cancelled,
        }

    @workflow.query
    def get_pending_approvals(self) -> list[dict[str, Any]]:
        """List of pending approval requests."""
        return list(self._pending_approvals)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowResult:
        self._last_output = input.initial_input
        self._total_steps = len(input.steps)

        # Validate ALL steps upfront before executing any of them
        # This prevents partially-executed workflows from bad step definitions
        try:
            for i, step_dict in enumerate(input.steps):
                try:
                    parse_step(step_dict)
                except (ValueError, Exception) as e:
                    self._status = "failed"
                    return WorkflowResult(
                        status="failed",
                        content=None,
                        step_results=[],
                        error=f"Step {i} validation failed: {e}",
                    )
        except Exception as e:
            self._status = "failed"
            return WorkflowResult(
                status="failed",
                content=None,
                step_results=[],
                error=f"Workflow step validation error: {e}",
            )

        try:
            for idx, step_dict in enumerate(input.steps):
                if self._cancelled:
                    self._status = "cancelled"
                    return WorkflowResult(
                        status="cancelled",
                        content=self._last_output or None,
                        step_results=self._step_results,
                        approval_decisions=self._approval_decisions,
                    )

                self._current_step_index = idx
                step = parse_step(step_dict)

                if isinstance(step, AgentStep):
                    await self._run_agent_step(step, input)
                elif isinstance(step, ApprovalStep):
                    approved = await self._run_approval_step(step, idx, input)
                    if not approved:
                        # Use self._status which is already set to "timed_out"
                        # or "rejected" by _run_approval_step
                        final_status = self._status if self._status in ("timed_out", "cancelled") else "rejected"
                        self._status = final_status
                        return WorkflowResult(
                            status=final_status,
                            content=self._last_output or None,
                            step_results=self._step_results,
                            approval_decisions=self._approval_decisions,
                        )
                elif isinstance(step, ParallelStep):
                    await self._run_parallel_step(step, input)
                elif isinstance(step, ConditionalStep):
                    await self._run_conditional_step(step, input)
                elif isinstance(step, WaitStep):
                    await workflow.sleep(step.duration_seconds)

            self._status = "completed"
            return WorkflowResult(
                status="completed",
                content=self._last_output or None,
                step_results=self._step_results,
                approval_decisions=self._approval_decisions,
            )

        except Exception as e:
            self._status = "failed"
            return WorkflowResult(
                status="failed",
                content=self._last_output or None,
                step_results=self._step_results,
                approval_decisions=self._approval_decisions,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    async def _run_agent_step(
        self, step: AgentStep, wf_input: WorkflowInput
    ) -> None:
        agent_input = step.input or self._last_output

        raw = await workflow.execute_activity(
            "run_agent_activity",
            AgentActivityParams(
                agent_name=step.agent_name,
                input=agent_input,
                session_id=wf_input.session_id,
                user_id=wf_input.user_id,
                metadata=step.metadata,
            ),
            start_to_close_timeout=timedelta(seconds=step.timeout),
            retry_policy=RetryPolicy(
                maximum_attempts=step.retries,
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
            self._last_output = result.content

    async def _run_approval_step(
        self,
        step: ApprovalStep,
        step_index: int,
        wf_input: WorkflowInput,
    ) -> bool:
        """Returns True if approved, False if rejected/timed-out."""
        request_id = f"approval-{uuid.uuid4().hex[:12]}"
        approval_info: dict[str, Any] = {
            "request_id": request_id,
            "workflow_id": workflow.info().workflow_id,
            "step_index": step_index,
            "description": step.description,
            "context": self._last_output,
            "approvers": step.approvers,
            "timeout": step.timeout,
        }
        self._pending_approvals.append(approval_info)

        # Send notification (failure shouldn't block the workflow, but should be logged)
        try:
            await workflow.execute_activity(
                "send_notification_activity",
                NotificationParams(
                    type="approval_required",
                    payload=approval_info,
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )
        except Exception as e:
            workflow.logger.warning(
                f"Notification activity failed for approval {request_id}: {e}. "
                "Workflow continues without notification."
            )

        self._status = "waiting_for_approval"
        self._pending_decision = None

        # Wait for decision or timeout
        try:
            await workflow.wait_condition(
                lambda: self._pending_decision is not None or self._cancelled,
                timeout=timedelta(seconds=step.timeout),
            )
        except asyncio.TimeoutError:
            self._pending_approvals = [
                a for a in self._pending_approvals if a["request_id"] != request_id
            ]
            self._status = "timed_out"
            return False

        if self._cancelled:
            return False

        decision = self._pending_decision
        self._pending_decision = None
        self._pending_approvals = [
            a for a in self._pending_approvals if a["request_id"] != request_id
        ]

        if decision:
            # Validate that the decision's request_id matches the current approval
            if decision.request_id != request_id:
                workflow.logger.warning(
                    f"Approval decision request_id mismatch: "
                    f"expected '{request_id}', got '{decision.request_id}'. "
                    f"Ignoring mismatched decision."
                )
                return False
            self._approval_decisions.append(decision)
            self._status = "running"
            return decision.decision == "approved"

        return False

    async def _run_parallel_step(
        self, step: ParallelStep, wf_input: WorkflowInput
    ) -> None:
        """Run multiple agents concurrently and merge results."""
        handles = []
        for agent_step in step.agents:
            agent_input = agent_step.input or self._last_output
            handle = workflow.start_activity(
                "run_agent_activity",
                AgentActivityParams(
                    agent_name=agent_step.agent_name,
                    input=agent_input,
                    session_id=wf_input.session_id,
                    user_id=wf_input.user_id,
                    metadata=agent_step.metadata,
                ),
                start_to_close_timeout=timedelta(seconds=agent_step.timeout),
                retry_policy=RetryPolicy(
                    maximum_attempts=agent_step.retries,
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=60),
                ),
                heartbeat_timeout=timedelta(seconds=60),
                result_type=AgentActivityResult,
            )
            handles.append(handle)

        raw_results = await asyncio.gather(*handles)
        results = [
            r if isinstance(r, AgentActivityResult) else AgentActivityResult.model_validate(r)
            for r in raw_results
        ]
        self._step_results.extend(results)

        # Merge strategy
        if step.merge_strategy == "first_success":
            for r in results:
                if r.status != "error":
                    self._last_output = r.content
                    break
        elif step.merge_strategy == "structured":
            import json as _json
            # Use indexed keys to prevent collision when agents share a name
            parts = {}
            for i, r in enumerate(results):
                agent_key = r.agents_used[0] if r.agents_used else f"agent-{i}"
                # Append index suffix if key already exists (collision prevention)
                unique_key = agent_key
                suffix = 1
                while unique_key in parts:
                    unique_key = f"{agent_key}_{suffix}"
                    suffix += 1
                parts[unique_key] = r.content
            # Use JSON serialization instead of str() to preserve structure
            self._last_output = _json.dumps(parts)
        else:
            self._last_output = "\n\n".join(r.content for r in results if r.content)

    async def _run_conditional_step(
        self, step: ConditionalStep, wf_input: WorkflowInput
    ) -> None:
        """Run condition agent and branch.

        The condition agent must return exactly "true" or "false" (case-insensitive).
        The raw result is stored as a side-effect to preserve Temporal determinism:
        on replay the same activity result is replayed, so the branch decision
        is consistent across replays.
        """
        raw_cond = await workflow.execute_activity(
            "run_agent_activity",
            AgentActivityParams(
                agent_name=step.condition_agent,
                input=self._last_output,
                session_id=wf_input.session_id,
                user_id=wf_input.user_id,
                metadata=step.metadata,
            ),
            start_to_close_timeout=timedelta(seconds=step.timeout),
            retry_policy=RetryPolicy(
                maximum_attempts=step.retries,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=60),
            ),
            heartbeat_timeout=timedelta(seconds=60),
            result_type=AgentActivityResult,
        )
        condition_result = raw_cond if isinstance(raw_cond, AgentActivityResult) else AgentActivityResult.model_validate(raw_cond)
        self._step_results.append(condition_result)

        # Deterministic evaluation: only accept explicit true/false values
        # to keep Temporal replay deterministic (LLM output is already fixed
        # in the event history on replay, so the same branch is always taken)
        condition_value = (condition_result.content or "").strip().lower()
        is_true = condition_value in ("true", "yes", "1", "approved", "continue")

        branch_steps = step.if_true if is_true else step.if_false
        for step_dict in branch_steps:
            if self._cancelled:
                return
            parsed = parse_step(step_dict)
            # Handle ALL step types in branches, not just AgentStep
            if isinstance(parsed, AgentStep):
                await self._run_agent_step(parsed, wf_input)
            elif isinstance(parsed, ApprovalStep):
                approved = await self._run_approval_step(
                    parsed, self._current_step_index, wf_input
                )
                if not approved:
                    self._status = "rejected"
                    return
            elif isinstance(parsed, ParallelStep):
                await self._run_parallel_step(parsed, wf_input)
            elif isinstance(parsed, WaitStep):
                await workflow.sleep(parsed.duration_seconds)
            elif isinstance(parsed, ConditionalStep):
                await self._run_conditional_step(parsed, wf_input)
