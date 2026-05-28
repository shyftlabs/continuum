"""
Serializable types for the Temporal integration.

All types are Pydantic models suitable for Temporal payloads.
No agent objects, no callables -- only plain serializable data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Step types
# ---------------------------------------------------------------------------


class StepType(str, Enum):
    AGENT = "agent"
    APPROVAL = "approval"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    WAIT = "wait"


class AgentStep(BaseModel):
    type: Literal["agent"] = "agent"
    agent_name: str
    input: str | None = None
    timeout: int = 300
    retries: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalStep(BaseModel):
    type: Literal["approval"] = "approval"
    description: str
    approvers: list[str] = Field(default_factory=list)
    timeout: int = 86400
    auto_approve_if: str | None = None


class ParallelStep(BaseModel):
    type: Literal["parallel"] = "parallel"
    agents: list[AgentStep]
    merge_strategy: str = "concatenate"


class ConditionalStep(BaseModel):
    type: Literal["conditional"] = "conditional"
    condition_agent: str
    condition_input: str | None = None
    if_true: list[dict[str, Any]] = Field(default_factory=list)
    if_false: list[dict[str, Any]] = Field(default_factory=list)
    timeout: int = 300
    retries: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class WaitStep(BaseModel):
    type: Literal["wait"] = "wait"
    duration_seconds: int = Field(ge=1, le=86400 * 7)  # Min 1s, max 7 days


WorkflowStep = AgentStep | ApprovalStep | ParallelStep | ConditionalStep | WaitStep


def parse_step(data: dict[str, Any]) -> WorkflowStep:
    """Parse a serialized step dict into the appropriate step model."""
    step_type = data.get("type")
    if step_type == "agent":
        return AgentStep.model_validate(data)
    elif step_type == "approval":
        return ApprovalStep.model_validate(data)
    elif step_type == "parallel":
        return ParallelStep.model_validate(data)
    elif step_type == "conditional":
        return ConditionalStep.model_validate(data)
    elif step_type == "wait":
        return WaitStep.model_validate(data)
    else:
        raise ValueError(f"Unknown step type: {step_type}")


# ---------------------------------------------------------------------------
# Activity params / results
# ---------------------------------------------------------------------------


class AgentActivityParams(BaseModel):
    agent_name: str
    input: str
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class AgentActivityResult(BaseModel):
    content: str
    status: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    agents_used: list[str] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_agent_response(cls, resp: Any) -> AgentActivityResult:
        """Create from an AgentResponse dataclass."""
        # Build usage dict with explicit None checks upfront
        usage: dict[str, int] = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": int(resp.usage.prompt_tokens or 0),
                "completion_tokens": int(resp.usage.completion_tokens or 0),
                "total_tokens": int(resp.usage.total_tokens or 0),
            }

        return cls(
            content=resp.content or "",
            status=resp.status.value if hasattr(resp.status, "value") else str(resp.status),
            structured_output=(
                resp.structured_output.model_dump() if resp.structured_output else None
            ),
            usage=usage,
            agents_used=list(resp.agents_used) if resp.agents_used else [],
            error=resp.error,
        )


class NotificationParams(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Approval types
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    request_id: str
    workflow_id: str
    step_index: int
    description: str
    context: str | None = None
    approvers: list[str] = Field(default_factory=list)
    timeout: int = 86400
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ApprovalDecision(BaseModel):
    request_id: str
    decision: str  # "approved", "rejected", "escalated"
    decided_by: str
    reason: str | None = None
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# Workflow input / output
# ---------------------------------------------------------------------------


class WorkflowInput(BaseModel):
    steps: list[dict[str, Any]]
    initial_input: str = Field(max_length=2_000_000)  # 2MB max to stay within Temporal gRPC limits
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    status: str  # completed, rejected, timed_out, failed, cancelled
    content: str | None = None
    step_results: list[AgentActivityResult] = Field(default_factory=list)
    approval_decisions: list[ApprovalDecision] = Field(default_factory=list)
    error: str | None = None
