"""
Temporal Workflow Integration for the Orchestrator SDK.

Optional module -- requires ``pip install shyftlabs-continuum[temporal]``.

Provides:
- Agent Registry: register any BaseAgent for Temporal execution
- TemporalClient: ergonomic wrapper over temporalio.client.Client
- WorkerManager: start/stop workers with built-in activities & workflows
- Generic AgentWorkflow: declarative step-based workflow execution
- Human-in-the-Loop: approval gates, notification hooks, escalation
- Convenience workflows: Sequential, Parallel, Loop patterns
"""

# Core
# Activities
from orchestrator.temporal.activities import run_agent_activity, send_notification_activity
from orchestrator.temporal.client import TemporalClient, get_temporal_client, reset_temporal_client
from orchestrator.temporal.config import TemporalConfig

# Exceptions
from orchestrator.temporal.exceptions import (
    AgentNotRegisteredError,
    ApprovalTimeoutError,
    TemporalActivityError,
    TemporalConnectionError,
    TemporalError,
    TemporalWorkflowError,
    WorkflowCancelledError,
)

# Human-in-the-loop
from orchestrator.temporal.human_in_loop import ApprovalNotificationConfig, HumanInLoopManager
from orchestrator.temporal.registry import AgentRegistry, get_agent_registry, reset_agent_registry

# Types
from orchestrator.temporal.types import (
    AgentActivityParams,
    AgentActivityResult,
    AgentStep,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStep,
    ConditionalStep,
    NotificationParams,
    ParallelStep,
    StepType,
    WaitStep,
    WorkflowInput,
    WorkflowResult,
    WorkflowStep,
    parse_step,
)
from orchestrator.temporal.worker import WorkerManager, get_worker_manager, reset_worker_manager

# Workflows
from orchestrator.temporal.workflows import (
    AgentWorkflow,
    LoopAgentWorkflow,
    ParallelAgentWorkflow,
    SequentialAgentWorkflow,
)

__all__ = [
    # Core
    "TemporalClient",
    "get_temporal_client",
    "reset_temporal_client",
    "TemporalConfig",
    "AgentRegistry",
    "get_agent_registry",
    "reset_agent_registry",
    "WorkerManager",
    "get_worker_manager",
    "reset_worker_manager",
    # Human-in-the-loop
    "HumanInLoopManager",
    "ApprovalNotificationConfig",
    # Types
    "StepType",
    "AgentStep",
    "ApprovalStep",
    "ParallelStep",
    "ConditionalStep",
    "WaitStep",
    "WorkflowStep",
    "parse_step",
    "AgentActivityParams",
    "AgentActivityResult",
    "NotificationParams",
    "ApprovalRequest",
    "ApprovalDecision",
    "WorkflowInput",
    "WorkflowResult",
    # Activities
    "run_agent_activity",
    "send_notification_activity",
    # Workflows
    "AgentWorkflow",
    "SequentialAgentWorkflow",
    "ParallelAgentWorkflow",
    "LoopAgentWorkflow",
    # Exceptions
    "TemporalError",
    "TemporalConnectionError",
    "TemporalWorkflowError",
    "TemporalActivityError",
    "AgentNotRegisteredError",
    "ApprovalTimeoutError",
    "WorkflowCancelledError",
]
