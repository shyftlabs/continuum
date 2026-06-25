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

# --------------------------------------------------------------------------- #
# Pure submodules — no ``temporalio`` dependency. These import even when the
# optional ``[temporal]`` extra is absent, so ``continuum.temporal.types`` (the
# dataclasses + ``is_authorized`` predicate), config, and exceptions stay usable
# without the Temporal runtime.
# --------------------------------------------------------------------------- #
from continuum.temporal.config import TemporalConfig

# Exceptions
from continuum.temporal.exceptions import (
    AgentNotRegisteredError,
    ApprovalTimeoutError,
    TemporalActivityError,
    TemporalConnectionError,
    TemporalError,
    TemporalWorkflowError,
    WorkflowCancelledError,
)
from continuum.temporal.registry import AgentRegistry, get_agent_registry, reset_agent_registry

# Types
from continuum.temporal.types import (
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

# --------------------------------------------------------------------------- #
# Runtime submodules — require the optional ``temporalio`` extra
# (``pip install -e '.[temporal]'``). Guarded so that importing this package
# without the extra does not hard-fail; the runtime names are simply
# unavailable until ``temporalio`` is installed.
# --------------------------------------------------------------------------- #
try:
    from continuum.temporal.activities import run_agent_activity, send_notification_activity
    from continuum.temporal.client import (
        TemporalClient,
        get_temporal_client,
        reset_temporal_client,
    )
    from continuum.temporal.human_in_loop import ApprovalNotificationConfig, HumanInLoopManager
    from continuum.temporal.worker import WorkerManager, get_worker_manager, reset_worker_manager
    from continuum.temporal.workflows import (
        AgentWorkflow,
        LoopAgentWorkflow,
        ParallelAgentWorkflow,
        SequentialAgentWorkflow,
    )
except ImportError:  # temporalio not installed — pure types/config/exceptions still import.
    pass

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
