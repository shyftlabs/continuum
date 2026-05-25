"""
Agent exceptions.

Defines all exception types for the agent orchestration module.
"""

from __future__ import annotations

from typing import Any

from orchestrator.exceptions import OrchestratorError


class AgentError(OrchestratorError):
    """Base exception for all agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, context=context or {})
        self.agent_name = agent_name
        self.run_id = run_id
        self.trace_id = trace_id
        self.original_error = original_error

        # Add to context
        if agent_name:
            self.context["agent_name"] = agent_name
        if run_id:
            self.context["run_id"] = run_id
        if trace_id:
            self.context["trace_id"] = trace_id


class AgentNotFoundError(AgentError):
    """Raised when an agent is not found in the registry."""

    def __init__(
        self,
        agent_name: str,
        message: str | None = None,
        **kwargs: Any,
    ):
        msg = message or f"Agent '{agent_name}' not found in registry"
        super().__init__(msg, agent_name=agent_name, **kwargs)


class AgentConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        if config_key:
            self.context["config_key"] = config_key


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""

    def __init__(
        self,
        message: str,
        turn: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.turn = turn
        if turn is not None:
            self.context["turn"] = turn


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""

    def __init__(
        self,
        message: str,
        timeout: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.timeout = timeout
        if timeout is not None:
            self.context["timeout"] = timeout


class MaxTurnsExceededError(AgentError):
    """Raised when maximum turns are exceeded."""

    def __init__(
        self,
        message: str | None = None,
        max_turns: int | None = None,
        current_turn: int | None = None,
        **kwargs: Any,
    ):
        msg = message or f"Maximum turns ({max_turns}) exceeded at turn {current_turn}"
        super().__init__(msg, **kwargs)
        self.max_turns = max_turns
        self.current_turn = current_turn
        self.partial_response: Any = None
        if max_turns is not None:
            self.context["max_turns"] = max_turns
        if current_turn is not None:
            self.context["current_turn"] = current_turn


# =============================================================================
# Handoff Errors
# =============================================================================


class HandoffError(AgentError):
    """Base exception for handoff-related errors."""

    def __init__(
        self,
        message: str,
        from_agent: str | None = None,
        to_agent: str | None = None,
        handoff_id: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.handoff_id = handoff_id

        if from_agent:
            self.context["from_agent"] = from_agent
        if to_agent:
            self.context["to_agent"] = to_agent
        if handoff_id:
            self.context["handoff_id"] = handoff_id


class HandoffNotAllowedError(HandoffError):
    """Raised when a handoff is not allowed."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        reason: str = "Handoff not defined",
        **kwargs: Any,
    ):
        message = f"Handoff from '{from_agent}' to '{to_agent}' not allowed: {reason}"
        super().__init__(message, from_agent=from_agent, to_agent=to_agent, **kwargs)
        self.reason = reason


class HandoffDepthExceededError(HandoffError):
    """Raised when handoff depth exceeds limit."""

    def __init__(
        self,
        current_depth: int,
        max_depth: int,
        **kwargs: Any,
    ):
        message = f"Handoff depth ({current_depth}) exceeds maximum ({max_depth})"
        super().__init__(message, **kwargs)
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.context["current_depth"] = current_depth
        self.context["max_depth"] = max_depth


class HandoffTargetNotFoundError(HandoffError):
    """Raised when handoff target agent is not found."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        **kwargs: Any,
    ):
        message = f"Handoff target '{to_agent}' not found (from '{from_agent}')"
        super().__init__(message, from_agent=from_agent, to_agent=to_agent, **kwargs)


class HandoffCycleDetectedError(HandoffError):
    """Raised when a cycle is detected in the handoff chain."""

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        agent_stack: list[str],
        **kwargs: Any,
    ):
        cycle_path = " → ".join(agent_stack + [to_agent])
        message = (
            f"Handoff cycle detected: {from_agent} → {to_agent}. "
            f"Agent '{to_agent}' already exists in handoff chain: {cycle_path}"
        )
        super().__init__(message, from_agent=from_agent, to_agent=to_agent, **kwargs)
        self.agent_stack = agent_stack
        self.context["agent_stack"] = agent_stack
        self.context["cycle_path"] = cycle_path


# =============================================================================
# Tool Errors
# =============================================================================


class AgentToolError(AgentError):
    """Raised when a tool call fails during agent execution."""

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.tool_args = tool_args

        if tool_name:
            self.context["tool_name"] = tool_name
        if tool_args:
            self.context["tool_args"] = tool_args


class ToolAccessDeniedError(AgentToolError):
    """Raised when an access policy denies a tool call."""

    def __init__(
        self,
        tool_name: str,
        policy_name: str | None = None,
        subject: str | None = None,
        denial_message: str = "",
        **kwargs: Any,
    ):
        message = f"Access denied: tool '{tool_name}' is blocked by policy"
        if policy_name:
            message += f" '{policy_name}'"
        super().__init__(message, tool_name=tool_name, **kwargs)
        if policy_name:
            self.context["policy_name"] = policy_name
        if subject:
            self.context["subject"] = subject
        if denial_message:
            self.context["denial_message"] = denial_message


class MemoryAccessDeniedError(AgentError):
    """Raised when an access policy denies a memory read or write."""

    def __init__(
        self,
        operation: str,
        scope: str | None = None,
        policy_name: str | None = None,
        **kwargs: Any,
    ):
        message = f"Access denied: memory {operation} is blocked by policy"
        if policy_name:
            message += f" '{policy_name}'"
        super().__init__(message, **kwargs)
        self.context["operation"] = operation
        if scope:
            self.context["scope"] = scope
        if policy_name:
            self.context["policy_name"] = policy_name


# =============================================================================
# Workflow Errors
# =============================================================================


class WorkflowError(AgentError):
    """Base exception for workflow-related errors."""

    def __init__(
        self,
        message: str,
        workflow_type: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.workflow_type = workflow_type
        self.step = step

        if workflow_type:
            self.context["workflow_type"] = workflow_type
        if step is not None:
            self.context["step"] = step


class SequentialWorkflowError(WorkflowError):
    """Raised when sequential workflow execution fails."""

    def __init__(
        self,
        message: str,
        failed_agent: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, workflow_type="sequential", **kwargs)
        self.failed_agent = failed_agent
        if failed_agent:
            self.context["failed_agent"] = failed_agent


class ParallelWorkflowError(WorkflowError):
    """Raised when parallel workflow execution fails."""

    def __init__(
        self,
        message: str,
        failed_agents: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, workflow_type="parallel", **kwargs)
        self.failed_agents = failed_agents or []
        if failed_agents:
            self.context["failed_agents"] = failed_agents


class LoopWorkflowError(WorkflowError):
    """Raised when loop workflow execution fails."""

    def __init__(
        self,
        message: str,
        iteration: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, workflow_type="loop", **kwargs)
        self.iteration = iteration
        if iteration is not None:
            self.context["iteration"] = iteration


class LoopMaxIterationsError(LoopWorkflowError):
    """Raised when loop reaches maximum iterations without termination."""

    def __init__(
        self,
        max_iterations: int,
        **kwargs: Any,
    ):
        message = f"Loop reached maximum iterations ({max_iterations}) without termination"
        super().__init__(message, iteration=max_iterations, **kwargs)
        self.max_iterations = max_iterations


class PlannerWorkflowError(WorkflowError):
    """Raised when planner workflow execution fails."""

    def __init__(
        self,
        message: str,
        failed_agent: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, workflow_type="planner", **kwargs)
        self.failed_agent = failed_agent
        if failed_agent:
            self.context["failed_agent"] = failed_agent


# =============================================================================
# State Errors
# =============================================================================


class RunStateError(AgentError):
    """Raised when there's an issue with run state."""

    pass


class RunStateNotFoundError(RunStateError):
    """Raised when run state is not found."""

    def __init__(
        self,
        run_id: str,
        **kwargs: Any,
    ):
        message = f"Run state not found for run_id '{run_id}'"
        super().__init__(message, run_id=run_id, **kwargs)


class RunStatePersistenceError(RunStateError):
    """Raised when run state persistence fails."""

    pass


# =============================================================================
# Router Errors
# =============================================================================


class RouterError(AgentError):
    """Raised when routing fails."""

    def __init__(
        self,
        message: str,
        input_text: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if input_text:
            # Truncate for context
            self.context["input_preview"] = (
                input_text[:200] + "..." if len(input_text) > 200 else input_text
            )


class NoRouteFoundError(RouterError):
    """Raised when no route matches the input."""

    def __init__(
        self,
        message: str | None = None,
        available_routes: list[str] | None = None,
        **kwargs: Any,
    ):
        msg = message or "No route found for the given input"
        super().__init__(msg, **kwargs)
        self.available_routes = available_routes or []
        if available_routes:
            self.context["available_routes"] = available_routes
