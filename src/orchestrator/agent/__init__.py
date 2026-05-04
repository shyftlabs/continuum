"""
Agent module for the Orchestrator SDK.

Provides agent abstraction and multi-agent orchestration capabilities.

Features:
- BaseAgent: Fundamental agent class with tools, handoffs, and memory
- AgentRunner: Executes agents with full observability
- Workflow Agents: RouterAgent, SequentialAgent, ParallelAgent, LoopAgent
- Handoff Management: OpenAI-style agent handoffs with history summarization
- State Persistence: Redis-based state management for pause/resume
- Full Langfuse Tracing: All operations traced for observability
"""

# Base agent
from orchestrator.agent.base import (
    BaseAgent,
    agent_as_tool,
    create_agent,
)

# Configuration
from orchestrator.agent.config import (
    AgentConfig,
    AgentMemoryConfig,
    HandoffConfig,
    LoopConfig,
    ParallelConfig,
    PlanningConfig,
    ReflectionConfig,
    RouterConfig,
    RunnerConfig,
    SequentialConfig,
    apply_llm_route_env_overrides,
)

# Exceptions
from orchestrator.agent.exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentNotFoundError,
    AgentTimeoutError,
    AgentToolError,
    HandoffDepthExceededError,
    HandoffError,
    HandoffNotAllowedError,
    HandoffTargetNotFoundError,
    LoopMaxIterationsError,
    LoopWorkflowError,
    MaxTurnsExceededError,
    NoRouteFoundError,
    ParallelWorkflowError,
    RouterError,
    RunStateError,
    RunStateNotFoundError,
    RunStatePersistenceError,
    SequentialWorkflowError,
    WorkflowError,
)

# Execution components (new architecture)
from orchestrator.agent.execution import (
    Executor,
    HandoffExecutor,
    MessageBuilder,
    StreamExecutor,
    ToolHandler,
)

# Handoff management
from orchestrator.agent.handoff import (
    HandoffManager,
    HistorySummarizer,
    default_history_mapper,
    summarize_conversation,
)

# Interfaces (new architecture)
from orchestrator.agent.interfaces import (
    IContextService,
    IExecutionService,
    IExecutor,
    IHandoffExecutor,
    IMemoryService,
    IMessageBuilder,
    ISessionService,
    IStreamExecutor,
    IToolHandler,
    IToolService,
)

# State persistence
from orchestrator.agent.persistence import (
    RunStateManager,
    get_global_state_manager,
    initialize_global_state_manager,
)

# Runner
from orchestrator.agent.runner import AgentRunner

# Services (new architecture)
from orchestrator.agent.services import (
    ContextService,
    MemoryService,
    SessionService,
    ToolService,
)

# Types
from orchestrator.agent.types import (
    AgentEvent,
    AgentResponse,
    EventType,
    FailStrategy,
    Handoff,
    HandoffData,
    HandoffResult,
    HistorySummarizationMode,
    MemoryScope,
    MergeStrategy,
    PrepareRunResult,
    ResponseStatus,
    Route,
    RunContext,
    RunState,
    RunStatus,
    StepResult,
    TerminationConfig,
    TerminationType,
    TokenUsage,
    ToolExecutionSummary,
    generate_handoff_id,
    generate_run_id,
)

# Utilities (new architecture)
from orchestrator.agent.utils import (
    create_run_context,
    inject_tool_context_to_prompt,
    message_to_dict,
    validate_input,
)

# Workflow agents
from orchestrator.agent.workflow import (
    LoopAgent,
    ParallelAgent,
    PlannerAgent,
    ReflectionAgent,
    RouterAgent,
    SequentialAgent,
    create_planner_agent,
    create_reflection_agent,
    generate_critique_prompt,
)
from orchestrator.agent.workflow.loop import create_loop_agent
from orchestrator.agent.workflow.parallel import create_parallel_agent

# Factory functions for workflow agents
from orchestrator.agent.workflow.router import create_router_agent
from orchestrator.agent.workflow.sequential import create_sequential_agent

__all__ = [
    # Base Agent
    "BaseAgent",
    "create_agent",
    "agent_as_tool",
    # Runner
    "AgentRunner",
    # Workflow Agents
    "RouterAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "ReflectionAgent",
    # Factory Functions
    "create_router_agent",
    "create_sequential_agent",
    "create_parallel_agent",
    "create_loop_agent",
    "create_planner_agent",
    "create_reflection_agent",
    "generate_critique_prompt",
    "PlannerAgent",
    "PlanningConfig",
    # Handoff Management
    "HandoffManager",
    "HistorySummarizer",
    "default_history_mapper",
    "summarize_conversation",
    # State Persistence
    "RunStateManager",
    "get_global_state_manager",
    "initialize_global_state_manager",
    # Configuration
    "AgentConfig",
    "AgentMemoryConfig",
    "HandoffConfig",
    "RunnerConfig",
    "RouterConfig",
    "apply_llm_route_env_overrides",
    "SequentialConfig",
    "ParallelConfig",
    "LoopConfig",
    "ReflectionConfig",
    # Types
    "AgentResponse",
    "AgentEvent",
    "EventType",
    "ResponseStatus",
    "RunStatus",
    "RunState",
    "RunContext",
    "StepResult",
    "TokenUsage",
    "ToolExecutionSummary",
    "Handoff",
    "HandoffData",
    "HandoffResult",
    "Route",
    "TerminationConfig",
    "TerminationType",
    "MemoryScope",
    "MergeStrategy",
    "FailStrategy",
    "HistorySummarizationMode",
    "generate_run_id",
    "generate_handoff_id",
    # Exceptions
    "AgentError",
    "AgentConfigurationError",
    "AgentNotFoundError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentToolError",
    "MaxTurnsExceededError",
    "HandoffError",
    "HandoffNotAllowedError",
    "HandoffDepthExceededError",
    "HandoffTargetNotFoundError",
    "WorkflowError",
    "SequentialWorkflowError",
    "ParallelWorkflowError",
    "LoopWorkflowError",
    "LoopMaxIterationsError",
    "RouterError",
    "NoRouteFoundError",
    "RunStateError",
    "RunStateNotFoundError",
    "RunStatePersistenceError",
    # Services (new architecture)
    "ContextService",
    "MemoryService",
    "SessionService",
    "ToolService",
    # Execution components (new architecture)
    "Executor",
    "StreamExecutor",
    "ToolHandler",
    "HandoffExecutor",
    "MessageBuilder",
    # Interfaces (new architecture)
    "IExecutor",
    "IStreamExecutor",
    "IExecutionService",
    "IContextService",
    "ISessionService",
    "IMemoryService",
    "IToolService",
    "IMessageBuilder",
    "IToolHandler",
    "IHandoffExecutor",
    # Utilities (new architecture)
    "message_to_dict",
    "validate_input",
    "create_run_context",
    "inject_tool_context_to_prompt",
]
