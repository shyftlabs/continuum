"""
Agent configuration.

Defines configuration classes for agents and agent execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

# Type aliases for memory hooks
MemoryPreStoreFilter = Callable[[list[str]], list[str]]
MemoryOnStoredCallback = Callable[[list[str]], None]

from orchestrator.agent.types import (
    FailStrategy,
    HistorySummarizationMode,
    MemoryScope,
    MergeStrategy,
)
from orchestrator.config import settings

if TYPE_CHECKING:
    from orchestrator.llm.context_management import ContextManagementConfig
    from orchestrator.tools.tool_attention.config import ToolAttentionConfig


# =============================================================================
# Agent Memory Configuration
# =============================================================================


@dataclass
class AgentMemoryConfig:
    """
    Configuration for agent memory behavior.

    Controls how the agent interacts with long-term memory (mem0).
    """

    # Read settings
    search_memories: bool = True  # Whether to search long-term memory
    search_scope: MemoryScope = MemoryScope.USER  # Scope for memory search
    search_limit: int = 5  # Number of memories to retrieve
    search_threshold: float = 0.0  # Minimum similarity score

    # Write settings
    store_memories: bool = True  # Whether to store new memories
    store_scope: MemoryScope = MemoryScope.USER  # Scope for memory storage
    store_assistant_messages: bool = True  # Store assistant responses
    store_user_messages: bool = True  # Store user messages

    # Memory policy hooks — product-level customization (domain-agnostic in SDK)
    # extraction_prompt: custom fact extraction prompt; if None, mem0 default is used
    extraction_prompt: str | None = None
    # pre_store_filter: called with extracted fact texts after storage;
    # facts not returned by the filter are deleted from the vector store (best-effort)
    pre_store_filter: MemoryPreStoreFilter | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    # on_stored: fired with the final list of stored fact texts after add + filter
    on_stored: MemoryOnStoredCallback | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    # Sharing settings (for multi-agent)
    broadcast_learnings: bool = False  # Share important learnings
    broadcast_to: list[str] | None = None  # Specific agents to share with
    broadcast_threshold: float = 0.8  # Importance threshold for broadcasting

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "search_memories": self.search_memories,
            "search_scope": self.search_scope.value,
            "search_limit": self.search_limit,
            "search_threshold": self.search_threshold,
            "store_memories": self.store_memories,
            "store_scope": self.store_scope.value,
            "store_assistant_messages": self.store_assistant_messages,
            "store_user_messages": self.store_user_messages,
            "broadcast_learnings": self.broadcast_learnings,
            "broadcast_to": self.broadcast_to,
            "broadcast_threshold": self.broadcast_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMemoryConfig:
        """Create from dictionary."""
        return cls(
            search_memories=data.get("search_memories", True),
            search_scope=MemoryScope(data.get("search_scope", "user")),
            search_limit=data.get("search_limit", 5),
            search_threshold=data.get("search_threshold", 0.0),
            store_memories=data.get("store_memories", True),
            store_scope=MemoryScope(data.get("store_scope", "user")),
            store_assistant_messages=data.get("store_assistant_messages", True),
            store_user_messages=data.get("store_user_messages", True),
            broadcast_learnings=data.get("broadcast_learnings", False),
            broadcast_to=data.get("broadcast_to"),
            broadcast_threshold=data.get("broadcast_threshold", 0.8),
        )


# =============================================================================
# Handoff Configuration
# =============================================================================


@dataclass
class HandoffConfig:
    """
    Configuration for agent handoff behavior.
    """

    # History handling
    transfer_history: bool = True  # Pass conversation history
    summarize_history: bool = True  # Summarize before transfer
    summarization_mode: HistorySummarizationMode = HistorySummarizationMode.HYBRID
    recent_messages: int = 5  # Recent messages for hybrid mode
    summary_model: str | None = None  # Model for summarization (default: agent's model)

    # Behavior
    return_to_parent: bool = True  # Return control after completion
    max_handoff_depth: int = 10  # Max nested handoffs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transfer_history": self.transfer_history,
            "summarize_history": self.summarize_history,
            "summarization_mode": self.summarization_mode.value,
            "recent_messages": self.recent_messages,
            "summary_model": self.summary_model,
            "return_to_parent": self.return_to_parent,
            "max_handoff_depth": self.max_handoff_depth,
        }


# =============================================================================
# Reflection Configuration
# =============================================================================


@dataclass
class ReflectionConfig:
    """
    Configuration for ReflectionAgent self-critique behavior.
    """

    critique_prompt: str = (
        "Review the response above. Reply ONLY 'PASS' if it fully answers the request, "
        "or 'NEEDS IMPROVEMENT: <reason>' if not."
    )
    max_reflections: int = 2
    reflection_model: str | None = None  # defaults to the inner agent's model


# =============================================================================
# Agent Configuration
# =============================================================================


@dataclass
class AgentConfig:
    """
    Configuration for a single agent.
    """

    # Model settings
    model: str = field(default_factory=lambda: settings.default_llm_model)
    temperature: float = 0.7
    max_tokens: int | None = None

    # Execution settings
    max_turns: int = 25  # Max conversation turns
    timeout: int = 300  # Timeout in seconds
    retry_count: int = 3  # Number of retries on failure

    # Memory settings
    memory: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)

    # Handoff settings
    handoff: HandoffConfig = field(default_factory=HandoffConfig)

    # Context management settings (can override global defaults)
    # Type hint uses string to avoid circular import (TYPE_CHECKING)
    context_management: "ContextManagementConfig | None" = None  # noqa: UP037  # None = use global defaults

    # Tool-attention: semantic per-turn tool schema routing to reduce token cost.
    # Only activates when the agent has >= tool_attention.min_tools tools.
    # None = disabled (all schemas sent every turn, original behaviour).
    tool_attention: "ToolAttentionConfig | None" = None  # noqa: UP037

    # Input sanitization
    input_sanitization: bool = True
    injection_detection: bool = False

    # Output settings
    output_type: Literal["text", "json", "structured"] = "text"

    # Reasoning modes
    reasoning_mode: bool = False  # two-pass: silent think-first LLM call before the turn loop
    react_mode: bool = False  # ReAct: appends Thought/Action/Observation template to system prompt

    # Session history
    session_history_turns: int | None = None  # None = use default (20 turns); number of complete request/response pairs to load

    # Context requirement
    require_context: bool = False  # if True, skip LLM and return no-knowledge message when no RAG context found

    # RAG retrieval hints — products read these in their retrieve_context() call.
    # None means "use the product's own default".
    retrieval_top_k: int | None = None      # how many chunks to return to the LLM
    rerank_enabled: bool | None = None      # whether to run a cross-encoder reranker
    rag_context: str | None = None          # RAG chunks injected after conversation history

    # Scanner hooks — products plug domain-specific scanners here instead of hardcoding in routers.
    # Input scanner signature:  (text: str) -> tuple[str, bool, str | None]
    #   returns (sanitized_text, is_safe, reason); is_safe=False → InputBlockedError raised
    # Output scanner signature: (prompt: str, output: str) -> tuple[str, bool, str | None]
    #   returns (sanitized_output, is_safe, reason); output with PII redacted in-place
    input_scanners: list[Callable[[str], tuple[str, bool, str | None]]] = field(default_factory=list)
    output_scanners: list[Callable[[str, str], tuple[str, bool, str | None]]] = field(default_factory=list)

    # Dispatch priority for this agent's LLM calls (1=lowest, 10=highest, 5=default).
    # Used as the stage-level weight in TwoLevelDispatcher for internal models:
    # a higher stage_priority agent's queue is served before lower-priority ones
    # when multiple agents are competing for the same inference backend.
    # For external APIs, RunContext.priority (request-level) is used instead.
    stage_priority: int = 5

    # Access control policies applied before each tool call and memory access.
    # Evaluated with deny-overrides semantics (Orla-style): an explicit deny always
    # wins over any allow. If no policy matches, access is open (default allow).
    # Use the resource prefixes "tool:", "memory:", "data:" to target different layers.
    # Example:
    #   AccessPolicy(name="no-delete", subjects=["billing_agent"],
    #                resources=["tool:delete_*"], effect="deny")
    access_policies: list[Any] = field(default_factory=list)  # list[AccessPolicy]

    # Tracing
    trace_all_turns: bool = True  # Trace every turn
    log_to_session: bool = True  # Log messages to session

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_turns": self.max_turns,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "memory": self.memory.to_dict(),
            "handoff": self.handoff.to_dict(),
            "context_management": (
                self.context_management.to_dict() if self.context_management else None
            ),
            "input_sanitization": self.input_sanitization,
            "injection_detection": self.injection_detection,
            "output_type": self.output_type,
            "reasoning_mode": self.reasoning_mode,
            "react_mode": self.react_mode,
            "trace_all_turns": self.trace_all_turns,
            "log_to_session": self.log_to_session,
            "tool_attention": (
                {
                    "k": self.tool_attention.k,
                    "min_tools": self.tool_attention.min_tools,
                    "threshold": self.tool_attention.threshold,
                    "always_promote": self.tool_attention.always_promote,
                    "collection_name": self.tool_attention.collection_name,
                    "embedding_model": self.tool_attention.embedding_model,
                    "embedding_dim": self.tool_attention.embedding_dim,
                }
                if self.tool_attention else None
            ),
        }


# =============================================================================
# Runner Configuration
# =============================================================================


@dataclass
class RunnerConfig:
    """
    Configuration for the agent runner.
    """

    # Execution
    default_max_turns: int = 25
    default_timeout: int = 300

    # State persistence
    persist_state: bool = True  # Persist run state to Redis
    state_ttl: int = 3600 * 24  # State TTL in seconds (24 hours)

    # Tool execution
    parallel_tool_calls: bool = True  # Execute tool calls in parallel
    max_parallel_tools: int = 5  # Max concurrent tool calls
    tool_timeout: int = 60  # Per-tool timeout

    # Error handling
    retry_on_error: bool = True
    max_retries: int = 3

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown: int = 60

    # Tracing
    trace_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_max_turns": self.default_max_turns,
            "default_timeout": self.default_timeout,
            "persist_state": self.persist_state,
            "state_ttl": self.state_ttl,
            "parallel_tool_calls": self.parallel_tool_calls,
            "max_parallel_tools": self.max_parallel_tools,
            "tool_timeout": self.tool_timeout,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_cooldown": self.circuit_breaker_cooldown,
            "trace_enabled": self.trace_enabled,
        }


# =============================================================================
# Workflow Configuration
# =============================================================================


@dataclass
class SequentialConfig:
    """Configuration for sequential agent execution."""

    pass_full_history: bool = False  # Pass full history vs just output
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST
    pipeline_context_max_chars: int | None = 300  # None = no truncation

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_full_history": self.pass_full_history,
            "fail_strategy": self.fail_strategy.value,
            "pipeline_context_max_chars": self.pipeline_context_max_chars,
        }


@dataclass
class ParallelConfig:
    """Configuration for parallel agent execution."""

    merge_strategy: MergeStrategy = MergeStrategy.LLM_SUMMARIZE
    fail_strategy: FailStrategy = FailStrategy.CONTINUE_ON_ERROR
    timeout: int = 300  # Overall timeout
    summary_model: str | None = None  # Model for LLM summarization
    summary_prompt: str | None = None  # Custom prompt for summarization

    def to_dict(self) -> dict[str, Any]:
        return {
            "merge_strategy": self.merge_strategy.value,
            "fail_strategy": self.fail_strategy.value,
            "timeout": self.timeout,
            "summary_model": self.summary_model,
            "summary_prompt": self.summary_prompt,
        }


@dataclass
class LoopConfig:
    """Configuration for loop agent execution."""

    max_iterations: int = 10
    check_interval: int = 1  # Check termination every N iterations

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "check_interval": self.check_interval,
        }


@dataclass
class PlanningConfig:
    """Configuration for planner agent."""

    max_steps: int = 10
    enable_replanning: bool = False
    replan_on_failure: bool = True
    planning_model: str | None = None
    fail_strategy: FailStrategy = FailStrategy.FAIL_FAST
    strict_agent_pool: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "enable_replanning": self.enable_replanning,
            "replan_on_failure": self.replan_on_failure,
            "planning_model": self.planning_model,
            "fail_strategy": self.fail_strategy.value,
            "strict_agent_pool": self.strict_agent_pool,
        }


TierClassifierMode = Literal["light_only", "heavy_only", "gpt_4o_mini", "qwen", "qwen_local"]


@dataclass
class RouterConfig:
    """Configuration for router agent."""

    routing_strategy: Literal["llm", "rule_based", "hybrid", "model_tier"] = "llm"
    routing_model: str | None = None  # Model for LLM routing (default: agent's model)
    routing_prompt: str | None = None  # Custom prompt for routing decision

    # --- Smart layer (model_tier) -------------------------------------------------
    tier_classifier: TierClassifierMode = "gpt_4o_mini"
    tier_classifier_llm_model: str | None = None  # gpt_4o_mini default id; qwen/qwen_local require explicit id
    tier_classifier_max_tokens: int = 128
    # Keyword / length heuristics before the classifier LLM (disable to always call the classifier).
    tier_classifier_heuristic_shortcut: bool = True
    tier_router_api_base: str | None = None  # Remote Hugging Face (or other) router URL — qwen mode only
    tier_router_api_key: str | None = None  # Remote router API key — qwen mode only
    tier_local_router_api_base: str | None = None  # Local OpenAI-compatible URL — qwen_local only (never tier_router_api_base)
    tier_local_router_api_key: str | None = None  # Optional key for local classifier server

    tier_nano_model: str | None = None
    tier_fast_model: str | None = None
    tier_balanced_model: str | None = None
    tier_specialist_model: str | None = None
    tier_frontier_model: str | None = None
    tier_light_model: str | None = None  # legacy → fast slot if tier_fast_model unset
    tier_heavy_model: str | None = None  # legacy → balanced slot if tier_balanced_model unset
    tier_force_completion_model: str | None = None  # if set, all completions use this model id

    tier_light_temperature: float = 0.5  # nano, fast, balanced
    tier_heavy_temperature: float = 0.3  # specialist, frontier
    tier_completion_max_tokens: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_strategy": self.routing_strategy,
            "routing_model": self.routing_model,
            "routing_prompt": self.routing_prompt,
            "tier_classifier": self.tier_classifier,
            "tier_classifier_llm_model": self.tier_classifier_llm_model,
            "tier_classifier_max_tokens": self.tier_classifier_max_tokens,
            "tier_classifier_heuristic_shortcut": self.tier_classifier_heuristic_shortcut,
            "tier_router_api_base": self.tier_router_api_base,
            "tier_router_api_key": self.tier_router_api_key,
            "tier_local_router_api_base": self.tier_local_router_api_base,
            "tier_local_router_api_key": self.tier_local_router_api_key,
            "tier_nano_model": self.tier_nano_model,
            "tier_fast_model": self.tier_fast_model,
            "tier_balanced_model": self.tier_balanced_model,
            "tier_specialist_model": self.tier_specialist_model,
            "tier_frontier_model": self.tier_frontier_model,
            "tier_light_model": self.tier_light_model,
            "tier_heavy_model": self.tier_heavy_model,
            "tier_force_completion_model": self.tier_force_completion_model,
            "tier_light_temperature": self.tier_light_temperature,
            "tier_heavy_temperature": self.tier_heavy_temperature,
            "tier_completion_max_tokens": self.tier_completion_max_tokens,
        }


_VALID_LLM_ROUTE_CLASSIFIERS: frozenset[str] = frozenset(
    {"light_only", "heavy_only", "gpt_4o_mini", "qwen", "qwen_local"}
)

# HF/local router model ids must not override tier_classifier_llm_model for OpenAI-hosted classifiers.
_ROUTER_MODEL_CLASSIFIER_MODES: frozenset[str] = frozenset({"qwen", "qwen_local"})


def apply_llm_route_env_overrides(rc: RouterConfig) -> RouterConfig:
    """
    Apply ``LLM_ROUTE_*`` fields from global :mod:`orchestrator.config.settings`
    onto ``rc`` (mutates in place). Safe to call before applying explicit API overrides.

    Env vars (via Settings): ``LLM_ROUTE_TIER_CLASSIFIER``, ``LLM_ROUTE_ROUTER_MODEL`` (only applied when
    ``tier_classifier`` is ``qwen`` or ``qwen_local`` — avoids sending HF model ids to OpenAI for
    ``gpt_4o_mini``), ``LLM_ROUTE_ROUTER_API_BASE``, ``LLM_ROUTE_ROUTER_API_KEY``
    (required for ``qwen``),
    ``LLM_ROUTE_FORCE_COMPLETION_MODEL``,
    ``LLM_ROUTE_TIER_CLASSIFIER_HEURISTIC_SHORTCUT`` (optional ``false`` to disable keyword shortcuts),
    ``LLM_ROUTE_LOCAL_ROUTER_API_BASE`` / ``LLM_ROUTE_LOCAL_ROUTER_API_KEY`` → ``tier_local_router_api_*`` (``qwen_local``),
    ``LLM_ROUTE_LOCAL_ROUTER_MODEL`` → classifier model id for ``qwen_local`` (e.g. MLX HF repo id).

    For ``qwen`` classifier, ``HF_API_KEY`` (global Settings) is used when router API key env fields are unset;
    default HF router URL and model are applied in :mod:`orchestrator.agent.smart_layer.classifier`.
    """
    tc_raw = (settings.llm_route_tier_classifier or "").strip().lower()
    if tc_raw and tc_raw in _VALID_LLM_ROUTE_CLASSIFIERS:
        rc.tier_classifier = tc_raw  # type: ignore[assignment]

    rm = (settings.llm_route_router_model or "").strip()
    lm = (settings.llm_route_local_router_model or "").strip()
    if rc.tier_classifier == "qwen" and rm:
        rc.tier_classifier_llm_model = rm
    elif rc.tier_classifier == "qwen_local":
        if lm:
            rc.tier_classifier_llm_model = lm
        elif rm:
            rc.tier_classifier_llm_model = rm

    if settings.llm_route_router_api_base and str(settings.llm_route_router_api_base).strip():
        rc.tier_router_api_base = str(settings.llm_route_router_api_base).strip()

    if settings.llm_route_router_api_key and str(settings.llm_route_router_api_key).strip():
        rc.tier_router_api_key = str(settings.llm_route_router_api_key).strip()

    if settings.llm_route_local_router_api_base and str(settings.llm_route_local_router_api_base).strip():
        rc.tier_local_router_api_base = str(settings.llm_route_local_router_api_base).strip()

    if settings.llm_route_local_router_api_key and str(settings.llm_route_local_router_api_key).strip():
        rc.tier_local_router_api_key = str(settings.llm_route_local_router_api_key).strip()

    if settings.llm_route_force_completion_model and str(settings.llm_route_force_completion_model).strip():
        rc.tier_force_completion_model = str(settings.llm_route_force_completion_model).strip()

    if settings.llm_route_tier_classifier_heuristic_shortcut is not None:
        rc.tier_classifier_heuristic_shortcut = settings.llm_route_tier_classifier_heuristic_shortcut

    return rc
