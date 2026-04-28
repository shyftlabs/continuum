"""
Base Agent class.

Defines the fundamental agent abstraction that all agents inherit from.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
from orchestrator.agent.exceptions import AgentConfigurationError
from orchestrator.logging import get_logger
from orchestrator.agent.types import (
    Handoff,
    MemoryScope,
)
from orchestrator.config import settings

if TYPE_CHECKING:
    from orchestrator.agent.types import RunContext
    from orchestrator.llm.types import ToolDefinition
    from orchestrator.security.policy import PolicyStore
    from orchestrator.tools import MCPServer, ToolExecutor

_logger = get_logger(__name__)


class _SafeFormatMap(dict):
    """Format-map that leaves unresolved {slots} in place instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


_THINK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Reason step by step before calling other tools or giving a final answer. Use this to plan your next action.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning about what to do next",
                }
            },
            "required": ["thought"],
        },
    },
}


@dataclass
class BaseAgent:
    """
    Base class for all agents.

    An agent is an AI entity with a specific role, instructions, tools,
    and the ability to hand off to other agents.

    Attributes:
        name: Unique identifier for the agent
        instructions: System prompt / role definition
        model: LLM model to use (defaults to settings)
        tools: List of tool definitions available to the agent
        handoffs: List of agents this agent can hand off to
        config: Agent configuration
        output_schema: Optional Pydantic model for structured output

    Example:
        ```python
        from orchestrator.agent import BaseAgent, Handoff

        support_agent = BaseAgent(
            name="support-agent",
            instructions="You are a helpful customer support agent...",
            model="gpt-4o",
            tools=[search_kb_tool, create_ticket_tool],
            handoffs=[
                Handoff(
                    target_agent="billing-agent",
                    description="Hand off billing-related inquiries",
                ),
            ],
        )
        ```
    """

    # Identity
    name: str
    instructions: str = ""
    description: str = ""  # Short description for when used as tool/route

    # Model configuration
    model: str = field(default_factory=lambda: settings.default_llm_model)
    temperature: float = 0.7
    max_tokens: int | None = None

    # Tools
    tools: list[ToolDefinition] | list[dict[str, Any]] = field(default_factory=list)
    tool_executor: ToolExecutor | None = None
    mcp_servers: list[MCPServer] = field(default_factory=list)
    policy_store: PolicyStore | None = None

    # Handoffs
    handoffs: list[Handoff] = field(default_factory=list)

    # Memory configuration
    memory_config: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)

    # Agent configuration
    config: AgentConfig = field(default_factory=AgentConfig)

    # Output schema for structured output
    output_schema: type[BaseModel] | None = None

    # JSON mode configuration for structured outputs
    enable_json_mode: bool = False
    json_schema: dict[str, Any] | type[BaseModel] | None = None
    json_strict: bool = True

    # Input validation
    input_schema: type[BaseModel] | None = None

    # Lifecycle hooks
    on_start: Callable[[BaseAgent, dict[str, Any]], None] | None = None
    on_end: Callable[[BaseAgent, dict[str, Any]], None] | None = None
    on_error: Callable[[BaseAgent, Exception, dict[str, Any]], None] | None = None
    on_tool_call: Callable[[BaseAgent, str, dict[str, Any]], None] | None = None
    on_handoff: Callable[[BaseAgent, str, dict[str, Any]], None] | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Prompt engineering
    # -------------------------------------------------------------------------

    # PromptTemplate — static variables merged with runtime context when
    # resolving {slot} placeholders in instructions.
    # Runtime slots available automatically: {user_id}, {session_id},
    # {run_id}, {agent_name}, {date}, plus any key from context.metadata.
    # template_vars take highest priority and override all of the above.
    #
    # Example:
    #   BaseAgent(
    #       instructions="You are helping {user_name}. Today is {date}.",
    #       template_vars={"user_name": "Alice"},
    #   )
    template_vars: dict[str, Any] = field(default_factory=dict)

    # Few-shot examples — injected into the system prompt automatically.
    # Each dict must have "input" and "output" keys.
    #
    # Example:
    #   BaseAgent(
    #       examples=[
    #           {"input": "What is 2+2?", "output": "4"},
    #           {"input": "Summarise in one word: the sky is blue.", "output": "Sky"},
    #       ]
    #   )
    examples: list[dict[str, str]] = field(default_factory=list)

    # Dynamic instruction modifiers — callables applied in order after
    # template rendering and few-shot injection.  Each receives the current
    # prompt string and the RunContext, and returns the modified prompt.
    # Useful for context-aware changes (user tier, session length, memory topics).
    #
    # Example:
    #   def add_tier_note(prompt: str, ctx: RunContext) -> str:
    #       tier = ctx.metadata.get("user_tier", "free")
    #       if tier == "enterprise":
    #           return prompt + "\nThis is an enterprise user — prioritise SLA."
    #       return prompt
    #
    #   BaseAgent(instruction_modifiers=[add_tier_note])
    instruction_modifiers: list[Callable[[str, Any], str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate agent configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate agent configuration."""
        if not self.name:
            raise AgentConfigurationError("Agent name is required")

        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise AgentConfigurationError(
                f"Agent name '{self.name}' must be alphanumeric with hyphens/underscores only"
            )

        # Validate handoffs don't reference self
        for handoff in self.handoffs:
            if handoff.target_agent == self.name:
                raise AgentConfigurationError(f"Agent '{self.name}' cannot hand off to itself")

    @property
    def system_prompt(self) -> str:
        """
        Get the system prompt without runtime context (backward-compatible).

        For full resolution including template variables, few-shot examples,
        and dynamic modifiers, call resolve_system_prompt(context) instead.
        """
        return self.resolve_system_prompt(context=None)

    def resolve_system_prompt(self, context: Any | None = None) -> str:
        """
        Build the final system prompt for this agent.

        Applies three layers in order:

        1. **Template rendering** — replaces ``{slot}`` placeholders in
           ``instructions`` using (in ascending priority):
           built-in runtime vars → ``context.metadata`` → ``template_vars``.

           Built-in runtime vars:
           - ``{agent_name}``  — this agent's name
           - ``{date}``        — today's date (ISO 8601)
           - ``{user_id}``     — context.user_id (empty string if None)
           - ``{session_id}``  — context.session_id (empty string if None)
           - ``{run_id}``      — context.run_id (empty string if None)

           Unknown slots are left as-is (no KeyError).

        2. **Few-shot examples** — if ``examples`` is non-empty, appends a
           formatted "Examples:" block to the prompt.

        3. **Instruction modifiers** — callables in ``instruction_modifiers``
           are applied in order, each receiving ``(prompt, context)`` and
           returning the modified prompt string.

        Args:
            context: RunContext for the current execution (may be None).

        Returns:
            The fully resolved system prompt string.
        """
        from datetime import date as _date

        prompt = self.instructions

        # ------------------------------------------------------------------
        # 1. Template rendering
        # ------------------------------------------------------------------
        if "{" in prompt:
            vars_map: dict[str, Any] = {
                "agent_name": self.name,
                "date": _date.today().isoformat(),
                "user_id": "",
                "session_id": "",
                "run_id": "",
            }
            if context is not None:
                vars_map["user_id"] = getattr(context, "user_id", None) or ""
                vars_map["session_id"] = getattr(context, "session_id", None) or ""
                vars_map["run_id"] = getattr(context, "run_id", None) or ""
                # Merge context.metadata (lower priority than template_vars)
                meta = getattr(context, "metadata", None)
                if isinstance(meta, dict):
                    vars_map.update(meta)
            # template_vars override everything
            vars_map.update(self.template_vars)
            try:
                prompt = prompt.format_map(_SafeFormatMap(vars_map))
            except Exception as e:
                _logger.warning(
                    f"Template rendering failed for agent '{self.name}': {e}. "
                    f"Using unrendered instructions.",
                    exc_info=True,
                )

        # ------------------------------------------------------------------
        # 2. Few-shot examples
        # ------------------------------------------------------------------
        if self.examples:
            lines = ["\n\nExamples:"]
            for ex in self.examples:
                lines.append(f"Input: {ex.get('input', '')}")
                lines.append(f"Output: {ex.get('output', '')}")
                lines.append("")
            prompt = prompt.rstrip() + "\n" + "\n".join(lines).rstrip()

        # ------------------------------------------------------------------
        # 3. Dynamic instruction modifiers
        # ------------------------------------------------------------------
        for modifier in self.instruction_modifiers:
            try:
                prompt = modifier(prompt, context)
            except Exception as e:
                _logger.warning(
                    f"Instruction modifier '{getattr(modifier, '__name__', repr(modifier))}' "
                    f"failed for agent '{self.name}': {e}. Skipping modifier.",
                    exc_info=True,
                )

        return prompt

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """
        Get all tools formatted for LLM consumption.

        Includes both regular tools and handoffs as special tools.
        When react_mode is enabled, prepends a 'think' tool so the LLM
        can express reasoning steps via function calling before acting.
        """
        tools = []

        # Add regular tools
        for tool in self.tools:
            if isinstance(tool, dict):
                tools.append(tool)
            elif hasattr(tool, "to_dict"):
                tools.append(tool.to_dict())
            else:
                tools.append(tool)

        # Capture whether there are regular tools before adding handoffs
        has_regular_tools = bool(tools)

        # Add handoffs as tools
        for handoff in self.handoffs:
            tools.append(handoff.to_tool_definition())

        # Inject think tool only when react_mode is enabled AND there are regular tools
        # Without regular tools, react_mode has no effect (same as normal mode)
        if self.config and self.config.react_mode and has_regular_tools:
            tools.insert(0, _THINK_TOOL)

        return tools

    def get_handoff(self, target_name: str) -> Handoff | None:
        """
        Get a handoff definition by target agent name.

        Args:
            target_name: Name of the target agent

        Returns:
            Handoff definition or None if not found
        """
        for handoff in self.handoffs:
            if handoff.target_agent == target_name:
                return handoff
        return None

    def can_handoff_to(self, target_name: str) -> bool:
        """
        Check if this agent can hand off to another agent.

        Args:
            target_name: Name of the target agent

        Returns:
            True if handoff is allowed
        """
        return self.get_handoff(target_name) is not None

    def is_handoff_tool_call(self, tool_name: str) -> tuple[bool, str | None]:
        """
        Check if a tool call is a handoff request.

        Args:
            tool_name: Name of the tool being called

        Returns:
            Tuple of (is_handoff, target_agent_name)
        """
        if tool_name.startswith("handoff_to_"):
            target = tool_name[len("handoff_to_") :]
            if self.can_handoff_to(target):
                return True, target
        return False, None

    def clone(self, **overrides: Any) -> BaseAgent:
        """
        Create a copy of this agent with optional overrides.

        Args:
            **overrides: Fields to override

        Returns:
            New agent instance
        """
        # Deep copy mutable nested structures to prevent shared-state mutations
        current = {
            "name": self.name,
            "instructions": self.instructions,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": copy.deepcopy(self.tools),
            "tool_executor": self.tool_executor,
            "mcp_servers": list(self.mcp_servers),  # shallow OK — server instances are shared
            "policy_store": self.policy_store,
            "handoffs": copy.deepcopy(self.handoffs),
            "memory_config": copy.deepcopy(self.memory_config),
            "config": copy.deepcopy(self.config),
            "output_schema": self.output_schema,
            "enable_json_mode": self.enable_json_mode,
            "json_schema": copy.deepcopy(self.json_schema) if isinstance(self.json_schema, dict) else self.json_schema,
            "json_strict": self.json_strict,
            "input_schema": self.input_schema,
            "on_start": self.on_start,
            "on_end": self.on_end,
            "on_error": self.on_error,
            "on_tool_call": self.on_tool_call,
            "on_handoff": self.on_handoff,
            "metadata": copy.deepcopy(self.metadata),
            "tags": list(self.tags),
            "template_vars": copy.deepcopy(self.template_vars),
            "examples": copy.deepcopy(self.examples),
            "instruction_modifiers": list(self.instruction_modifiers),
        }

        # Apply overrides
        current.update(overrides)

        return BaseAgent(**current)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert agent to dictionary representation.

        Used for serialization and logging.
        """
        return {
            "name": self.name,
            "instructions": self.instructions,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": [t if isinstance(t, dict) else str(t) for t in self.tools],
            "handoffs": [
                {
                    "target_agent": h.target_agent,
                    "description": h.description,
                    "return_to_parent": h.return_to_parent,
                }
                for h in self.handoffs
            ],
            "memory_config": self.memory_config.to_dict(),
            "config": self.config.to_dict(),
            "output_schema": self.output_schema.__name__ if self.output_schema else None,
            "enable_json_mode": self.enable_json_mode,
            "json_schema": (
                self.json_schema.__name__
                if isinstance(self.json_schema, type)
                else self.json_schema
                if isinstance(self.json_schema, dict)
                else None
            ),
            "json_strict": self.json_strict,
            "metadata": self.metadata,
            "tags": self.tags,
            "template_vars": self.template_vars,
            "examples": self.examples,
            "has_instruction_modifiers": len(self.instruction_modifiers) > 0,
        }

    def to_tool_definition(self, description: str | None = None) -> dict[str, Any]:
        """
        Convert this agent to a tool definition.

        Allows this agent to be used as a tool by another agent.

        Args:
            description: Override description for the tool

        Returns:
            Tool definition dict
        """
        desc = description or self.description or f"Consult {self.name} agent"

        return {
            "type": "function",
            "function": {
                "name": f"consult_{self.name.replace('-', '_')}",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query or task for the agent",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for the agent",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BaseAgent(name={self.name!r}, model={self.model!r}, "
            f"tools={len(self.tools)}, handoffs={len(self.handoffs)})"
        )


def agent_as_tool(
    agent: BaseAgent,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Create a tool definition that wraps an agent.

    This allows using one agent as a tool for another agent,
    enabling hierarchical agent compositions.

    Args:
        agent: The agent to wrap as a tool
        description: Optional description override

    Returns:
        Tool definition dict

    Example:
        ```python
        math_expert = BaseAgent(
            name="math-expert",
            instructions="You are a mathematics expert...",
        )

        main_agent = BaseAgent(
            name="main-agent",
            tools=[
                agent_as_tool(math_expert, "Use for complex math"),
            ],
        )
        ```
    """
    return agent.to_tool_definition(description)


def create_agent(
    name: str,
    instructions: str,
    *,
    model: str | None = None,
    tools: list[Any] | None = None,
    handoffs: list[Handoff] | None = None,
    memory_scope: MemoryScope = MemoryScope.USER,
    store_memories: bool = True,
    search_memories: bool = True,
    output_schema: type[BaseModel] | None = None,
    **kwargs: Any,
) -> BaseAgent:
    """
    Factory function to create an agent with common defaults.

    Args:
        name: Agent name
        instructions: System prompt
        model: LLM model (defaults to settings)
        tools: List of tools
        handoffs: List of handoffs
        memory_scope: Memory scope for this agent
        store_memories: Whether to store new memories
        search_memories: Whether to search memories
        output_schema: Optional structured output schema
        **kwargs: Additional BaseAgent arguments

    Returns:
        Configured BaseAgent instance

    Example:
        ```python
        agent = create_agent(
            name="support-agent",
            instructions="You help customers with their issues.",
            tools=[search_tool],
            memory_scope=MemoryScope.USER,
        )
        ```
    """
    memory_config = AgentMemoryConfig(
        search_memories=search_memories,
        store_memories=store_memories,
        search_scope=memory_scope,
        store_scope=memory_scope,
    )

    return BaseAgent(
        name=name,
        instructions=instructions,
        model=model or settings.default_llm_model,
        tools=tools or [],
        handoffs=handoffs or [],
        memory_config=memory_config,
        output_schema=output_schema,
        **kwargs,
    )
