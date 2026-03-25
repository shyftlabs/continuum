"""
Agent Registry for Temporal integration.

Users register their agents here; activities look them up by name at runtime.
The registry also stores a runner factory so activities can create AgentRunner instances.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from orchestrator.temporal.exceptions import AgentNotRegisteredError

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.runner import AgentRunner


class AgentRegistry:
    """Registry of agents available to Temporal activities.

    Users register their agents here; activities look them up by name at runtime.
    Also stores an AgentRunner factory so activities can create runners.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._runner_factory: Callable[[], AgentRunner] | None = None
        self._runner: AgentRunner | None = None
        self._notification_handler: Callable[..., Any] | None = None
        self._lock = threading.Lock()

    def register(self, agent: BaseAgent) -> None:
        """Register an agent by its name."""
        with self._lock:
            self._agents[agent.name] = agent

    def register_many(self, agents: list[BaseAgent]) -> None:
        """Register multiple agents."""
        with self._lock:
            for agent in agents:
                self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        """Get agent by name. Raises AgentNotRegisteredError if missing."""
        with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                available = list(self._agents.keys())
        if agent is None:
            raise AgentNotRegisteredError(
                f"Agent '{name}' is not registered",
                agent_name=name,
                available_agents=available,
            )
        return agent

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        with self._lock:
            return list(self._agents.keys())

    def set_runner_factory(self, factory: Callable[[], AgentRunner]) -> None:
        """Set factory for creating AgentRunner instances."""
        self._runner_factory = factory
        self._runner = None

    def get_runner(self) -> AgentRunner:
        """Get or create an AgentRunner.

        If a runner factory has been set, uses it to create a new runner on first call.
        Otherwise creates a default AgentRunner from the global container.
        """
        if self._runner is not None:
            return self._runner

        if self._runner_factory is not None:
            self._runner = self._runner_factory()
            return self._runner

        from orchestrator.agent.runner import AgentRunner

        self._runner = AgentRunner()
        return self._runner

    def set_notification_handler(self, handler: Callable[..., Any]) -> None:
        """Set handler for approval notifications."""
        self._notification_handler = handler

    def get_notification_handler(self) -> Callable[..., Any] | None:
        """Get the notification handler."""
        return self._notification_handler

    def clear(self) -> None:
        """Remove all registered agents and reset runner."""
        with self._lock:
            self._agents.clear()
            self._runner = None
            self._runner_factory = None
            self._notification_handler = None


_global_registry: AgentRegistry | None = None
_registry_lock = threading.Lock()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry (singleton)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = AgentRegistry()
    return _global_registry


def reset_agent_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
