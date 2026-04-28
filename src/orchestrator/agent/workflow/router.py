"""
Router Agent - Dynamic routing/triage agent.

Routes user requests to appropriate specialist agents based on
LLM analysis or rule-based conditions.

NOTE: Workflow agents now include Langfuse span tracing for full observability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import RouterConfig
from orchestrator.agent.types import Route, RunContext
from orchestrator.config import settings
from orchestrator.llm.config import LLMConfig
from orchestrator.logging import get_logger
from orchestrator.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient

logger = get_logger(__name__)


@dataclass
class RouterAgent(BaseAgent):
    """
    Routes requests to appropriate specialist agents.

    The RouterAgent inspects incoming requests and decides which
    specialist agent should handle them. It supports:
    - LLM-based routing (intelligent analysis)
    - Rule-based routing (pattern matching)
    - Hybrid routing (rules first, then LLM)

    Example:
        ```python
        from orchestrator.agent import BaseAgent
        from orchestrator.agent.workflow import RouterAgent
        from orchestrator.agent.types import Route

        # Define specialist agents
        billing_agent = BaseAgent(
            name="billing-agent",
            instructions="Handle billing inquiries...",
        )

        technical_agent = BaseAgent(
            name="technical-agent",
            instructions="Handle technical issues...",
        )

        # Create router
        router = RouterAgent(
            name="triage-agent",
            instructions="Route user requests to appropriate specialist.",
            routes=[
                Route(
                    agent_name="billing-agent",
                    description="Billing, payments, invoices, refunds",
                ),
                Route(
                    agent_name="technical-agent",
                    description="Technical issues, bugs, outages",
                ),
            ],
            fallback_agent_name="general-agent",
        )
        ```
    """

    # Routes to available specialists
    routes: list[Route] = field(default_factory=list)

    # Fallback agent if no route matches
    fallback_agent_name: str | None = None

    # Router configuration
    router_config: RouterConfig = field(default_factory=RouterConfig)

    # Custom routing function
    custom_router: Callable[[str, list[Route]], str | None] | None = None

    def __post_init__(self) -> None:
        """Initialize router agent."""
        super().__post_init__()

        # Build routing prompt if not provided
        if not self.instructions:
            self.instructions = self._build_routing_instructions()

    def _build_routing_instructions(self) -> str:
        """Build system prompt for routing."""
        route_descriptions = []
        for route in self.routes:
            route_descriptions.append(f"- {route.agent_name}: {route.description}")

        routes_text = "\n".join(route_descriptions)

        return f"""You are a routing agent that analyzes user requests and decides which specialist agent should handle them.

Available specialist agents:
{routes_text}

Your task:
1. Analyze the user's request
2. Determine which specialist is best suited to handle it
3. Respond with ONLY the agent name (e.g., "billing-agent")

If the request doesn't clearly fit any specialist, respond with "none".
"""

    async def route(
        self,
        input_text: str,
        llm_client: LLMClient | None = None,
        context: RunContext | None = None,
    ) -> str | None:
        """
        Determine which agent should handle the input.

        Args:
            input_text: User input to route
            llm_client: LLM client for intelligent routing

        Returns:
            Agent name or None if no route found

        Raises:
            RouterError: If routing fails
        """
        available_routes = [r.agent_name for r in self.routes]
        strategy = self.router_config.routing_strategy

        # Create span for routing decision
        async with SpanScope(
            "router.route",
            input={"input_preview": input_text[:200], "strategy": strategy},
            metadata={
                "router_name": self.name,
                "available_routes": available_routes,
                "fallback_agent": self.fallback_agent_name,
            },
        ) as span:
            # Try custom router first
            if self.custom_router:
                result = self.custom_router(input_text, self.routes)
                if result:
                    span.set_output({"selected_route": result, "method": "custom_router"})
                    self._stamp_priority(result, context)
                    return result

            if strategy == "rule_based":
                result = self._rule_based_route(input_text)
                span.set_output({"selected_route": result, "method": "rule_based"})
                self._stamp_priority(result, context)
                return result
            elif strategy == "llm":
                result = await self._llm_route(input_text, llm_client)
                span.set_output({"selected_route": result, "method": "llm"})
                self._stamp_priority(result, context)
                return result
            elif strategy == "hybrid":
                # Try rules first
                rule_result = self._rule_based_route(input_text)
                if rule_result:
                    span.set_output({"selected_route": rule_result, "method": "hybrid_rule"})
                    self._stamp_priority(rule_result, context)
                    return rule_result
                # Fall back to LLM
                llm_result = await self._llm_route(input_text, llm_client)
                span.set_output({"selected_route": llm_result, "method": "hybrid_llm"})
                self._stamp_priority(llm_result, context)
                return llm_result

            span.set_output({"selected_route": None, "method": "none"})
            return None

    def _stamp_priority(self, agent_name: str | None, context: RunContext | None) -> None:
        """Stamp RunContext.priority from the selected route's dispatch_priority."""
        if context is None or agent_name is None:
            return
        route = self.get_route(agent_name)
        if route is not None:
            context.priority = route.dispatch_priority

    def _rule_based_route(self, input_text: str) -> str | None:
        """Route based on rules/conditions."""
        input_lower = input_text.lower()

        # Sort by priority
        sorted_routes = sorted(self.routes, key=lambda r: r.priority, reverse=True)

        for route in sorted_routes:
            # Check callable condition
            if callable(route.condition):
                if route.condition(input_text):
                    return route.agent_name
            # Check string pattern
            elif isinstance(route.condition, str):
                if route.condition.lower() in input_lower:
                    return route.agent_name
            # Check description keywords
            else:
                keywords = route.description.lower().split()
                if any(kw in input_lower for kw in keywords if len(kw) > 3):
                    return route.agent_name

        return None

    async def _llm_route(
        self,
        input_text: str,
        llm_client: LLMClient | None = None,
    ) -> str | None:
        """Route using LLM analysis."""
        if llm_client is None:
            from orchestrator.core.container import get_container

            llm_client = get_container().llm_client

        try:
            # Build routing prompt
            prompt = f"""Analyze this user request and determine which agent should handle it.

Available agents:
{chr(10).join(f"- {r.agent_name}: {r.description}" for r in self.routes)}

User request: {input_text}

Respond with ONLY the agent name from the list above, or "none" if no agent fits.
Agent name:"""

            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(
                    model=self.router_config.routing_model or self.model,
                    temperature=0.1,
                    max_tokens=50,
                ),
                auto_session=False,
            )

            # Parse response
            result = (response.content or "").strip().lower()

            # Find matching agent
            for route in self.routes:
                if route.agent_name.lower() in result or result in route.agent_name.lower():
                    return route.agent_name

            if "none" in result:
                return None

            return None

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}")
            return None

    def get_route(self, agent_name: str) -> Route | None:
        """Get route definition by agent name."""
        for route in self.routes:
            if route.agent_name == agent_name:
                return route
        return None

    def add_route(self, route: Route) -> None:
        """Add a route to the router."""
        # Remove existing route with same name
        self.routes = [r for r in self.routes if r.agent_name != route.agent_name]
        self.routes.append(route)

    def remove_route(self, agent_name: str) -> bool:
        """Remove a route by agent name."""
        original_len = len(self.routes)
        self.routes = [r for r in self.routes if r.agent_name != agent_name]
        return len(self.routes) < original_len

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "routes": [r.to_dict() for r in self.routes],
                "fallback_agent_name": self.fallback_agent_name,
                "router_config": self.router_config.to_dict(),
            }
        )
        return base


def create_router_agent(
    name: str,
    routes: list[tuple[str, str]],  # List of (agent_name, description)
    *,
    fallback: str | None = None,
    strategy: Literal["llm", "rule_based", "hybrid"] = "hybrid",
    model: str | None = None,
) -> RouterAgent:
    """
    Factory function to create a router agent.

    Args:
        name: Router agent name
        routes: List of (agent_name, description) tuples
        fallback: Fallback agent name
        strategy: Routing strategy
        model: Model for LLM routing

    Returns:
        Configured RouterAgent

    Example:
        ```python
        router = create_router_agent(
            name="triage",
            routes=[
                ("billing", "Handle billing and payment questions"),
                ("technical", "Handle technical issues and bugs"),
            ],
            fallback="general",
        )
        ```
    """
    route_objects = [
        Route(agent_name=agent_name, description=description) for agent_name, description in routes
    ]

    return RouterAgent(
        name=name,
        routes=route_objects,
        fallback_agent_name=fallback,
        model=model or settings.default_llm_model,
        router_config=RouterConfig(routing_strategy=strategy),
    )
