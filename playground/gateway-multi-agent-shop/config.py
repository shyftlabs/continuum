import os

# Must be set before orchestrator settings are imported (they are cached on first import).
os.environ.setdefault("SMART_GATEWAY_URL", "http://localhost:8787/v1")
os.environ.setdefault("SMART_GATEWAY_API_KEY", "your-smart-gateway-api-key")

from dataclasses import dataclass, field


@dataclass
class WorkflowShopConfig:
    mcp_url: str = "http://localhost:8890/mcp"
    mcp_timeout: float = 10.0

    # Placeholder model — GatewayProvider translates to auto/<tier> at runtime.
    model: str = "gpt-4o-mini"
    max_turns: int = 10
    enable_memory: bool = True
    enable_session: bool = True

    # Per-agent gateway routing mode: "strict" | "modest" | "quality"
    # None → falls back to SMART_GATEWAY_DEFAULT_MODE env var (default "modest")
    gateway_mode: str | None = None

    mode_descriptions: dict = field(
        default_factory=lambda: {
            "sequential": "search → recommend → add to cart  (3-step pipeline)",
            "parallel": "search dogs + cats simultaneously, merge results",
            "loop": "keep searching until a match under your budget is found",
            "scatter": "analyse each of p1/p2/p3 in parallel, pick best value",
            "supervised": "write a buying guide — supervisor retries if quality < 0.7",
            "planner": "dynamic plan: LLM decides which steps to run",
            "debate": "pro-premium vs pro-budget dog food — judge decides",
            "reflection": "write a recommendation email — self-critique until PASS",
            "router": "triage: route to search, cart, or support based on intent",
            "handoff": "orchestrator plans, hands off to executor which calls MCP tools",
        }
    )


default_config = WorkflowShopConfig()
