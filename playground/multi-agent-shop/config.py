from dataclasses import dataclass, field


@dataclass
class WorkflowShopConfig:
    mcp_url: str = "http://localhost:8890/mcp"
    mcp_timeout: float = 10.0
    model: str = "gemini/gemini-2.5-flash"
    max_turns: int = 10
    enable_memory: bool = True
    enable_session: bool = True

    # Per-mode descriptions shown in the CLI header
    mode_descriptions: dict = field(default_factory=lambda: {
        "sequential": "search → recommend → add to cart  (3-step pipeline)",
        "parallel":   "search dogs + cats simultaneously, merge results",
        "loop":       "keep searching until a match under your budget is found",
        "scatter":    "analyse each of p1/p2/p3 in parallel, pick best value",
        "supervised": "write a buying guide — supervisor retries if quality < 0.7",
        "planner":    "dynamic plan: LLM decides which steps to run",
        "debate":     "pro-premium vs pro-budget dog food — judge decides",
        "reflection": "write a recommendation email — self-critique until PASS",
        "router":     "triage: route to search, cart, or support based on intent",
        "handoff":    "orchestrator plans, hands off to executor which calls MCP tools",
    })


default_config = WorkflowShopConfig()
