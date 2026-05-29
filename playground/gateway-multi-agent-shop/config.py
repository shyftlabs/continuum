import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Load the project root .env. override=True ensures .env values win over any
# stale shell exports. Must run before orchestrator settings are imported
# (they are module-level singletons, cached on first import).
#
# Gateway mode is controlled entirely by the root .env:
#   Local gateway:  SMART_GATEWAY_URL=http://localhost:8787/v1
#                   SMART_GATEWAY_API_KEY=ck-prod-2026-05-19
#   Hosted gateway: SMART_GATEWAY_URL=https://continuum.shyftops.io/v1
#                   SMART_GATEWAY_API_KEY=<hosted-key>
#   No gateway:     omit both vars — orchestrator routes directly to the provider
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=True)

# load_dotenv(override=True) sets vars that ARE in .env, but does not clear vars
# that are absent from .env yet still live in os.environ as stale shell exports.
# Explicitly remove gateway vars when they are not present in the .env file so
# that a previous shell session can never accidentally activate the gateway.
_file_env = dotenv_values(_ENV_PATH)
for _var in ("SMART_GATEWAY_URL", "SMART_GATEWAY_API_KEY"):
    if _var not in _file_env:
        os.environ.pop(_var, None)

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
