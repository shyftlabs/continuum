import os

# Point Continuum at the Smart Gateway running via docker compose.
# Must be set before orchestrator settings are imported (they are cached on first import).
# os.environ.setdefault("SMART_GATEWAY_URL", "https://continuum.shyftops.io/v1")
os.environ.setdefault("SMART_GATEWAY_URL", "http://localhost:8787/v1")
os.environ.setdefault("SMART_GATEWAY_API_KEY", "your-smart-gateway-api-key")

from dataclasses import dataclass


@dataclass
class ShopConfig:
    mcp_url: str = "http://localhost:8888/mcp"
    mcp_timeout: float = 10.0

    agent_name: str = "shop-assistant"
    # Model name is sent as-is to the gateway — gateway routes to the provider.
    # Must be in the virtual key's allowed_models list (conf.json).
    # "auto" delegates model selection to the gateway based on gateway_mode.
    agent_model: str = "auto"
    agent_temperature: float = 0.7
    max_turns: int = 3

    # Per-agent gateway routing mode: "strict" | "modest" | "quality"
    # None → falls back to SMART_GATEWAY_DEFAULT_MODE env var (default "modest")
    gateway_mode: str | None = None

    enable_memory: bool = True
    enable_session: bool = True

    system_instructions: str = """You are a friendly pet shop assistant.
Help users find the right products for their pets, answer pet care questions, manage their cart, and checkout.
Be concise and helpful."""


default_config = ShopConfig()
