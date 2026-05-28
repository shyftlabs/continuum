import os
from pathlib import Path

from dotenv import load_dotenv

# Load the project root .env first so hosted gateway settings (SMART_GATEWAY_URL,
# SMART_GATEWAY_API_KEY) take precedence over shell env vars and the localhost
# fallbacks below. override=True is required because the shell may already have
# stale localhost values exported from a previous session.
# Must be done before orchestrator settings are imported (they are cached on first import).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

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
    agent_model: str = "gpt-4o-mini"
    agent_temperature: float = 0.7
    max_turns: int = 10

    # Per-agent gateway routing mode: "strict" | "modest" | "quality"
    # None → falls back to SMART_GATEWAY_DEFAULT_MODE env var (default "modest")
    gateway_mode: str | None = None

    enable_memory: bool = True
    enable_session: bool = True

    system_instructions: str = """You are a friendly pet shop assistant.
Help users find the right products for their pets, answer pet care questions, manage their cart, and checkout.
Be concise and helpful."""


default_config = ShopConfig()
