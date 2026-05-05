from dataclasses import dataclass


@dataclass
class ShopConfig:
    mcp_url: str = "http://localhost:8888/mcp"
    mcp_timeout: float = 10.0

    agent_name: str = "shop-assistant"
    # agent_model: str = "gemini/gemini-2.5-flash"
    # agent_model: str = "gpt-4o"
    agent_model: str = "gpt-5"
    agent_temperature: float = 1.0 # 0.7, gpt-5 only accepts 1.0
    max_turns: int = 10

    enable_memory: bool = True
    enable_session: bool = True

    system_instructions: str = """You are a friendly pet shop assistant.
Help users find the right products for their pets, answer pet care questions, manage their cart, and checkout.
Be concise and helpful."""


default_config = ShopConfig()
