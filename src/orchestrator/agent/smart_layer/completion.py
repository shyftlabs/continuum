"""Single-turn chat completion helpers for model_tier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.agent.smart_layer.defaults import MODEL_TIER_DEFAULT_INSTRUCTIONS
from orchestrator.agent.smart_layer.types import ProductTier
from orchestrator.llm.config import LLMConfig

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient


def temperature_for_tier(tier: ProductTier, router_config: Any) -> float:
    if tier in (ProductTier.specialist, ProductTier.frontier):
        return router_config.tier_heavy_temperature
    return router_config.tier_light_temperature


def build_model_tier_messages(system_instructions: str, user_text: str) -> list[dict[str, str]]:
    sys_content = (system_instructions or "").strip() or MODEL_TIER_DEFAULT_INSTRUCTIONS
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_text.strip()},
    ]


async def complete_non_stream(
    llm_client: LLMClient,
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    priority: int = 5,
) -> str:
    cfg = LLMConfig(model=model, temperature=temperature, max_tokens=max_tokens)
    resp = await llm_client.chat(
        messages=messages,
        config=cfg,
        auto_session=False,
        priority=priority,
    )
    return (resp.content or "").strip()
