"""Resolve concrete model id per tier from RouterConfig + settings."""

from __future__ import annotations

from orchestrator.agent.config import RouterConfig
from orchestrator.agent.smart_layer.defaults import default_model_for_tier
from orchestrator.agent.smart_layer.types import ProductTier


def resolve_model_for_tier(
    tier: ProductTier, router_config: RouterConfig, default_llm_model: str
) -> str:
    """
    Per-tier override fields, legacy light/heavy slots, then gap-fill with default_llm_model.

    Legacy: tier_light_model → fast if tier_fast_model unset;
            tier_heavy_model → balanced if tier_balanced_model unset.
    """
    rc = router_config

    explicit = {
        ProductTier.nano: rc.tier_nano_model,
        ProductTier.fast: rc.tier_fast_model or rc.tier_light_model,
        ProductTier.balanced: rc.tier_balanced_model or rc.tier_heavy_model,
        ProductTier.specialist: rc.tier_specialist_model,
        ProductTier.frontier: rc.tier_frontier_model,
    }

    mid = (explicit[tier] or "").strip()
    if mid:
        return mid

    base = default_model_for_tier(tier)
    if base:
        return base

    return default_llm_model or "gpt-4o-mini"


def effective_completion_model(
    tier: ProductTier, router_config: RouterConfig, default_llm_model: str
) -> str:
    """If tier_force_completion_model is set, use it for every tier (cheap testing)."""
    forced = (router_config.tier_force_completion_model or "").strip()
    if forced:
        return forced
    return resolve_model_for_tier(tier, router_config, default_llm_model)
