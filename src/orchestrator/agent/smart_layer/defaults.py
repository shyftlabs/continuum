"""Default tier → model mapping using IDs supported in this repo."""

from __future__ import annotations

from orchestrator.agent.smart_layer.types import ProductTier

# OpenAI-style IDs already referenced in orchestrator.llm.utils / providers.
DEFAULT_TIER_MODELS: dict[ProductTier, str] = {
    ProductTier.nano: "gpt-4o-mini",
    ProductTier.fast: "gpt-4o-mini",
    ProductTier.balanced: "gpt-4o",
    ProductTier.specialist: "gpt-4o",
    ProductTier.frontier: "gpt-4o-turbo",
}

MODEL_TIER_DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer clearly and concisely based on the user's message."
)

CLASSIFIER_SYSTEM_PROMPT = """You classify user tasks into exactly one product tier for LLM routing.

Tiers (capability intent):
- nano: trivial triage, skim, yes/no, ultra-short replies
- fast: short answers, quick lookups, simple single-step tasks
- balanced: general multi-step reasoning, explanations, mixed tasks
- specialist: code, debugging, repository/tooling style work, implementation detail
- frontier: hardest reasoning, proofs, novel architecture, deep analysis

Respond with a single JSON object only, no markdown: {"tier": "<one of nano,fast,balanced,specialist,frontier>"}
"""


def default_model_for_tier(tier: ProductTier) -> str:
    return DEFAULT_TIER_MODELS[tier]
