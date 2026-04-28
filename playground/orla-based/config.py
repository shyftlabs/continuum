"""
Configuration for the orla-based playground.

Wires together:
- User tiers (free / premium) with different dispatch priorities
- PolicyStore: free users cannot call tool:checkout
- PriorityDispatcher for LLM calls
- Model and agent settings
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from dataclasses import dataclass, field

from orchestrator.llm.dispatcher import PriorityDispatcher
from orchestrator.security.policy import AccessPolicy, PolicyStore


# ---------------------------------------------------------------------------
# User tiers
# ---------------------------------------------------------------------------

TIER_PRIORITY = {
    "premium": 9,
    "free": 2,
}

# ---------------------------------------------------------------------------
# Policy store
# ---------------------------------------------------------------------------

def build_policy_store() -> PolicyStore:
    """Build the default policy store.

    Rules:
    - free_agent cannot call tool:checkout (premium only)
    - All agents can call search, cart, and product tools
    """
    store = PolicyStore()

    store.add_policy(AccessPolicy(
        name="block_free_checkout",
        subjects=["free-agent"],
        resources=["tool:checkout"],
        effect="deny",
        denial_message="'checkout' is only available on the premium tier. Tell the user to type '/tier premium' to upgrade.",
    ))

    store.add_policy(AccessPolicy(
        name="allow_all_search",
        subjects=["*"],
        resources=["tool:search_products", "tool:get_product",
                   "tool:add_to_cart", "tool:view_cart"],
        effect="allow",
    ))

    store.add_policy(AccessPolicy(
        name="allow_premium_checkout",
        subjects=["premium-agent"],
        resources=["tool:checkout"],
        effect="allow",
    ))

    return store


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def build_dispatcher() -> PriorityDispatcher:
    return PriorityDispatcher(max_concurrent=5)


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    model: str = "gemini/gemini-2.5-flash"
    temperature: float = 0.7
    max_turns: int = 8
    enable_memory: bool = True
    enable_session: bool = True
    default_tier: str = "free"
    policy_store: PolicyStore = field(default_factory=build_policy_store)
    dispatcher: PriorityDispatcher = field(default_factory=build_dispatcher)


default_config = AppConfig()
