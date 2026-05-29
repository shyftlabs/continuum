"""Product tiers and smart-layer result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

TierClassifierMode = Literal["light_only", "heavy_only", "gpt_4o_mini", "qwen", "qwen_local"]

ClassifierSkipReason = Literal[
    "classifier_llm",
    "heuristic_shortcut",
    "forced_tier",
    "light_only",
    "heavy_only",
]

PRODUCT_TIERS = ("nano", "fast", "balanced", "specialist", "frontier")


class ProductTier(str, Enum):
    nano = "nano"
    fast = "fast"
    balanced = "balanced"
    specialist = "specialist"
    frontier = "frontier"


def parse_product_tier(value: str | None) -> ProductTier | None:
    if not value:
        return None
    v = value.strip().lower()
    try:
        return ProductTier(v)
    except ValueError:
        return None


def tier_dispatch_priority(tier: ProductTier) -> int:
    """Runtime request priority for PriorityDispatcher (higher = more urgent)."""
    return {
        ProductTier.nano: 2,
        ProductTier.fast: 4,
        ProductTier.balanced: 5,
        ProductTier.specialist: 8,
        ProductTier.frontier: 10,
    }[tier]


@dataclass
class ModelTierResult:
    """Outcome of a single model_tier turn (classify + complete)."""

    content: str
    routing: dict[str, Any]
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None


@dataclass
class ClassifyOutcome:
    tier: ProductTier
    skipped_classifier: bool
    classify_ms: float
    skip_reason: ClassifierSkipReason = "classifier_llm"


@dataclass
class StreamYield:
    """Internal event from streaming model_tier execution."""

    kind: Literal["routing", "content_delta", "done"]
    routing: dict[str, Any] | None = None
    text: str | None = None
