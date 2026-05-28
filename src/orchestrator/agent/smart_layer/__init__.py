"""Smart layer (model_tier): tier classification + single-turn completion.

Import execution helpers from ``orchestrator.agent.smart_layer.runner_facade`` to avoid
import cycles with ``RouterAgent``.
"""

from __future__ import annotations

from orchestrator.agent.smart_layer.defaults import (
    CLASSIFIER_SYSTEM_PROMPT,
    MODEL_TIER_DEFAULT_INSTRUCTIONS,
)
from orchestrator.agent.smart_layer.types import (
    ClassifyOutcome,
    ModelTierResult,
    ProductTier,
    StreamYield,
    parse_product_tier,
    tier_dispatch_priority,
)

__all__ = [
    "CLASSIFIER_SYSTEM_PROMPT",
    "MODEL_TIER_DEFAULT_INSTRUCTIONS",
    "ClassifyOutcome",
    "ModelTierResult",
    "ProductTier",
    "StreamYield",
    "parse_product_tier",
    "tier_dispatch_priority",
    "classify_product_tier",
    "effective_completion_model",
    "extract_last_user_text",
    "heuristic_tier",
    "TierClassifierError",
    "parse_classifier_json",
    "parse_classifier_tier_strict",
    "resolve_model_for_tier",
    "run_model_tier_turn",
    "stream_model_tier_turn",
]


def __getattr__(name: str):
    if name == "classify_product_tier":
        from orchestrator.agent.smart_layer.classifier import classify_product_tier

        return classify_product_tier
    if name == "effective_completion_model":
        from orchestrator.agent.smart_layer.resolve import effective_completion_model

        return effective_completion_model
    if name == "extract_last_user_text":
        from orchestrator.agent.smart_layer.runner_facade import extract_last_user_text

        return extract_last_user_text
    if name == "heuristic_tier":
        from orchestrator.agent.smart_layer.heuristics import heuristic_tier

        return heuristic_tier
    if name == "TierClassifierError":
        from orchestrator.agent.smart_layer.errors import TierClassifierError

        return TierClassifierError
    if name == "parse_classifier_json":
        from orchestrator.agent.smart_layer.json_parse import parse_classifier_json

        return parse_classifier_json
    if name == "parse_classifier_tier_strict":
        from orchestrator.agent.smart_layer.json_parse import parse_classifier_tier_strict

        return parse_classifier_tier_strict
    if name == "resolve_model_for_tier":
        from orchestrator.agent.smart_layer.resolve import resolve_model_for_tier

        return resolve_model_for_tier
    if name == "run_model_tier_turn":
        from orchestrator.agent.smart_layer.runner_facade import run_model_tier_turn

        return run_model_tier_turn
    if name == "stream_model_tier_turn":
        from orchestrator.agent.smart_layer.runner_facade import stream_model_tier_turn

        return stream_model_tier_turn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
