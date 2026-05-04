"""Orchestrate classify + completion for model_tier (sync + async stream)."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from orchestrator.agent.smart_layer.classifier import classify_product_tier
from orchestrator.agent.smart_layer.completion import (
    build_model_tier_messages,
    complete_non_stream,
    temperature_for_tier,
)
from orchestrator.agent.smart_layer.resolve import effective_completion_model, resolve_model_for_tier
from orchestrator.agent.smart_layer.types import ModelTierResult, ProductTier, StreamYield, tier_dispatch_priority
from orchestrator.config import settings
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.llm.config import LLMConfig
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient

logger = get_logger(__name__)


def _routing_payload(
    *,
    tier: ProductTier,
    routed_model: str,
    execution_model: str,
    classify_ms: float,
    llm_ms: float,
    skipped_classifier: bool,
    classifier_skip_reason: str,
    tier_classifier: str,
    done: bool,
) -> dict[str, Any]:
    return {
        "tier": tier.value,
        "routed_model": routed_model,
        "execution_model": execution_model,
        "timings_ms": {"classify": round(classify_ms, 3), "llm": round(llm_ms, 3)},
        "skipped_classifier": skipped_classifier,
        "classifier_skip_reason": classifier_skip_reason,
        "tier_classifier": tier_classifier,
        "cache_hit": False,
        "similarity": None,
        "done": done,
    }


def extract_last_user_text(
    input_data: str | list[dict[str, Any]],
    messages: list[dict[str, Any]] | None = None,
) -> str:
    """Single-turn semantics: last user message from string or chat messages."""
    if isinstance(input_data, str):
        return input_data.strip()

    src = messages if messages else input_data
    for m in reversed(src):
        if m.get("role") == "user":
            c = m.get("content")
            return str(c or "").strip()
    return ""


async def run_model_tier_turn(
    agent: RouterAgent,
    llm_client: LLMClient,
    *,
    user_text: str,
    forced_tier: ProductTier | None = None,
) -> ModelTierResult:
    """Non-streaming: aggregate streaming internally if needed, or direct completion."""
    rc = agent.router_config

    if os.environ.get("ROUTER_SHADOW_MODE"):
        logger.info("ROUTER_SHADOW_MODE set: shadow dual-path is not implemented; running model_tier normally")

    classify_out = await classify_product_tier(
        user_text=user_text,
        router_config=rc,
        llm_client=llm_client,
        forced_tier=forced_tier,
    )
    tier = classify_out.tier
    classify_ms = classify_out.classify_ms

    routed_model = resolve_model_for_tier(tier, rc, settings.default_llm_model)
    execution_model = effective_completion_model(tier, rc, settings.default_llm_model)
    temp = temperature_for_tier(tier, rc)
    max_tok = rc.tier_completion_max_tokens
    messages = build_model_tier_messages(agent.instructions, user_text)
    priority = tier_dispatch_priority(tier)

    t_llm = time.perf_counter()
    parts: list[str] = []
    cfg = LLMConfig(model=execution_model, temperature=temp, max_tokens=max_tok)

    async for chunk in llm_client.chat_stream(
        messages=messages,
        config=cfg,
    ):
        if chunk.content:
            parts.append(chunk.content)

    content = "".join(parts).strip()
    llm_ms = (time.perf_counter() - t_llm) * 1000

    if not content:
        logger.warning("model_tier stream produced no text; attempting non-stream fallback")
        t_fb = time.perf_counter()
        content = await complete_non_stream(
            llm_client,
            messages=messages,
            model=execution_model,
            temperature=temp,
            max_tokens=max_tok,
            priority=priority,
        )
        llm_ms += (time.perf_counter() - t_fb) * 1000

    if not content:
        raise RuntimeError("model_tier completion produced no assistant text")

    routing = _routing_payload(
        tier=tier,
        routed_model=routed_model,
        execution_model=execution_model,
        classify_ms=classify_ms,
        llm_ms=llm_ms,
        skipped_classifier=classify_out.skipped_classifier,
        classifier_skip_reason=classify_out.skip_reason,
        tier_classifier=rc.tier_classifier,
        done=True,
    )

    return ModelTierResult(content=content, routing=routing)


async def stream_model_tier_turn(
    agent: RouterAgent,
    llm_client: LLMClient,
    *,
    user_text: str,
    forced_tier: ProductTier | None = None,
) -> AsyncIterator[StreamYield]:
    """Emit routing (partial + final), content deltas, then done routing snapshot."""
    rc = agent.router_config

    if os.environ.get("ROUTER_SHADOW_MODE"):
        logger.info("ROUTER_SHADOW_MODE set: shadow dual-path is not implemented; running model_tier normally")

    classify_out = await classify_product_tier(
        user_text=user_text,
        router_config=rc,
        llm_client=llm_client,
        forced_tier=forced_tier,
    )
    tier = classify_out.tier
    classify_ms = classify_out.classify_ms

    routed_model = resolve_model_for_tier(tier, rc, settings.default_llm_model)
    execution_model = effective_completion_model(tier, rc, settings.default_llm_model)
    temp = temperature_for_tier(tier, rc)
    max_tok = rc.tier_completion_max_tokens
    messages = build_model_tier_messages(agent.instructions, user_text)
    priority = tier_dispatch_priority(tier)

    partial = _routing_payload(
        tier=tier,
        routed_model=routed_model,
        execution_model=execution_model,
        classify_ms=classify_ms,
        llm_ms=0.0,
        skipped_classifier=classify_out.skipped_classifier,
        classifier_skip_reason=classify_out.skip_reason,
        tier_classifier=rc.tier_classifier,
        done=False,
    )
    yield StreamYield(kind="routing", routing=partial)

    t_llm = time.perf_counter()
    parts: list[str] = []
    cfg = LLMConfig(model=execution_model, temperature=temp, max_tokens=max_tok)

    async for chunk in llm_client.chat_stream(messages=messages, config=cfg):
        if chunk.content:
            parts.append(chunk.content)
            yield StreamYield(kind="content_delta", text=chunk.content)

    content = "".join(parts).strip()
    llm_ms = (time.perf_counter() - t_llm) * 1000

    if not content:
        logger.warning("model_tier stream produced no text; attempting non-stream fallback")
        t_fb = time.perf_counter()
        content = await complete_non_stream(
            llm_client,
            messages=messages,
            model=execution_model,
            temperature=temp,
            max_tokens=max_tok,
            priority=priority,
        )
        llm_ms += (time.perf_counter() - t_fb) * 1000
        if content:
            yield StreamYield(kind="content_delta", text=content)

    if not content:
        raise RuntimeError("model_tier completion produced no assistant text")

    final_routing = _routing_payload(
        tier=tier,
        routed_model=routed_model,
        execution_model=execution_model,
        classify_ms=classify_ms,
        llm_ms=llm_ms,
        skipped_classifier=classify_out.skipped_classifier,
        classifier_skip_reason=classify_out.skip_reason,
        tier_classifier=rc.tier_classifier,
        done=True,
    )
    yield StreamYield(kind="routing", routing=final_routing)
    yield StreamYield(kind="done", routing=final_routing)
