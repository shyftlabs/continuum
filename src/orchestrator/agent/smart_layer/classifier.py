"""Tier classification: fixed slots, host JSON classifier, remote Qwen, or local Qwen."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from orchestrator.agent.config import RouterConfig
from orchestrator.agent.smart_layer.defaults import CLASSIFIER_SYSTEM_PROMPT
from orchestrator.agent.smart_layer.errors import TierClassifierError
from orchestrator.agent.smart_layer.heuristics import heuristic_tier
from orchestrator.agent.smart_layer.json_parse import parse_classifier_tier_strict
from orchestrator.agent.smart_layer.types import ClassifyOutcome, ProductTier
from orchestrator.config import settings
from orchestrator.llm.config import LLMConfig

if TYPE_CHECKING:
    from orchestrator.llm import LLMClient

_DEFAULT_CLASSIFIER_MODEL = "gpt-4o-mini"

# Defaults when tier_classifier=qwen and LLM_ROUTE_ROUTER_* / tier_router_* are omitted (HF_API_KEY only).
_DEFAULT_HF_ROUTER_API_BASE = "https://router.huggingface.co/v1"
_DEFAULT_HF_TIER_CLASSIFIER_MODEL = "Qwen/Qwen3-4B-Instruct-2507:fastest"

# Modes that send classification to a dedicated HTTP endpoint (HF router or local server).
# Keyword heuristics must not short-circuit these — otherwise tier is chosen locally while the
# remote/local classifier is never contacted (completion still uses the host stack, so runs
# appear to "work" without the classifier server).
_ROUTED_HTTP_CLASSIFIER_MODES = frozenset({"qwen", "qwen_local"})


def _classifier_llm_config(
    *,
    router_config: RouterConfig,
    mode: str,
) -> LLMConfig:
    """Build LLMConfig for the tier classifier call (OpenAI-compatible chat completions)."""
    explicit_model = (router_config.tier_classifier_llm_model or "").strip()
    env_router_model = (settings.llm_route_router_model or "").strip()
    env_local_model = (settings.llm_route_local_router_model or "").strip()

    if mode == "qwen":
        cls_model = explicit_model or env_router_model or _DEFAULT_HF_TIER_CLASSIFIER_MODEL
    elif mode == "qwen_local":
        # Prefer explicit/rc, then LOCAL_ROUTER_MODEL (MLX), then HF ROUTER_MODEL fallback.
        cls_model = explicit_model or env_local_model or env_router_model
        if not cls_model:
            raise TierClassifierError(
                "tier_classifier=qwen_local requires tier_classifier_llm_model, LLM_ROUTE_LOCAL_ROUTER_MODEL "
                "(e.g. mlx-community/Qwen2.5-3B-Instruct-4bit), or LLM_ROUTE_ROUTER_MODEL."
            )
    else:
        cls_model = explicit_model or _DEFAULT_CLASSIFIER_MODEL

    cls_config = LLMConfig(
        model=cls_model,
        temperature=0.1,
        max_tokens=router_config.tier_classifier_max_tokens,
        json_mode=True,
    )

    api_base: str | None = None
    api_key: str | None = None

    if mode == "gpt_4o_mini":
        # Always use the host LLM stack (e.g. OPENAI_API_KEY / default provider) — never the remote router URL.
        pass

    elif mode == "qwen":
        api_base = (
            (router_config.tier_router_api_base or "").strip()
            or (settings.llm_route_router_api_base or "").strip()
            or _DEFAULT_HF_ROUTER_API_BASE
        )
        rk = (router_config.tier_router_api_key or "").strip()
        sk = (settings.llm_route_router_api_key or "").strip()
        hk = (settings.hf_api_key or "").strip()
        api_key = rk or sk or hk or None

    elif mode == "qwen_local":
        # Never use tier_router_api_base here — that slot is for remote Qwen (HF). Mixing them caused
        # qwen_local to call a leftover HF/OpenAI URL while localhost was down.
        api_base = (
            (router_config.tier_local_router_api_base or "").strip()
            or (settings.llm_route_local_router_api_base or "").strip()
            or None
        )
        lk = (router_config.tier_local_router_api_key or "").strip()
        api_key = lk or settings.llm_route_local_router_api_key

    if api_base:
        cls_config = cls_config.model_copy(
            update={
                "api_base": api_base,
                "api_key": api_key,
            }
        )

    if mode == "qwen":
        rk = (router_config.tier_router_api_key or "").strip()
        sk = (settings.llm_route_router_api_key or "").strip()
        hk = (settings.hf_api_key or "").strip()
        if not (rk or sk or hk):
            raise TierClassifierError(
                "tier_classifier=qwen requires HF_API_KEY, tier_router_api_key, or LLM_ROUTE_ROUTER_API_KEY "
                "(OpenAI is not used for this classifier call)."
            )

    if mode == "qwen_local" and not (cls_config.api_base or "").strip():
        raise TierClassifierError(
            "tier_classifier=qwen_local requires tier_local_router_api_base or "
            "LLM_ROUTE_LOCAL_ROUTER_API_BASE (local OpenAI-compatible classifier URL only; "
            "remote HF URL belongs in tier_router_api_base for mode=qwen)."
        )

    return cls_config


async def classify_product_tier(
    *,
    user_text: str,
    router_config: RouterConfig,
    llm_client: LLMClient,
    forced_tier: ProductTier | None = None,
) -> ClassifyOutcome:
    """
    Choose product tier using heuristic shortcut (gpt_4o_mini only) and/or classifier LLM.

    For ``qwen`` and ``qwen_local``, keyword heuristics are not applied so the configured
    classifier endpoint is always used when this function reaches the LLM path.
    """
    t0 = time.perf_counter()

    if forced_tier is not None:
        elapsed = (time.perf_counter() - t0) * 1000
        return ClassifyOutcome(
            tier=forced_tier,
            skipped_classifier=True,
            classify_ms=elapsed,
            skip_reason="forced_tier",
        )

    mode = router_config.tier_classifier

    if mode == "light_only":
        elapsed = (time.perf_counter() - t0) * 1000
        return ClassifyOutcome(
            tier=ProductTier.fast,
            skipped_classifier=True,
            classify_ms=elapsed,
            skip_reason="light_only",
        )

    if mode == "heavy_only":
        elapsed = (time.perf_counter() - t0) * 1000
        return ClassifyOutcome(
            tier=ProductTier.balanced,
            skipped_classifier=True,
            classify_ms=elapsed,
            skip_reason="heavy_only",
        )

    if (
        router_config.tier_classifier_heuristic_shortcut
        and mode not in _ROUTED_HTTP_CLASSIFIER_MODES
    ):
        h = heuristic_tier(user_text)
        if h is not None:
            elapsed = (time.perf_counter() - t0) * 1000
            return ClassifyOutcome(
                tier=h,
                skipped_classifier=True,
                classify_ms=elapsed,
                skip_reason="heuristic_shortcut",
            )

    user_prompt = f'User message:\n"""{user_text.strip()}"""\n\nJSON tier classification:'

    cls_config = _classifier_llm_config(router_config=router_config, mode=mode)

    resp = await llm_client.chat(
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        config=cls_config,
        auto_session=False,
    )
    tier = parse_classifier_tier_strict(resp.content)

    elapsed = (time.perf_counter() - t0) * 1000
    return ClassifyOutcome(
        tier=tier,
        skipped_classifier=False,
        classify_ms=elapsed,
        skip_reason="classifier_llm",
    )
