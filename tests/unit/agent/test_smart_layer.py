"""Unit tests for smart_layer (model_tier) helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agent.config import RouterConfig
from orchestrator.agent.smart_layer import classifier as classifier_mod
from orchestrator.agent.smart_layer.classifier import classify_product_tier
from orchestrator.agent.smart_layer.errors import TierClassifierError
from orchestrator.agent.smart_layer.heuristics import heuristic_tier
from orchestrator.agent.smart_layer.json_parse import (
    parse_classifier_json,
    parse_classifier_tier_strict,
)
from orchestrator.agent.smart_layer.resolve import resolve_model_for_tier
from orchestrator.agent.smart_layer.types import ProductTier
from orchestrator.llm.types import LLMResponse


class TestJsonParse:
    def test_strict_tier(self):
        assert parse_classifier_json('{"tier": "specialist"}') == ProductTier.specialist

    def test_legacy_complexity(self):
        assert parse_classifier_json('{"complexity": "simple"}') == ProductTier.fast
        assert parse_classifier_json('{"complexity": "complex"}') == ProductTier.balanced

    def test_regex_tier(self):
        assert parse_classifier_json('prefix {"tier": "frontier"} suffix') == ProductTier.frontier

    def test_fallback_balanced(self):
        assert parse_classifier_json("") == ProductTier.balanced
        assert parse_classifier_json("no json here") == ProductTier.balanced


class TestParseClassifierTierStrict:
    def test_explicit_balanced_ok(self):
        assert parse_classifier_tier_strict('{"tier": "balanced"}') == ProductTier.balanced

    def test_empty_raises(self):
        with pytest.raises(TierClassifierError, match="empty"):
            parse_classifier_tier_strict("")
        with pytest.raises(TierClassifierError, match="empty"):
            parse_classifier_tier_strict("   ")

    def test_garbage_raises(self):
        with pytest.raises(TierClassifierError, match="valid tier"):
            parse_classifier_tier_strict("no tier here at all")


class TestHeuristics:
    def test_specialist_keyword(self):
        assert heuristic_tier("please debug this stack trace") == ProductTier.specialist

    def test_frontier_keyword(self):
        assert heuristic_tier("give a formal proof that np-complete") == ProductTier.frontier

    def test_short_nano(self):
        assert heuristic_tier("ok") == ProductTier.nano

    def test_no_match_returns_none(self):
        assert heuristic_tier("explain how databases use b-trees in moderate detail") is None


class TestResolveModel:
    def test_explicit_override(self):
        rc = RouterConfig(tier_nano_model="custom-nano")
        assert resolve_model_for_tier(ProductTier.nano, rc, "gpt-4o-mini") == "custom-nano"

    def test_legacy_light_heavy(self):
        rc = RouterConfig(tier_light_model="fast-model", tier_heavy_model="heavy-model")
        assert resolve_model_for_tier(ProductTier.fast, rc, "fallback") == "fast-model"
        assert resolve_model_for_tier(ProductTier.balanced, rc, "fallback") == "heavy-model"

    def test_default_table(self):
        rc = RouterConfig()
        assert resolve_model_for_tier(ProductTier.nano, rc, "gpt-4o-mini") == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_classify_light_only_skips_llm():
    rc = RouterConfig(tier_classifier="light_only")
    llm = MagicMock()
    llm.chat = AsyncMock()
    out = await classify_product_tier(
        user_text="anything",
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert out.tier == ProductTier.fast
    assert out.skipped_classifier is True
    assert out.skip_reason == "light_only"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_classify_forced_tier():
    rc = RouterConfig()
    llm = MagicMock()
    llm.chat = AsyncMock()
    out = await classify_product_tier(
        user_text="x",
        router_config=rc,
        llm_client=llm,
        forced_tier=ProductTier.frontier,
    )
    assert out.tier == ProductTier.frontier
    assert out.skipped_classifier is True
    assert out.skip_reason == "forced_tier"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_gpt_4o_mini_classifier_ignores_remote_router_url(monkeypatch):
    """gpt_4o_mini mode must not send classifier to tier_router_api_base."""
    rc = RouterConfig(
        tier_classifier="gpt_4o_mini",
        tier_router_api_base="https://router.example/v1",
        tier_router_api_key="should-not-be-used-for-mini",
    )
    llm = MagicMock()
    seen: dict = {}

    async def fake_chat(**kwargs):
        cfg = kwargs["config"]
        seen["api_base"] = getattr(cfg, "api_base", None)
        return LLMResponse(model="gpt-4o-mini", content='{"tier":"fast"}')

    llm.chat = AsyncMock(side_effect=fake_chat)
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    await classify_product_tier(
        user_text="longer message without heuristic match " * 5,
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert seen.get("api_base") is None


@pytest.mark.asyncio
async def test_classify_llm_json(monkeypatch):
    rc = RouterConfig(tier_classifier="gpt_4o_mini")
    llm = MagicMock()

    async def fake_chat(**kwargs):
        return LLMResponse(model="gpt-4o-mini", content='{"tier": "balanced"}')

    llm.chat = AsyncMock(side_effect=fake_chat)

    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )

    out = await classify_product_tier(
        user_text="longer message without heuristic match " * 5,
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert out.tier == ProductTier.balanced
    assert out.skipped_classifier is False
    assert out.skip_reason == "classifier_llm"
    llm.chat.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_uses_hf_defaults_when_only_hf_api_key(monkeypatch):
    """qwen mode: HF router URL + model default; auth from HF_API_KEY."""
    import orchestrator.config as oc

    monkeypatch.setattr(oc.settings, "hf_api_key", "hf_test_token")
    monkeypatch.setattr(oc.settings, "llm_route_router_api_base", None)
    monkeypatch.setattr(oc.settings, "llm_route_router_api_key", None)
    monkeypatch.setattr(oc.settings, "llm_route_router_model", None)

    rc = RouterConfig(tier_classifier="qwen")
    llm = MagicMock()
    seen: dict = {}

    async def fake_chat(**kwargs):
        seen["config"] = kwargs["config"]
        return LLMResponse(model="m", content='{"tier":"fast"}')

    llm.chat = AsyncMock(side_effect=fake_chat)
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    out = await classify_product_tier(
        user_text="longer message without heuristic match " * 5,
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert out.tier == ProductTier.fast
    cfg = seen["config"]
    assert cfg.api_base == classifier_mod._DEFAULT_HF_ROUTER_API_BASE
    assert cfg.model == classifier_mod._DEFAULT_HF_TIER_CLASSIFIER_MODEL
    assert cfg.api_key == "hf_test_token"


@pytest.mark.asyncio
async def test_qwen_local_short_prompt_still_requires_classifier_no_heuristic(monkeypatch):
    """Heuristics must not bypass qwen_local — otherwise local server is never called."""
    import orchestrator.config as oc

    monkeypatch.setattr(oc.settings, "llm_route_local_router_api_base", None)
    rc = RouterConfig(
        tier_classifier="qwen_local",
        tier_classifier_llm_model="qwen2.5",
    )
    llm = MagicMock()
    llm.chat = AsyncMock()
    with pytest.raises(TierClassifierError, match="LLM_ROUTE_LOCAL_ROUTER_API_BASE"):
        await classify_product_tier(
            user_text="ok",
            router_config=rc,
            llm_client=llm,
            forced_tier=None,
        )
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_qwen_local_always_calls_classifier_llm_for_short_message():
    """Without routed modes, 'ok' → nano heuristic; qwen_local must hit the classifier."""
    rc = RouterConfig(
        tier_classifier="qwen_local",
        tier_classifier_llm_model="qwen2.5",
        tier_local_router_api_base="http://127.0.0.1:11434/v1",
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=LLMResponse(model="local-qwen", content='{"tier": "balanced"}')
    )
    out = await classify_product_tier(
        user_text="ok",
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert out.tier == ProductTier.balanced
    assert out.skip_reason == "classifier_llm"
    llm.chat.assert_called_once()
    call_kw = llm.chat.call_args.kwargs
    assert (call_kw["config"].api_base or "").startswith("http://127.0.0.1:11434")


@pytest.mark.asyncio
async def test_qwen_local_ignores_remote_router_api_base(monkeypatch):
    """HF URL in tier_router_api_base must not satisfy qwen_local (common playground mistake)."""
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    rc = RouterConfig(
        tier_classifier="qwen_local",
        tier_classifier_llm_model="qwen2.5",
        tier_router_api_base="https://router.huggingface.co/v1",
        tier_router_api_key="hf_fake",
    )
    llm = MagicMock()
    llm.chat = AsyncMock()
    with pytest.raises(TierClassifierError, match="tier_local_router_api_base"):
        await classify_product_tier(
            user_text="hello world classifier path",
            router_config=rc,
            llm_client=llm,
            forced_tier=None,
        )
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_qwen_local_requires_local_or_router_base(monkeypatch):
    import orchestrator.config as oc

    monkeypatch.setattr(oc.settings, "llm_route_local_router_api_base", None)
    rc = RouterConfig(
        tier_classifier="qwen_local",
        tier_classifier_llm_model="qwen2.5",
    )
    llm = MagicMock()
    llm.chat = AsyncMock()
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    with pytest.raises(TierClassifierError, match="LLM_ROUTE_LOCAL_ROUTER_API_BASE"):
        await classify_product_tier(
            user_text="longer message without heuristic match " * 5,
            router_config=rc,
            llm_client=llm,
            forced_tier=None,
        )
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_qwen_requires_hf_openai_or_router_token(monkeypatch):
    import orchestrator.config as oc

    monkeypatch.setattr(oc.settings, "hf_api_key", None)
    monkeypatch.setattr(oc.settings, "llm_route_router_api_key", None)

    rc = RouterConfig(
        tier_classifier="qwen",
        tier_classifier_llm_model="Qwen/Qwen3-Test",
        tier_router_api_base="https://router.example/v1",
    )
    llm = MagicMock()
    llm.chat = AsyncMock()
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    with pytest.raises(TierClassifierError, match="HF_API_KEY"):
        await classify_product_tier(
            user_text="longer message without heuristic match " * 5,
            router_config=rc,
            llm_client=llm,
            forced_tier=None,
        )
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_classifier_llm_failure_propagates(monkeypatch):
    rc = RouterConfig(tier_classifier="gpt_4o_mini")
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=ConnectionError("refused"))
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    with pytest.raises(ConnectionError, match="refused"):
        await classify_product_tier(
            user_text="longer message without heuristic match " * 5,
            router_config=rc,
            llm_client=llm,
            forced_tier=None,
        )


@pytest.mark.asyncio
async def test_classifier_unparseable_response_raises(monkeypatch):
    rc = RouterConfig(tier_classifier="gpt_4o_mini")
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=LLMResponse(model="gpt-4o-mini", content="thanks"))
    monkeypatch.setattr(
        "orchestrator.agent.smart_layer.classifier.heuristic_tier",
        lambda _: None,
    )
    with pytest.raises(TierClassifierError, match="valid tier"):
        await classify_product_tier(
            user_text="longer message without heuristic match " * 5,
            router_config=rc,
            llm_client=llm,
            forced_tier=None,
        )


@pytest.mark.asyncio
async def test_heuristic_shortcut_triggers_for_short_message():
    rc = RouterConfig(tier_classifier="gpt_4o_mini")
    llm = MagicMock()
    llm.chat = AsyncMock()
    out = await classify_product_tier(
        user_text="ok",
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert out.tier == ProductTier.nano
    assert out.skipped_classifier is True
    assert out.skip_reason == "heuristic_shortcut"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_heuristic_shortcut_disabled_calls_classifier_llm():
    rc = RouterConfig(tier_classifier="gpt_4o_mini", tier_classifier_heuristic_shortcut=False)
    llm = MagicMock()

    async def fake_chat(**kwargs):
        return LLMResponse(model="gpt-4o-mini", content='{"tier": "balanced"}')

    llm.chat = AsyncMock(side_effect=fake_chat)
    out = await classify_product_tier(
        user_text="ok",
        router_config=rc,
        llm_client=llm,
        forced_tier=None,
    )
    assert out.tier == ProductTier.balanced
    assert out.skipped_classifier is False
    assert out.skip_reason == "classifier_llm"
    llm.chat.assert_called_once()
