"""Tests for LLM_ROUTE_* env applied to RouterConfig."""

from __future__ import annotations

import pytest

from orchestrator.agent.config import RouterConfig, apply_llm_route_env_overrides


@pytest.fixture
def clear_overrides(monkeypatch):
    """Ensure llm_route_* don't leak from real .env during tests."""
    import orchestrator.config as oc

    s = oc.settings
    for name in (
        "llm_route_tier_classifier",
        "llm_route_router_model",
        "llm_route_router_api_base",
        "llm_route_router_api_key",
        "llm_route_force_completion_model",
        "llm_route_tier_classifier_heuristic_shortcut",
        "llm_route_local_router_api_base",
        "llm_route_local_router_api_key",
        "llm_route_local_router_model",
        "hf_api_key",
    ):
        monkeypatch.setattr(s, name, None, raising=False)


def test_apply_llm_route_env_overrides_full(monkeypatch, clear_overrides):
    import orchestrator.config as oc

    s = oc.settings
    monkeypatch.setattr(s, "llm_route_tier_classifier", "qwen")
    monkeypatch.setattr(s, "llm_route_router_model", "Qwen/Test")
    monkeypatch.setattr(s, "llm_route_router_api_base", "https://router.example/v1")
    monkeypatch.setattr(s, "llm_route_router_api_key", "hf_secret")
    monkeypatch.setattr(s, "llm_route_force_completion_model", "gpt-4o-mini")

    rc = RouterConfig()
    apply_llm_route_env_overrides(rc)

    assert rc.tier_classifier == "qwen"
    assert rc.tier_classifier_llm_model == "Qwen/Test"
    assert rc.tier_router_api_base == "https://router.example/v1"
    assert rc.tier_router_api_key == "hf_secret"
    assert rc.tier_force_completion_model == "gpt-4o-mini"


def test_local_router_model_applied_for_qwen_local_only(monkeypatch, clear_overrides):
    import orchestrator.config as oc

    s = oc.settings
    monkeypatch.setattr(s, "llm_route_tier_classifier", "qwen_local")
    monkeypatch.setattr(s, "llm_route_local_router_model", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    monkeypatch.setattr(s, "llm_route_router_model", "Qwen/HF-Only-Id")

    rc = RouterConfig()
    apply_llm_route_env_overrides(rc)

    assert rc.tier_classifier == "qwen_local"
    assert rc.tier_classifier_llm_model == "mlx-community/Qwen2.5-3B-Instruct-4bit"


def test_apply_llm_route_local_router_env(monkeypatch, clear_overrides):
    import orchestrator.config as oc

    s = oc.settings
    monkeypatch.setattr(s, "llm_route_local_router_api_base", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(s, "llm_route_local_router_api_key", "local-dev-key")

    rc = RouterConfig()
    apply_llm_route_env_overrides(rc)

    assert rc.tier_local_router_api_base == "http://127.0.0.1:11434/v1"
    assert rc.tier_local_router_api_key == "local-dev-key"


def test_heuristic_shortcut_false_from_env(monkeypatch, clear_overrides):
    import orchestrator.config as oc

    s = oc.settings
    monkeypatch.setattr(s, "llm_route_tier_classifier_heuristic_shortcut", False)

    rc = RouterConfig()
    assert rc.tier_classifier_heuristic_shortcut is True
    apply_llm_route_env_overrides(rc)
    assert rc.tier_classifier_heuristic_shortcut is False


def test_router_model_env_not_applied_for_gpt_4o_mini(monkeypatch, clear_overrides):
    """LLM_ROUTE_ROUTER_MODEL must not set tier_classifier_llm_model when mode is OpenAI classifier."""
    import orchestrator.config as oc

    s = oc.settings
    monkeypatch.setattr(s, "llm_route_router_model", "Qwen/Qwen3-4B-Instruct-2507:fastest")

    rc = RouterConfig()
    assert rc.tier_classifier == "gpt_4o_mini"
    apply_llm_route_env_overrides(rc)

    assert rc.tier_classifier_llm_model is None


def test_invalid_classifier_env_ignored(monkeypatch, clear_overrides):
    import orchestrator.config as oc

    s = oc.settings
    monkeypatch.setattr(s, "llm_route_tier_classifier", "not_a_mode")

    rc = RouterConfig()
    before = rc.tier_classifier
    apply_llm_route_env_overrides(rc)
    assert rc.tier_classifier == before
