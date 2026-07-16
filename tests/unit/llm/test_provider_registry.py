"""
Provider registry tests — routing algorithm, built-in behavior preservation, and
the third-party extension API (register_provider).

The routing-algorithm cases replace the registered factories with sentinels so we
exercise get_provider's selection logic (prefix match, longest-wins, default,
gateway short-circuit) without constructing real SDK clients. A separate case
checks the built-in factories still bind to the correct provider classes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import continuum.llm.providers as registry  # the package module, for state save/restore
from continuum.llm.config import LLMConfig
from continuum.llm.providers import (
    BaseProvider,
    get_provider,
    register_default_provider,
    register_provider,
)


@pytest.fixture
def clean_registry():
    """Snapshot and restore the module-global registry around each test."""
    saved = dict(registry._PROVIDERS)
    saved_default = registry._default_factory
    yield
    registry._PROVIDERS.clear()
    registry._PROVIDERS.update(saved)
    registry._default_factory = saved_default


class _Sentinel(BaseProvider):
    """A do-nothing provider tagged with which registration produced it."""

    def __init__(self, tag: str):
        self.tag = tag

    def complete(self, *a, **k): ...  # noqa: D102
    async def acomplete(self, *a, **k): ...  # noqa: D102
    def stream(self, *a, **k): ...  # noqa: D102
    async def astream(self, *a, **k): ...  # noqa: D102


def _tag_factory(tag: str):
    return lambda config, settings: _Sentinel(tag)


# ---------------------------------------------------------------------------
# Routing algorithm — built-in prefix mapping is preserved exactly.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_registry")
class TestRoutingAlgorithm:
    @pytest.fixture(autouse=True)
    def _sentinels(self):
        # Replace every built-in registration with a tagged sentinel so we can
        # assert *which* provider get_provider selects without SDK construction.
        registry._PROVIDERS.clear()
        register_provider("gemini/", _tag_factory("gemini"))
        register_provider("google/", _tag_factory("gemini"))
        register_provider("claude/", _tag_factory("anthropic"))
        register_provider("anthropic/", _tag_factory("anthropic"))
        register_provider("claude-", _tag_factory("anthropic"))
        register_default_provider(_tag_factory("openai"))

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4o-mini", "openai"),
            ("openai/gpt-4o", "openai"),
            ("azure/my-deployment", "openai"),
            ("unknown-model-xyz", "openai"),  # default fallback
            ("claude-haiku-4-5-20251001", "anthropic"),
            ("claude/opus", "anthropic"),
            ("anthropic/claude-3", "anthropic"),
            ("gemini/gemini-2.5-flash", "gemini"),
            ("google/gemini-pro", "gemini"),
            ("GEMINI/UPPERCASE", "gemini"),  # case-insensitive
        ],
    )
    def test_model_routes_to_expected_provider(self, model, expected):
        provider = get_provider(LLMConfig(model=model))
        assert provider.tag == expected

    def test_longest_prefix_wins(self):
        # A more specific prefix must beat a broader one when both match.
        register_provider("claude-3-5/", _tag_factory("specific"))
        provider = get_provider(LLMConfig(model="claude-3-5/sonnet"))
        assert provider.tag == "specific"  # not the broad "claude-" anthropic match


# ---------------------------------------------------------------------------
# Gateway short-circuit — when configured, ALL models bypass the registry.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_registry")
class TestGatewayShortCircuit:
    def test_gateway_url_overrides_registry(self, monkeypatch):
        from continuum.config import settings
        from continuum.llm.providers.gateway_provider import GatewayProvider

        # Even with a sentinel registered for the model, the gateway wins.
        registry._PROVIDERS.clear()
        register_provider("gpt-", _tag_factory("openai"))
        register_default_provider(_tag_factory("openai"))

        monkeypatch.setattr(settings, "smart_gateway_url", "https://gw.example/v1", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_api_key", "k", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_default_mode", "balanced", raising=False)

        provider = get_provider(LLMConfig(model="gpt-4o-mini"))
        assert isinstance(provider, GatewayProvider)


# ---------------------------------------------------------------------------
# Built-in factories bind to the correct provider classes (dummy keys, offline).
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_registry")
class TestBuiltinFactoryBindings:
    def test_builtin_prefixes_registered(self):
        for prefix in ("gemini/", "google/", "claude/", "anthropic/", "claude-"):
            assert prefix in registry._PROVIDERS
        assert registry._default_factory is not None

    def test_factories_construct_expected_classes(self):
        from continuum.llm.providers.anthropic_provider import AnthropicProvider
        from continuum.llm.providers.gemini_provider import GeminiProvider
        from continuum.llm.providers.openai_provider import OpenAIProvider

        cfg = LLMConfig(model="x")
        s = SimpleNamespace(
            gemini_api_key="test",
            anthropic_api_key="test",
            openai_api_key="test",
            openai_organization=None,
        )
        assert isinstance(registry._make_gemini(cfg, s), GeminiProvider)
        assert isinstance(registry._make_anthropic(cfg, s), AnthropicProvider)
        assert isinstance(registry._make_openai(cfg, s), OpenAIProvider)


# ---------------------------------------------------------------------------
# Extension API — third parties add/override providers without editing core.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_registry")
class TestExtensionAPI:
    def test_register_new_provider(self):
        register_provider("myco/", _tag_factory("myco"))
        provider = get_provider(LLMConfig(model="myco/super-model"))
        assert provider.tag == "myco"

    def test_reregister_overrides_builtin(self):
        # A project can swap a built-in for its own implementation.
        register_provider("gemini/", _tag_factory("my-gemini"))
        provider = get_provider(LLMConfig(model="gemini/gemini-2.5-flash"))
        assert provider.tag == "my-gemini"


# ---------------------------------------------------------------------------
# temperature=None must be omitted so providers/models that reject it never
# receive the parameter (mirrors Anthropic's existing guard).
# ---------------------------------------------------------------------------


class TestTemperatureOmission:
    @pytest.mark.parametrize(
        "provider_path,model",
        [
            ("continuum.llm.providers.openai_provider.OpenAIProvider", "gpt-4o-mini"),
            ("continuum.llm.providers.gemini_provider.GeminiProvider", "gemini/gemini-2.5-flash"),
        ],
    )
    def test_temperature_omitted_when_none(self, provider_path, model):
        import importlib

        mod_name, cls_name = provider_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        provider = cls(api_key="test-key")

        omitted = provider._build_kwargs(LLMConfig(model=model, temperature=None), None, None)
        assert "temperature" not in omitted

        present = provider._build_kwargs(LLMConfig(model=model, temperature=0.4), None, None)
        assert present["temperature"] == 0.4
