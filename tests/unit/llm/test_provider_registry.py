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

    def test_shadowed_custom_registration_warns_once(self, monkeypatch, caplog):
        """A custom register_provider() going silently inert is a footgun —
        the gateway short-circuit must warn (once) that it is bypassed."""
        import logging

        from continuum.config import settings

        register_provider("myco/", _tag_factory("myco"))
        monkeypatch.setattr(settings, "smart_gateway_url", "https://gw.example/v1", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_api_key", "k", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_default_mode", "modest", raising=False)
        monkeypatch.setattr(registry, "_gateway_shadow_warned", False)
        # continuum's root logger sets propagate=False; re-enable so caplog sees it.
        monkeypatch.setattr(logging.getLogger("continuum"), "propagate", True)

        with caplog.at_level(logging.WARNING, logger="continuum.llm.providers"):
            get_provider(LLMConfig(model="myco/super-model"))
            get_provider(LLMConfig(model="myco/super-model"))

        shadow_warnings = [r for r in caplog.records if "bypassed" in r.getMessage()]
        assert len(shadow_warnings) == 1  # warned, and only once
        assert "myco/" in shadow_warnings[0].getMessage()

    def test_no_shadow_warning_without_custom_registrations(self, monkeypatch, caplog):
        import logging

        from continuum.config import settings

        # Only built-ins registered (clean_registry restores them).
        monkeypatch.setattr(settings, "smart_gateway_url", "https://gw.example/v1", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_api_key", "k", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_default_mode", "modest", raising=False)
        monkeypatch.setattr(registry, "_gateway_shadow_warned", False)

        with caplog.at_level(logging.WARNING, logger="continuum.llm.providers"):
            get_provider(LLMConfig(model="gpt-4o-mini"))

        assert not [r for r in caplog.records if "bypassed" in r.getMessage()]


# ---------------------------------------------------------------------------
# Gateway model fidelity + error attribution — a named model must reach the
# gateway verbatim, substitutions must be loud, and gateway errors must not be
# blamed on OpenAI.
# ---------------------------------------------------------------------------


class TestGatewayModelFidelity:
    def _gw(self, mode="modest"):
        from continuum.llm.providers.gateway_provider import GatewayProvider

        return GatewayProvider(gateway_url="https://gw.example/v1", api_key="k", router_mode=mode)

    def test_auto_tier_passes_through(self):
        assert self._gw()._normalize_model("auto/quality") == "auto/quality"

    def test_provider_qualified_model_passes_through_verbatim(self):
        """Regression: qualified names were ALSO silently replaced with
        auto/<tier> — there was no way to pin a model through the gateway."""
        gw = self._gw()
        assert gw._normalize_model("claude/claude-haiku-4-5") == "claude/claude-haiku-4-5"
        assert gw._normalize_model("gemini/gemini-2.5-flash") == "gemini/gemini-2.5-flash"

    def test_bare_model_is_tier_routed_with_warning(self, caplog, monkeypatch):
        import logging

        from continuum.llm.providers.gateway_provider import GatewayProvider

        GatewayProvider._warned_models.discard("gpt-4o-mini")
        gw = self._gw(mode="modest")
        # continuum's root logger sets propagate=False; re-enable so caplog sees it.
        monkeypatch.setattr(logging.getLogger("continuum"), "propagate", True)
        with caplog.at_level(logging.WARNING, logger="continuum.llm.providers.gateway_provider"):
            assert gw._normalize_model("gpt-4o-mini") == "auto/mid"
            assert gw._normalize_model("gpt-4o-mini") == "auto/mid"

        subs = [r for r in caplog.records if "replacing requested model" in r.getMessage()]
        assert len(subs) == 1  # loud, but once per model

    def test_gateway_errors_attributed_to_gateway_not_openai(self):
        """Regression: a gateway 401 raised provider=openai, sending users to
        debug the wrong system."""
        import httpx
        import openai as openai_sdk

        from continuum.llm.exceptions import LLMAuthenticationError

        gw = self._gw()
        err = openai_sdk.AuthenticationError(
            "Incorrect API key",
            response=httpx.Response(
                401, request=httpx.Request("POST", "https://gw.example/v1/chat/completions")
            ),
            body=None,
        )
        with pytest.raises(LLMAuthenticationError) as exc_info:
            gw._handle_exception(err, "claude/claude-haiku-4-5")
        assert exc_info.value.provider == "gateway"
        assert exc_info.value.context["gateway_url"] == "https://gw.example/v1"

    def test_plain_openai_errors_still_attributed_to_openai(self):
        import httpx
        import openai as openai_sdk

        from continuum.llm.exceptions import LLMAuthenticationError
        from continuum.llm.providers.openai_provider import OpenAIProvider

        p = OpenAIProvider(api_key="k")
        err = openai_sdk.AuthenticationError(
            "Incorrect API key",
            response=httpx.Response(
                401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            ),
            body=None,
        )
        with pytest.raises(LLMAuthenticationError) as exc_info:
            p._handle_exception(err, "gpt-4o-mini")
        assert exc_info.value.provider == "openai"
        assert "gateway_url" not in exc_info.value.context


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


class TestMaxRetriesWiring:
    """config.max_retries must reach the underlying SDK clients.

    Regression: providers built their SDK clients without max_retries, so
    llm_max_retries was a dead knob — the SDK used its own default (2) and a
    hung call retried uncontrollably (llm_request_timeout is per-attempt, not a
    total ceiling). Every provider must now propagate the configured budget so
    a caller can set it to 0 for a real single-attempt timeout.
    """

    _S = SimpleNamespace(
        gemini_api_key="test",
        anthropic_api_key="test",
        openai_api_key="test",
        openai_organization=None,
    )

    def test_openai_client_receives_max_retries(self):
        p = registry._make_openai(LLMConfig(model="gpt-4o-mini", max_retries=0), self._S)
        assert p._client.max_retries == 0
        assert p._async_client.max_retries == 0

    def test_anthropic_client_receives_max_retries(self):
        p = registry._make_anthropic(LLMConfig(model="claude-opus-4-8", max_retries=1), self._S)
        assert p._client.max_retries == 1
        assert p._async_client.max_retries == 1

    def test_gemini_client_receives_max_retries(self):
        p = registry._make_gemini(LLMConfig(model="gemini/gemini-2.5-flash", max_retries=4), self._S)
        assert p._client.max_retries == 4
        assert p._async_client.max_retries == 4

    def test_gateway_client_receives_max_retries(self, monkeypatch):
        from continuum.config import settings

        monkeypatch.setattr(settings, "smart_gateway_url", "https://gw.example/v1", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_api_key", "k", raising=False)
        monkeypatch.setattr(settings, "smart_gateway_default_mode", "modest", raising=False)
        p = get_provider(LLMConfig(model="gpt-4o-mini", max_retries=0))
        assert p._client.max_retries == 0
        assert p._async_client.max_retries == 0

    def test_zero_retries_gives_single_attempt(self):
        # The point of the fix: retries=0 -> one attempt -> timeout is the ceiling.
        p = registry._make_openai(LLMConfig(model="gpt-4o-mini", max_retries=0), self._S)
        assert p._client.max_retries == 0


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
