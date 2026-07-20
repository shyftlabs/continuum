"""
Unit tests for TL-65 — remove the hardcoded OpenAI default.

The framework historically defaulted every meta-operation (router LLM routing,
reflection critique, tier classifier, memory fact-extraction, summarization) to
``gpt-4o-mini``, so an Anthropic-only or Gemini-only deployment still needed an
OpenAI key. These tests verify the default is now provider-aware and that memory
and summarization inherit it, while explicit settings are preserved.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from continuum.config import Settings, _resolve_default_model

# Env vars that feed the provider-aware resolver. A developer shell (or CI)
# commonly exports DEFAULT_LLM_MODEL / MEMORY_LLM_MODEL etc.; pydantic-settings
# treats env-sourced values as explicitly set (model_fields_set), which is the
# resolver's intended "explicit wins" behavior — so these tests must scrub them
# to observe the auto-detection path deterministically.
_RESOLVER_ENV_VARS = (
    "DEFAULT_LLM_MODEL",
    "MEMORY_LLM_MODEL",
    "CONTEXT_SUMMARIZATION_MODEL",
    "ANTHROPIC_DEFAULT_MODEL",
    "GEMINI_DEFAULT_MODEL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)


def _settings(**overrides):
    """Build Settings deterministically: scrub resolver env vars, keys default to None."""
    base = dict(openai_api_key=None, anthropic_api_key=None, gemini_api_key=None)
    base.update(overrides)
    clean_env = {k: v for k, v in os.environ.items() if k not in _RESOLVER_ENV_VARS}
    with patch.dict(os.environ, clean_env, clear=True):
        return Settings(_env_file=None, **base)


# ---------------------------------------------------------------------------
# Pure resolver
# ---------------------------------------------------------------------------


class TestResolveDefaultModel:
    def test_openai_key_keeps_gpt4o_mini(self) -> None:
        assert (
            _resolve_default_model(
                None,
                has_openai=True,
                has_anthropic=False,
                has_gemini=False,
                anthropic_model="A",
                gemini_model="G",
            )
            == "gpt-4o-mini"
        )

    def test_anthropic_only_uses_anthropic_default(self) -> None:
        assert (
            _resolve_default_model(
                None,
                has_openai=False,
                has_anthropic=True,
                has_gemini=False,
                anthropic_model="claude-x",
                gemini_model="G",
            )
            == "claude-x"
        )

    def test_gemini_only_uses_gemini_default(self) -> None:
        assert (
            _resolve_default_model(
                None,
                has_openai=False,
                has_anthropic=False,
                has_gemini=True,
                anthropic_model="A",
                gemini_model="gemini-x",
            )
            == "gemini-x"
        )

    def test_nothing_configured_falls_back_to_gpt4o_mini(self) -> None:
        assert (
            _resolve_default_model(
                None,
                has_openai=False,
                has_anthropic=False,
                has_gemini=False,
                anthropic_model="A",
                gemini_model="G",
            )
            == "gpt-4o-mini"
        )

    def test_explicit_always_wins(self) -> None:
        assert (
            _resolve_default_model(
                "my-model",
                has_openai=True,
                has_anthropic=True,
                has_gemini=True,
                anthropic_model="A",
                gemini_model="G",
            )
            == "my-model"
        )

    def test_openai_preferred_when_multiple_keys(self) -> None:
        # Back-compat: an OpenAI key present keeps today's behavior.
        assert (
            _resolve_default_model(
                None,
                has_openai=True,
                has_anthropic=True,
                has_gemini=True,
                anthropic_model="A",
                gemini_model="G",
            )
            == "gpt-4o-mini"
        )


# ---------------------------------------------------------------------------
# Settings — provider-aware defaults + inheritance
# ---------------------------------------------------------------------------


class TestSettingsProviderAwareDefaults:
    def test_anthropic_only(self) -> None:
        s = _settings(anthropic_api_key="x")
        assert s.default_llm_model == "claude-haiku-4-5"

    def test_gemini_only(self) -> None:
        s = _settings(gemini_api_key="x")
        assert s.default_llm_model == "gemini/gemini-2.5-flash"

    def test_openai_unchanged(self) -> None:
        s = _settings(openai_api_key="x")
        assert s.default_llm_model == "gpt-4o-mini"

    def test_none_configured_falls_back(self) -> None:
        s = _settings()
        assert s.default_llm_model == "gpt-4o-mini"

    def test_explicit_model_preserved(self) -> None:
        s = _settings(anthropic_api_key="x", default_llm_model="claude-sonnet-4-6")
        assert s.default_llm_model == "claude-sonnet-4-6"

    def test_memory_and_summarization_inherit(self) -> None:
        s = _settings(anthropic_api_key="x")
        assert s.memory_llm_model == "claude-haiku-4-5"
        assert s.context_summarization_model == "claude-haiku-4-5"

    def test_explicit_memory_model_preserved(self) -> None:
        s = _settings(anthropic_api_key="x", memory_llm_model="claude-custom-mem")
        assert s.memory_llm_model == "claude-custom-mem"
        # summarization still inherits the resolved default
        assert s.context_summarization_model == "claude-haiku-4-5"

    def test_per_provider_default_is_overridable(self) -> None:
        s = _settings(anthropic_api_key="x", anthropic_default_model="claude-opus-4-8")
        assert s.default_llm_model == "claude-opus-4-8"
