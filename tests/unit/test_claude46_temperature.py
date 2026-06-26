"""
Unit tests for Claude 4.6+ temperature compatibility.

Claude 4.6 and newer (adaptive-thinking models such as Claude Opus 4.8) reject
an explicit ``temperature`` parameter. These tests verify:

  * ``LLMConfig`` accepts ``temperature=None`` and omits it from kwargs.
  * The Anthropic provider auto-detects Claude 4.6+ models and omits temperature.
  * Older Claude models still receive temperature.
  * The OpenAI/Gemini providers omit temperature when it is None.
"""

from __future__ import annotations

import pytest

from continuum.llm.config import LLMConfig
from continuum.llm.providers.anthropic_provider import (
    AnthropicProvider,
    _model_rejects_temperature,
    _parse_claude_version,
)

# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


class TestParseClaudeVersion:
    @pytest.mark.parametrize(
        "model,expected",
        [
            # New family-first naming
            ("claude-sonnet-4-6", (4, 6)),
            ("claude-opus-4-8", (4, 8)),
            ("claude-haiku-4-5-20251001", (4, 5)),
            ("claude-fable-5", (5, 0)),
            ("anthropic/claude-opus-4-8", (4, 8)),
            # Legacy version-first naming
            ("claude-3-5-sonnet-20241022", (3, 5)),
            ("claude-3-opus-20240229", (3, 0)),
            ("claude-2-1", (2, 1)),
            # Unparseable
            ("gpt-4o", None),
            ("some-random-model", None),
        ],
    )
    def test_parse(self, model: str, expected: tuple[int, int] | None) -> None:
        assert _parse_claude_version(model) == expected


class TestModelRejectsTemperature:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "claude-opus-4-8",
            "claude-fable-5",
            "anthropic/claude-opus-4-8",
        ],
    )
    def test_rejects_46_plus(self, model: str) -> None:
        assert _model_rejects_temperature(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5-20251001",  # 4.5 < 4.6
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "gpt-4o",  # non-Claude / unparseable -> keep temperature
        ],
    )
    def test_keeps_for_older_or_unknown(self, model: str) -> None:
        assert _model_rejects_temperature(model) is False


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfigTemperature:
    def test_accepts_none(self) -> None:
        config = LLMConfig(model="claude-opus-4-8", temperature=None)
        assert config.temperature is None

    def test_to_kwargs_omits_none(self) -> None:
        config = LLMConfig(model="claude-opus-4-8", temperature=None)
        assert "temperature" not in config.to_kwargs()

    def test_to_kwargs_includes_value(self) -> None:
        config = LLMConfig(model="gpt-4o", temperature=0.5)
        assert config.to_kwargs()["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Anthropic provider _build_kwargs
# ---------------------------------------------------------------------------


class TestAnthropicBuildKwargs:
    @staticmethod
    def _provider() -> AnthropicProvider:
        # A dummy key is enough; the SDK does not make a network call at init.
        return AnthropicProvider(api_key="test-key")

    def _build(self, model: str, temperature: float | None) -> dict:
        provider = self._provider()
        config = LLMConfig(model=model, temperature=temperature)
        messages = [{"role": "user", "content": "hi"}]
        return provider._build_kwargs(messages, config, tools=None, tool_choice=None)

    def test_omits_temperature_for_46_model(self) -> None:
        kwargs = self._build("claude-opus-4-8", 0.7)
        assert "temperature" not in kwargs

    def test_omits_temperature_for_sonnet_46(self) -> None:
        kwargs = self._build("claude-sonnet-4-6", 0.7)
        assert "temperature" not in kwargs

    def test_includes_temperature_for_older_model(self) -> None:
        kwargs = self._build("claude-3-5-sonnet-20241022", 0.3)
        assert kwargs["temperature"] == 0.3

    def test_omits_when_none(self) -> None:
        kwargs = self._build("claude-3-5-sonnet-20241022", None)
        assert "temperature" not in kwargs


# ---------------------------------------------------------------------------
# OpenAI / Gemini providers omit temperature when None
# ---------------------------------------------------------------------------


class TestOtherProvidersTemperatureNone:
    def test_openai_omits_none(self) -> None:
        from continuum.llm.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        config = LLMConfig(model="gpt-4o", temperature=None)
        kwargs = provider._build_kwargs(config, tools=None, tool_choice=None)
        assert "temperature" not in kwargs

    def test_openai_includes_value(self) -> None:
        from continuum.llm.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="test-key")
        config = LLMConfig(model="gpt-4o", temperature=0.2)
        kwargs = provider._build_kwargs(config, tools=None, tool_choice=None)
        assert kwargs["temperature"] == 0.2

    def test_gemini_omits_none(self) -> None:
        from continuum.llm.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="test-key")
        config = LLMConfig(model="gemini-2.5-flash", temperature=None)
        kwargs = provider._build_kwargs(config, tools=None, tool_choice=None)
        assert "temperature" not in kwargs
