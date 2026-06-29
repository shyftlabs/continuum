"""
Mocked integration tests for AnthropicProvider temperature handling.

These exercise the FULL provider flow end to end:

    LLMConfig -> _build_kwargs() -> client.messages.create(**payload)
              -> LLMResponse.from_anthropic_response()

...without any real network call. The Anthropic SDK client (created in
AnthropicProvider.__init__) is replaced with a mock that:
  - captures the exact request payload passed to messages.create(), and
  - returns a realistically-shaped response object, so the response-parsing
    path runs too and generate/complete returns a real LLMResponse.

No real API key is needed — a dummy "test-key" is enough because the mock
intercepts the call before the SDK would ever reach the network.

Run:
  pytest tests/integration/ -v -k temperature
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.llm.config import LLMConfig
from continuum.llm.providers.anthropic_provider import AnthropicProvider

MESSAGES = [{"role": "user", "content": "Reply with exactly one word: hello"}]


def _fake_response(text: str = "hello") -> SimpleNamespace:
    """A response shaped like the real Anthropic SDK message object.

    Matches what LLMResponse.from_anthropic_response() reads: .content (a list
    of typed blocks), .usage (.input_tokens/.output_tokens), .stop_reason, .id.
    """
    return SimpleNamespace(
        id="msg_test_123",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        stop_reason="end_turn",
    )


def _provider_with_mock_sync() -> AnthropicProvider:
    """Provider whose sync SDK client is mocked to capture the request payload."""
    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.create.return_value = _fake_response()
    return provider


def _captured_payload(provider: AnthropicProvider) -> dict:
    """The kwargs that were passed to messages.create()."""
    return provider._client.messages.create.call_args.kwargs


class TestAnthropicTemperatureFlowMocked:
    # (1) 4.6+ model with a temperature set -> temperature MUST be omitted.
    def test_temperature_omitted_for_opus_4_8(self) -> None:
        provider = _provider_with_mock_sync()
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)

        resp = provider.complete(MESSAGES, cfg)

        payload = _captured_payload(provider)
        assert "temperature" not in payload, "4.6+ model must not receive temperature"
        # full payload was still built correctly
        assert payload["model"] == "claude-opus-4-8"
        assert payload["messages"][0]["content"] == MESSAGES[0]["content"]
        # (5) response-parsing path ran and produced a real LLMResponse
        provider._client.messages.create.assert_called_once()
        assert resp.content == "hello"
        assert resp.usage.total_tokens == 7

    # (2) another 4.6+ model -> temperature omitted.
    def test_temperature_omitted_for_sonnet_4_6(self) -> None:
        provider = _provider_with_mock_sync()
        cfg = LLMConfig(model="claude-sonnet-4-6", temperature=0.7, max_tokens=16)

        resp = provider.complete(MESSAGES, cfg)

        assert "temperature" not in _captured_payload(provider)
        assert resp.content == "hello"

    # (3) older model with a temperature -> temperature MUST be forwarded as-is.
    def test_temperature_kept_for_legacy_sonnet(self) -> None:
        provider = _provider_with_mock_sync()
        cfg = LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=16)

        resp = provider.complete(MESSAGES, cfg)

        payload = _captured_payload(provider)
        assert payload["temperature"] == 0.3, "older model must keep temperature"
        assert resp.content == "hello"

    # (4) temperature=None on ANY model -> omitted.
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-8", "claude-sonnet-4-6", "claude-3-5-sonnet-20241022"],
    )
    def test_temperature_none_is_omitted(self, model: str) -> None:
        provider = _provider_with_mock_sync()
        cfg = LLMConfig(model=model, temperature=None, max_tokens=16)

        provider.complete(MESSAGES, cfg)

        assert "temperature" not in _captured_payload(provider)

    # (5 + async) the async path (acomplete) behaves identically and parses the response.
    async def test_async_flow_omits_temperature_for_4_6(self) -> None:
        provider = AnthropicProvider(api_key="test-key")
        provider._async_client = MagicMock()
        provider._async_client.messages.create = AsyncMock(return_value=_fake_response("hi"))
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)

        resp = await provider.acomplete(MESSAGES, cfg)

        payload = provider._async_client.messages.create.call_args.kwargs
        assert "temperature" not in payload
        assert resp.content == "hi"
