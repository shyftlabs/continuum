"""
Unit tests for the Anthropic provider's error-driven temperature drop-back.

Claude 4.6+ adaptive-thinking models (e.g. Claude Opus 4.8) reject an explicit
``temperature`` parameter with a 400. Rather than guessing from the model name,
the provider sends temperature normally and, if the API rejects it, strips it and
retries once — then caches the model so later calls omit temperature up front.

These are mock-first tests: the anthropic SDK client is replaced with a mock, so
no network or API key is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest

from continuum.llm.config import LLMConfig
from continuum.llm.exceptions import LLMContextLengthError
from continuum.llm.providers.anthropic_provider import AnthropicProvider

MESSAGES = [{"role": "user", "content": "Reply with exactly one word: hello"}]


@pytest.fixture(autouse=True)
def _clear_temp_cache() -> None:
    """The temp-unsupported cache is class-level (shared across instances for the
    process), so reset it before each test to keep them isolated."""
    AnthropicProvider._temp_unsupported.clear()


def _bad_request(message: str) -> anthropic.BadRequestError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req)
    return anthropic.BadRequestError(message, response=resp, body=None)


def _temperature_400() -> anthropic.BadRequestError:
    return _bad_request("temperature: unexpected parameter; temperature is not supported")


def _context_400() -> anthropic.BadRequestError:
    return _bad_request("prompt is too long: 300000 tokens > 200000 maximum")


def _fake_response(text: str = "hello") -> SimpleNamespace:
    """A response shaped like the real Anthropic SDK message object."""
    return SimpleNamespace(
        id="msg_test_123",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        stop_reason="end_turn",
    )


def _provider_sync() -> AnthropicProvider:
    provider = AnthropicProvider(api_key="test-key")
    provider._client = MagicMock()
    return provider


class TestCompleteDropBack:
    def test_retries_once_without_temperature(self) -> None:
        provider = _provider_sync()
        # First call rejects temperature, second (retry) succeeds.
        provider._client.messages.create.side_effect = [_temperature_400(), _fake_response()]
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)

        resp = provider.complete(MESSAGES, cfg)

        assert provider._client.messages.create.call_count == 2
        first_kwargs = provider._client.messages.create.call_args_list[0].kwargs
        retry_kwargs = provider._client.messages.create.call_args_list[1].kwargs
        assert first_kwargs["temperature"] == 0.7  # first attempt carried it
        assert "temperature" not in retry_kwargs  # retry dropped it
        assert "claude-opus-4-8" in provider._temp_unsupported  # model learned
        assert resp.content == "hello"

    def test_cached_model_omits_temperature_up_front(self) -> None:
        provider = _provider_sync()
        provider._temp_unsupported.add("claude-opus-4-8")  # already learned
        provider._client.messages.create.return_value = _fake_response()
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)

        resp = provider.complete(MESSAGES, cfg)

        # Single call, no wasted retry, temperature never sent.
        provider._client.messages.create.assert_called_once()
        assert "temperature" not in provider._client.messages.create.call_args.kwargs
        assert resp.content == "hello"

    def test_cache_persists_across_provider_instances(self) -> None:
        """Regression: get_provider() builds a NEW provider per call, so the cache
        must be shared across instances or every call re-pays the retry."""
        # First instance learns the model rejects temperature.
        p1 = _provider_sync()
        p1._client.messages.create.side_effect = [_temperature_400(), _fake_response()]
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)
        p1.complete(MESSAGES, cfg)
        assert p1._client.messages.create.call_count == 2  # 400 + retry

        # A brand-new instance (as get_provider would create) must skip the retry.
        p2 = _provider_sync()
        p2._client.messages.create.return_value = _fake_response()
        p2.complete(MESSAGES, cfg)
        p2._client.messages.create.assert_called_once()  # no retry — learned globally
        assert "temperature" not in p2._client.messages.create.call_args.kwargs

    def test_non_temperature_400_does_not_retry(self) -> None:
        provider = _provider_sync()
        provider._client.messages.create.side_effect = _context_400()
        cfg = LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=16)

        with pytest.raises(LLMContextLengthError):
            provider.complete(MESSAGES, cfg)

        # Exactly one attempt; the model is NOT marked temp-unsupported.
        provider._client.messages.create.assert_called_once()
        assert provider._temp_unsupported == set()

    def test_successful_call_keeps_temperature_for_normal_model(self) -> None:
        provider = _provider_sync()
        provider._client.messages.create.return_value = _fake_response()
        cfg = LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=16)

        provider.complete(MESSAGES, cfg)

        provider._client.messages.create.assert_called_once()
        assert provider._client.messages.create.call_args.kwargs["temperature"] == 0.3


class TestAcompleteDropBack:
    async def test_async_retries_once_without_temperature(self) -> None:
        provider = AnthropicProvider(api_key="test-key")
        provider._async_client = MagicMock()
        provider._async_client.messages.create = AsyncMock(
            side_effect=[_temperature_400(), _fake_response("hi")]
        )
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)

        resp = await provider.acomplete(MESSAGES, cfg)

        assert provider._async_client.messages.create.call_count == 2
        assert "temperature" not in provider._async_client.messages.create.call_args_list[1].kwargs
        assert resp.content == "hi"


class _FakeStreamCM:
    """Minimal stand-in for anthropic's MessageStreamManager context manager."""

    def __init__(self, texts: list[str], final: SimpleNamespace):
        self._texts = texts
        self._final = final

    def __enter__(self):
        return SimpleNamespace(
            text_stream=iter(self._texts),
            get_final_message=lambda: self._final,
        )

    def __exit__(self, *exc):
        return False


class TestStreamDropBack:
    def test_stream_retries_without_double_emit(self) -> None:
        provider = _provider_sync()
        # First stream open rejects temperature; retry opens a working stream.
        provider._client.messages.stream.side_effect = [
            _temperature_400(),
            _FakeStreamCM(["hel", "lo"], _fake_response("hello")),
        ]
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7, max_tokens=16)

        chunks = list(provider.stream(MESSAGES, cfg))

        assert provider._client.messages.stream.call_count == 2
        assert "temperature" not in provider._client.messages.stream.call_args_list[1].kwargs
        # Only the retry's content is emitted — no chunk from the failed attempt.
        text = "".join(c.content for c in chunks if c.content)
        assert text == "hello"
        assert "claude-opus-4-8" in provider._temp_unsupported


class TestBuildKwargsCache:
    def test_build_kwargs_omits_temperature_for_cached_model(self) -> None:
        provider = AnthropicProvider(api_key="test-key")
        provider._temp_unsupported.add("claude-opus-4-8")
        cfg = LLMConfig(model="claude-opus-4-8", temperature=0.7)

        kwargs = provider._build_kwargs(MESSAGES, cfg, tools=None, tool_choice=None)

        assert "temperature" not in kwargs

    def test_build_kwargs_keeps_temperature_for_unknown_model(self) -> None:
        provider = AnthropicProvider(api_key="test-key")
        cfg = LLMConfig(model="claude-3-5-sonnet-20241022", temperature=0.5)

        kwargs = provider._build_kwargs(MESSAGES, cfg, tools=None, tool_choice=None)

        assert kwargs["temperature"] == 0.5
