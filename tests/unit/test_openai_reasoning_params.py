"""
Unit tests for the OpenAI provider's GPT-5 / o-series parameter handling.

The GPT-5 family and the o-series reasoning models reject several Chat
Completions parameters that every earlier OpenAI model accepts:

* ``max_tokens`` — must be sent as ``max_completion_tokens`` instead
* ``temperature`` — only the default (1) is accepted
* ``top_p`` / ``frequency_penalty`` / ``presence_penalty`` / ``stop`` — rejected

Two mechanisms cover this, mirroring ``AnthropicProvider``'s temperature
drop-back:

1. **Static** — a model-name prefix match adapts the kwargs up front, so a known
   reasoning model never pays a failed request.
2. **Learned** — if the API rejects a parameter anyway (a model id we do not
   recognise yet), the provider drops/renames it, retries once, and caches the
   model so later calls adapt up front.

These are mock-first tests: the openai SDK client is replaced with a mock, so no
network or API key is needed. The parameter lists below were verified against the
live API — see ``_REASONING_UNSUPPORTED_PARAMS`` in the provider.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from continuum.llm.config import LLMConfig
from continuum.llm.providers.openai_provider import OpenAIProvider

MESSAGES = [{"role": "user", "content": "Reply with exactly one word: hello"}]


@contextmanager
def _captured_provider_logs():
    """Collect records from the provider's logger.

    caplog cannot see these: the "continuum" parent sets propagate=False, so
    records never reach the root logger. Same pattern as
    tests/unit/test_local_shop_resources.py.
    """
    records: list[tuple[int, str]] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append((record.levelno, record.getMessage()))

    handler = _Collector()
    logger = logging.getLogger("continuum.llm.providers.openai_provider")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _clear_param_cache() -> None:
    """The unsupported-param cache is class-level (shared across instances for the
    process), so reset it before each test to keep them isolated."""
    OpenAIProvider._unsupported_params.clear()


def _bad_request(param: str, code: str, message: str) -> openai.BadRequestError:
    """Build a 400 shaped like the real OpenAI error body (verified live)."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    body = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    return openai.BadRequestError(message, response=resp, body=body)


def _max_tokens_400() -> openai.BadRequestError:
    return _bad_request(
        "max_tokens",
        "unsupported_parameter",
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead.",
    )


def _temperature_400() -> openai.BadRequestError:
    return _bad_request(
        "temperature",
        "unsupported_value",
        "Unsupported value: 'temperature' does not support 0.7 with this model. "
        "Only the default (1) value is supported.",
    )


def _context_400() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    msg = "This model's maximum context length is 128000 tokens."
    body = {"message": msg, "type": "invalid_request_error", "code": "context_length_exceeded"}
    return openai.BadRequestError(msg, response=resp, body=body)


def _fake_response(text: str = "hello") -> SimpleNamespace:
    """A response shaped like the real OpenAI SDK completion object."""
    return SimpleNamespace(
        id="chatcmpl_test_123",
        model="gpt-5-mini",
        created=0,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant", content=text, tool_calls=None, function_call=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def _provider_sync() -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key")
    provider._client = MagicMock()
    return provider


class TestStaticReasoningDetection:
    """Known reasoning models adapt up front — no failed request, no retry."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-2025-08-07",
            "gpt-5.1",
            "gpt-5.2-chat-latest",
            "gpt-5.4-mini",
            "gpt-5.6-luna",  # codename variants still belong to the family
            "o1",
            "o3-mini",
            "o4-mini",
        ],
    )
    def test_renames_max_tokens(self, model: str) -> None:
        provider = OpenAIProvider(api_key="test-key")
        cfg = LLMConfig(model=model, max_tokens=256, temperature=None)

        kwargs = provider._build_kwargs(cfg, tools=None, tool_choice=None)

        assert kwargs["max_completion_tokens"] == 256
        assert "max_tokens" not in kwargs

    @pytest.mark.parametrize(
        "param,value",
        [
            ("temperature", 0.7),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.5),
            ("stop", ["END"]),
        ],
    )
    def test_omits_unsupported_params(self, param: str, value: object) -> None:
        provider = OpenAIProvider(api_key="test-key")
        cfg = LLMConfig(model="gpt-5-mini", **{param: value})  # type: ignore[arg-type]

        kwargs = provider._build_kwargs(cfg, tools=None, tool_choice=None)

        assert param not in kwargs

    def test_keeps_supported_params(self) -> None:
        """seed and user ARE accepted by reasoning models — do not strip them."""
        provider = OpenAIProvider(api_key="test-key")
        cfg = LLMConfig(model="gpt-5-mini", seed=42, user="u1", temperature=None)

        kwargs = provider._build_kwargs(cfg, tools=None, tool_choice=None)

        assert kwargs["seed"] == 42
        assert kwargs["user"] == "u1"

    def test_default_temperature_is_dropped(self) -> None:
        """Regression: settings default temperature (0.7) must not reach GPT-5.

        This is the second 400 a caller hits after the max_tokens fix alone.
        """
        provider = OpenAIProvider(api_key="test-key")
        cfg = LLMConfig(model="gpt-5-mini")  # temperature defaults to 0.7

        kwargs = provider._build_kwargs(cfg, tools=None, tool_choice=None)

        assert "temperature" not in kwargs


class TestNonReasoningModelsUnchanged:
    """Regression guard: the fix must not alter existing model behaviour."""

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
    def test_legacy_models_keep_max_tokens_and_temperature(self, model: str) -> None:
        provider = OpenAIProvider(api_key="test-key")
        cfg = LLMConfig(model=model, max_tokens=256, temperature=0.7, top_p=0.9)

        kwargs = provider._build_kwargs(cfg, tools=None, tool_choice=None)

        assert kwargs["max_tokens"] == 256
        assert "max_completion_tokens" not in kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9

    def test_openai_prefix_is_stripped_before_matching(self) -> None:
        """`openai/gpt-5-mini` must be detected as a reasoning model too."""
        provider = OpenAIProvider(api_key="test-key")
        cfg = LLMConfig(model="openai/gpt-5-mini", max_tokens=64)

        kwargs = provider._build_kwargs(cfg, tools=None, tool_choice=None)

        assert kwargs["max_completion_tokens"] == 64
        assert "max_tokens" not in kwargs


class TestLearnedDropBack:
    """An unrecognised model that rejects a param is retried once, then cached."""

    def test_retries_once_renaming_max_tokens(self) -> None:
        provider = _provider_sync()
        provider._client.chat.completions.create.side_effect = [
            _max_tokens_400(),
            _fake_response(),
        ]
        # A model the static list does NOT match.
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)

        resp = provider.complete(MESSAGES, cfg)

        assert provider._client.chat.completions.create.call_count == 2
        first = provider._client.chat.completions.create.call_args_list[0].kwargs
        retry = provider._client.chat.completions.create.call_args_list[1].kwargs
        assert first["max_tokens"] == 64  # first attempt carried it
        assert "max_tokens" not in retry  # retry renamed it
        assert retry["max_completion_tokens"] == 64  # limit preserved, not dropped
        assert "max_tokens" in provider._unsupported_params["futuremodel-x"]
        assert resp.content == "hello"

    def test_retries_once_dropping_temperature(self) -> None:
        provider = _provider_sync()
        provider._client.chat.completions.create.side_effect = [
            _temperature_400(),
            _fake_response(),
        ]
        cfg = LLMConfig(model="futuremodel-x", temperature=0.7)

        provider.complete(MESSAGES, cfg)

        retry = provider._client.chat.completions.create.call_args_list[1].kwargs
        assert "temperature" not in retry
        assert "temperature" in provider._unsupported_params["futuremodel-x"]

    def test_adapts_repeatedly_when_several_params_are_rejected(self) -> None:
        """Each 400 names only ONE parameter, so an unrecognised reasoning model
        rejects max_tokens, then temperature, then top_p on successive attempts.

        Regression: a single retry handled only the first and the call still
        failed — caught by a live run against gpt-5-mini, not by the mocks.
        """
        provider = _provider_sync()
        provider._client.chat.completions.create.side_effect = [
            _max_tokens_400(),
            _temperature_400(),
            _bad_request(
                "top_p",
                "unsupported_parameter",
                "Unsupported parameter: 'top_p' is not supported with this model.",
            ),
            _fake_response(),
        ]
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=0.7, top_p=0.9)

        resp = provider.complete(MESSAGES, cfg)

        assert provider._client.chat.completions.create.call_count == 4
        final = provider._client.chat.completions.create.call_args_list[-1].kwargs
        assert final["max_completion_tokens"] == 64
        assert "max_tokens" not in final
        assert "temperature" not in final
        assert "top_p" not in final
        assert provider._unsupported_params["futuremodel-x"] == {
            "max_tokens",
            "temperature",
            "top_p",
        }
        assert resp.content == "hello"

    def test_adaptation_is_bounded(self) -> None:
        """A model that rejects every attempt must not spin forever."""
        provider = _provider_sync()
        provider._client.chat.completions.create.side_effect = _max_tokens_400()
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)

        with pytest.raises(Exception):
            provider.complete(MESSAGES, cfg)

        # First attempt renames max_tokens; the second 400 names a param that is
        # no longer present, so adaptation stops immediately rather than looping.
        assert provider._client.chat.completions.create.call_count == 2

    def test_cached_model_adapts_up_front(self) -> None:
        provider = _provider_sync()
        provider._unsupported_params["futuremodel-x"] = {"max_tokens"}
        provider._client.chat.completions.create.return_value = _fake_response()
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)

        provider.complete(MESSAGES, cfg)

        # Single call, no wasted retry.
        provider._client.chat.completions.create.assert_called_once()
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in kwargs
        assert kwargs["max_completion_tokens"] == 64

    def test_cache_persists_across_provider_instances(self) -> None:
        """Regression: get_provider() builds a NEW provider per call, so the cache
        must be shared across instances or every call re-pays the retry."""
        p1 = _provider_sync()
        p1._client.chat.completions.create.side_effect = [_max_tokens_400(), _fake_response()]
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)
        p1.complete(MESSAGES, cfg)
        assert p1._client.chat.completions.create.call_count == 2

        p2 = _provider_sync()
        p2._client.chat.completions.create.return_value = _fake_response()
        p2.complete(MESSAGES, cfg)
        p2._client.chat.completions.create.assert_called_once()  # learned globally

    def test_non_param_400_does_not_retry(self) -> None:
        from continuum.llm.exceptions import LLMContextLengthError

        provider = _provider_sync()
        provider._client.chat.completions.create.side_effect = _context_400()
        cfg = LLMConfig(model="gpt-4o-mini", max_tokens=64)

        with pytest.raises(LLMContextLengthError):
            provider.complete(MESSAGES, cfg)

        provider._client.chat.completions.create.assert_called_once()
        assert provider._unsupported_params == {}


class TestErrorClassification:
    """The max_tokens 400 must not be misreported as a context-length error.

    ``_handle_exception`` classifies any BadRequestError whose message contains
    "token" as LLMContextLengthError — and "max_tokens" contains "token". A
    reasoning-param rejection that survives the retry must still surface as an
    invalid-request error so the real cause is visible.
    """

    def test_unsupported_param_is_not_context_length_error(self) -> None:
        from continuum.llm.exceptions import LLMContextLengthError, LLMInvalidRequestError

        provider = _provider_sync()
        # Rejected twice: the retry does not help, so the error must propagate.
        provider._client.chat.completions.create.side_effect = [
            _max_tokens_400(),
            _max_tokens_400(),
        ]
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)

        with pytest.raises(LLMInvalidRequestError) as exc:
            provider.complete(MESSAGES, cfg)

        assert not isinstance(exc.value, LLMContextLengthError)


class TestAsyncAndStreaming:
    async def test_acomplete_retries_once(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=[_max_tokens_400(), _fake_response("hi")]
        )
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)

        resp = await provider.acomplete(MESSAGES, cfg)

        assert provider._async_client.chat.completions.create.call_count == 2
        retry = provider._async_client.chat.completions.create.call_args_list[1].kwargs
        assert retry["max_completion_tokens"] == 64
        assert resp.content == "hi"

    def test_stream_retries_without_double_emit(self) -> None:
        provider = _provider_sync()

        def _chunks():
            for text in ("hel", "lo"):
                yield SimpleNamespace(
                    id="c1",
                    model="gpt-5-mini",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=text, role="assistant", tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                )

        provider._client.chat.completions.create.side_effect = [_max_tokens_400(), _chunks()]
        cfg = LLMConfig(model="futuremodel-x", max_tokens=64, temperature=None)

        out = list(provider.stream(MESSAGES, cfg))

        assert provider._client.chat.completions.create.call_count == 2
        # Only the retry's content is emitted — no chunk from the failed attempt.
        assert "".join(c.content for c in out if c.content) == "hello"


class TestTruncationWarning:
    """`max_completion_tokens` is a COMBINED budget: reasoning tokens + visible
    output. A limit that was fine as `max_tokens` on a legacy model can be spent
    entirely on invisible reasoning, returning finish_reason='length' and empty
    content. Verified live: gpt-5-mini at 32 returns '' (32/32 reasoning); at 128
    it returns 'hello' (64 reasoning + output).

    Renaming the parameter must not silently convert a loud 400 into an empty
    string — the caller gets a warning naming the cause.
    """

    @staticmethod
    def _response(content: str | None, finish_reason: str) -> SimpleNamespace:
        r = _fake_response(content or "")
        r.choices[0].message.content = content
        r.choices[0].finish_reason = finish_reason
        return r

    @staticmethod
    def _warnings_for(content: str | None, finish_reason: str, max_tokens: int) -> list[str]:
        """Run a completion and return the provider's warning messages."""
        provider = _provider_sync()
        provider._client.chat.completions.create.return_value = TestTruncationWarning._response(
            content, finish_reason
        )
        cfg = LLMConfig(model="gpt-5-mini", max_tokens=max_tokens)
        with _captured_provider_logs() as records:
            provider.complete(MESSAGES, cfg)
        return [msg for level, msg in records if level >= logging.WARNING]

    def test_warns_when_reasoning_budget_consumed(self) -> None:
        msgs = self._warnings_for("", "length", 32)
        assert any("max_completion_tokens" in m for m in msgs)

    def test_no_warning_on_normal_completion(self) -> None:
        msgs = self._warnings_for("hello", "stop", 512)
        assert not any("max_completion_tokens" in m for m in msgs)

    def test_no_warning_when_truncated_but_content_present(self) -> None:
        """Ordinary truncation (content produced, then cut off) is not this bug."""
        msgs = self._warnings_for("partial", "length", 32)
        assert not any("max_completion_tokens" in m for m in msgs)


class TestGatewayInheritance:
    """GatewayProvider extends OpenAIProvider and calls super()._build_kwargs,
    so it must inherit the reasoning-model handling for free.

    Note on responsibility: when SMART_GATEWAY_URL is set every model routes
    through the gateway, and the gateway already rewrites max_tokens →
    max_completion_tokens for GPT-5/o-series itself. Continuum's own handling is
    therefore belt-and-braces here — it matters for a *pinned* id, because the
    gateway's rewrite is gated on the model having a reasoning-alias policy in
    its registry and is skipped otherwise.
    """

    @staticmethod
    def _gateway():
        from continuum.llm.providers.gateway_provider import GatewayProvider

        return GatewayProvider(
            gateway_url="https://gw.example/v1", api_key="test-key", router_mode=None
        )

    def test_gateway_renames_max_tokens_for_pinned_model(self) -> None:
        """A gateway-pinned model keeps its `openai/` qualifier, so detection
        must look past the qualifier rather than matching the raw string."""
        cfg = LLMConfig(model="openai/gpt-5-mini", max_tokens=128)

        kwargs = self._gateway()._build_kwargs(cfg, tools=None, tool_choice=None)

        assert kwargs["max_completion_tokens"] == 128
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_gateway_tier_keeps_max_tokens_for_the_gateway_to_rewrite(self) -> None:
        """A bare name is rewritten to a tier (`auto/mid`), whose upstream model is
        unknowable here — so static detection correctly declines to fire and
        `max_tokens` is left intact.

        That is not a gap: the gateway performs the rename itself once it has
        resolved the tier to a physical model. Continuum must NOT pre-empt it,
        because guessing wrong for a tier that resolves to a non-reasoning model
        would send `max_completion_tokens` where `max_tokens` was wanted.
        """
        cfg = LLMConfig(model="gpt-5-mini", max_tokens=128)

        kwargs = self._gateway()._build_kwargs(cfg, tools=None, tool_choice=None)

        assert kwargs["model"] == "auto/mid"
        assert kwargs["max_tokens"] == 128
