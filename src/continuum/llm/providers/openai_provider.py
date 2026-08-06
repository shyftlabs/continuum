"""
OpenAI provider — direct openai SDK.

Handles: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, gpt-4o-turbo, the GPT-5 family,
the o-series reasoning models, and Azure OpenAI.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI

from continuum.llm.config import LLMConfig
from continuum.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from continuum.llm.providers.base import BaseProvider
from continuum.llm.types import FunctionCall, LLMResponse, StreamChunk, ToolCall
from continuum.logging import get_logger

logger = get_logger(__name__)

_PROVIDER = "openai"

# Model-name prefixes for the GPT-5 family and the o-series reasoning models.
# `gpt-5` covers every point release and suffix (gpt-5-mini, gpt-5.4-nano,
# gpt-5.2-chat-latest, the gpt-5.6-<codename> line, dated snapshots, ...) —
# verified against the live model list, where all of them reject `max_tokens`.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Chat Completions parameters these models reject outright (400). Verified live
# against gpt-5-mini: each returns `unsupported_parameter` — except temperature,
# which returns `unsupported_value` ("only the default (1) is supported").
# `seed` and `user` ARE accepted and must not be stripped.
_REASONING_UNSUPPORTED_PARAMS = frozenset(
    {"temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"}
)

# Renamed rather than dropped: the caller's output limit is preserved, since
# dropping it would silently uncap the response.
_RENAMED_PARAMS = {"max_tokens": "max_completion_tokens"}

# Error codes that mean "this model does not accept that parameter". The body's
# `param` field names the offending kwarg, so no message parsing is needed.
_UNSUPPORTED_PARAM_CODES = frozenset({"unsupported_parameter", "unsupported_value"})

# Upper bound on adaptation retries for one call. Each 400 names a single
# parameter, so an unrecognised reasoning model needs one round trip per
# rejected parameter — at most every param we know how to drop, plus the rename.
_MAX_PARAM_ADAPT_ATTEMPTS = len(_REASONING_UNSUPPORTED_PARAMS) + len(_RENAMED_PARAMS) + 1


def _is_reasoning_model(model: str) -> bool:
    """Whether `model` belongs to a family that rejects the legacy chat params.

    Any provider qualifier is stripped first, so a gateway-pinned
    ``openai/gpt-5-mini`` (which GatewayProvider passes through verbatim) is
    matched as well as a bare ``gpt-5-mini``. A gateway tier like ``auto/mid``
    resolves upstream and cannot be classified here — the error-driven path
    below covers it.
    """
    bare = model.rsplit("/", 1)[-1]
    return bare.startswith(_REASONING_MODEL_PREFIXES)


class OpenAIProvider(BaseProvider):
    """Calls OpenAI (or Azure OpenAI) directly via the openai SDK."""

    # Parameters each model has been observed to reject, keyed by normalized
    # model name. Seeded by the static prefix match above; extended at runtime
    # when an unrecognised model returns an unsupported-parameter 400, so a
    # future model id costs one retry instead of blocking every call.
    #
    # CLASS-level on purpose: get_provider() constructs a fresh OpenAIProvider on
    # every LLM call, so an instance-level cache would never survive between
    # requests and every call would re-pay the retry. A class attribute persists
    # for the process lifetime and is shared across all instances (the rejection
    # is a property of the model, not of the client/key). Mirrors
    # AnthropicProvider._temp_unsupported.
    _unsupported_params: dict[str, set[str]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int | None = None,
    ):
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        if api_base:
            kwargs["base_url"] = api_base
        # Wire the configured retry budget into the SDK client. Without this the
        # SDK falls back to its own default (2), so llm_max_retries did nothing
        # and a hanging call retried uncontrollably (the per-attempt timeout is
        # not a total ceiling). None → leave the SDK default untouched.
        if max_retries is not None:
            kwargs["max_retries"] = max_retries

        default_headers: dict[str, str] = {}
        if api_version:
            default_headers["api-version"] = api_version
        if extra_headers:
            default_headers.update(extra_headers)
        if default_headers:
            kwargs["default_headers"] = default_headers

        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)

    def _normalize_model(self, model: str) -> str:
        return model.removeprefix("openai/").removeprefix("azure/")

    def _unsupported_for(self, model: str) -> set[str]:
        """Parameters to omit/rename for `model` — static family match plus
        anything learned from a previous 400."""
        unsupported = set(self._unsupported_params.get(model, ()))
        if _is_reasoning_model(model):
            unsupported |= _REASONING_UNSUPPORTED_PARAMS
            unsupported |= set(_RENAMED_PARAMS)
        return unsupported

    def _build_kwargs(
        self,
        config: LLMConfig,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        model = self._normalize_model(config.model)
        kwargs: dict[str, Any] = {"model": model}

        # GPT-5 / o-series reject several legacy chat params. Omit them up front
        # (or send the renamed equivalent) so a known reasoning model never pays
        # a failed request. `skip` also carries anything learned at runtime.
        skip = self._unsupported_for(model)

        def put(name: str, value: Any) -> None:
            if value is None or name in skip:
                return
            kwargs[name] = value

        put("temperature", config.temperature)
        put("top_p", config.top_p)
        put("frequency_penalty", config.frequency_penalty)
        put("presence_penalty", config.presence_penalty)
        put("stop", config.stop)
        put("seed", config.seed)
        put("user", config.user)

        if config.max_tokens is not None:
            # Reasoning models take the same limit under a different name.
            token_param = _RENAMED_PARAMS["max_tokens"] if "max_tokens" in skip else "max_tokens"
            kwargs[token_param] = config.max_tokens

        if config.timeout:
            kwargs["timeout"] = config.timeout

        # Response format
        if config.response_format is not None:
            kwargs["response_format"] = config.response_format
        elif config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if config.extra_body is not None:
            kwargs["extra_body"] = config.extra_body

        return kwargs

    # Overridden by subclasses that speak the OpenAI wire protocol to a
    # different upstream (e.g. GatewayProvider → "gateway"), so errors are
    # attributed to the system that actually served the request.
    _provider_label: str = _PROVIDER

    def _warn_if_budget_consumed(
        self, model: str, finish_reason: str | None, has_content: bool
    ) -> None:
        """Warn when a response was truncated before producing any visible text.

        On reasoning models ``max_completion_tokens`` is a COMBINED budget —
        internal reasoning tokens plus the visible answer — so a limit that was
        adequate as ``max_tokens`` on a legacy model can be spent entirely on
        reasoning, yielding ``finish_reason='length'`` and empty content. That
        would otherwise be a silent empty response, so name the cause.
        """
        if finish_reason != "length" or has_content:
            return
        if _is_reasoning_model(model):
            logger.warning(
                "Model '%s' returned no content: the whole max_completion_tokens "
                "budget was consumed by reasoning tokens. This budget covers "
                "reasoning AND visible output, so it must be set higher than the "
                "max_tokens you would use on a non-reasoning model — raise it.",
                model,
            )
        else:
            logger.warning(
                "Model '%s' returned no content — the output token limit was "
                "reached before any text was produced. Raise max_tokens.",
                model,
            )

    def _check_response_content(self, model: str, response: Any) -> None:
        """Extract finish_reason/content from a completion and warn if empty."""
        try:
            choice = response.choices[0]
            finish_reason, content = choice.finish_reason, choice.message.content
        except (AttributeError, IndexError):  # non-standard response shape
            return
        self._warn_if_budget_consumed(model, finish_reason, bool(content))

    @staticmethod
    def _rejected_param(e: Exception) -> str | None:
        """Return the parameter name a 400 rejected, or None if it is not one.

        OpenAI reports these in structured form — ``code`` is
        ``unsupported_parameter``/``unsupported_value`` and ``param`` names the
        kwarg — so this reads the body instead of pattern-matching the message.
        """
        if not isinstance(e, openai.BadRequestError):
            return None
        body = e.body if isinstance(e.body, dict) else {}
        if body.get("code") not in _UNSUPPORTED_PARAM_CODES:
            return None
        param = body.get("param")
        return param if isinstance(param, str) and param else None

    def _call_adapting(self, make_call: Callable[[], Any], kwargs: dict[str, Any]) -> Any:
        """Run `make_call`, adapting away rejected parameters and retrying.

        A model can reject several parameters, but each 400 names only one — so
        one retry is not enough for an unrecognised reasoning model (it reports
        `max_tokens`, then `temperature`, then `top_p`, ...). Loop until the call
        succeeds or nothing further can be adapted. Bounded by the number of
        parameters we could ever drop, so a persistently failing call cannot spin.
        `kwargs` is mutated in place, so `make_call` sees each adaptation.
        """
        for attempt in range(_MAX_PARAM_ADAPT_ATTEMPTS):
            try:
                return make_call()
            except Exception as e:
                if attempt == _MAX_PARAM_ADAPT_ATTEMPTS - 1 or not self._adapt_after_rejection(
                    kwargs, e
                ):
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _acall_adapting(
        self, make_call: Callable[[], Awaitable[Any]], kwargs: dict[str, Any]
    ) -> Any:
        """Async counterpart of `_call_adapting`."""
        for attempt in range(_MAX_PARAM_ADAPT_ATTEMPTS):
            try:
                return await make_call()
            except Exception as e:
                if attempt == _MAX_PARAM_ADAPT_ATTEMPTS - 1 or not self._adapt_after_rejection(
                    kwargs, e
                ):
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _adapt_after_rejection(self, kwargs: dict[str, Any], e: Exception) -> bool:
        """Drop (or rename) the parameter a 400 rejected and remember the model.

        Returns True if `kwargs` changed, so a retry is worth attempting; False
        otherwise (nothing to adjust — let the error surface).
        """
        param = self._rejected_param(e)
        if param is None or param not in kwargs:
            return False

        model = kwargs.get("model", "")
        self._unsupported_params.setdefault(model, set()).add(param)

        value = kwargs.pop(param)
        renamed = _RENAMED_PARAMS.get(param)
        if renamed:
            # Preserve the caller's intent rather than uncapping the response.
            kwargs[renamed] = value
            logger.warning(
                "Model '%s' rejected '%s'; retrying with '%s' and using it for this "
                "model for the rest of this process.",
                model,
                param,
                renamed,
            )
        else:
            logger.warning(
                "Model '%s' rejected parameter '%s'; retrying without it and omitting "
                "it for this model for the rest of this process.",
                model,
                param,
            )
        return True

    def _handle_exception(self, e: Exception, model: str) -> None:
        provider = self._provider_label
        ctx: dict[str, Any] = {"model": model, "provider": provider}
        gateway_url = getattr(self, "_gateway_url", None)
        if gateway_url:
            ctx["gateway_url"] = gateway_url
        if isinstance(e, openai.AuthenticationError):
            raise LLMAuthenticationError(
                str(e), model=model, provider=provider, original_error=e, context=ctx
            ) from e
        if isinstance(e, openai.RateLimitError):
            raise LLMRateLimitError(
                str(e), model=model, provider=provider, original_error=e, context=ctx
            ) from e
        if isinstance(e, openai.APITimeoutError):
            raise LLMTimeoutError(
                str(e), model=model, provider=provider, original_error=e, context=ctx
            ) from e
        if isinstance(e, openai.BadRequestError):
            msg = str(e)
            # An unsupported-parameter 400 is NOT a context-length problem, but
            # its message contains "token" (e.g. "'max_tokens' is not supported
            # with this model"), which the substring check below would otherwise
            # misclassify — hiding the real cause behind a context-length error.
            if self._rejected_param(e) is None and (
                "context" in msg.lower() or "token" in msg.lower()
            ):
                raise LLMContextLengthError(
                    msg, model=model, provider=provider, original_error=e, context=ctx
                ) from e
            raise LLMInvalidRequestError(
                msg, model=model, provider=provider, original_error=e, context=ctx
            ) from e
        if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
            raise LLMServiceUnavailableError(
                str(e), model=model, provider=provider, original_error=e, context=ctx
            ) from e
        raise LLMError(str(e), model=model, provider=provider, original_error=e, context=ctx) from e

    def complete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(config, tools, tool_choice)
        try:
            response = self._call_adapting(
                lambda: self._client.chat.completions.create(messages=messages, **kwargs), kwargs
            )
            self._check_response_content(kwargs["model"], response)
            return LLMResponse.from_openai_response(response)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(config, tools, tool_choice)
        try:
            response = await self._acall_adapting(
                lambda: self._async_client.chat.completions.create(messages=messages, **kwargs),
                kwargs,
            )
            self._check_response_content(kwargs["model"], response)
            return LLMResponse.from_openai_response(response)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise

    @staticmethod
    def _accumulate_tool_call(acc: dict[int, dict[str, str]], raw_tc: Any) -> None:
        """Merge one raw OpenAI tool-call delta into the accumulator dict."""
        idx = raw_tc.index
        if idx not in acc:
            acc[idx] = {"id": "", "name": "", "arguments": ""}
        if raw_tc.id:
            acc[idx]["id"] = raw_tc.id
        if raw_tc.function:
            if raw_tc.function.name:
                acc[idx]["name"] += raw_tc.function.name
            if raw_tc.function.arguments:
                acc[idx]["arguments"] += raw_tc.function.arguments

    @staticmethod
    def _build_tool_calls_from_acc(acc: dict[int, dict[str, str]]) -> list[ToolCall]:
        return [
            ToolCall(
                id=acc[i]["id"],
                type="function",
                function=FunctionCall(name=acc[i]["name"], arguments=acc[i]["arguments"]),
            )
            for i in sorted(acc)
        ]

    def stream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        kwargs = self._build_kwargs(config, tools, tool_choice)
        kwargs["stream"] = True
        try:
            # Only the stream *open* is retried — nothing has been yielded yet,
            # so a retry here cannot emit a chunk twice.
            response = self._call_adapting(
                lambda: self._client.chat.completions.create(messages=messages, **kwargs), kwargs
            )
            tc_acc: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            emitted_content = False
            for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                delta = choice.delta if choice else None
                if choice and choice.finish_reason:
                    finish_reason = choice.finish_reason
                if delta and delta.tool_calls:
                    for raw_tc in delta.tool_calls:
                        self._accumulate_tool_call(tc_acc, raw_tc)
                if delta and delta.content:
                    emitted_content = True
                    yield StreamChunk(
                        id=chunk.id,
                        model=chunk.model,
                        content=delta.content,
                        role=delta.role,
                        is_finished=False,
                    )
            if tc_acc:
                yield StreamChunk(
                    tool_calls=self._build_tool_calls_from_acc(tc_acc),
                    finish_reason=finish_reason or "tool_calls",
                    is_finished=True,
                )
            elif finish_reason:
                # Same silent-empty case as the non-streaming path.
                self._warn_if_budget_consumed(kwargs["model"], finish_reason, emitted_content)
                yield StreamChunk(finish_reason=finish_reason, is_finished=True)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise

    async def astream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = self._build_kwargs(config, tools, tool_choice)
        kwargs["stream"] = True
        try:
            # Only the stream *open* is retried — see the sync `stream` above.
            response = await self._acall_adapting(
                lambda: self._async_client.chat.completions.create(messages=messages, **kwargs),
                kwargs,
            )
            tc_acc: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            emitted_content = False
            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                delta = choice.delta if choice else None
                if choice and choice.finish_reason:
                    finish_reason = choice.finish_reason
                if delta and delta.tool_calls:
                    for raw_tc in delta.tool_calls:
                        self._accumulate_tool_call(tc_acc, raw_tc)
                if delta and delta.content:
                    emitted_content = True
                    yield StreamChunk(
                        id=chunk.id,
                        model=chunk.model,
                        content=delta.content,
                        role=delta.role,
                        is_finished=False,
                    )
            if tc_acc:
                yield StreamChunk(
                    tool_calls=self._build_tool_calls_from_acc(tc_acc),
                    finish_reason=finish_reason or "tool_calls",
                    is_finished=True,
                )
            elif finish_reason:
                # Same silent-empty case as the non-streaming path.
                self._warn_if_budget_consumed(kwargs["model"], finish_reason, emitted_content)
                yield StreamChunk(finish_reason=finish_reason, is_finished=True)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise
