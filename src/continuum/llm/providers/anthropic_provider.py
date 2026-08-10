"""
Anthropic provider — direct anthropic SDK.

Handles message format conversion between the OpenAI-style format used
throughout the codebase and Anthropic's native API format.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import anthropic

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
from continuum.llm.structured_output import (
    STRUCTURED_OUTPUT_TOOL,
    STRUCTURED_OUTPUT_TOOL_DESCRIPTION,
    forces_structured_tool,
    schema_from_response_format,
    unwrap_structured_tool_call,
)
from continuum.llm.types import LLMResponse, StreamChunk
from continuum.logging import get_logger

logger = get_logger(__name__)

_PROVIDER = "anthropic"


class AnthropicProvider(BaseProvider):
    """Calls Anthropic Claude directly via the anthropic SDK."""

    # Models learned at runtime to reject an explicit `temperature` parameter
    # (Claude 4.6+ adaptive-thinking models return a 400 if one is supplied).
    # Populated on the first such 400; consulted by _build_kwargs thereafter so
    # subsequent calls omit temperature up front instead of retrying every time.
    #
    # CLASS-level on purpose: get_provider() constructs a fresh AnthropicProvider
    # on every LLM call, so an instance-level cache would never survive between
    # requests and every call would re-pay the retry. A class attribute persists
    # for the process lifetime and is shared across all instances (the rejection
    # is a property of the model, not of the client/key).
    _temp_unsupported: set[str] = set()

    def __init__(self, api_key: str | None = None, max_retries: int | None = None):
        # Wire the configured retry budget into the SDK client; without it the
        # Anthropic SDK uses its own default (2), so llm_max_retries did nothing
        # and a hang retried uncontrollably. None → leave the SDK default.
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = anthropic.Anthropic(**client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(**client_kwargs)

    def _normalize_model(self, model: str) -> str:
        return model.removeprefix("anthropic/").removeprefix("claude/")

    @staticmethod
    def _is_temperature_rejection(e: Exception) -> bool:
        """True if `e` is a 400 specifically about the temperature parameter."""
        return isinstance(e, anthropic.BadRequestError) and "temperature" in str(e).lower()

    def _mark_temp_unsupported(self, kwargs: dict[str, Any]) -> bool:
        """Record that this model rejects temperature and strip it from `kwargs`.

        Returns True if temperature was present and removed (so a retry is worth
        attempting), False otherwise (nothing changed — do not retry).
        """
        if "temperature" not in kwargs:
            return False
        model = kwargs.get("model", "")
        self._temp_unsupported.add(model)
        kwargs.pop("temperature")
        logger.warning(
            "Model '%s' rejected an explicit temperature; retrying without it and "
            "omitting it for this model for the rest of this process.",
            model,
        )
        return True

    def _split_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Split OpenAI-format messages into (system_prompt, anthropic_messages).

        Anthropic takes system as a top-level param, not inside the messages list.
        Tool results (role=tool) must be wrapped as user messages with tool_result blocks.
        Tool calls from assistant must be converted to tool_use content blocks.
        """
        system_blocks: list[dict[str, Any]] = []
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "system":
                if content:
                    block: dict[str, Any] = {"type": "text", "text": content}
                    cc = msg.get("cache_control")
                    if cc:
                        block["cache_control"] = cc
                    system_blocks.append(block)

            elif role == "tool":
                # Tool result — must live inside a user message
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content or "",
                }
                # Merge into last user message if possible, otherwise create new one
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    prev_content = anthropic_messages[-1]["content"]
                    if isinstance(prev_content, list):
                        prev_content.append(tool_result_block)
                    else:
                        anthropic_messages[-1]["content"] = [tool_result_block]
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_result_block]})

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    content_blocks: list[dict[str, Any]] = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            args_str = func.get("arguments", "{}")
                        else:
                            func = tc.function
                            args_str = func.arguments or "{}"
                        try:
                            input_data = json.loads(args_str)
                        except json.JSONDecodeError:
                            input_data = {}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", "") if isinstance(tc, dict) else tc.id,
                                "name": func.get("name", "")
                                if isinstance(func, dict)
                                else func.name,
                                "input": input_data,
                            }
                        )
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content or ""})

            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content or ""})

        if not system_blocks:
            system: str | list | None = None
        elif any("cache_control" in b for b in system_blocks):
            system = system_blocks  # list form required for cache_control
        else:
            system = "\n\n".join(b["text"] for b in system_blocks)
        return system, anthropic_messages

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert OpenAI tool format to Anthropic tool format."""
        if not tools:
            return None
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                func = tool.get("function", {})
                result.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            else:
                result.append(
                    {
                        "name": tool.function.name,
                        "description": tool.function.description or "",
                        "input_schema": tool.function.parameters
                        or {"type": "object", "properties": {}},
                    }
                )
        return result

    def _convert_tool_choice(
        self, tool_choice: str | dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if tool_choice is None:
            return None
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict) and "function" in tool_choice:
            return {"type": "tool", "name": tool_choice["function"]["name"]}
        return {"type": "auto"}

    @staticmethod
    def supports_native_schema() -> bool:
        """Yes — via forced tool use (see _schema_tool_kwargs)."""
        return True

    @staticmethod
    def _schema_tool_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
        """Declare the throwaway schema tool and leave the model no alternative.

        Anthropic's own tool spelling — `input_schema`, and a `tool_choice` of
        `{"type": "tool", "name": …}` — rather than the OpenAI wire form.
        """
        return {
            "tools": [
                {
                    "name": STRUCTURED_OUTPUT_TOOL,
                    "description": STRUCTURED_OUTPUT_TOOL_DESCRIPTION,
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL},
        }

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        *,
        enforce_schema: bool = False,
    ) -> dict[str, Any]:
        """Build the Anthropic request.

        ``enforce_schema`` opts into forced tool use for a requested output
        schema. It is off by default because the streaming paths cannot use it:
        a forced tool answers with ``input_json_delta`` and no text at all, so
        ``text_stream`` would yield nothing and the run would go silently empty.
        complete()/acomplete() turn it on.
        """
        system, anthropic_messages = self._split_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._normalize_model(config.model),
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens or 4096,
        }

        # Forcing the synthetic tool would take the caller's own tools off the
        # table, so real tool-calling turns keep the prompt-only floor. The
        # executor only asks for a schema on tool-less calls, so in practice the
        # enforced path is the one that runs for structured output.
        schema = None if tools else schema_from_response_format(config.response_format)
        if enforce_schema and schema is not None:
            kwargs.update(self._schema_tool_kwargs(schema))
        elif config.json_mode or config.response_format:
            # Nothing enforceable (bare json_object, or tools are in play): the
            # cross-provider floor is all that is left. The field names reach the
            # model separately, via structured_output.schema_prompt.
            #
            # `or ""`: _split_messages returns None when the caller sent no
            # system message, and concatenating onto that raised TypeError —
            # JSON mode on an agent with no instructions could not run at all.
            system = ((system or "") + "\nRespond with valid JSON only.").strip()

        if system:
            kwargs["system"] = system
        if (
            config.temperature is not None
            and self._normalize_model(config.model) not in self._temp_unsupported
        ):
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop is not None:
            stop = config.stop if isinstance(config.stop, list) else [config.stop]
            kwargs["stop_sequences"] = stop
        if config.timeout:
            kwargs["timeout"] = config.timeout

        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            tc = self._convert_tool_choice(tool_choice)
            if tc:
                kwargs["tool_choice"] = tc

        return kwargs

    @staticmethod
    def _to_response(raw: Any, config: LLMConfig, kwargs: dict[str, Any]) -> LLMResponse:
        """Build the LLMResponse, undoing the forced-tool delivery if used."""
        response = LLMResponse.from_anthropic_response(raw, config.model)
        if forces_structured_tool(kwargs):
            return unwrap_structured_tool_call(response)
        return response

    def _handle_exception(self, e: Exception, model: str) -> None:
        ctx = {"model": model, "provider": _PROVIDER}
        if isinstance(e, anthropic.AuthenticationError):
            raise LLMAuthenticationError(
                str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx
            ) from e
        if isinstance(e, anthropic.RateLimitError):
            raise LLMRateLimitError(
                str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx
            ) from e
        if isinstance(e, anthropic.APITimeoutError):
            raise LLMTimeoutError(
                str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx
            ) from e
        if isinstance(e, anthropic.BadRequestError):
            msg = str(e)
            if "context" in msg.lower() or "token" in msg.lower():
                raise LLMContextLengthError(
                    msg, model=model, provider=_PROVIDER, original_error=e, context=ctx
                ) from e
            raise LLMInvalidRequestError(
                msg, model=model, provider=_PROVIDER, original_error=e, context=ctx
            ) from e
        if isinstance(e, (anthropic.APIConnectionError, anthropic.InternalServerError)):
            raise LLMServiceUnavailableError(
                str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx
            ) from e
        raise LLMError(
            str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx
        ) from e

    def complete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(messages, config, tools, tool_choice, enforce_schema=True)
        try:
            response = self._client.messages.create(**kwargs)
            return self._to_response(response, config, kwargs)
        except Exception as e:
            # Error-driven drop: if the model rejected temperature, strip it and
            # retry once. The model is cached so future calls skip it up front.
            if self._is_temperature_rejection(e) and self._mark_temp_unsupported(kwargs):
                try:
                    response = self._client.messages.create(**kwargs)
                    return self._to_response(response, config, kwargs)
                except Exception as e2:
                    self._handle_exception(e2, config.model)
                    raise
            self._handle_exception(e, config.model)
            raise

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(messages, config, tools, tool_choice, enforce_schema=True)
        try:
            response = await self._async_client.messages.create(**kwargs)
            return self._to_response(response, config, kwargs)
        except Exception as e:
            if self._is_temperature_rejection(e) and self._mark_temp_unsupported(kwargs):
                try:
                    response = await self._async_client.messages.create(**kwargs)
                    return self._to_response(response, config, kwargs)
                except Exception as e2:
                    self._handle_exception(e2, config.model)
                    raise
            self._handle_exception(e, config.model)
            raise

    def _do_stream(self, kwargs: dict[str, Any], model: str) -> Iterator[StreamChunk]:
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield StreamChunk(content=text, is_finished=False)
            # Yield a final chunk with finish reason
            final = stream.get_final_message()
            yield StreamChunk.from_anthropic_response(final, model)

    def stream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            yield from self._do_stream(kwargs, config.model)
        except Exception as e:
            # A temperature-400 fires on stream open, before any chunk is yielded,
            # so retrying without temperature cannot double-emit content.
            if self._is_temperature_rejection(e) and self._mark_temp_unsupported(kwargs):
                try:
                    yield from self._do_stream(kwargs, config.model)
                    return
                except Exception as e2:
                    self._handle_exception(e2, config.model)
                    raise
            self._handle_exception(e, config.model)
            raise

    async def _do_astream(self, kwargs: dict[str, Any], model: str) -> AsyncIterator[StreamChunk]:
        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content=text, is_finished=False)
            final = await stream.get_final_message()
            yield StreamChunk.from_anthropic_response(final, model)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            async for chunk in self._do_astream(kwargs, config.model):
                yield chunk
        except Exception as e:
            if self._is_temperature_rejection(e) and self._mark_temp_unsupported(kwargs):
                try:
                    async for chunk in self._do_astream(kwargs, config.model):
                        yield chunk
                    return
                except Exception as e2:
                    self._handle_exception(e2, config.model)
                    raise
            self._handle_exception(e, config.model)
            raise
