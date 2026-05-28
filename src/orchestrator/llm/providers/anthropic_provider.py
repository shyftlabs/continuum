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

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from orchestrator.llm.providers.base import BaseProvider
from orchestrator.llm.types import LLMResponse, StreamChunk
from orchestrator.logging import get_logger

logger = get_logger(__name__)

_PROVIDER = "anthropic"


class AnthropicProvider(BaseProvider):
    """Calls Anthropic Claude directly via the anthropic SDK."""

    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _normalize_model(self, model: str) -> str:
        return model.removeprefix("anthropic/").removeprefix("claude/")

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

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        system, anthropic_messages = self._split_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._normalize_model(config.model),
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens or 4096,
        }
        if config.json_mode or config.response_format:
            system = (system + "\nRespond with valid JSON only.").strip()
        if system:
            kwargs["system"] = system
        if config.temperature is not None:
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
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            response = self._client.messages.create(**kwargs)
            return LLMResponse.from_anthropic_response(response, config.model)
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
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            response = await self._async_client.messages.create(**kwargs)
            return LLMResponse.from_anthropic_response(response, config.model)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise

    def stream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(content=text, is_finished=False)
                # Yield a final chunk with finish reason
                final = stream.get_final_message()
                yield StreamChunk.from_anthropic_response(final, config.model)
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
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            async with self._async_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(content=text, is_finished=False)
                final = await stream.get_final_message()
                yield StreamChunk.from_anthropic_response(final, config.model)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise
