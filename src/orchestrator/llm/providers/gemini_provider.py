"""
Google Gemini provider via the OpenAI-compatible endpoint.

Google exposes a fully OpenAI-compatible REST API at:
  https://generativelanguage.googleapis.com/v1beta/openai/

This means we can reuse the openai SDK — no google-generativeai package needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import openai
from openai import AsyncOpenAI, OpenAI

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

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_PROVIDER = "gemini"


class GeminiProvider(BaseProvider):
    """Calls Google Gemini via its OpenAI-compatible endpoint."""

    def __init__(self, api_key: str | None = None):
        kwargs: dict[str, Any] = {
            "base_url": _GEMINI_BASE_URL,
            "api_key": api_key or "placeholder",  # SDK requires a non-empty key
        }
        self._client = OpenAI(**kwargs)
        self._async_client = AsyncOpenAI(**kwargs)

    def _normalize_model(self, model: str) -> str:
        # Strip provider prefix: "gemini/gemini-2.5-flash" → "gemini-2.5-flash"
        return model.removeprefix("gemini/").removeprefix("google/")

    def _build_kwargs(
        self,
        config: LLMConfig,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._normalize_model(config.model),
            "temperature": config.temperature,
        }
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop is not None:
            kwargs["stop"] = config.stop
        if config.timeout:
            kwargs["timeout"] = config.timeout

        # Gemini supports json_object mode but not Pydantic schema via compat endpoint
        if config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        return kwargs

    def _handle_exception(self, e: Exception, model: str) -> None:
        ctx = {"model": model, "provider": _PROVIDER}
        if isinstance(e, openai.AuthenticationError):
            raise LLMAuthenticationError(str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx) from e
        if isinstance(e, openai.RateLimitError):
            raise LLMRateLimitError(str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx) from e
        if isinstance(e, openai.APITimeoutError):
            raise LLMTimeoutError(str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx) from e
        if isinstance(e, openai.BadRequestError):
            msg = str(e)
            if "context" in msg.lower() or "token" in msg.lower():
                raise LLMContextLengthError(msg, model=model, provider=_PROVIDER, original_error=e, context=ctx) from e
            raise LLMInvalidRequestError(msg, model=model, provider=_PROVIDER, original_error=e, context=ctx) from e
        if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
            raise LLMServiceUnavailableError(str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx) from e
        raise LLMError(str(e), model=model, provider=_PROVIDER, original_error=e, context=ctx) from e

    def complete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(config, tools, tool_choice)
        try:
            response = self._client.chat.completions.create(messages=messages, **kwargs)
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
            response = await self._async_client.chat.completions.create(messages=messages, **kwargs)
            return LLMResponse.from_openai_response(response)
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
        kwargs = self._build_kwargs(config, tools, tool_choice)
        kwargs["stream"] = True
        try:
            response = self._client.chat.completions.create(messages=messages, **kwargs)
            for chunk in response:
                yield StreamChunk.from_openai_chunk(chunk)
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
            response = await self._async_client.chat.completions.create(messages=messages, **kwargs)
            async for chunk in response:
                yield StreamChunk.from_openai_chunk(chunk)
        except Exception as e:
            self._handle_exception(e, config.model)
            raise
