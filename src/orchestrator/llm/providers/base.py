"""
Abstract base class for LLM providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from orchestrator.llm.config import LLMConfig
from orchestrator.llm.types import LLMResponse, StreamChunk


class BaseProvider(ABC):
    """Abstract base for all LLM provider implementations."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]: ...

    @abstractmethod
    async def astream(
        self,
        messages: list[dict[str, Any]],
        config: LLMConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
