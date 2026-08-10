"""
Abstract base class for LLM providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from continuum.llm.config import LLMConfig
from continuum.llm.types import LLMResponse, StreamChunk


class BaseProvider(ABC):
    """Abstract base for all LLM provider implementations."""

    @staticmethod
    def supports_native_schema() -> bool:
        """Can this provider make the model's answer match a JSON schema?

        Answering honestly is the point: when it is False, Continuum falls back
        to asking for the shape in the prompt and salvaging whatever comes back,
        and callers deserve to know which of the two they are getting.

        Defaults to False so a third-party provider that has not implemented
        enforcement is never reported as enforcing one.
        """
        return False

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
