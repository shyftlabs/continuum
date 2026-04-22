"""
LLM Client - Unified interface for multi-LLM provider support.

Calls provider SDKs directly (OpenAI, Anthropic, Gemini).
Supports synchronous, asynchronous, and streaming modes.
Includes full observability integration with Langfuse via @observe decorator.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from orchestrator.config import settings
from orchestrator.llm.callbacks import (
    get_langfuse_metadata,
    setup_langfuse,
)
from orchestrator.llm.config import LLMConfig
from orchestrator.llm.exceptions import (
    LLMError,
)
from orchestrator.llm.providers import get_provider
from orchestrator.llm.types import (
    ChatMessage,
    LLMResponse,
    StreamChunk,
    ToolDefinition,
)
from orchestrator.llm.utils import supports_tools_with_json_mode
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe
from orchestrator.observability.trace_context import (
    get_current_session_id,
)

logger = get_logger(__name__)


class _LLMRateLimiter:
    """Simple token-bucket rate limiter for LLM API requests (RPM)."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.tokens = float(rpm)
        self._lock = asyncio.Lock()
        self._last_update: float | None = None

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._last_update is None:
                self._last_update = now
                self.tokens -= 1
                return

            elapsed = now - self._last_update
            self._last_update = now
            self.tokens = min(float(self.rpm), self.tokens + elapsed * (self.rpm / 60.0))

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (60.0 / self.rpm)
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class LLMClient:
    """
    Unified LLM client that routes to OpenAI, Anthropic, or Gemini
    based on the model name in LLMConfig.

    All LLM calls are automatically traced via the @observe decorator.
    Session management and context compression are handled transparently.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        enable_langfuse: bool = True,
    ):
        self.default_config = config or LLMConfig()
        self._langfuse_enabled = enable_langfuse
        self._rate_limiter: _LLMRateLimiter | None = None

        if self.default_config.rate_limit_rpm and self.default_config.rate_limit_rpm > 0:
            self._rate_limiter = _LLMRateLimiter(self.default_config.rate_limit_rpm)

        if enable_langfuse:
            try:
                setup_langfuse()
            except Exception:
                logger.warning("Langfuse setup skipped (not configured or unavailable)")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _convert_messages(
        self, messages: list[ChatMessage] | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                result.append(msg.to_dict())
            else:
                result.append(msg)
        return result

    def _convert_tools(
        self, tools: list[ToolDefinition] | list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if tools is None:
            return None
        result = []
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                result.append(tool.to_dict())
            else:
                result.append(tool)
        return result

    def _build_metadata(
        self,
        tools: list[dict[str, Any]] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._langfuse_enabled:
            return trace_metadata or {}
        metadata = get_langfuse_metadata(custom_metadata=trace_metadata)
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            metadata["tools_available"] = tool_names
        return metadata

    def _apply_json_mode_compat(
        self,
        config: LLMConfig,
        tools_dict: list[dict[str, Any]] | None,
    ) -> LLMConfig:
        """Disable JSON mode when tools are present and model doesn't support both."""
        if tools_dict and (config.json_mode or config.response_format):
            if not supports_tools_with_json_mode(config.model, config.custom_llm_provider):
                logger.warning(
                    f"Model '{config.model}' does not support tools + JSON mode simultaneously. "
                    "Disabling JSON mode to allow tool usage."
                )
                return config.model_copy(update={"json_mode": False, "response_format": None})
        return config

    def _log_json_mode_status(self, config: LLMConfig) -> None:
        if config.json_mode:
            logger.info(f"JSON mode active: json_object for model {config.model}")
        elif config.response_format is not None:
            logger.info(f"JSON mode active: schema for model {config.model}")

    def _validate_json_response(self, content: str | None, config: LLMConfig) -> None:
        if not (config.json_mode or config.response_format) or not content:
            return
        try:
            stripped = content.strip()
            is_json = (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            )
            if not is_json:
                logger.warning(
                    "LLM response is not JSON despite JSON mode being enabled",
                    extra={"model": config.model, "preview": stripped[:100]},
                )
            else:
                json.loads(content)
        except json.JSONDecodeError:
            logger.warning(
                "LLM response is not valid JSON despite JSON mode",
                extra={"model": config.model, "preview": content[:100]},
            )

    def _get_provider_from_model(self, model: str) -> str:
        if "/" in model:
            return model.split("/")[0]
        if model.startswith("gpt"):
            return "openai"
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith("gemini"):
            return "google"
        return "unknown"

    # =========================================================================
    # Synchronous Methods
    # =========================================================================

    @observe(name="llm_chat_sync", capture_output=True)
    def chat_sync(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous chat completion. Auto-traced via @observe."""
        effective_config = config or self.default_config
        messages_dict = self._convert_messages(messages)
        tools_dict = self._convert_tools(tools)
        effective_config = self._apply_json_mode_compat(effective_config, tools_dict)
        self._log_json_mode_status(effective_config)

        provider = get_provider(effective_config)
        logger.debug(f"Sync completion: model={effective_config.model}")

        response = provider.complete(messages_dict, effective_config, tools_dict, tool_choice)
        self._validate_json_response(response.content, effective_config)
        return response

    @observe(name="llm_chat_stream_sync", capture_output=False)
    def chat_stream_sync(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Synchronous streaming chat completion. Auto-traced via @observe."""
        effective_config = config or self.default_config
        messages_dict = self._convert_messages(messages)
        tools_dict = self._convert_tools(tools)
        effective_config = self._apply_json_mode_compat(effective_config, tools_dict)

        provider = get_provider(effective_config)
        logger.debug(f"Sync stream: model={effective_config.model}")

        yield from provider.stream(messages_dict, effective_config, tools_dict, tool_choice)

    # =========================================================================
    # Asynchronous Methods
    # =========================================================================

    @observe(name="llm_chat", capture_output=True)
    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        auto_session: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Asynchronous chat completion. Primary method for async contexts.
        Auto-traced via @observe. Session history loaded/saved automatically.
        """
        effective_config = config or self.default_config
        effective_session_id = session_id or get_current_session_id()

        # Load conversation history from session
        if effective_session_id and auto_session:
            try:
                from orchestrator.core.container import get_container

                session_client = get_container().session_client
                if session_client.is_enabled:
                    history_messages = await session_client.get_conversation_history(
                        session_id=effective_session_id,
                    )
                    if history_messages:
                        history_dicts = [
                            msg.to_dict() if hasattr(msg, "to_dict") else msg
                            for msg in history_messages
                        ]
                        messages_dict = history_dicts + self._convert_messages(messages)
                        logger.debug(
                            f"Loaded {len(history_messages)} messages from session: {effective_session_id}",
                            extra={"total_messages": len(messages_dict)},
                        )
                    else:
                        messages_dict = self._convert_messages(messages)
                else:
                    messages_dict = self._convert_messages(messages)
            except Exception as e:
                logger.warning(f"Failed to load session history: {e}, continuing with provided messages")
                messages_dict = self._convert_messages(messages)
        else:
            messages_dict = self._convert_messages(messages)

        # Context compression
        try:
            from orchestrator.llm.context_management import get_progressive_context_manager

            context_manager = get_progressive_context_manager()
            if context_manager.config.enabled:
                messages_dict, compression_result = await context_manager.compress_if_needed(
                    messages=messages_dict,
                    model=effective_config.model,
                )
                if compression_result.was_compressed:
                    logger.info(
                        f"Context compressed: {compression_result.original_token_count} → "
                        f"{compression_result.compressed_token_count} tokens "
                        f"({compression_result.compression_ratio:.1%} ratio)"
                    )
        except Exception as e:
            logger.warning(f"Context management failed, continuing without compression: {e}")

        tools_dict = self._convert_tools(tools)
        effective_config = self._apply_json_mode_compat(effective_config, tools_dict)
        self._log_json_mode_status(effective_config)

        if self._rate_limiter:
            await self._rate_limiter.acquire()

        provider = get_provider(effective_config)
        logger.debug(f"Async completion: model={effective_config.model}")

        llm_response = await provider.acomplete(messages_dict, effective_config, tools_dict, tool_choice)
        self._validate_json_response(llm_response.content, effective_config)

        # Save messages to session
        if effective_session_id and auto_session:
            try:
                from orchestrator.core.container import get_container

                session_client = get_container().session_client
                if session_client.is_enabled:
                    new_messages = self._convert_messages(messages)
                    for msg_dict in new_messages:
                        if isinstance(msg_dict, dict):
                            msg_role = msg_dict.get("role")
                            msg_content = msg_dict.get("content")
                            if msg_role == "user" and msg_content:
                                user_msg = ChatMessage(**msg_dict)
                                await session_client.add_message(
                                    session_id=effective_session_id,
                                    message=user_msg,
                                    store_in_memory=True,
                                )

                    assistant_message = ChatMessage(
                        role="assistant",
                        content=llm_response.content,
                        tool_calls=llm_response.tool_calls,
                        function_call=llm_response.function_call,
                    )
                    should_store_in_memory = (
                        llm_response.content is not None
                        and llm_response.content.strip()
                        and not llm_response.tool_calls
                    )
                    await session_client.add_message(
                        session_id=effective_session_id,
                        message=assistant_message,
                        store_in_memory=should_store_in_memory,
                    )
                    logger.debug(f"Saved messages to session: {effective_session_id}")
            except Exception as e:
                logger.warning(f"Failed to save messages to session: {e}")

        return llm_response

    @observe(name="llm_chat_stream", capture_output=False)
    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous streaming chat completion. Auto-traced via @observe."""
        effective_config = config or self.default_config
        messages_dict = self._convert_messages(messages)

        # Context compression before streaming
        try:
            from orchestrator.llm.context_management import get_progressive_context_manager

            context_manager = get_progressive_context_manager()
            if context_manager.config.enabled:
                messages_dict, compression_result = await context_manager.compress_if_needed(
                    messages=messages_dict,
                    model=effective_config.model,
                )
                if compression_result.was_compressed:
                    logger.info(
                        f"Context compressed before streaming: {compression_result.original_token_count} → "
                        f"{compression_result.compressed_token_count} tokens"
                    )
        except Exception as e:
            logger.warning(f"Context management failed, continuing without compression: {e}")

        tools_dict = self._convert_tools(tools)
        effective_config = self._apply_json_mode_compat(effective_config, tools_dict)

        provider = get_provider(effective_config)
        logger.debug(f"Async stream: model={effective_config.model}")

        async for chunk in provider.astream(messages_dict, effective_config, tools_dict, tool_choice):
            yield chunk

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @observe(name="llm_get_model_info", capture_output=True)
    def get_model_info(self, model: str | None = None) -> dict[str, Any]:
        """Get context window information for a model."""
        from orchestrator.llm.context_window import get_context_window_manager

        model = model or self.default_config.model
        try:
            limits = get_context_window_manager().get_model_limits(model)
            return limits.to_dict()
        except Exception as e:
            logger.warning(f"Could not get model info for {model}: {e}")
            return {}

    def get_supported_models(self) -> list[str]:
        """Return the list of known supported models."""
        return [
            # OpenAI
            "gpt-4o", "gpt-4o-mini", "gpt-4o-turbo", "gpt-3.5-turbo",
            # Anthropic
            "claude-haiku-4.5", "claude-sonnet-4.5", "claude-opus-4.5",
            # Gemini
            "gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite",
        ]

    @observe(name="llm_count_tokens", capture_output=True)
    def count_tokens(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """Count tokens for a list of messages."""
        from orchestrator.llm.context_window import get_context_window_manager

        model = model or self.default_config.model
        messages_dict = self._convert_messages(messages)
        try:
            return get_context_window_manager().count_tokens(messages_dict, model)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return 0

    def complete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Alias for chat_sync."""
        return self.chat_sync(messages, config, **kwargs)

    async def acomplete(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Alias for chat."""
        return await self.chat(messages, config, **kwargs)

    def stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Alias for chat_stream_sync."""
        yield from self.chat_stream_sync(messages, config, **kwargs)
