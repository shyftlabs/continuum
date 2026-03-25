"""
Progressive Context Management - Advanced dynamic context compression.

Provides intelligent context management with:
- Proactive summarization when approaching context limits
- Async summarization for scalability
- Fallback to truncation on failure
- Full observability and metrics
- Multi-agent workflow support
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from orchestrator.config import settings
from orchestrator.llm import LLMClient
from orchestrator.llm.context_window import (
    ContextWindowManager,
    TruncationStrategy,
    get_context_window_manager,
)
from orchestrator.llm.types import ChatMessage
from orchestrator.logging import get_logger
from orchestrator.observability.metrics import get_metrics_collector
from orchestrator.observability.trace_context import SpanScope, truncate_data

logger = get_logger(__name__)


class CompressionStrategy(str, Enum):
    """Strategy for compressing context when approaching limits."""

    # Summarize older messages, keep recent intact
    SUMMARIZE_OLD = "summarize_old"

    # Truncate oldest messages (fallback)
    TRUNCATE_OLDEST = "truncate_oldest"

    # Smart: Try summarization, fallback to truncation
    SMART = "smart"


@dataclass
class ContextManagementConfig:
    """Configuration for context management."""

    # Enable/disable
    enabled: bool = field(default_factory=lambda: settings.context_management_enabled)

    # Thresholds
    compression_threshold: float = field(
        default_factory=lambda: settings.context_compression_threshold
    )  # 0.0-1.0

    # Summarization settings
    summarization_model: str = field(default_factory=lambda: settings.context_summarization_model)
    summarization_temperature: float = field(
        default_factory=lambda: settings.context_summarization_temperature
    )
    summarization_timeout: int = field(
        default_factory=lambda: settings.context_summarization_timeout
    )
    summarization_max_retries: int = field(
        default_factory=lambda: settings.context_summarization_max_retries
    )

    # Compression settings
    keep_recent_messages: int = field(default_factory=lambda: settings.context_keep_recent_messages)
    compression_strategy: CompressionStrategy = CompressionStrategy.SMART

    # Caching
    enable_caching: bool = field(default_factory=lambda: settings.context_enable_caching)
    cache_ttl_seconds: int = field(default_factory=lambda: settings.context_cache_ttl_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "compression_threshold": self.compression_threshold,
            "summarization_model": self.summarization_model,
            "summarization_temperature": self.summarization_temperature,
            "summarization_timeout": self.summarization_timeout,
            "summarization_max_retries": self.summarization_max_retries,
            "keep_recent_messages": self.keep_recent_messages,
            "compression_strategy": self.compression_strategy.value,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }


@dataclass
class CompressionResult:
    """Result of context compression operation."""

    original_token_count: int
    compressed_token_count: int
    messages_before: int
    messages_after: int
    was_compressed: bool
    strategy_used: str
    compression_ratio: float
    latency_ms: float
    summarization_used: bool = False
    truncation_used: bool = False
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_token_count": self.original_token_count,
            "compressed_token_count": self.compressed_token_count,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "was_compressed": self.was_compressed,
            "strategy_used": self.strategy_used,
            "compression_ratio": round(self.compression_ratio, 3),
            "latency_ms": round(self.latency_ms, 2),
            "summarization_used": self.summarization_used,
            "truncation_used": self.truncation_used,
            "cache_hit": self.cache_hit,
        }


class SummaryCache:
    """Simple in-memory LRU cache for summaries with bounded size."""

    # Default max cache entries to prevent unbounded memory growth
    DEFAULT_MAX_SIZE = 128

    def __init__(self, ttl_seconds: int = 3600, max_size: int = DEFAULT_MAX_SIZE):
        self._cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _generate_key(self, messages: list[dict[str, Any]]) -> str:
        """Generate cache key from messages."""
        # Create hash of message content (excluding timestamps/metadata)
        content = []
        for msg in messages:
            content.append(f"{msg.get('role')}:{msg.get('content', '')[:100]}")
        content_str = "|".join(content)
        return hashlib.md5(content_str.encode()).hexdigest()

    def _evict_expired_and_lru(self) -> None:
        """Evict expired entries, then oldest if still over max_size. Must be called under lock."""
        now = time.time()
        # Remove expired entries first
        expired_keys = [
            k for k, (_, ts) in self._cache.items() if now - ts >= self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]

        # If still over limit, evict oldest entries (LRU by timestamp)
        while len(self._cache) > self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    def get(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Get cached summary if available and not expired."""
        key = self._generate_key(messages)
        with self._lock:
            if key in self._cache:
                cached_messages, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    # Refresh timestamp for LRU behavior
                    self._cache[key] = (cached_messages, time.time())
                    logger.debug(f"Cache hit for summary key: {key[:8]}")
                    return cached_messages
                else:
                    # Expired, remove
                    del self._cache[key]
        return None

    def set(self, messages: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
        """Cache a summary."""
        key = self._generate_key(messages)
        with self._lock:
            self._cache[key] = (summary, time.time())
            self._evict_expired_and_lru()
            logger.debug(f"Cached summary key: {key[:8]} (cache size: {len(self._cache)})")

    def clear(self) -> None:
        """Clear all cached summaries."""
        with self._lock:
            self._cache.clear()


class ProgressiveContextManager:
    """
    Progressive context manager with intelligent compression.

    Features:
        - Proactive compression when approaching context limits
        - Async summarization for scalability
        - Fallback to truncation on failure
        - Caching to avoid re-summarization
        - Full observability

    Example:
        ```python
        from orchestrator.llm.context_management import ProgressiveContextManager

        manager = ProgressiveContextManager()

        # Compress messages if needed
        compressed, result = await manager.compress_if_needed(
            messages=messages,
            model="gpt-4o",
            config=ContextManagementConfig()
        )
        ```
    """

    def __init__(
        self,
        config: ContextManagementConfig | None = None,
        llm_client: LLMClient | None = None,
        context_window_manager: ContextWindowManager | None = None,
    ):
        """
        Initialize progressive context manager.

        Args:
            config: Context management configuration
            llm_client: LLM client for summarization (uses global if not provided)
            context_window_manager: Context window manager (uses global if not provided)
        """
        self._config = config or ContextManagementConfig()
        self._llm_client = llm_client
        self._context_window_manager = context_window_manager or get_context_window_manager()
        self._cache = (
            SummaryCache(ttl_seconds=self._config.cache_ttl_seconds)
            if self._config.enable_caching
            else None
        )
        self._metrics = get_metrics_collector()

    @property
    def config(self) -> ContextManagementConfig:
        """Get current configuration."""
        return self._config

    def _get_llm_client(self) -> LLMClient | None:
        """Get LLM client for summarization."""
        if self._llm_client:
            return self._llm_client

        # Create a new LLMClient instance for summarization if needed
        # This avoids circular dependencies and ensures we have a client
        try:
            return LLMClient(enable_langfuse=True)
        except Exception as e:
            logger.warning(f"Failed to create LLM client for summarization: {e}")
            return None

    async def compress_if_needed(
        self,
        messages: list[dict[str, Any]],
        model: str,
        config: ContextManagementConfig | None = None,
    ) -> tuple[list[dict[str, Any]], CompressionResult]:
        """
        Compress messages if they exceed the compression threshold.

        This is the main method to use for automatic context management.
        Trace context is automatically captured from contextvars.

        Args:
            messages: Messages to check/compress
            model: Model name for context limit checking
            config: Optional config override

        Returns:
            Tuple of (compressed_messages, compression_result)
        """
        effective_config = config or self._config

        if not effective_config.enabled:
            return messages, CompressionResult(
                original_token_count=0,
                compressed_token_count=0,
                messages_before=len(messages),
                messages_after=len(messages),
                was_compressed=False,
                strategy_used="disabled",
                compression_ratio=1.0,
                latency_ms=0.0,
            )

        start_time = time.time()
        original_count = len(messages)

        # Get model limits
        limits = self._context_window_manager.get_model_limits(model)
        current_tokens = self._context_window_manager.count_tokens(messages, model)
        threshold_tokens = int(
            limits.effective_input_limit * effective_config.compression_threshold
        )

        # Check if compression needed
        if current_tokens <= threshold_tokens:
            return messages, CompressionResult(
                original_token_count=current_tokens,
                compressed_token_count=current_tokens,
                messages_before=original_count,
                messages_after=original_count,
                was_compressed=False,
                strategy_used="none",
                compression_ratio=1.0,
                latency_ms=(time.time() - start_time) * 1000,
            )

        logger.info(
            f"Context compression needed: {current_tokens} tokens "
            f"(threshold: {threshold_tokens}, limit: {limits.effective_input_limit})"
        )

        # Create span for compression operation (uses contextvars)
        async with SpanScope(
            "context.compression",
            input=truncate_data(
                {
                    "message_count": len(messages),
                    "current_tokens": current_tokens,
                    "threshold_tokens": threshold_tokens,
                    "strategy": effective_config.compression_strategy.value,
                    "model": model,
                }
            ),
            metadata={
                "compression": True,
                "strategy": effective_config.compression_strategy.value,
                "model": model,
            },
        ) as compression_span:
            try:
                # Compress based on strategy
                if effective_config.compression_strategy == CompressionStrategy.SMART:
                    compressed, result = await self._compress_smart(
                        messages=messages,
                        model=model,
                        config=effective_config,
                        current_tokens=current_tokens,
                        threshold_tokens=threshold_tokens,
                        limits=limits,
                    )
                elif effective_config.compression_strategy == CompressionStrategy.SUMMARIZE_OLD:
                    compressed, result = await self._compress_summarize(
                        messages=messages,
                        model=model,
                        config=effective_config,
                    )
                else:  # TRUNCATE_OLDEST
                    compressed, result = await self._compress_truncate(
                        messages=messages,
                        model=model,
                        config=effective_config,
                        limits=limits,
                    )

                result.latency_ms = (time.time() - start_time) * 1000
                result.original_token_count = current_tokens
                result.messages_before = original_count
                result.messages_after = len(compressed)

                # Record metrics
                self._metrics.record_latency(
                    "context_compression",
                    result.latency_ms,
                    metadata={
                        "strategy": result.strategy_used,
                        "compression_ratio": result.compression_ratio,
                        "summarization_used": result.summarization_used,
                        "cache_hit": result.cache_hit,
                    },
                )

                # Update span with results
                compression_span.set_output(
                    truncate_data(
                        {
                            "compressed_token_count": result.compressed_token_count,
                            "compression_ratio": result.compression_ratio,
                            "strategy_used": result.strategy_used,
                            "messages_before": result.messages_before,
                            "messages_after": result.messages_after,
                        }
                    )
                )
                compression_span.add_metadata("compression_ratio", result.compression_ratio)
                compression_span.add_metadata("strategy_used", result.strategy_used)

                logger.info(
                    f"Context compressed: {current_tokens} → {result.compressed_token_count} tokens "
                    f"({result.compression_ratio:.1%} ratio, {result.latency_ms:.1f}ms)"
                )

                return compressed, result

            except Exception as e:
                # Track error in metrics
                self._metrics.track_error(
                    "context_compression",
                    e,
                    metadata={
                        "strategy": effective_config.compression_strategy.value,
                        "model": model,
                        "current_tokens": current_tokens,
                    },
                )
                compression_span.set_error(str(e))
                logger.error(f"Context compression failed: {e}, falling back to truncation")

                # Fallback to truncation on any error
                try:
                    compressed, result = await self._compress_truncate(
                        messages=messages,
                        model=model,
                        config=effective_config,
                        limits=limits,
                    )
                    result.latency_ms = (time.time() - start_time) * 1000
                    result.original_token_count = current_tokens
                    result.messages_before = original_count
                    result.messages_after = len(compressed)
                    result.strategy_used = "fallback_truncate"

                    compression_span.add_metadata("fallback_used", True)
                    return compressed, result
                except Exception as fallback_error:
                    # Even truncation failed - return original messages
                    logger.error(f"Fallback truncation also failed: {fallback_error}")
                    compression_span.set_error(f"Compression and fallback failed: {fallback_error}")
                    return messages, CompressionResult(
                        original_token_count=current_tokens,
                        compressed_token_count=current_tokens,
                        messages_before=original_count,
                        messages_after=original_count,
                        was_compressed=False,
                        strategy_used="error_no_compression",
                        compression_ratio=1.0,
                        latency_ms=(time.time() - start_time) * 1000,
                    )

    async def _compress_smart(
        self,
        messages: list[dict[str, Any]],
        model: str,
        config: ContextManagementConfig,
        current_tokens: int,
        threshold_tokens: int,
        limits: Any,
    ) -> tuple[list[dict[str, Any]], CompressionResult]:
        """Smart compression: try summarization, fallback to truncation."""
        # Try summarization first
        try:
            compressed, result = await self._compress_summarize(
                messages=messages,
                model=model,
                config=config,
            )

            # Check if summarization was sufficient
            compressed_tokens = self._context_window_manager.count_tokens(compressed, model)
            if compressed_tokens <= threshold_tokens:
                result.strategy_used = "smart_summarize"
                result.compressed_token_count = compressed_tokens
                result.compression_ratio = (
                    compressed_tokens / current_tokens if current_tokens > 0 else 1.0
                )
                return compressed, result

            # Summarization wasn't enough, also truncate
            logger.debug("Summarization insufficient, applying truncation as well")
            truncated, trunc_result = await self._compress_truncate(
                messages=compressed,
                model=model,
                config=config,
                limits=limits,
            )
            result.truncation_used = True
            result.strategy_used = "smart_summarize_truncate"
            result.compressed_token_count = trunc_result.compressed_token_count
            result.compression_ratio = (
                trunc_result.compressed_token_count / current_tokens if current_tokens > 0 else 1.0
            )
            return truncated, result

        except Exception as e:
            logger.warning(f"Summarization failed, falling back to truncation: {e}")
            # Fallback to truncation
            return await self._compress_truncate(
                messages=messages,
                model=model,
                config=config,
                limits=limits,
            )

    async def _compress_summarize(
        self,
        messages: list[dict[str, Any]],
        model: str,
        config: ContextManagementConfig,
    ) -> tuple[list[dict[str, Any]], CompressionResult]:
        """Compress by summarizing older messages."""
        if not messages:
            return [], CompressionResult(
                original_token_count=0,
                compressed_token_count=0,
                messages_before=0,
                messages_after=0,
                was_compressed=False,
                strategy_used="summarize_old",
                compression_ratio=1.0,
                latency_ms=0.0,
            )

        # Separate system messages, older messages, and recent messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        conversation_messages = [m for m in messages if m.get("role") != "system"]

        if len(conversation_messages) <= config.keep_recent_messages:
            # Not enough messages to compress
            return messages, CompressionResult(
                original_token_count=0,
                compressed_token_count=0,
                messages_before=len(messages),
                messages_after=len(messages),
                was_compressed=False,
                strategy_used="summarize_old",
                compression_ratio=1.0,
                latency_ms=0.0,
            )

        # Split into older and recent
        older_messages = conversation_messages[: -config.keep_recent_messages]
        recent_messages = conversation_messages[-config.keep_recent_messages :]

        # Check cache
        summary_messages = None
        cache_hit = False
        if self._cache:
            summary_messages = self._cache.get(older_messages)
            if summary_messages:
                cache_hit = True
                logger.debug("Using cached summary")

        # Summarize if not cached
        if not summary_messages:
            summary_messages = await self._summarize_messages(
                messages=older_messages,
                model=model,
                config=config,
            )

            # Cache the summary
            if self._cache and summary_messages:
                self._cache.set(older_messages, summary_messages)

        # Combine: system + summary + recent
        compressed = system_messages + summary_messages + recent_messages

        compressed_tokens = self._context_window_manager.count_tokens(compressed, model)
        original_tokens = self._context_window_manager.count_tokens(messages, model)

        return compressed, CompressionResult(
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            messages_before=len(messages),
            messages_after=len(compressed),
            was_compressed=True,
            strategy_used="summarize_old",
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            latency_ms=0.0,  # Will be set by caller
            summarization_used=True,
            cache_hit=cache_hit,
        )

    async def _compress_truncate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        config: ContextManagementConfig,
        limits: Any,
    ) -> tuple[list[dict[str, Any]], CompressionResult]:
        """Compress by truncating oldest messages."""
        truncated, trunc_result = self._context_window_manager.truncate_messages(
            messages=messages,
            model=model,
            strategy=TruncationStrategy.KEEP_SYSTEM_AND_RECENT,
            response_buffer_percent=0.25,
        )

        compressed_tokens = self._context_window_manager.count_tokens(truncated, model)
        original_tokens = self._context_window_manager.count_tokens(messages, model)

        return truncated, CompressionResult(
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            messages_before=len(messages),
            messages_after=len(truncated),
            was_compressed=trunc_result.was_truncated,
            strategy_used="truncate_oldest",
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            latency_ms=0.0,  # Will be set by caller
            truncation_used=True,
        )

    async def _summarize_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
        config: ContextManagementConfig,
    ) -> list[dict[str, Any]]:
        """Summarize a list of messages using LLM."""
        llm_client = self._get_llm_client()
        if not llm_client:
            logger.warning("No LLM client available for summarization, using text fallback")
            return self._text_summary(messages)

        # Build summarization prompt
        conversation_text = self._format_messages_for_summary(messages)
        summary_prompt = f"""Summarize the following conversation concisely, preserving:
- Key decisions and outcomes
- Important facts and information
- User preferences and context
- Any critical details needed to continue the conversation

CONVERSATION:
{conversation_text}

SUMMARY:"""

        # Create span for summarization with full observability (uses contextvars)
        async with SpanScope(
            "context.summarization",
            input=truncate_data(
                {
                    "message_count": len(messages),
                    "model": config.summarization_model,
                    "original_model": model,
                }
            ),
            metadata={
                "compression": True,
                "original_model": model,
                "summarization_model": config.summarization_model,
            },
        ) as span:
            try:
                # Summarize with timeout
                from orchestrator.llm.config import LLMConfig

                summary_llm_config = LLMConfig(
                    model=config.summarization_model,
                    temperature=config.summarization_temperature,
                    max_tokens=1000,  # Limit summary length
                )

                summary_response = await asyncio.wait_for(
                    llm_client.chat(
                        messages=[ChatMessage(role="user", content=summary_prompt)],
                        config=summary_llm_config,
                        auto_session=False,  # Don't save summarization to session
                    ),
                    timeout=config.summarization_timeout,
                )

                summary_content = summary_response.content or ""

                # Create summary message
                summary_message = {
                    "role": "assistant",
                    "content": f"[Previous conversation summary: {summary_content}]",
                }

                span.set_output(truncate_data({"summary_length": len(summary_content)}))

                return [summary_message]

            except TimeoutError:
                logger.warning("Summarization timed out, using text fallback")
                span.set_error("Timeout")
                # Track timeout in metrics
                self._metrics.track_error(
                    "context_summarization",
                    TimeoutError("Summarization timed out"),
                    metadata={
                        "model": config.summarization_model,
                        "timeout": config.summarization_timeout,
                    },
                )
                return self._text_summary(messages)
            except Exception as e:
                logger.warning(f"Summarization failed: {e}, using text fallback")
                span.set_error(str(e))
                # Track error in metrics
                self._metrics.track_error(
                    "context_summarization",
                    e,
                    metadata={
                        "model": config.summarization_model,
                        "original_model": model,
                    },
                )
                return self._text_summary(messages)

    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for summarization prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                formatted.append(f"{role.upper()}: {content[:500]}")  # Truncate long messages
        return "\n".join(formatted)

    def _text_summary(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create a simple text-based summary without LLM."""
        summary_lines = [f"Previous conversation ({len(messages)} messages):"]
        for i, msg in enumerate(messages[:10], 1):  # First 10 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            if content:
                summary_lines.append(f"{i}. {role}: {content}...")
        if len(messages) > 10:
            summary_lines.append(f"... and {len(messages) - 10} more messages")

        return [
            {
                "role": "assistant",
                "content": "\n".join(summary_lines),
            }
        ]


# Global context manager instance
_global_context_manager: ProgressiveContextManager | None = None
_global_lock = threading.Lock()


def get_progressive_context_manager(
    config: ContextManagementConfig | None = None,
) -> ProgressiveContextManager:
    """
    Get the global progressive context manager.

    Args:
        config: Optional configuration override

    Returns:
        ProgressiveContextManager instance
    """
    global _global_context_manager

    if _global_context_manager is None:
        with _global_lock:
            if _global_context_manager is None:
                _global_context_manager = ProgressiveContextManager(config=config)

    return _global_context_manager
