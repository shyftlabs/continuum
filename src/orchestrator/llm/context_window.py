"""
Context Window Manager - Per-model context window management.

Provides automatic context window management with:
- Hardcoded model limits per provider
- Automatic truncation when approaching limits
- Buffer reservation for response tokens
- Configurable truncation strategies
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from orchestrator.logging import get_logger

logger = get_logger(__name__)


class TruncationStrategy(str, Enum):
    """Strategy for truncating messages when context limit is exceeded."""

    # Remove oldest messages first (keep recent context)
    OLDEST_FIRST = "oldest_first"

    # Keep system prompt and recent messages
    KEEP_SYSTEM_AND_RECENT = "keep_system_and_recent"

    # Smart truncation: keep system, first user message, and recent
    SMART = "smart"

    # No truncation - raise error instead
    NONE = "none"


@dataclass
class ModelLimits:
    """Context window limits for a model."""

    model: str
    max_tokens: int  # Total context window
    max_input_tokens: int | None = None  # Max input tokens (if different)
    max_output_tokens: int | None = None  # Max output tokens

    # Reserved buffer for response (percentage of max_tokens)
    response_buffer_percent: float = 0.25

    @property
    def effective_input_limit(self) -> int:
        """Get effective input token limit after reserving buffer for response."""
        if self.max_input_tokens:
            return self.max_input_tokens
        # Reserve buffer for response
        return int(self.max_tokens * (1 - self.response_buffer_percent))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "effective_input_limit": self.effective_input_limit,
            "response_buffer_percent": self.response_buffer_percent,
        }


@dataclass
class TruncationResult:
    """Result of message truncation."""

    original_token_count: int
    truncated_token_count: int
    messages_removed: int
    was_truncated: bool
    strategy_used: TruncationStrategy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_token_count": self.original_token_count,
            "truncated_token_count": self.truncated_token_count,
            "messages_removed": self.messages_removed,
            "was_truncated": self.was_truncated,
            "strategy_used": self.strategy_used.value,
        }


class ContextWindowManager:
    """
    Manages context window limits for LLM models.

    Features:
        - Auto-detects model limits from hardcoded provider tables
        - Caches model info for performance
        - Provides token counting
        - Automatic message truncation when needed
        - Configurable truncation strategies

    Example:
        ```python
        from orchestrator.llm.context_window import ContextWindowManager

        manager = ContextWindowManager()

        # Get model limits
        limits = manager.get_model_limits("gpt-4o")
        print(f"Max tokens: {limits.max_tokens}")

        # Check if messages fit
        messages = [{"role": "user", "content": "Hello"}]
        if manager.will_exceed_limit(messages, "gpt-4o"):
            messages = manager.truncate_messages(messages, "gpt-4o")

        # Or use ensure_fits which handles everything
        messages = manager.ensure_fits(messages, "gpt-4o")
        ```
    """

    # Context window limits per model (tokens)
    DEFAULT_LIMITS: dict[str, int] = {
        # OpenAI
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-turbo": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        # Anthropic
        "claude-haiku-4.5": 200000,
        "claude-sonnet-4.5": 200000,
        "claude-opus-4.5": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        # Google Gemini
        "gemini-2.5-pro": 1000000,
        "gemini-2.5-flash": 1000000,
        "gemini-2.5-flash-lite": 1000000,
        "gemini-1.5-pro": 1000000,
        "gemini-1.5-flash": 1000000,
        "gemini-pro": 32000,
    }

    def __init__(
        self,
        default_buffer_percent: float = 0.25,
        default_strategy: TruncationStrategy = TruncationStrategy.KEEP_SYSTEM_AND_RECENT,
    ):
        """
        Initialize context window manager.

        Args:
            default_buffer_percent: Default percentage to reserve for response (0.0-1.0)
            default_strategy: Default truncation strategy
        """
        self._default_buffer_percent = default_buffer_percent
        self._default_strategy = default_strategy
        self._limits_cache: dict[str, ModelLimits] = {}
        self._lock = threading.Lock()

    def get_model_limits(
        self,
        model: str,
        response_buffer_percent: float | None = None,
    ) -> ModelLimits:
        """
        Get context window limits for a model.

        Auto-detects limits from hardcoded provider tables, with fallback to defaults.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-sonnet")
            response_buffer_percent: Override default buffer percentage

        Returns:
            ModelLimits with context window information
        """
        buffer = response_buffer_percent or self._default_buffer_percent

        # Check cache first
        cache_key = f"{model}:{buffer}"
        if cache_key in self._limits_cache:
            return self._limits_cache[cache_key]

        with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._limits_cache:
                return self._limits_cache[cache_key]

            limits = self._fetch_model_limits(model, buffer)
            self._limits_cache[cache_key] = limits
            return limits

    def _fetch_model_limits(
        self,
        model: str,
        buffer_percent: float,
    ) -> ModelLimits:
        """
        Look up model limits.

        Order:
        1. Hardcoded table (fast, covers all known models)
        2. Gemini API (for unknown Gemini models — API returns inputTokenLimit)
        3. Conservative fallback: 4096
        """
        max_tokens: int | None = None
        max_input_tokens: int | None = None
        max_output_tokens: int | None = None
        model_lower = model.lower()

        # 1. Hardcoded table
        for key, default_max in self.DEFAULT_LIMITS.items():
            if key in model_lower:
                max_tokens = default_max
                logger.debug(f"Using hardcoded limit for {model}: {max_tokens}")
                break

        # 2. Gemini API for unknown Gemini models
        if max_tokens is None and any(p in model_lower for p in ("gemini", "google")):
            limits = self._fetch_gemini_limits(model)
            if limits:
                max_input_tokens, max_output_tokens = limits
                max_tokens = max_input_tokens  # total context = input limit
                logger.info(f"Fetched Gemini limits for {model}: input={max_input_tokens}, output={max_output_tokens}")

        # 3. Conservative fallback
        if max_tokens is None:
            max_tokens = 4096
            logger.warning(
                f"Unknown context limit for {model}, using conservative default: {max_tokens}"
            )

        return ModelLimits(
            model=model,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            response_buffer_percent=buffer_percent,
        )

    def _fetch_gemini_limits(self, model: str) -> tuple[int, int] | None:
        """
        Fetch token limits from Gemini model info API.

        Returns (input_token_limit, output_token_limit) or None on failure.
        """
        import json
        import urllib.error
        import urllib.request

        from orchestrator.config import settings

        api_key = settings.gemini_api_key
        if not api_key:
            return None

        # Strip provider prefix to get bare model name
        bare = model.lower()
        for prefix in ("gemini/", "google/"):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{bare}?key={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read())
            input_limit = data.get("inputTokenLimit")
            output_limit = data.get("outputTokenLimit")
            if input_limit and output_limit:
                return int(input_limit), int(output_limit)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            logger.debug(f"Gemini model info API failed for {model}: {e}")
        return None

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> int:
        """
        Count tokens in messages using tiktoken.

        Uses cl100k_base encoding (GPT-4/Claude/Gemini all use similar tokenization).
        Falls back to character estimate if tiktoken is unavailable.
        """
        try:
            import tiktoken

            # Use model-specific encoding for OpenAI models, cl100k_base for others
            try:
                enc = tiktoken.encoding_for_model(model.split("/")[-1])
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            total = 0
            for msg in messages:
                total += 4  # per-message overhead
                content = msg.get("content") or ""
                if isinstance(content, str):
                    total += len(enc.encode(content))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            total += len(enc.encode(str(block.get("text") or block.get("content") or "")))
                total += len(enc.encode(msg.get("role", "")))
            total += 2  # priming tokens
            return total
        except Exception as e:
            logger.warning(f"Token counting failed, using estimate: {e}")
            total_chars = sum(
                len(str(msg.get("content", ""))) + len(str(msg.get("role", ""))) for msg in messages
            )
            return total_chars // 4

    def will_exceed_limit(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_buffer_percent: float | None = None,
    ) -> bool:
        """
        Check if messages will exceed context window limit.

        Args:
            messages: List of messages
            model: Model name
            response_buffer_percent: Override default buffer percentage

        Returns:
            True if messages exceed limit
        """
        limits = self.get_model_limits(model, response_buffer_percent)
        token_count = self.count_tokens(messages, model)
        return token_count > limits.effective_input_limit

    def get_available_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_buffer_percent: float | None = None,
    ) -> int:
        """
        Get number of tokens available for additional content.

        Args:
            messages: Current messages
            model: Model name
            response_buffer_percent: Override default buffer percentage

        Returns:
            Number of available tokens (negative if over limit)
        """
        limits = self.get_model_limits(model, response_buffer_percent)
        token_count = self.count_tokens(messages, model)
        return limits.effective_input_limit - token_count

    def truncate_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
        strategy: TruncationStrategy | None = None,
        response_buffer_percent: float | None = None,
    ) -> tuple[list[dict[str, Any]], TruncationResult]:
        """
        Truncate messages to fit within context window.

        Args:
            messages: List of messages to truncate
            model: Model name
            strategy: Truncation strategy to use
            response_buffer_percent: Override default buffer percentage

        Returns:
            Tuple of (truncated_messages, truncation_result)

        Raises:
            ValueError: If strategy is NONE and messages exceed limit
        """
        strategy = strategy or self._default_strategy
        limits = self.get_model_limits(model, response_buffer_percent)

        original_count = self.count_tokens(messages, model)

        # Check if truncation needed
        if original_count <= limits.effective_input_limit:
            return messages, TruncationResult(
                original_token_count=original_count,
                truncated_token_count=original_count,
                messages_removed=0,
                was_truncated=False,
                strategy_used=strategy,
            )

        if strategy == TruncationStrategy.NONE:
            raise ValueError(
                f"Messages ({original_count} tokens) exceed context limit "
                f"({limits.effective_input_limit} tokens) and truncation is disabled"
            )

        # Apply truncation strategy
        if strategy == TruncationStrategy.OLDEST_FIRST:
            truncated = self._truncate_oldest_first(messages, limits, model)
        elif strategy == TruncationStrategy.KEEP_SYSTEM_AND_RECENT:
            truncated = self._truncate_keep_system_and_recent(messages, limits, model)
        elif strategy == TruncationStrategy.SMART:
            truncated = self._truncate_smart(messages, limits, model)
        else:
            truncated = messages

        truncated_count = self.count_tokens(truncated, model)

        logger.info(
            f"Truncated messages from {original_count} to {truncated_count} tokens "
            f"(removed {len(messages) - len(truncated)} messages) using {strategy.value}"
        )

        return truncated, TruncationResult(
            original_token_count=original_count,
            truncated_token_count=truncated_count,
            messages_removed=len(messages) - len(truncated),
            was_truncated=True,
            strategy_used=strategy,
        )

    def _truncate_oldest_first(
        self,
        messages: list[dict[str, Any]],
        limits: ModelLimits,
        model: str,
    ) -> list[dict[str, Any]]:
        """Remove oldest messages first until under limit."""
        result = messages.copy()

        while len(result) > 1 and self.count_tokens(result, model) > limits.effective_input_limit:
            # Remove the oldest non-system message
            for i, msg in enumerate(result):
                if msg.get("role") != "system":
                    result.pop(i)
                    break
            else:
                # Only system messages left, remove oldest
                result.pop(0)

        return result

    def _truncate_keep_system_and_recent(
        self,
        messages: list[dict[str, Any]],
        limits: ModelLimits,
        model: str,
    ) -> list[dict[str, Any]]:
        """Keep system prompt(s) and most recent messages."""
        # Separate system messages and others
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        # Start with system messages
        result = system_messages.copy()
        system_tokens = self.count_tokens(result, model)
        available = limits.effective_input_limit - system_tokens

        # Add messages from the end (most recent) until we hit the limit
        for msg in reversed(other_messages):
            msg_tokens = self.count_tokens([msg], model)
            if msg_tokens <= available:
                result.append(msg)
                available -= msg_tokens
            else:
                break

        # Restore message order (system first, then chronological)
        non_system = [m for m in result if m.get("role") != "system"]
        result = system_messages + list(reversed(non_system))

        return result

    def _truncate_smart(
        self,
        messages: list[dict[str, Any]],
        limits: ModelLimits,
        model: str,
    ) -> list[dict[str, Any]]:
        """
        Smart truncation: keep system, first user message, and recent.

        This preserves context about what the conversation started with.
        """
        # Separate by type
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        if not other_messages:
            return system_messages

        # Find first user message
        first_user_msg = None
        first_user_idx = -1
        for i, msg in enumerate(other_messages):
            if msg.get("role") == "user":
                first_user_msg = msg
                first_user_idx = i
                break

        # Calculate available tokens
        result = system_messages.copy()
        if first_user_msg:
            result.append(first_user_msg)

        used_tokens = self.count_tokens(result, model)
        available = limits.effective_input_limit - used_tokens

        # Add recent messages (excluding first user if already added)
        recent_messages = (
            other_messages[first_user_idx + 1 :] if first_user_idx >= 0 else other_messages
        )

        # Add from end
        to_add = []
        for msg in reversed(recent_messages):
            msg_tokens = self.count_tokens([msg], model)
            if msg_tokens <= available:
                to_add.append(msg)
                available -= msg_tokens
            else:
                break

        # Build final result
        result.extend(reversed(to_add))

        return result

    def ensure_fits(
        self,
        messages: list[dict[str, Any]],
        model: str,
        strategy: TruncationStrategy | None = None,
        response_buffer_percent: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ensure messages fit within context window, truncating if needed.

        This is the main method to use for automatic context management.

        Args:
            messages: Messages to check/truncate
            model: Model name
            strategy: Truncation strategy
            response_buffer_percent: Override default buffer

        Returns:
            Messages that fit within context window
        """
        truncated, result = self.truncate_messages(
            messages, model, strategy, response_buffer_percent
        )
        return truncated

    def clear_cache(self) -> None:
        """Clear the model limits cache."""
        with self._lock:
            self._limits_cache.clear()


# Global context window manager
_global_context_manager: ContextWindowManager | None = None
_global_lock = threading.Lock()


def get_context_window_manager() -> ContextWindowManager:
    """
    Get the global context window manager.

    Returns:
        ContextWindowManager instance
    """
    global _global_context_manager

    if _global_context_manager is None:
        with _global_lock:
            if _global_context_manager is None:
                _global_context_manager = ContextWindowManager()

    return _global_context_manager
