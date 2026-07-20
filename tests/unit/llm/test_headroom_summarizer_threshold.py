"""Decision #2 (second half): when Headroom is enabled, the summarizer's
trigger threshold is raised (default 0.92) so the cache-hostile summarizer
fires only as a rare last resort behind Headroom's cache-friendly per-turn
compression. `max()` semantics: a user-configured HIGHER threshold wins.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from continuum.config import settings
from continuum.llm.context_management import (
    CompressionStrategy,
    ContextManagementConfig,
    ProgressiveContextManager,
)
from continuum.llm.context_window import TruncationResult, TruncationStrategy

MESSAGES = [{"role": "user", "content": "hello"}]


def _manager(token_count: int, limit: int = 1000) -> ProgressiveContextManager:
    """Manager with a stubbed window manager: fixed limit + fixed token count."""
    cwm = MagicMock()
    cwm.get_model_limits.return_value = MagicMock(effective_input_limit=limit)
    cwm.count_tokens.return_value = token_count
    cwm.truncate_messages.return_value = (
        MESSAGES,
        TruncationResult(
            original_token_count=token_count,
            truncated_token_count=10,
            messages_removed=0,
            was_truncated=True,
            strategy_used=TruncationStrategy.KEEP_SYSTEM_AND_RECENT,
        ),
    )
    config = ContextManagementConfig(
        enabled=True,
        compression_threshold=0.8,
        compression_strategy=CompressionStrategy.TRUNCATE_OLDEST,  # no LLM call needed
    )
    return ProgressiveContextManager(config=config, context_window_manager=cwm)


class TestHeadroomRaisesSummarizerThreshold:
    async def test_between_thresholds_no_compression_when_headroom_on(self, monkeypatch):
        """850/1000 tokens: above 0.8, below 0.92 → with Headroom on, do nothing."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(settings, "headroom_context_threshold", 0.92)
        _, result = await _manager(token_count=850).compress_if_needed(MESSAGES, "gpt-4o")
        assert result.was_compressed is False

    async def test_between_thresholds_compresses_when_headroom_off(self, monkeypatch):
        """Same 850/1000: with Headroom off, the 0.8 threshold applies → compress."""
        monkeypatch.setattr(settings, "headroom_enabled", False)
        _, result = await _manager(token_count=850).compress_if_needed(MESSAGES, "gpt-4o")
        assert result.was_compressed is True

    async def test_above_raised_threshold_still_compresses(self, monkeypatch):
        """950/1000 > 0.92 → summarizer still fires even with Headroom on (safety net)."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(settings, "headroom_context_threshold", 0.92)
        _, result = await _manager(token_count=950).compress_if_needed(MESSAGES, "gpt-4o")
        assert result.was_compressed is True

    async def test_user_configured_higher_threshold_wins(self, monkeypatch):
        """max() semantics: user threshold 0.95 > bump 0.92 → 0.93 usage doesn't fire."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(settings, "headroom_context_threshold", 0.92)
        manager = _manager(token_count=930)
        manager._config.compression_threshold = 0.95
        _, result = await manager.compress_if_needed(MESSAGES, "gpt-4o")
        assert result.was_compressed is False

    async def test_bump_never_lowers_threshold(self, monkeypatch):
        """A (mis)configured low bump can't LOWER an explicit user threshold."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(settings, "headroom_context_threshold", 0.5)
        manager = _manager(token_count=700)  # 0.7: above 0.5, below 0.8
        _, result = await manager.compress_if_needed(MESSAGES, "gpt-4o")
        assert result.was_compressed is False  # user's 0.8 still governs
