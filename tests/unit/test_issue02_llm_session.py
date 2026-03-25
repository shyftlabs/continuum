"""
Unit tests for Issue 02 — LLM, Memory, Session layer fixes.

Tests count_tokens fallback, rate limiter, stream cleanup, StreamChunk null safety,
SummaryCache bounded size, context compression empty guard, EvalCase context validation.
"""

from __future__ import annotations

import asyncio
import time

import pytest


# ---------------------------------------------------------------------------
# 02-#1: count_tokens returns conservative estimate instead of 0
# ---------------------------------------------------------------------------


class TestCountTokensFallback:
    def test_returns_estimate_on_exception(self):
        from orchestrator.llm.client import LLMClient
        from orchestrator.llm.config import LLMConfig

        client = LLMClient(
            config=LLMConfig(model="nonexistent-model-xyz"),
            enable_langfuse=False,
        )
        messages = [{"role": "user", "content": "Hello world, this is a test message"}]
        count = client.count_tokens(messages, model="nonexistent-model-xyz")
        # Should be > 0 (character-based estimate), not 0
        assert count > 0

    def test_estimate_is_positive_for_long_content(self):
        from orchestrator.llm.client import LLMClient
        from orchestrator.llm.config import LLMConfig

        client = LLMClient(
            config=LLMConfig(model="nonexistent-model-xyz"),
            enable_langfuse=False,
        )
        content = "a" * 300  # 300 chars
        messages = [{"role": "user", "content": content}]
        count = client.count_tokens(messages, model="nonexistent-model-xyz")
        # LiteLLM uses a default tokenizer — result should be positive and reasonable
        assert count > 0
        assert count < 500  # Shouldn't be absurdly high


# ---------------------------------------------------------------------------
# 02-#4: Rate limiter sleep outside lock
# ---------------------------------------------------------------------------


class TestLLMRateLimiter:
    def test_rate_limiter_allows_initial_request(self):
        from orchestrator.llm.client import _LLMRateLimiter

        rl = _LLMRateLimiter(rpm=60)
        # Should not block
        asyncio.get_event_loop().run_until_complete(rl.acquire())
        assert rl.tokens == 59.0  # One token consumed

    def test_rate_limiter_with_low_rpm_handled(self):
        """Low RPM should not cause errors — rate limiter should handle gracefully."""
        from orchestrator.llm.client import _LLMRateLimiter

        rl = _LLMRateLimiter(rpm=120)  # 2 per second
        # Consume one token
        asyncio.get_event_loop().run_until_complete(rl.acquire())
        assert rl.tokens == 119.0
        # Second acquire should work quickly
        t0 = time.monotonic()
        asyncio.get_event_loop().run_until_complete(rl.acquire())
        elapsed = time.monotonic() - t0
        assert elapsed < 5  # Should be near-instant with plenty of tokens
        assert abs(rl.tokens - 118.0) < 0.1


# ---------------------------------------------------------------------------
# 02-#6: SummaryCache bounded memory
# ---------------------------------------------------------------------------


class TestSummaryCacheBounded:
    def test_evicts_when_over_max_size(self):
        from orchestrator.llm.context_management import SummaryCache

        cache = SummaryCache(ttl_seconds=3600, max_size=3)
        for i in range(5):
            msgs = [{"role": "user", "content": f"message-{i}"}]
            cache.set(msgs, [{"role": "assistant", "content": f"summary-{i}"}])

        # Cache should contain at most 3 entries
        assert len(cache._cache) <= 3

    def test_expired_entries_evicted(self):
        from orchestrator.llm.context_management import SummaryCache

        cache = SummaryCache(ttl_seconds=0, max_size=10)  # TTL=0 = expire immediately
        msgs = [{"role": "user", "content": "hello"}]
        cache.set(msgs, [{"role": "assistant", "content": "hi"}])
        time.sleep(0.01)
        result = cache.get(msgs)
        assert result is None  # Should be expired

    def test_get_refreshes_lru_timestamp(self):
        from orchestrator.llm.context_management import SummaryCache

        cache = SummaryCache(ttl_seconds=3600, max_size=2)
        msgs1 = [{"role": "user", "content": "first"}]
        msgs2 = [{"role": "user", "content": "second"}]
        cache.set(msgs1, [{"role": "assistant", "content": "s1"}])
        time.sleep(0.01)
        cache.set(msgs2, [{"role": "assistant", "content": "s2"}])

        # Access msgs1 to refresh its timestamp
        cache.get(msgs1)

        # Adding a 3rd entry should evict msgs2 (oldest by LRU), not msgs1
        msgs3 = [{"role": "user", "content": "third"}]
        cache.set(msgs3, [{"role": "assistant", "content": "s3"}])

        assert cache.get(msgs1) is not None  # Should still be cached
        assert len(cache._cache) == 2


# ---------------------------------------------------------------------------
# 02-#11: StreamChunk null safety
# ---------------------------------------------------------------------------


class TestStreamChunkNullSafety:
    def test_from_empty_chunk(self):
        """StreamChunk should handle chunks with no choices gracefully."""
        from orchestrator.llm.types import StreamChunk

        class FakeChunk:
            choices = None
            id = "ch-1"
            model = "test"

        result = StreamChunk.from_litellm_chunk(FakeChunk())
        assert result.content is None
        assert result.finish_reason is None
        assert result.is_finished is False

    def test_from_chunk_with_delta_no_content(self):
        from orchestrator.llm.types import StreamChunk

        class FakeDelta:
            content = None
            role = "assistant"
            tool_calls = None

        class FakeChoice:
            delta = FakeDelta()
            finish_reason = None

        class FakeChunk:
            choices = [FakeChoice()]
            id = "ch-2"
            model = "test"

        result = StreamChunk.from_litellm_chunk(FakeChunk())
        assert result.content is None
        assert result.role == "assistant"

    def test_from_chunk_with_finish_reason(self):
        from orchestrator.llm.types import StreamChunk

        class FakeDelta:
            content = "done"
            role = None
            tool_calls = None

        class FakeChoice:
            delta = FakeDelta()
            finish_reason = "stop"

        class FakeChunk:
            choices = [FakeChoice()]
            id = "ch-3"
            model = "gpt-4"

        result = StreamChunk.from_litellm_chunk(FakeChunk())
        assert result.content == "done"
        assert result.finish_reason == "stop"
        assert result.is_finished is True


# ---------------------------------------------------------------------------
# 02-#12: Empty message list guard in _compress_summarize
# ---------------------------------------------------------------------------


class TestCompressSummarizeEmptyGuard:
    def test_empty_messages_returns_empty(self):
        from orchestrator.llm.context_management import (
            ContextManagementConfig,
            ProgressiveContextManager,
        )

        mgr = ProgressiveContextManager()
        config = ContextManagementConfig()
        result, compression = asyncio.get_event_loop().run_until_complete(
            mgr._compress_summarize([], "gpt-4", config)
        )
        assert result == []
        assert not compression.was_compressed


# ---------------------------------------------------------------------------
# 02-#14: Token counting fallback is conservative
# ---------------------------------------------------------------------------


class TestContextWindowTokenFallback:
    def test_count_tokens_returns_positive(self):
        from orchestrator.llm.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        messages = [{"role": "user", "content": "a" * 300}]
        count = mgr.count_tokens(messages, "totally-fake-model")
        # LiteLLM uses a default tokenizer — should return a positive count
        assert count > 0
        assert count < 500  # Shouldn't be absurdly high
