"""Unit tests for LLM context window management."""

import logging

from continuum.llm.context_window import (
    ContextWindowManager,
    ModelLimits,
    TruncationResult,
    TruncationStrategy,
)

logger = logging.getLogger(__name__)


class TestModelLimits:
    def test_effective_input_limit(self):
        logger.info("ModelLimits: effective input limit")
        ml = ModelLimits(model="gpt-4", max_tokens=8192)
        assert ml.effective_input_limit == int(8192 * 0.75)

    def test_effective_input_limit_with_max_input(self):
        logger.info("ModelLimits: effective input limit with max input")
        ml = ModelLimits(model="gpt-4", max_tokens=8192, max_input_tokens=6000)
        assert ml.effective_input_limit == 6000

    def test_to_dict(self):
        logger.info("ModelLimits: to dict")
        ml = ModelLimits(model="gpt-4", max_tokens=8192)
        d = ml.to_dict()
        assert d["model"] == "gpt-4"
        assert d["max_tokens"] == 8192
        assert "effective_input_limit" in d


class TestTruncationResult:
    def test_to_dict(self):
        logger.info("TruncationResult: to dict")
        tr = TruncationResult(
            original_token_count=1000,
            truncated_token_count=500,
            messages_removed=3,
            was_truncated=True,
            strategy_used=TruncationStrategy.OLDEST_FIRST,
        )
        d = tr.to_dict()
        assert d["was_truncated"] is True
        assert d["messages_removed"] == 3
        assert d["strategy_used"] == "oldest_first"


class TestTruncationStrategy:
    def test_values(self):
        logger.info("TruncationStrategy: values")
        assert TruncationStrategy.OLDEST_FIRST == "oldest_first"
        assert TruncationStrategy.KEEP_SYSTEM_AND_RECENT == "keep_system_and_recent"
        assert TruncationStrategy.SMART == "smart"
        assert TruncationStrategy.NONE == "none"


class TestContextWindowManager:
    def test_init(self):
        logger.info("ContextWindowManager: init")
        cwm = ContextWindowManager()
        assert cwm is not None

    def test_get_model_limits(self):
        logger.info("ContextWindowManager: get model limits")
        cwm = ContextWindowManager()
        limits = cwm.get_model_limits("gpt-4")
        assert limits.max_tokens == 8192

    def test_get_model_limits_fallback(self):
        logger.info("ContextWindowManager: get model limits fallback")
        cwm = ContextWindowManager()
        limits = cwm.get_model_limits("unknown-model-xyz")
        assert limits.max_tokens > 0

    def test_new_claude_model_uses_family_default(self):
        """A Claude id not in the exact table (hyphen minor) must get 200k, not 4096."""
        cwm = ContextWindowManager()
        assert cwm.get_model_limits("claude-opus-4-8").max_tokens == 200000
        assert cwm.get_model_limits("claude-fable-5").max_tokens == 200000
        assert cwm.get_model_limits("anthropic/claude-sonnet-4-6").max_tokens == 200000

    def test_unknown_non_family_model_uses_conservative_default(self):
        """A model outside any known family still falls back to 4096."""
        cwm = ContextWindowManager()
        assert cwm.get_model_limits("totally-unknown-model-x").max_tokens == 4096

    def test_unknown_gemini_falls_back_to_family_when_api_unavailable(self, monkeypatch):
        """If the Gemini API lookup yields nothing, the gemini family default applies."""
        cwm = ContextWindowManager()
        monkeypatch.setattr(cwm, "_fetch_gemini_limits", lambda model: None)
        assert cwm.get_model_limits("gemini-9-ultra").max_tokens == 1000000
