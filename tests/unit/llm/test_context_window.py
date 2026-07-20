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


class TestCountTokensToolBlocks:
    """count_tokens must not undercount tool-call payloads.

    Regression: an Anthropic tool_use block keeps its payload under `input`
    (and tool_result under a nested `content` list), and OpenAI tool_calls
    live outside `content`. The old counter read only `text`/`content` string
    fields, counting these as ~0 tokens — so an oversized request slipped past
    compression and the provider rejected it with a 400.
    """

    def _big(self, approx_tokens: int) -> str:
        return "word " * approx_tokens

    def test_anthropic_tool_use_payload_is_counted(self):
        cwm = ContextWindowManager()
        payload = self._big(4000)
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": payload}}
                ],
            }
        ]
        # Payload is ~4000 tokens; counting it as ~0 would be the bug.
        assert cwm.count_tokens(msgs, "gpt-4o") > 3000

    def test_anthropic_tool_result_nested_content_is_counted(self):
        cwm = ContextWindowManager()
        payload = self._big(4000)
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": payload}],
                    }
                ],
            }
        ]
        assert cwm.count_tokens(msgs, "gpt-4o") > 3000

    def test_openai_tool_calls_arguments_are_counted(self):
        cwm = ContextWindowManager()
        args = self._big(4000)
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": args},
                    }
                ],
            }
        ]
        assert cwm.count_tokens(msgs, "gpt-4o") > 3000

    def test_plain_text_still_counted_normally(self):
        cwm = ContextWindowManager()
        msgs = [{"role": "user", "content": "hello world"}]
        n = cwm.count_tokens(msgs, "gpt-4o")
        assert 0 < n < 50  # small, unchanged behaviour

    def test_openai_text_block_list_unchanged(self):
        cwm = ContextWindowManager()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
        assert cwm.count_tokens(msgs, "gpt-4o") > 0


class TestTruncationToolPairBoundary:
    """truncate_messages() must not leave a tool result orphaned from its call.

    Regression: the truncation strategies drop oldest messages without regard
    to tool pairing, so the first kept turn could be a tool result whose call
    was truncated away. The provider then 400s — OpenAI on an unpaired tool
    result, Anthropic with "unexpected tool_use_id found in tool_result
    blocks". Reproduced live at ~162K tokens. The boundary must be repaired for
    both message formats.
    """

    @staticmethod
    def _anthropic_orphans(messages):
        tool_use_ids = {
            b.get("id")
            for m in messages
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        return [
            b.get("tool_use_id")
            for m in messages
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict)
            and b.get("type") == "tool_result"
            and b.get("tool_use_id") not in tool_use_ids
        ]

    def test_anthropic_tool_result_not_orphaned_by_truncation(self):
        cwm = ContextWindowManager()
        # Size turns (gpt-4 window ~8192, ~6144 effective) so the recent suffix
        # survives but the older tool_use turn is dropped — leaving the
        # tool_result as the boundary orphan unless repaired.
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "very old filler"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "s", "input": {"q": "word " * 2000}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "r"}],
            },
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "word " * 4500},
        ]
        out, res = cwm.truncate_messages(
            msgs, "gpt-4", strategy=TruncationStrategy.KEEP_SYSTEM_AND_RECENT
        )
        assert res.was_truncated
        assert self._anthropic_orphans(out) == []

    def test_openai_tool_result_not_orphaned_by_truncation(self):
        cwm = ContextWindowManager()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "very old filler"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "s", "arguments": "word " * 2000},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "word " * 4500},
        ]
        out, res = cwm.truncate_messages(
            msgs, "gpt-4", strategy=TruncationStrategy.KEEP_SYSTEM_AND_RECENT
        )
        assert res.was_truncated
        # No leading orphaned tool result (role=="tool") left after system.
        non_system = [m for m in out if m.get("role") != "system"]
        assert not (non_system and non_system[0].get("role") == "tool")

    def test_no_truncation_leaves_messages_untouched(self):
        cwm = ContextWindowManager()
        msgs = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "r"}],
            },
        ]
        out, res = cwm.truncate_messages(
            msgs, "gpt-4", strategy=TruncationStrategy.KEEP_SYSTEM_AND_RECENT
        )
        # Under the limit: nothing truncated, nothing stripped (no boundary cut).
        assert not res.was_truncated
        assert out == msgs
