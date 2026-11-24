"""A long chat reuses its last summary instead of summarizing again every turn.

Each turn, the "older" part of a chat (everything before the kept recent
messages) grows by a turn, so its cache key changed and the whole older history
was summarized again: one extra model call per turn, with a prompt that grew
with the chat. Measured on a 12-turn chat, summarizing cost more than sending
the full history.

Now, on a cache miss, the summary cached for the longest earlier prefix is
reused and the messages added since follow it word for word, until more than
``summary_reuse_messages`` were added; then the older history is summarized
afresh (a full summary, so facts are not lost by rewriting a summary of a
summary).

Covers:
- prefix keys equal the full-content key of each prefix
- turns within the limit make no summary call and keep new messages verbatim
- past the limit, one fresh full summary, which later turns reuse
- an edited history, summary_reuse_messages=0 and another config namespace
  summarize in full
- reuse is off by default and without a token budget
- a reused summary must fit under the threshold (small windows summarize afresh)
- a failed summary call's text fallback is not cached or reused
- the kept messages never start inside a tool exchange
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.llm.context_management import (
    ContextManagementConfig,
    ProgressiveContextManager,
    SummaryCache,
)


def _chat(turns: int) -> list[dict]:
    msgs = []
    for i in range(1, turns + 1):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def _manager(config: ContextManagementConfig):
    llm = MagicMock()
    counter = {"n": 0}

    async def chat(messages, config, auto_session):
        counter["n"] += 1
        return MagicMock(content=f"summary v{counter['n']}", usage=None)

    llm.chat = AsyncMock(side_effect=chat)
    manager = ProgressiveContextManager(config=config, llm_client=llm)
    manager._context_window_manager = MagicMock(count_tokens=MagicMock(return_value=10))
    return manager, llm


def _prompt(llm, call: int) -> str:
    return llm.chat.await_args_list[call].kwargs["messages"][0].content


async def _turn(manager, config, turns: int, chat=None, threshold_tokens=1000):
    messages = (chat or _chat(turns)) + [{"role": "user", "content": "latest"}]
    return await manager._compress_summarize(
        messages, "gpt-4o-mini", config, threshold_tokens=threshold_tokens
    )


def test_prefix_keys_match_full_keys():
    cache = SummaryCache()
    msgs = _chat(3)
    keys = cache._prefix_keys(msgs, "ns")
    assert keys == [cache._generate_key(msgs[:n], "ns") for n in range(1, len(msgs) + 1)]


async def test_turns_within_the_limit_reuse_the_summary_without_a_call():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)

    _, first = await _turn(manager, config, 3)
    compressed, second = await _turn(manager, config, 4)
    compressed3, third = await _turn(manager, config, 5)

    assert llm.chat.await_count == 1
    assert not first.summary_reused
    assert second.summary_reused and third.summary_reused
    assert compressed[0]["content"] == "[Previous conversation summary: summary v1]"
    # turn 4 follows the summary word for word, then the latest message
    assert [m["content"] for m in compressed[1:]] == ["question 4", "answer 4", "latest"]
    assert [m["content"] for m in compressed3[1:5]] == [
        "question 4",
        "answer 4",
        "question 5",
        "answer 5",
    ]


async def test_past_the_limit_summarizes_afresh_and_reuses_that():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)  # v1 covers turns 1-3
    await _turn(manager, config, 5)  # 4 new messages: reuse
    _, sixth = await _turn(manager, config, 6)  # 6 new: fresh summary v2
    compressed, seventh = await _turn(manager, config, 7)  # reuse v2

    assert llm.chat.await_count == 2
    assert not sixth.summary_reused
    full = _prompt(llm, 1)
    assert "question 1" in full and "answer 6" in full  # a full summary, not of v1
    assert "summary v1" not in full
    assert seventh.summary_reused
    assert compressed[0]["content"] == "[Previous conversation summary: summary v2]"
    assert [m["content"] for m in compressed[1:3]] == ["question 7", "answer 7"]


async def test_same_history_is_still_a_plain_cache_hit():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)
    _, again = await _turn(manager, config, 3)

    assert llm.chat.await_count == 1
    assert again.cache_hit and not again.summary_reused


async def test_edited_history_is_summarized_in_full():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)
    edited = _chat(4)
    edited[0]["content"] = "question 1, edited"
    _, result = await _turn(manager, config, 4, chat=edited)

    assert not result.summary_reused
    assert "question 1, edited" in _prompt(llm, 1)


async def test_reuse_off_summarizes_every_turn():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=0)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)
    _, result = await _turn(manager, config, 4)

    assert llm.chat.await_count == 2
    assert not result.summary_reused


async def test_other_settings_never_reuse_a_summary():
    a = ContextManagementConfig(
        summarization_model="model-a", keep_recent_messages=1, summary_reuse_messages=4
    )
    b = ContextManagementConfig(
        summarization_model="model-b", keep_recent_messages=1, summary_reuse_messages=4
    )
    manager, llm = _manager(a)

    await _turn(manager, a, 3)
    _, result = await _turn(manager, b, 4)

    assert not result.summary_reused
    assert llm.chat.await_count == 2


def test_reuse_is_off_by_default():
    assert ContextManagementConfig().summary_reuse_messages == 0


async def test_default_config_summarizes_every_turn():
    config = ContextManagementConfig(keep_recent_messages=1)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)
    _, result = await _turn(manager, config, 4)

    assert llm.chat.await_count == 2
    assert not result.summary_reused


async def test_no_token_budget_means_no_reuse():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)
    _, result = await _turn(manager, config, 4, threshold_tokens=None)

    assert llm.chat.await_count == 2
    assert not result.summary_reused


async def test_small_window_summarizes_afresh_when_reuse_would_not_fit():
    """On an 8k-class window the reused tail can push the result over the threshold.

    Then a later truncation would drop the summary first (it is the oldest
    message) and a second compression would summarize the summary. Reuse is
    skipped and the older history is summarized afresh, as before.
    """
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)
    manager._context_window_manager = MagicMock(
        count_tokens=MagicMock(side_effect=lambda msgs, _model: 1000 * len(msgs))
    )

    await _turn(manager, config, 3, threshold_tokens=3000)
    # summary + 2 reused + latest = 4 messages = 4000 tokens > 3000
    compressed, result = await _turn(manager, config, 4, threshold_tokens=3000)

    assert not result.summary_reused
    assert llm.chat.await_count == 2
    assert compressed[0]["content"] == "[Previous conversation summary: summary v2]"
    assert len(compressed) == 2

    # with room for the tail, the same turn reuses
    _, roomy = await _turn(manager, config, 5, threshold_tokens=10_000)
    assert roomy.summary_reused


async def test_compress_if_needed_on_a_small_window_keeps_the_summary():
    """End to end through compress_if_needed: the summary is never truncated away."""
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=10)
    manager, llm = _manager(config)
    window = MagicMock()
    window.count_tokens = MagicMock(side_effect=lambda msgs, _model: 1000 * len(msgs))
    window.get_model_limits = MagicMock(return_value=MagicMock(effective_input_limit=5000))
    manager._context_window_manager = window

    reused = 0
    for turns in range(3, 9):
        messages = _chat(turns) + [{"role": "user", "content": "latest"}]
        out, result = await manager.compress_if_needed(messages, "gpt-4", config)
        reused += result.summary_reused
        assert result.was_compressed
        assert out[0]["content"].startswith("[Previous conversation summary:")
        assert not result.truncation_used
        assert window.count_tokens(out, "gpt-4") <= 4000
    assert reused  # reuse still happens when it fits


async def test_failed_summary_is_not_cached_or_reused():
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=4)
    manager, llm = _manager(config)
    llm.chat = AsyncMock(side_effect=RuntimeError("summarizer down"))

    first, _ = await _turn(manager, config, 3)
    assert first[0]["content"].startswith("Previous conversation (")  # text fallback

    # same history again: no cache hit, the model is tried again
    _, again = await _turn(manager, config, 3)
    assert not again.cache_hit
    assert llm.chat.await_count == 2

    # next turn: nothing to reuse, the model is tried again
    _, later = await _turn(manager, config, 4)
    assert not later.summary_reused
    assert llm.chat.await_count == 3


async def test_reuse_never_starts_the_kept_messages_inside_a_tool_exchange():
    """The kept messages would start with a tool result, cut off from the call
    that the cached summary covers. Providers reject that with a 400, so the
    older history is summarized afresh instead."""
    config = ContextManagementConfig(keep_recent_messages=1, summary_reuse_messages=6)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)  # summary of q1..a3
    history = _chat(3) + [
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "assistant", "content": "answer 4"},
        {"role": "user", "content": "question 5"},
        {"role": "assistant", "content": "answer 5"},
    ]
    compressed, result = await _turn(manager, config, 0, chat=history)

    assert not result.summary_reused
    assert llm.chat.await_count == 2
    assert all(m.get("role") != "tool" for m in compressed)
