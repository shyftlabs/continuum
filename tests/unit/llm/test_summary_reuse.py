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


async def _turn(manager, config, turns: int, chat=None):
    messages = (chat or _chat(turns)) + [{"role": "user", "content": "latest"}]
    return await manager._compress_summarize(messages, "gpt-4o-mini", config)


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
    config = ContextManagementConfig(keep_recent_messages=1)
    manager, llm = _manager(config)

    await _turn(manager, config, 3)
    _, again = await _turn(manager, config, 3)

    assert llm.chat.await_count == 1
    assert again.cache_hit and not again.summary_reused


async def test_edited_history_is_summarized_in_full():
    config = ContextManagementConfig(keep_recent_messages=1)
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
    a = ContextManagementConfig(summarization_model="model-a", keep_recent_messages=1)
    b = ContextManagementConfig(summarization_model="model-b", keep_recent_messages=1)
    manager, llm = _manager(a)

    await _turn(manager, a, 3)
    _, result = await _turn(manager, b, 4)

    assert not result.summary_reused
    assert llm.chat.await_count == 2
