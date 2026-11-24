"""Summary caching follows each agent's own config, and keys on the whole conversation.

get_progressive_context_manager() returns one manager per process, built with the
config of whichever agent asked first. Its summary cache was created (or not) from
that first config, so a later agent with enable_caching=False still got cached
summaries, and one with caching on got none if the first agent had it off. Measured
with two agents in one process: the agent with caching off made 22 model calls for
21 questions instead of re-summarising.

The cache key also hashed only the first 100 characters of each message, so two
conversations that start alike (a template, a pasted document) shared a summary.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.llm.context_management import (
    ContextManagementConfig,
    ProgressiveContextManager,
    SummaryCache,
)

OLDER = [
    {"role": "user", "content": "question one"},
    {"role": "assistant", "content": "answer one"},
]


def _manager(first: ContextManagementConfig):
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="A summary.", usage=None))
    manager = ProgressiveContextManager(config=first, llm_client=llm)
    manager._context_window_manager = MagicMock(count_tokens=MagicMock(return_value=10))
    return manager, llm


async def _summarise(manager, config, older=OLDER):
    messages = older + [{"role": "user", "content": "latest"}]
    await manager._compress_summarize(messages, "gpt-4o-mini", config)


async def test_agent_with_caching_off_is_not_served_from_another_agents_cache():
    cached = ContextManagementConfig(enable_caching=True, keep_recent_messages=1)
    uncached = ContextManagementConfig(enable_caching=False, keep_recent_messages=1)
    manager, llm = _manager(first=cached)

    await _summarise(manager, cached)
    await _summarise(manager, uncached)
    await _summarise(manager, uncached)

    assert llm.chat.await_count == 3


async def test_agent_with_caching_on_gets_a_cache_even_if_the_first_agent_had_none():
    uncached = ContextManagementConfig(enable_caching=False, keep_recent_messages=1)
    cached = ContextManagementConfig(enable_caching=True, keep_recent_messages=1)
    manager, llm = _manager(first=uncached)

    await _summarise(manager, cached)
    await _summarise(manager, cached)

    assert llm.chat.await_count == 1


async def test_different_summarization_settings_do_not_share_summaries():
    a = ContextManagementConfig(summarization_model="model-a", keep_recent_messages=1)
    b = ContextManagementConfig(summarization_model="model-b", keep_recent_messages=1)
    manager, llm = _manager(first=a)

    await _summarise(manager, a)
    await _summarise(manager, b)
    await _summarise(manager, a)

    assert llm.chat.await_count == 2


def test_conversations_that_differ_after_100_characters_do_not_share_a_key():
    cache = SummaryCache()
    template = "Summarise the quarterly letter for the fund below. " * 3
    fund_a = [{"role": "user", "content": template + "Fund A: 12% gross"}]
    fund_b = [{"role": "user", "content": template + "Fund B: -4% gross"}]

    cache.set(fund_a, [{"role": "assistant", "content": "about A"}])

    assert cache.get(fund_b) is None
    assert cache.get(fund_a) == [{"role": "assistant", "content": "about A"}]


def test_same_text_with_a_different_tool_exchange_is_a_different_key():
    cache = SummaryCache()
    plain = [{"role": "assistant", "content": "Looking it up."}]
    with_call = [
        {
            "role": "assistant",
            "content": "Looking it up.",
            "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}],
        }
    ]

    cache.set(plain, [{"role": "assistant", "content": "plain"}])

    assert cache.get(with_call) is None


def test_message_boundaries_cannot_collide():
    cache = SummaryCache()
    joined = [{"role": "user", "content": "a\x1fuser:b"}]
    split = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]

    cache.set(joined, [{"role": "assistant", "content": "joined"}])

    assert cache.get(split) is None


# compress_if_needed(config=None): the call falls back to the manager's own config.

LONG = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]


def _manager_over_threshold(config: ContextManagementConfig):
    manager, llm = _manager(first=config)
    window = MagicMock()
    window.count_tokens = MagicMock(side_effect=lambda msgs, _model: 1000 * len(msgs))
    window.get_model_limits = MagicMock(return_value=MagicMock(effective_input_limit=4000))
    manager._context_window_manager = window
    return manager, llm


async def test_compress_without_config_uses_the_managers_cache_setting():
    manager, llm = _manager_over_threshold(
        ContextManagementConfig(enable_caching=True, keep_recent_messages=1)
    )

    _, first = await manager.compress_if_needed(list(LONG), "gpt-4o-mini")
    _, second = await manager.compress_if_needed(list(LONG), "gpt-4o-mini")

    assert first.was_compressed and second.was_compressed
    assert llm.chat.await_count == 1
    assert second.cache_hit


async def test_compress_without_config_respects_the_managers_caching_off():
    manager, llm = _manager_over_threshold(
        ContextManagementConfig(enable_caching=False, keep_recent_messages=1)
    )

    await manager.compress_if_needed(list(LONG), "gpt-4o-mini")
    await manager.compress_if_needed(list(LONG), "gpt-4o-mini")

    assert llm.chat.await_count == 2


async def test_compress_without_config_is_off_when_the_manager_is_disabled():
    manager, llm = _manager_over_threshold(ContextManagementConfig(enabled=False))

    out, result = await manager.compress_if_needed(list(LONG), "gpt-4o-mini")

    assert out == LONG
    assert result.strategy_used == "disabled"
    llm.chat.assert_not_awaited()
