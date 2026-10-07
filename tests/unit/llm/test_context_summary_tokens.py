"""The context-summarization call is not starved of tokens.

ProgressiveContextManager._summarize_messages capped the summary call at
max_tokens=1000. The default summarization model (CONTEXT_SUMMARIZATION_MODEL)
can be a reasoning model, whose hidden reasoning counts against the cap: live on
gemini/gemini-2.5-flash (2026-09-29), summarising a 24-message conversation,
4 of 9 replies ended finish_reason=length after 826-959 hidden tokens -- the
summary that replaces the older turns stopped mid-sentence ("... Keep
dual-writes").

Now the cap is ContextManagementConfig.summarization_max_tokens, defaulting to
the normal LLM default (DEFAULT_LLM_MAX_TOKENS). The prompt still asks for a
concise summary; the cap is a guard, not the length control.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.llm.context_management import ContextManagementConfig, ProgressiveContextManager


async def _max_tokens(**config):
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="A summary.", usage=None))
    cfg = ContextManagementConfig(**config)
    manager = ProgressiveContextManager(config=cfg, llm_client=llm)
    out = await manager._summarize_messages(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "gpt-4o-mini",
        cfg,
    )
    assert "A summary." in out[0]["content"]
    return llm.chat.await_args.kwargs["config"].max_tokens


async def test_by_default_it_uses_the_normal_llm_default():
    from continuum.config import settings

    assert await _max_tokens() == settings.default_llm_max_tokens


async def test_a_cap_can_still_be_set():
    assert await _max_tokens(summarization_max_tokens=1200) == 1200


def test_the_setting_is_serialised():
    assert (
        ContextManagementConfig(summarization_max_tokens=1200).to_dict()["summarization_max_tokens"]
        == 1200
    )
    assert ContextManagementConfig().summarization_max_tokens is None
