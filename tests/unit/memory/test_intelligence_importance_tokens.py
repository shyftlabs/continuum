"""The importance-scoring call is not starved of tokens.

IntelligentMemoryClient._score_importance capped its reply at max_tokens=16. A
reasoning model spends that on hidden reasoning first: measured live on
gemini/gemini-2.5-flash (2026-09-29), 5/5 calls for a clearly high-importance
text ("I just got promoted to VP of Engineering ...") came back empty with
finish_reason=length, so every memory silently got the 0.5 fallback.

Now the cap is configurable (IntelligenceConfig.importance_max_tokens) and
defaults to the normal LLM default (DEFAULT_LLM_MAX_TOKENS).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.memory.intelligence import IntelligenceConfig, IntelligentMemoryClient


async def _max_tokens(**config):
    client = object.__new__(IntelligentMemoryClient)
    client._intel = IntelligenceConfig(**config)
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="high", usage=None))
    assert await client._score_importance("I was promoted to VP.", llm) == 0.8
    return llm.chat.await_args.kwargs["config"].max_tokens


async def test_by_default_it_uses_the_normal_llm_default():
    from continuum.config import settings

    assert await _max_tokens() == settings.default_llm_max_tokens


async def test_a_cap_can_still_be_set():
    assert await _max_tokens(importance_max_tokens=24) == 24


def test_the_default_is_none():
    assert IntelligenceConfig().importance_max_tokens is None
