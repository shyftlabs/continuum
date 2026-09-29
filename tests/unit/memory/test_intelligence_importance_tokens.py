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

from unittest.mock import AsyncMock, MagicMock, patch

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


# The user-profile call had the same defect at max_tokens=300: live
# (gemini-2.5-flash, 2026-09-29) 3/3 replies were cut inside the JSON
# ('```json\n{"employer": "Stripe", "expertise'), so of employer, expertise,
# preferences and topics only employer was kept.


async def _profile_max_tokens(**config):
    client = object.__new__(IntelligentMemoryClient)
    client._intel = IntelligenceConfig(**config)
    client.get_user_profile = AsyncMock(return_value=None)
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="{}", usage=None))
    with patch("asyncio.sleep", AsyncMock()):
        await client._update_user_profile("u1", "I work at Stripe.", llm)
    return llm.chat.await_args.kwargs["config"].max_tokens


async def test_the_profile_call_uses_the_normal_llm_default():
    from continuum.config import settings

    assert await _profile_max_tokens() == settings.default_llm_max_tokens


async def test_the_profile_cap_can_still_be_set():
    assert await _profile_max_tokens(profile_max_tokens=600) == 600


def test_the_profile_cap_default_is_none():
    assert IntelligenceConfig().profile_max_tokens is None
