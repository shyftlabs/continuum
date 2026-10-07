"""An importance label is read only when the reply names exactly one.

_score_importance matched `label in reply` over the labels in list order, so
"not high, just medium" scored whichever came first in the dict, and "highly
relevant" scored "high". And a call that failed, or a reply naming no label,
was stored as importance 0.5 -- indistinguishable from a real "medium".

Now a reply is read only when exactly one label appears as a whole word
(markdown and punctuation around it are fine). Otherwise the memory is stored
without an importance score and a WARNING says so. Search re-ranking and
pruning already read a missing score as 0.5, so ranking is unchanged; what
changes is that an unscored memory no longer claims to have been scored.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.memory.intelligence import IntelligenceConfig, IntelligentMemoryClient


def _client(**config):
    client = object.__new__(IntelligentMemoryClient)
    client._intel = IntelligenceConfig(**config)
    return client


def _llm(reply=None, raises=None):
    llm = MagicMock()
    if raises is not None:
        llm.chat = AsyncMock(side_effect=raises)
    else:
        llm.chat = AsyncMock(return_value=MagicMock(content=reply, usage=None))
    return llm


class _Logs:
    def __init__(self, name):
        self.name, self.records = name, []

    def __enter__(self):
        outer = self

        class H(logging.Handler):
            def emit(self, record):
                outer.records.append(record)

        self._h = H()
        logging.getLogger(self.name).addHandler(self._h)
        return self

    def __exit__(self, *exc):
        logging.getLogger(self.name).removeHandler(self._h)

    def warnings(self):
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]


class TestTheLabelIsRead:
    @pytest.mark.parametrize(
        ("reply", "score"),
        [
            ("high", 0.8),
            ("Critical", 0.95),
            ("**critical**", 0.95),
            ("Trivial.", 0.1),
            ("low", 0.25),
            ("medium - useful context about their tools", 0.5),
        ],
    )
    async def test_one_whole_label(self, reply, score):
        assert await _client()._score_importance("text", _llm(reply)) == pytest.approx(score)


class TestAnUnreadableReplyIsUnscored:
    @pytest.mark.parametrize(
        "reply",
        ["not high, just medium", "low or medium", "highly relevant", "important", "", None],
    )
    async def test_no_score(self, reply):
        assert await _client()._score_importance("text", _llm(reply)) is None

    async def test_a_failed_call_is_no_score(self):
        assert await _client()._score_importance("text", _llm(raises=RuntimeError("down"))) is None


class TestAddStoresOnlyARealScore:
    async def _add(self, llm):
        client = _client(enable_entity_memory=False, enable_user_profiles=False)
        base_add = AsyncMock(return_value=MagicMock())
        with (
            patch.object(IntelligentMemoryClient, "_get_llm", return_value=llm),
            patch("continuum.memory.intelligence.MemoryClient.add", base_add),
            _Logs("continuum.memory.intelligence") as logs,
        ):
            await client.add("I was promoted to VP.", user_id="u1")
        return base_add.await_args.kwargs["metadata"], logs

    async def test_a_scored_memory_carries_its_importance(self):
        metadata, logs = await self._add(_llm("high"))
        assert metadata["importance"] == pytest.approx(0.8)
        assert not logs.warnings()

    @pytest.mark.parametrize(
        "llm",
        [
            pytest.param(_llm(raises=RuntimeError("down")), id="call-fails"),
            pytest.param(_llm(""), id="empty"),
            pytest.param(_llm("not high, just medium"), id="ambiguous"),
        ],
    )
    async def test_an_unscored_memory_has_no_importance_and_is_reported(self, llm):
        metadata, logs = await self._add(llm)
        assert "importance" not in metadata
        assert any("unscored" in m for m in logs.warnings())
