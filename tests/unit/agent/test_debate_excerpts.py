"""The judge must not silently receive half the debate.

DebateAgent hands the judge an excerpt of each side, not the whole argument.
By default (summarise_arguments=False, truncate_chars=2000) each side is cut to
its first 2000 characters. A live gateway-multi-agent-shop run showed:

    DebateAgent 'debate-shop': both sides complete — pro=4056 chars, con=3436 chars
    FINAL PROMPT [food-judge]  <4383 chars>

The judge saw 49% of one argument and 58% of the other -- missing the end of
each, which is usually where an argument lands its conclusion -- and nothing in
the log said so. The one line that mentions the lengths reads as if the judge
saw all of it. The summarise path already logs what it did ("summarised
arguments — pro 4056→812 chars"); the truncation path, which is the default,
logged nothing.

The line added here carries lengths only, never the arguments, so it is safe
under LOG_PROMPT_CONTENT's rules without a wrapper.
"""

from __future__ import annotations

import logging

import pytest

from continuum.agent import BaseAgent
from continuum.agent.workflow.debate import DebateAgent, DebateConfig

SENTINEL = "ZZQX-DEBATE-CANARY"


def _debate(truncate_chars: int | None = 2000) -> DebateAgent:
    return DebateAgent(
        name="debate-shop",
        pro_agent=BaseAgent(name="pro-premium", instructions="argue for"),
        con_agent=BaseAgent(name="pro-budget", instructions="argue against"),
        judge_agent=BaseAgent(name="food-judge", instructions="judge"),
        debate_config=DebateConfig(summarise_arguments=False, truncate_chars=truncate_chars),
    )


@pytest.fixture
def records():
    captured: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    root = logging.getLogger("continuum")
    handler, level = Collector(), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield captured
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _cut_lines(records) -> list[logging.LogRecord]:
    return [r for r in records if "judge sees" in r.getMessage()]


@pytest.mark.asyncio
class TestTruncationIsReported:
    async def test_a_cut_is_logged_with_what_the_judge_actually_sees(self, records):
        """The shape of the live run: both sides over the limit."""
        await _debate()._prepare_excerpts(pro_content="p" * 4056, con_content="c" * 3436)

        lines = _cut_lines(records)
        assert len(lines) == 1, [r.getMessage() for r in records]
        message = lines[0].getMessage()
        assert "2000 of 4056" in message
        assert "2000 of 3436" in message
        assert "truncate_chars=2000" in message

    async def test_nothing_is_said_when_nothing_is_cut(self, records):
        """Arguments within the limit reach the judge whole. A line saying so on
        every debate would be noise, and would train people to skip it."""
        await _debate()._prepare_excerpts(pro_content="p" * 500, con_content="c" * 800)
        assert _cut_lines(records) == []

    async def test_one_side_cut_is_reported_honestly(self, records):
        """Only the pro side is over the limit. The line must not imply the con
        side lost anything."""
        await _debate()._prepare_excerpts(pro_content="p" * 3000, con_content="c" * 800)

        message = _cut_lines(records)[0].getMessage()
        assert "2000 of 3000" in message
        assert "800 of 800" in message

    async def test_no_limit_means_no_cut_and_no_line(self, records):
        pro, con, _ = await _debate(truncate_chars=None)._prepare_excerpts(
            pro_content="p" * 9000, con_content="c" * 9000
        )
        assert len(pro) == 9000 and len(con) == 9000
        assert _cut_lines(records) == []

    async def test_it_carries_lengths_not_the_arguments(self, records):
        """Safe to log unwrapped only because it contains no content."""
        await _debate()._prepare_excerpts(pro_content=SENTINEL * 400, con_content=SENTINEL * 400)
        assert all(SENTINEL not in r.getMessage() for r in records)

    async def test_it_is_at_info_like_its_summarise_sibling(self, records):
        """The summarise path reports at INFO; the truncation path is the same
        kind of event and should sit beside it, not louder or quieter."""
        await _debate()._prepare_excerpts(pro_content="p" * 4056, con_content="c" * 3436)
        assert _cut_lines(records)[0].levelno == logging.INFO

    async def test_the_excerpts_themselves_are_unchanged(self, records):
        """Reporting the cut must not change what the judge receives."""
        pro, con, _ = await _debate()._prepare_excerpts(
            pro_content="p" * 4056, con_content="c" * 800
        )
        assert pro == "p" * 2000 + "…"
        assert con == "c" * 800
