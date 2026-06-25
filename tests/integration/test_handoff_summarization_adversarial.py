"""
Adversarial integration tests for handoff history summarization.

Covers the issue #25 requirement to exercise *every* summarization mode used when
one agent hands off to another:

    FULL       - pass the entire history through untouched
    RECENT_N   - keep only the last N conversation turns
    SUMMARY    - collapse history into a single summary message (LLM or text)
    HYBRID     - summarize older turns + keep the last N verbatim

The machinery under test is ``continuum.agent.handoff.history`` — both the async
``HistorySummarizer.summarize`` (used by ``HandoffManager.prepare_handoff``) and
the sync ``summarize_conversation`` convenience wrapper. Both route turn-boundary
selection through ``_find_turn_boundary``.

Hostile angles probed here:
  - empty history, single message, history with no user turns
  - RECENT_N / HYBRID with N larger than the conversation
  - RECENT_N / HYBRID with N == 0   (boundary bug, see HSUM-01)
  - RECENT_N / HYBRID with negative N (IndexError crash, see HSUM-02)
  - SUMMARY with no LLM client (text fallback) and with an LLM that raises
  - deepcopy isolation: a returned message must not alias the input

These are pure-Python paths (no external service), but the SUMMARY LLM path is
driven with an injected fake async client so nothing real is contacted. Marked
@pytest.mark.integration per the issue's deliverable convention.

FIXED DEFECTS (these were originally asserted as xfail(strict=True); the fix in
``_find_turn_boundary`` guards ``n_turns <= 0`` by returning ``len(messages)``,
so the tests below now assert the corrected behavior directly):
  HSUM-01  RECENT_N/HYBRID with recent_turns=0 kept the WHOLE history instead of
           zero recent turns. Root cause: ``user_indices[-n_turns]`` with n==0 is
           ``user_indices[0]`` (no negative-zero index in Python), so the boundary
           collapsed to 0 == "keep everything". Fixed.
  HSUM-02  RECENT_N/HYBRID with a negative recent_turns raised IndexError instead
           of degrading gracefully. Fixed by the same ``n_turns <= 0`` guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from continuum.agent.handoff.history import (
    HistorySummarizer,
    _find_turn_boundary,
    summarize_conversation,
)
from continuum.agent.types import HistorySummarizationMode as Mode

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _convo(n_turns: int) -> list[dict[str, Any]]:
    """Build a conversation of ``n_turns`` user/assistant pairs."""
    msgs: list[dict[str, Any]] = []
    for i in range(1, n_turns + 1):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _user_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


class _FakeLLM:
    """Minimal async LLM stub for the SUMMARY path — contacts nothing real."""

    def __init__(self, content: str = "CONDENSED SUMMARY", raises: bool = False):
        self._content = content
        self._raises = raises
        self.calls = 0

    async def chat(self, messages: Any, config: Any = None, auto_session: bool = False) -> Any:
        self.calls += 1
        if self._raises:
            raise RuntimeError("summarizer LLM unavailable")
        return SimpleNamespace(content=self._content)


# --------------------------------------------------------------------------- #
# FULL
# --------------------------------------------------------------------------- #
class TestFullMode:
    async def test_full_returns_entire_history(self) -> None:
        msgs = _convo(3)
        out = await HistorySummarizer(mode=Mode.FULL).summarize(msgs)
        assert [m["content"] for m in out] == [m["content"] for m in msgs]

    async def test_full_is_a_deepcopy_not_an_alias(self) -> None:
        msgs = _convo(2)
        out = await HistorySummarizer(mode=Mode.FULL).summarize(msgs)
        out[0]["content"] = "MUTATED"
        # Mutating the handoff payload must not corrupt the caller's history.
        assert msgs[0]["content"] == "q1"

    async def test_full_empty_history(self) -> None:
        assert await HistorySummarizer(mode=Mode.FULL).summarize([]) == []


# --------------------------------------------------------------------------- #
# RECENT_N
# --------------------------------------------------------------------------- #
class TestRecentNMode:
    async def test_keeps_last_n_turns(self) -> None:
        msgs = _convo(5)  # 5 user turns, 10 messages
        out = await HistorySummarizer(mode=Mode.RECENT_N, recent_turns=2).summarize(msgs)
        # Last 2 turns => q4,a4,q5,a5
        assert [m["content"] for m in out] == ["q4", "a4", "q5", "a5"]

    async def test_n_larger_than_history_keeps_all(self) -> None:
        msgs = _convo(2)
        out = await HistorySummarizer(mode=Mode.RECENT_N, recent_turns=99).summarize(msgs)
        assert len(out) == len(msgs)

    async def test_single_message_no_user_turn(self) -> None:
        # History with no user message at all (e.g. a lone system/assistant note).
        msgs = [{"role": "assistant", "content": "preamble"}]
        out = await HistorySummarizer(mode=Mode.RECENT_N, recent_turns=1).summarize(msgs)
        assert out == msgs

    async def test_recent_turns_zero_keeps_nothing(self) -> None:
        msgs = _convo(3)
        out = await HistorySummarizer(mode=Mode.RECENT_N, recent_turns=0).summarize(msgs)
        # "Keep the last 0 turns" must not retain the entire conversation.
        assert _user_count(out) == 0

    async def test_negative_recent_turns_does_not_crash(self) -> None:
        msgs = _convo(3)
        # A negative N is invalid input; it should be clamped/handled, not crash.
        out = await HistorySummarizer(mode=Mode.RECENT_N, recent_turns=-100).summarize(msgs)
        assert isinstance(out, list)


# --------------------------------------------------------------------------- #
# SUMMARY
# --------------------------------------------------------------------------- #
class TestSummaryMode:
    async def test_summary_with_llm_collapses_to_one_message(self) -> None:
        msgs = _convo(4)
        llm = _FakeLLM(content="the gist")
        out = await HistorySummarizer(mode=Mode.SUMMARY).summarize(msgs, llm_client=llm)
        assert llm.calls == 1
        assert len(out) == 1
        assert "the gist" in out[0]["content"]

    async def test_summary_without_llm_falls_back_to_text(self) -> None:
        msgs = _convo(3)
        out = await HistorySummarizer(mode=Mode.SUMMARY).summarize(msgs, llm_client=None)
        assert len(out) == 1
        # Text fallback embeds the original turns inline.
        assert "q1" in out[0]["content"]

    async def test_summary_llm_failure_falls_back_to_text(self) -> None:
        msgs = _convo(3)
        llm = _FakeLLM(raises=True)
        out = await HistorySummarizer(mode=Mode.SUMMARY).summarize(msgs, llm_client=llm)
        # LLM raised, but the handoff must still produce a usable summary.
        assert len(out) == 1
        assert "q1" in out[0]["content"]

    async def test_summary_empty_history(self) -> None:
        out = await HistorySummarizer(mode=Mode.SUMMARY).summarize([], llm_client=None)
        assert out == []


# --------------------------------------------------------------------------- #
# HYBRID
# --------------------------------------------------------------------------- #
class TestHybridMode:
    async def test_hybrid_summary_plus_recent(self) -> None:
        msgs = _convo(5)
        llm = _FakeLLM(content="older stuff")
        out = await HistorySummarizer(mode=Mode.HYBRID, recent_turns=2).summarize(
            msgs, llm_client=llm
        )
        # 1 summary message for the older 3 turns + the last 2 turns (4 messages).
        assert out[0]["content"].find("older stuff") != -1
        assert [m["content"] for m in out[-4:]] == ["q4", "a4", "q5", "a5"]

    async def test_hybrid_n_covers_all_skips_summary(self) -> None:
        # When recent_turns covers the whole history, boundary==0 -> no summary,
        # full history returned verbatim.
        msgs = _convo(2)
        out = await HistorySummarizer(mode=Mode.HYBRID, recent_turns=10).summarize(msgs)
        assert [m["content"] for m in out] == [m["content"] for m in msgs]

    async def test_hybrid_zero_summarizes_everything(self) -> None:
        msgs = _convo(3)
        llm = _FakeLLM(content="summary")
        out = await HistorySummarizer(mode=Mode.HYBRID, recent_turns=0).summarize(
            msgs, llm_client=llm
        )
        # recent_turns=0 => keep 0 verbatim, so everything is summarized into 1 msg.
        assert len(out) == 1


# --------------------------------------------------------------------------- #
# Sync wrapper parity (summarize_conversation) — same boundary engine
# --------------------------------------------------------------------------- #
class TestSyncWrapperParity:
    def test_sync_recent_n_matches_async_semantics(self) -> None:
        msgs = _convo(4)
        out = summarize_conversation(msgs, Mode.RECENT_N, recent_turns=2)
        assert [m["content"] for m in out] == ["q3", "a3", "q4", "a4"]

    def test_sync_recent_n_zero_keeps_nothing(self) -> None:
        msgs = _convo(3)
        out = summarize_conversation(msgs, Mode.RECENT_N, recent_turns=0)
        assert _user_count(out) == 0


# --------------------------------------------------------------------------- #
# Boundary engine, isolated
# --------------------------------------------------------------------------- #
class TestTurnBoundary:
    def test_boundary_last_turn(self) -> None:
        msgs = _convo(3)  # user indices 0,2,4
        assert _find_turn_boundary(msgs, 1) == 4

    def test_boundary_n_too_large_returns_zero(self) -> None:
        msgs = _convo(2)
        assert _find_turn_boundary(msgs, 99) == 0

    def test_boundary_zero_should_be_end_of_history(self) -> None:
        msgs = _convo(3)
        # Keeping 0 turns means the boundary is at the very end (len), not the start.
        assert _find_turn_boundary(msgs, 0) == len(msgs)
