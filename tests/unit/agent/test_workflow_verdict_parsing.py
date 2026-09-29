"""Verdict parsing tolerates the usual ways a model formats its answer.

SupervisedSequentialAgent read a score only from a line starting exactly
``SCORE:``; ReflectionAgent counted a critique as passing only if it started
exactly ``PASS``. Models routinely write ``Score: 0.8``, ``**SCORE:** 0.8``,
``**PASS**`` or ``Pass.``.

That mattered more once an unreadable supervisor reply became "unscored"
(kept, reported, not retried): a model writing ``Score: 0.8`` would skip the
step's quality gate. And ``**PASS**`` read as NEEDS IMPROVEMENT, spending a
retry on a draft the critic had approved.

Tolerated: any capitalisation, markdown emphasis, a leading bullet, ``=`` for
``:``, and words after the value. Not changed: what a number means -- the score
is still read as 0.0-1.0 and clamped, as before.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _llm(reply):
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content=reply, usage=None))
    return llm


async def _score(reply):
    from continuum.agent.base import BaseAgent
    from continuum.agent.workflow.supervised import SupervisedSequentialAgent

    sup = SupervisedSequentialAgent(name="sup", agents=[BaseAgent(name="a", instructions="x")])
    score, feedback, _ = await sup._score_output(
        step_num=1, agent_name="a", original_input="task", output="text", llm_client=_llm(reply)
    )
    return score, feedback


class TestSupervisorScoreFormats:
    @pytest.mark.parametrize(
        ("reply", "score", "feedback"),
        [
            ("SCORE: 0.3\nFEEDBACK: needs a season word", 0.3, "needs a season word"),
            ("Score: 0.8\nFeedback: ok", 0.8, "ok"),
            ("score: 0.8\nfeedback: ok", 0.8, "ok"),
            ("**SCORE:** 0.8\n**FEEDBACK:** ok", 0.8, "ok"),
            ("**Score**: 0.75\n**Feedback**: fine", 0.75, "fine"),
            ("- SCORE: 0.6\n- FEEDBACK: thin", 0.6, "thin"),
            ("SCORE = 0.9\nFEEDBACK = good", 0.9, "good"),
            ("SCORE:0.9\nFEEDBACK:good", 0.9, "good"),
            ("SCORE: 0.9 (good)\nFEEDBACK: covers it", 0.9, "covers it"),
            ("Here is my evaluation.\nSCORE: 0.4\nFEEDBACK: vague", 0.4, "vague"),
            ("SCORE: .5\nFEEDBACK: half", 0.5, "half"),
        ],
    )
    async def test_the_score_and_feedback_are_read(self, reply, score, feedback):
        got_score, got_feedback = await _score(reply)
        assert got_score == pytest.approx(score)
        assert got_feedback == feedback

    async def test_out_of_range_is_still_clamped(self):
        assert (await _score("SCORE: 1.5\nFEEDBACK: x"))[0] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "reply",
        [
            "Looks decent to me.",
            "SCORE: high\nFEEDBACK: fine",
            "The score explanation: none given",
            "Scoreboard: 3\nFEEDBACK: x",
        ],
    )
    async def test_a_reply_without_a_score_is_still_unscored(self, reply):
        """No false positives: only a score field with a number counts."""
        assert (await _score(reply))[0] is None


class TestCritiqueVerdictFormats:
    @pytest.mark.parametrize(
        "verdict",
        [
            "PASS",
            "Pass.",
            "pass",
            "**PASS**",
            "PASS - all three points are covered",
            "PASSED",
            "  \n PASS",
            "> PASS",
        ],
    )
    def test_these_are_passes(self, verdict):
        from continuum.agent.workflow.reflection import is_pass_verdict

        assert is_pass_verdict(verdict) is True

    @pytest.mark.parametrize(
        "verdict",
        [
            "NEEDS IMPROVEMENT: add figures",
            "**NEEDS IMPROVEMENT**: add figures",
            "Needs improvement - missing the date",
            "Passable, but it misses the date",
            "The response is fine overall",
            "I would not pass this: it misses the date",
        ],
    )
    def test_these_are_not(self, verdict):
        from continuum.agent.workflow.reflection import is_pass_verdict

        assert is_pass_verdict(verdict) is False

    async def test_a_bold_pass_stops_after_one_attempt(self):
        """It used to read as NEEDS IMPROVEMENT and spend a retry."""
        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig, AgentMemoryConfig
        from continuum.agent.types import AgentResponse, RunContext
        from continuum.agent.workflow.reflection import ReflectionAgent

        worker = BaseAgent(
            name="writer",
            instructions="Write.",
            config=AgentConfig(log_to_session=False),
            memory_config=AgentMemoryConfig(search_memories=False),
        )
        runner = MagicMock()
        runner.run = AsyncMock(side_effect=[AgentResponse(content=f"draft {i}") for i in (1, 2, 3)])
        result = await ReflectionAgent(name="r", agent=worker).execute(
            "Summarise.", runner, RunContext(run_id="r"), llm_client=_llm("**PASS**")
        )
        assert runner.run.await_count == 1
        assert result.content == "draft 1"
