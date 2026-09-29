"""A quality check that could not run is neither a pass nor a fail.

ReflectionAgent and SupervisedSequentialAgent each ask an LLM to judge a draft.
Both used to turn "the judge did not answer" into a verdict:

* ReflectionAgent treated a critique call that raised, or an empty critique
  reply, as PASS -- and logged it as passed. A broken critic approved every
  draft while the logs said each one had been checked.
* SupervisedSequentialAgent scored an unreachable supervisor, an unparseable
  reply, or a missing LLM client as 0.5 -- below the 0.7 threshold -- so the
  step was retried with "No feedback provided", or with the exception text fed
  to the worker as "Quality feedback". The missing-client path even said
  "defaulting to pass" while failing the step.

Now the draft is kept, marked as unchecked (a WARNING, and ``scored: False`` on
the supervised step span), and not retried: a retry has nothing to improve on.
An empty agent output is different -- that is a real verdict (0.0), and it retries.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.types import AgentResponse, RunContext


def _agent(name="writer"):
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name=name,
        instructions="Write.",
        config=AgentConfig(log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _runner(*contents):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=[AgentResponse(content=c) for c in contents])
    runner.save_turn = AsyncMock()
    runner.ensure_recorder = MagicMock(return_value=False)
    runner.persist_decision_trace = AsyncMock()
    return runner


def _llm(*, reply=None, raises=None):
    llm = MagicMock()
    if raises is not None:
        llm.chat = AsyncMock(side_effect=raises)
    else:
        llm.chat = AsyncMock(return_value=MagicMock(content=reply, usage=None))
    return llm


class _Logs:
    """Collect records from a continuum logger (caplog cannot: propagate=False)."""

    def __init__(self, name):
        self.name = name
        self.records: list[logging.LogRecord] = []

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

    def messages(self, level=None):
        return [r.getMessage() for r in self.records if level is None or r.levelno >= level]


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------


def _reflector():
    from continuum.agent.workflow.reflection import ReflectionAgent

    return ReflectionAgent(name="reflector", agent=_agent())


async def _reflect(llm, *drafts):
    runner = _runner(*drafts)
    with _Logs("continuum.agent.workflow.reflection") as logs:
        result = await _reflector().execute(
            "Summarise the report.", runner, RunContext(run_id="r"), llm_client=llm
        )
    return result, runner, logs


class TestReflectionWhenTheCriticCannotAnswer:
    @pytest.mark.parametrize(
        "llm",
        [
            pytest.param(_llm(raises=RuntimeError("provider down")), id="critique-raises"),
            pytest.param(_llm(reply=""), id="empty-reply"),
            pytest.param(_llm(reply="   \n"), id="whitespace-reply"),
            pytest.param(_llm(reply=None), id="no-content"),
        ],
    )
    async def test_the_draft_is_returned_unverified_not_passed(self, llm):
        result, runner, logs = await _reflect(llm, "draft 1", "draft 2", "draft 3")

        assert result.content == "draft 1"
        assert runner.run.await_count == 1, "nothing to improve on: no retry"
        assert not any("critique passed" in m for m in logs.messages()), (
            "an unanswered critique must not be logged as a pass"
        )
        assert any("unverified" in m for m in logs.messages(logging.WARNING))

    async def test_critique_reports_no_verdict_rather_than_pass(self):
        out = await _reflector()._critique(
            response_content="draft", llm_client=_llm(raises=TimeoutError())
        )
        assert out["verdict"] is None

    async def test_an_empty_reply_is_no_verdict_either(self):
        out = await _reflector()._critique(response_content="draft", llm_client=_llm(reply=""))
        assert out["verdict"] is None


class TestReflectionStillWorksWhenTheCriticAnswers:
    async def test_pass_stops_after_one_attempt(self):
        result, runner, logs = await _reflect(_llm(reply="PASS"), "draft 1", "draft 2")
        assert runner.run.await_count == 1
        assert any("critique passed" in m for m in logs.messages())

    async def test_needs_improvement_retries_with_the_feedback(self):
        llm = MagicMock()
        llm.chat = AsyncMock(
            side_effect=[
                MagicMock(content="NEEDS IMPROVEMENT: add figures", usage=None),
                MagicMock(content="PASS", usage=None),
            ]
        )
        result, runner, _ = await _reflect(llm, "draft 1", "draft 2")
        assert runner.run.await_count == 2
        assert "add figures" in runner.run.await_args_list[1].kwargs["input"]
        assert result.content == "draft 2"


# ---------------------------------------------------------------------------
# SupervisedSequentialAgent
# ---------------------------------------------------------------------------


def _span():
    span = MagicMock()
    span.__aenter__ = AsyncMock(return_value=span)
    span.__aexit__ = AsyncMock(return_value=False)
    return span


async def _supervise(llm, *outputs):
    from continuum.agent.workflow.supervised import SupervisedSequentialAgent

    sup = SupervisedSequentialAgent(name="sup", agents=[_agent("step-a")])
    runner = _runner(*outputs)
    span = _span()
    with (
        patch.object(type(sup), "_get_llm", return_value=llm),
        patch("continuum.agent.workflow.supervised.SpanScope", return_value=span),
        _Logs("continuum.agent.workflow.supervised") as logs,
    ):
        result = await sup.execute("Write a haiku.", runner, RunContext(run_id="r"))
    outputs_set = [c.args[0] for c in span.set_output.call_args_list if c.args]
    return result, runner, logs, outputs_set


class TestSupervisedWhenTheSupervisorCannotScore:
    @pytest.mark.parametrize(
        "llm",
        [
            pytest.param(_llm(raises=RuntimeError("provider down")), id="supervisor-raises"),
            pytest.param(_llm(reply="Looks decent to me."), id="no-score-line"),
            pytest.param(_llm(reply="SCORE: high\nFEEDBACK: fine"), id="score-not-a-number"),
            pytest.param(None, id="no-llm-client"),
        ],
    )
    async def test_the_output_is_kept_unscored_and_not_retried(self, llm):
        result, runner, logs, spans = await _supervise(llm, "haiku 1", "haiku 2", "haiku 3")

        assert runner.run.await_count == 1, "a retry has no feedback to act on"
        assert result.content == "haiku 1"
        assert any(s.get("scored") is False for s in spans), "the step span must say unscored"
        assert any("unscored" in m for m in logs.messages(logging.WARNING))

    async def test_an_error_is_never_fed_to_the_worker_as_feedback(self):
        _, runner, _, _ = await _supervise(
            _llm(raises=RuntimeError("SECRET-UPSTREAM-DETAIL")), "haiku 1", "haiku 2"
        )
        for call in runner.run.await_args_list:
            assert "SECRET-UPSTREAM-DETAIL" not in call.kwargs["input"]

    async def test_score_output_reports_no_score(self):
        from continuum.agent.workflow.supervised import SupervisedSequentialAgent

        sup = SupervisedSequentialAgent(name="sup", agents=[_agent()])
        score, _feedback, _usage = await sup._score_output(
            step_num=1,
            agent_name="a",
            original_input="task",
            output="text",
            llm_client=_llm(reply="no score here"),
        )
        assert score is None


class TestSupervisedStillScoresRealVerdicts:
    async def test_an_empty_output_is_a_failing_score_and_retries(self):
        """Empty output is a verdict, not an unanswered check."""
        llm = _llm(reply="SCORE: 0.9\nFEEDBACK: good")
        _, runner, _, _ = await _supervise(llm, "", "haiku 2")
        assert runner.run.await_count == 2
        assert "empty" in runner.run.await_args_list[1].kwargs["input"].lower()

    async def test_a_low_score_retries_with_the_supervisors_feedback(self):
        llm = MagicMock()
        llm.chat = AsyncMock(
            side_effect=[
                MagicMock(content="SCORE: 0.3\nFEEDBACK: needs a season word", usage=None),
                MagicMock(content="SCORE: 0.9\nFEEDBACK: good", usage=None),
            ]
        )
        result, runner, _, spans = await _supervise(llm, "haiku 1", "haiku 2")
        assert runner.run.await_count == 2
        assert "needs a season word" in runner.run.await_args_list[1].kwargs["input"]
        assert result.content == "haiku 2"
        assert any(s.get("success") is True and s.get("scored", True) for s in spans)

    async def test_a_passing_score_is_scored(self):
        _, runner, _, spans = await _supervise(_llm(reply="SCORE: 0.8\nFEEDBACK: ok"), "haiku 1")
        assert runner.run.await_count == 1
        assert any(s.get("success") is True and s.get("scored", True) for s in spans)
