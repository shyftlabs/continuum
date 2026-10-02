"""A System One fast path in front of the reflection and supervised quality gates.

ReflectionAgent's critic and SupervisedSequentialAgent's supervisor each spend an
LLM call judging every draft. With ``verdict_mode="system_one_classifier"`` a
System One classifier is asked first, and it may only *approve*: a draft passes
without the LLM judge when P(pass) reaches ``system_one_pass_threshold``
(default 0.9). Anything else -- a lower score, a backend error, the kill switch
-- goes to the LLM judge exactly as today, so a fail and its feedback never come
from the classifier. The one new risk is a confident false pass; the default
threshold is set against that.

Measured before building (2026-09-29, openrouter:typesafe/jev-1.13, 22 hand-
written request/draft cases x 2 runs): good drafts scored 0.87-0.99, subtly bad
ones 0.01-0.13 (a missing point, 8.05 for 3.11 miles, ASC for DESC, order #4421
for #4412, "1991" for the Berlin Wall). 0 bad drafts reached 0.9; 13 of 16 good
ones did. The LLM critic (gemini-2.5-flash) was 42/44 on the same cases.

Off by default: ``verdict_mode="llm"`` asks no classifier.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.types import AgentResponse, RunContext
from tests.unit.system_one.conftest import FakeClassifier


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from continuum.config import settings
    from continuum.core.container import reset_container
    from continuum.system_one.registry import clear_classifier_cache

    monkeypatch.setattr(settings, "system_one_backend", None)
    monkeypatch.setattr(settings, "system_one_disabled", False)
    reset_container()
    clear_classifier_cache()
    yield
    clear_classifier_cache()
    reset_container()


def _use(p_pass=None, *, raises=None):
    """Put a scripted backend in the container; P(pass) = ``p_pass``."""
    from continuum.core.container import get_container
    from continuum.system_one import SystemOneBackendError

    def answer(qid, q):
        if raises:
            raise SystemOneBackendError("backend down")
        return {"true": p_pass, "false": 1 - p_pass}

    backend = FakeClassifier(answer=answer)
    get_container().set_system_one_classifier(backend)
    return backend


def _worker(name="writer"):
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
    runner.ensure_recorder = MagicMock(return_value=False)
    runner.save_turn = AsyncMock()
    runner.persist_decision_trace = AsyncMock()
    return runner


def _llm(*replies):
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=[MagicMock(content=r, usage=None) for r in replies])
    return llm


class _Logs:
    def __init__(self, name):
        self.name, self.records = name, []

    def __enter__(self):
        import logging

        outer = self

        class H(logging.Handler):
            def emit(self, record):
                outer.records.append(record)

        self._h = H()
        logging.getLogger(self.name).addHandler(self._h)
        return self

    def __exit__(self, *exc):
        import logging

        logging.getLogger(self.name).removeHandler(self._h)

    def messages(self, level=0):
        return [r.getMessage() for r in self.records if r.levelno >= level]


# =============================================================================
# ReflectionAgent
# =============================================================================


def _reflector(**config):
    from continuum.agent.config import ReflectionConfig
    from continuum.agent.workflow.reflection import ReflectionAgent

    return ReflectionAgent(
        name="reflector", agent=_worker(), reflection_config=ReflectionConfig(**config)
    )


async def _reflect(reflector, runner, llm, request="What is the capital of France?"):
    return await reflector.execute(request, runner, RunContext(run_id="r"), llm_client=llm)


class TestReflectionOffByDefault:
    def test_the_default_verdict_mode_is_llm(self):
        from continuum.agent.config import ReflectionConfig

        assert ReflectionConfig().verdict_mode == "llm"

    async def test_by_default_no_classifier_is_asked(self):
        backend = _use(0.99)
        llm = _llm("PASS")
        await _reflect(_reflector(), _runner("Paris."), llm)
        assert backend.calls == []
        llm.chat.assert_awaited_once()


class TestReflectionFastPath:
    async def test_a_confident_pass_skips_the_critic(self):
        _use(0.95)
        llm = _llm("NEEDS IMPROVEMENT: should never be asked")
        runner = _runner("Paris is the capital of France.", "unused")
        result = await _reflect(_reflector(verdict_mode="system_one_classifier"), runner, llm)
        assert result.content == "Paris is the capital of France."
        assert runner.run.await_count == 1
        llm.chat.assert_not_awaited()

    async def test_below_the_threshold_the_critic_decides(self):
        _use(0.6)
        llm = _llm("NEEDS IMPROVEMENT: name the capital", "PASS")
        runner = _runner("France is in Europe.", "Paris is the capital of France.")
        result = await _reflect(_reflector(verdict_mode="system_one_classifier"), runner, llm)
        assert llm.chat.await_count == 2, "the classifier never fails a draft on its own"
        assert "name the capital" in runner.run.await_args_list[1].kwargs["input"]
        assert result.content == "Paris is the capital of France."

    async def test_the_threshold_is_configurable(self):
        _use(0.8)
        llm = _llm("NEEDS IMPROVEMENT: x")
        await _reflect(
            _reflector(verdict_mode="system_one_classifier", system_one_pass_threshold=0.75),
            _runner("Paris."),
            llm,
        )
        llm.chat.assert_not_awaited()

    async def test_the_default_threshold_is_conservative(self):
        from continuum.agent.config import ReflectionConfig

        assert ReflectionConfig().system_one_pass_threshold == pytest.approx(0.9)
        _use(0.89)
        llm = _llm("PASS")
        await _reflect(_reflector(verdict_mode="system_one_classifier"), _runner("Paris."), llm)
        llm.chat.assert_awaited_once()

    async def test_the_classifier_sees_the_request_and_the_draft(self):
        backend = _use(0.95)
        await _reflect(_reflector(verdict_mode="system_one_classifier"), _runner("Paris."), _llm())
        state, questions = backend.calls[0]
        assert state["request"] == "What is the capital of France?"
        assert state["draft"] == "Paris."
        (question,) = questions.values()
        assert question.kind == "binary"
        assert question.true_criteria and question.false_criteria

    async def test_a_retry_is_judged_against_the_original_request(self):
        backend = _use(0.5)
        await _reflect(
            _reflector(verdict_mode="system_one_classifier"),
            _runner("France.", "Paris."),
            _llm("NEEDS IMPROVEMENT: name the capital", "PASS"),
        )
        second_state, _ = backend.calls[1]
        assert second_state["request"] == "What is the capital of France?"
        assert second_state["draft"] == "Paris."

    async def test_a_custom_critique_prompt_reaches_the_classifier(self):
        """A task-specific checklist (e.g. from generate_critique_prompt) is what
        "good enough" means here; the default prompt is only reply-format
        instructions, so it is not sent."""
        backend = _use(0.95)
        checklist = "Reply ONLY 'PASS' if the response names the capital AND its population."
        await _reflect(
            _reflector(verdict_mode="system_one_classifier", critique_prompt=checklist),
            _runner("Paris."),
            _llm(),
        )
        assert backend.calls[0][0]["review_criteria"] == checklist

        default = _use(0.95)
        await _reflect(_reflector(verdict_mode="system_one_classifier"), _runner("Paris."), _llm())
        assert "review_criteria" not in default.calls[0][0]

    async def test_the_reflectors_own_backend_is_used(self):
        own = FakeClassifier(answer=lambda qid, q: {"true": 0.97, "false": 0.03})
        default = _use(0.1)
        with patch("continuum.system_one.registry.resolve_classifier", return_value=own) as resolve:
            llm = _llm("PASS")
            await _reflect(
                _reflector(verdict_mode="system_one_classifier", system_one_backend="local:x"),
                _runner("Paris."),
                llm,
            )
        assert resolve.call_args.args[0] == "local:x"
        assert own.calls and not default.calls
        llm.chat.assert_not_awaited()


class TestReflectionFallsBackToTheCritic:
    async def test_a_backend_error_goes_to_the_critic(self):
        _use(raises=True)
        llm = _llm("PASS")
        with _Logs("continuum.agent.workflow._quality_gate") as logs:
            result = await _reflect(
                _reflector(verdict_mode="system_one_classifier"), _runner("Paris."), llm
            )
        llm.chat.assert_awaited_once()
        assert result.content == "Paris."
        import logging

        assert any("using the LLM" in m for m in logs.messages(logging.WARNING))

    async def test_the_kill_switch_goes_to_the_critic_without_asking(self, monkeypatch):
        from continuum.config import settings

        backend = _use(0.99)
        monkeypatch.setattr(settings, "system_one_disabled", True)
        llm = _llm("PASS")
        await _reflect(_reflector(verdict_mode="system_one_classifier"), _runner("Paris."), llm)
        assert backend.calls == []
        llm.chat.assert_awaited_once()

    async def test_the_last_attempt_is_still_not_judged(self):
        """With max_reflections=0 there is nothing to retry into, as today."""
        backend = _use(0.1)
        llm = _llm()
        await _reflect(
            _reflector(verdict_mode="system_one_classifier", max_reflections=0),
            _runner("Paris."),
            llm,
        )
        assert backend.calls == []
        llm.chat.assert_not_awaited()


class TestReflectionConstruction:
    def test_opting_in_with_no_backend_is_an_error(self):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError) as exc:
            _reflector(verdict_mode="system_one_classifier")
        assert "ReflectionAgent 'reflector'" in str(exc.value)

    def test_a_container_backend_or_own_spec_satisfies_it(self):
        _reflector(verdict_mode="system_one_classifier", system_one_backend="local:x")
        _use(0.5)
        _reflector(verdict_mode="system_one_classifier")

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_the_threshold_must_be_a_probability(self, threshold):
        from continuum.agent.config import ReflectionConfig

        with pytest.raises(ValueError):
            ReflectionConfig(system_one_pass_threshold=threshold)

    def test_an_unknown_verdict_mode_is_rejected(self):
        from continuum.agent.config import ReflectionConfig

        with pytest.raises(ValueError):
            ReflectionConfig(verdict_mode="jev")

    def test_the_settings_are_serialised(self):
        _use(0.5)
        d = _reflector(
            verdict_mode="system_one_classifier",
            system_one_backend="local:x",
            system_one_pass_threshold=0.95,
        ).to_dict()["reflection_config"]
        assert d["verdict_mode"] == "system_one_classifier"
        assert d["system_one_backend"] == "local:x"
        assert d["system_one_pass_threshold"] == pytest.approx(0.95)


# =============================================================================
# SupervisedSequentialAgent
# =============================================================================


def _span():
    span = MagicMock()
    span.__aenter__ = AsyncMock(return_value=span)
    span.__aexit__ = AsyncMock(return_value=False)
    return span


def _supervised(**config):
    from continuum.agent.workflow.supervised import SupervisedConfig, SupervisedSequentialAgent

    return SupervisedSequentialAgent(
        name="sup", agents=[_worker("step-a")], supervised_config=SupervisedConfig(**config)
    )


async def _supervise(sup, runner, llm, request="Write a haiku about autumn."):
    span = _span()
    with (
        patch.object(type(sup), "_get_llm", return_value=llm),
        patch("continuum.agent.workflow.supervised.SpanScope", return_value=span),
    ):
        result = await sup.execute(request, runner, RunContext(run_id="r"))
    outputs = [c.args[0] for c in span.set_output.call_args_list if c.args]
    return result, outputs


class TestSupervisedOffByDefault:
    def test_the_default_verdict_mode_is_llm(self):
        from continuum.agent.workflow.supervised import SupervisedConfig

        assert SupervisedConfig().verdict_mode == "llm"

    async def test_by_default_no_classifier_is_asked(self):
        backend = _use(0.99)
        llm = _llm("SCORE: 0.9\nFEEDBACK: good")
        await _supervise(_supervised(), _runner("haiku"), llm)
        assert backend.calls == []
        llm.chat.assert_awaited_once()


class TestSupervisedFastPath:
    async def test_a_confident_pass_skips_the_supervisor(self):
        _use(0.95)
        llm = _llm("SCORE: 0.1\nFEEDBACK: should never be asked")
        runner = _runner("haiku 1", "unused")
        result, outputs = await _supervise(
            _supervised(verdict_mode="system_one_classifier"), runner, llm
        )
        assert result.content == "haiku 1"
        assert runner.run.await_count == 1
        llm.chat.assert_not_awaited()
        step = next(o for o in outputs if "attempts" in o)
        assert step["success"] is True
        assert step["decided_by"] == "system_one"
        assert step["p_pass"] == pytest.approx(0.95)

    async def test_below_the_threshold_the_supervisor_scores_it(self):
        _use(0.5)
        llm = _llm("SCORE: 0.3\nFEEDBACK: needs a season word", "SCORE: 0.9\nFEEDBACK: good")
        runner = _runner("haiku 1", "haiku 2")
        result, _ = await _supervise(_supervised(verdict_mode="system_one_classifier"), runner, llm)
        assert llm.chat.await_count == 2
        assert "needs a season word" in runner.run.await_args_list[1].kwargs["input"]
        assert result.content == "haiku 2"

    async def test_the_classifier_sees_the_steps_task_and_output(self):
        backend = _use(0.95)
        await _supervise(
            _supervised(verdict_mode="system_one_classifier"), _runner("haiku 1"), _llm()
        )
        state, _ = backend.calls[0]
        assert state["request"] == "Write a haiku about autumn."
        assert state["draft"] == "haiku 1"

    async def test_the_threshold_is_configurable(self):
        _use(0.8)
        llm = _llm("SCORE: 0.9\nFEEDBACK: good")
        await _supervise(
            _supervised(verdict_mode="system_one_classifier", system_one_pass_threshold=0.75),
            _runner("haiku 1"),
            llm,
        )
        llm.chat.assert_not_awaited()


class TestSupervisedFallsBackToTheSupervisor:
    async def test_a_backend_error_goes_to_the_supervisor(self):
        _use(raises=True)
        llm = _llm("SCORE: 0.9\nFEEDBACK: good")
        await _supervise(_supervised(verdict_mode="system_one_classifier"), _runner("h"), llm)
        llm.chat.assert_awaited_once()

    async def test_the_kill_switch_goes_to_the_supervisor_without_asking(self, monkeypatch):
        from continuum.config import settings

        backend = _use(0.99)
        monkeypatch.setattr(settings, "system_one_disabled", True)
        llm = _llm("SCORE: 0.9\nFEEDBACK: good")
        await _supervise(_supervised(verdict_mode="system_one_classifier"), _runner("h"), llm)
        assert backend.calls == []
        llm.chat.assert_awaited_once()


class TestSupervisedConstruction:
    def test_opting_in_with_no_backend_is_an_error(self):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError) as exc:
            _supervised(verdict_mode="system_one_classifier")
        assert "SupervisedSequentialAgent 'sup'" in str(exc.value)

    @pytest.mark.parametrize("threshold", [0.0, 1.5])
    def test_the_threshold_must_be_a_probability(self, threshold):
        from continuum.agent.workflow.supervised import SupervisedConfig

        with pytest.raises(ValueError):
            SupervisedConfig(system_one_pass_threshold=threshold)

    def test_an_unknown_verdict_mode_is_rejected(self):
        from continuum.agent.workflow.supervised import SupervisedConfig

        with pytest.raises(ValueError):
            SupervisedConfig(verdict_mode="jev")

    def test_the_settings_are_serialised(self):
        from continuum.agent.workflow.supervised import SupervisedConfig

        d = SupervisedConfig(
            verdict_mode="system_one_classifier",
            system_one_backend="local:x",
            system_one_pass_threshold=0.95,
        ).to_dict()
        assert d["verdict_mode"] == "system_one_classifier"
        assert d["system_one_backend"] == "local:x"
        assert d["system_one_pass_threshold"] == pytest.approx(0.95)
