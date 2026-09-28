"""LoopAgent with TerminationType.SYSTEM_ONE_CLASSIFIER.

Today's LLM check stops when the reply CONTAINS "COMPLETE", so "INCOMPLETE" and
"NOT COMPLETE" end the loop early. A System One classifier answers "is the task
complete?" as a probability, and the loop stops when it reaches a configurable
threshold. Off unless a loop opts in; a missing backend is a construction error;
the kill switch and a failing classifier both fall back to the existing LLM check.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _agent():
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    return BaseAgent(
        name="writer",
        instructions="Write.",
        config=AgentConfig(log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _loop(**termination):
    from continuum.agent.types import TerminationConfig, TerminationType
    from continuum.agent.workflow.loop import LoopAgent

    termination.setdefault("type", TerminationType.SYSTEM_ONE_CLASSIFIER)
    return LoopAgent(name="refiner", agent=_agent(), termination=TerminationConfig(**termination))


def _use(p_complete: float | list[float]):
    from continuum.core.container import get_container

    seq = list(p_complete) if isinstance(p_complete, list) else None

    def answer(qid, q):
        p = seq.pop(0) if seq is not None else p_complete
        return {"true": p, "false": 1 - p}

    backend = FakeClassifier(answer=answer)
    get_container().set_system_one_classifier(backend)
    return backend


def _response(content="draft"):
    from continuum.agent.types import AgentResponse

    return AgentResponse(content=content)


def _llm(reply):
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content=reply))
    return llm


async def _check(loop, content="draft", llm=None, original_input="Write a haiku."):
    return await loop._check_termination(
        response=_response(content),
        iteration=1,
        history=[{"iteration": 1, "input": original_input, "output": content}],
        llm_client=llm or _llm("CONTINUE"),
        original_input=original_input,
    )


class TestOffByDefault:
    def test_the_default_termination_is_still_the_llm_decision(self):
        from continuum.agent.types import TerminationConfig, TerminationType

        assert TerminationConfig().type == TerminationType.LLM_DECISION


class TestConstruction:
    def test_opting_in_with_no_backend_is_an_error(self):
        from continuum.system_one import SystemOneNotConfiguredError

        with pytest.raises(SystemOneNotConfiguredError) as exc:
            _loop()
        assert "refiner" in str(exc.value)

    def test_the_loops_own_backend_spec_satisfies_it(self):
        _loop(system_one_backend="jev:jev-latest")

    def test_a_threshold_outside_zero_to_one_is_refused(self):
        _use(0.5)
        with pytest.raises(ValueError):
            _loop(system_one_threshold=1.5)


class TestTheDecision:
    async def test_it_stops_when_p_complete_reaches_the_threshold(self):
        _use(0.8)
        assert await _check(_loop()) is True

    async def test_it_continues_below_the_threshold(self):
        _use(0.3)
        assert await _check(_loop()) is False

    async def test_the_default_threshold_is_one_half(self):
        from continuum.agent.types import TerminationConfig

        assert TerminationConfig().system_one_threshold == 0.5

    async def test_the_threshold_is_configurable(self):
        _use(0.8)
        assert await _check(_loop(system_one_threshold=0.9)) is False

    async def test_the_classifier_sees_the_task_and_the_latest_output(self):
        backend = _use(0.8)
        await _check(_loop(), content="An old silent pond", original_input="Write a haiku.")

        state, questions = backend.calls[0]
        assert state["task"] == "Write a haiku."
        assert state["latest_output"] == "An old silent pond"
        (question,) = questions.values()
        assert question.kind == "binary"
        assert question.true_criteria, "local NLI backends need a statement to test"

    async def test_the_llm_check_is_not_called(self):
        _use(0.8)
        llm = _llm("CONTINUE")
        await _check(_loop(), llm=llm)
        llm.chat.assert_not_called()

    async def test_the_decision_is_logged_with_its_source(self, caplog):
        _use(0.8)
        with caplog.at_level(logging.INFO, logger="continuum.agent.workflow.loop"):
            await _check(_loop())
        assert any("decided_by=system_one" in r.getMessage() for r in caplog.records)


class TestFallbacks:
    async def test_a_failing_classifier_falls_back_to_the_llm_check(self):
        from continuum.core.container import get_container

        class Down(FakeClassifier):
            async def classify(self, state, questions):
                from continuum.system_one import SystemOneTimeoutError

                raise SystemOneTimeoutError("slow", backend="fake")

        get_container().set_system_one_classifier(Down())
        llm = _llm("COMPLETE")
        assert await _check(_loop(), llm=llm) is True
        llm.chat.assert_called_once()

    async def test_the_kill_switch_uses_the_llm_check(self, monkeypatch):
        from continuum.config import settings

        backend = _use(0.9)
        monkeypatch.setattr(settings, "system_one_disabled", True)
        llm = _llm("CONTINUE")

        assert await _check(_loop(), llm=llm) is False
        llm.chat.assert_called_once()
        assert backend.calls == []


class TestEndToEnd:
    async def test_the_loop_runs_until_the_classifier_says_complete(self):
        from continuum.agent.types import AgentResponse, RunContext

        _use([0.2, 0.3, 0.9])
        runner = MagicMock()
        runner.run = AsyncMock(side_effect=[AgentResponse(content=f"draft {i}") for i in (1, 2, 3)])
        runner.save_turn = AsyncMock()
        runner.ensure_recorder = MagicMock(return_value=False)
        runner.persist_decision_trace = AsyncMock()

        span = MagicMock()
        span.__aenter__ = AsyncMock(return_value=span)
        span.__aexit__ = AsyncMock(return_value=False)
        with patch("continuum.observability.trace_context.SpanScope", return_value=span):
            result = await _loop(max_iterations=5).execute(
                "Write a haiku.", runner, RunContext(run_id="r"), llm_client=_llm("CONTINUE")
            )

        assert result.turn_count == 3
        assert result.content == "draft 3"
