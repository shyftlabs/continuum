"""The judging calls in ReflectionAgent and SupervisedSequentialAgent get what they need.

Two defects, both measured live on gemini/gemini-2.5-flash (2026-09-29):

1. **Token caps too tight for reasoning models.** The supervisor call was capped
   at max_tokens=200 and the critique at 256. A reasoning model spends the budget
   on hidden reasoning first (185-252 tokens here), so the visible reply is cut:
   at 200, 5/5 supervisor calls ended finish_reason=length with the FEEDBACK line
   lost; at 2000, 5/5 ended normally. A longer reasoning trace leaves no visible
   reply at all -- an unscored output, i.e. a skipped quality gate. 1 of 6
   critique calls at 256 was also cut ("NEEDS IMPROVEMENT: The user provided a
   statement"), rejecting a correct answer.
   Now: both caps are configurable and default to the normal LLM default
   (DEFAULT_LLM_MAX_TOKENS) instead of a hard-coded small number.

2. **The critic never saw the request.** _critique sent only the draft and the
   critique prompt, which asks whether the draft "fully answers the request".
   With truncation ruled out, a correct answer was still rejected 2 times in 6
   ("The original request or question ... was not provided"); with the request
   included, 6/6 passed. Now: the original request is sent first.
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


def _recording_llm(reply):
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content=reply, usage=None))
    return llm


def _runner(*contents):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=[AgentResponse(content=c) for c in contents])
    runner.ensure_recorder = MagicMock(return_value=False)
    runner.save_turn = AsyncMock()
    runner.persist_decision_trace = AsyncMock()
    return runner


# ---------------------------------------------------------------------------
# 1. token caps
# ---------------------------------------------------------------------------


class TestTheSupervisorCallIsNotStarvedOfTokens:
    async def _max_tokens(self, **config):
        from continuum.agent.workflow.supervised import SupervisedConfig, SupervisedSequentialAgent

        sup = SupervisedSequentialAgent(
            name="sup", agents=[_agent()], supervised_config=SupervisedConfig(**config)
        )
        llm = _recording_llm("SCORE: 0.9\nFEEDBACK: good")
        await sup._score_output(
            step_num=1, agent_name="a", original_input="task", output="text", llm_client=llm
        )
        return llm.chat.await_args.kwargs["config"].max_tokens

    async def test_by_default_it_uses_the_normal_llm_default(self):
        from continuum.config import settings

        assert await self._max_tokens() == settings.default_llm_max_tokens

    async def test_a_cap_can_still_be_set(self):
        assert await self._max_tokens(supervisor_max_tokens=500) == 500

    def test_the_setting_is_serialised(self):
        from continuum.agent.workflow.supervised import SupervisedConfig

        assert SupervisedConfig(supervisor_max_tokens=500).to_dict()["supervisor_max_tokens"] == 500
        assert SupervisedConfig().supervisor_max_tokens is None


class TestTheCritiqueCallIsNotStarvedOfTokens:
    async def _max_tokens(self, **config):
        from continuum.agent.config import ReflectionConfig
        from continuum.agent.workflow.reflection import ReflectionAgent

        ref = ReflectionAgent(
            name="r", agent=_agent(), reflection_config=ReflectionConfig(**config)
        )
        llm = _recording_llm("PASS")
        await ref._critique(response_content="draft", llm_client=llm)
        return llm.chat.await_args.kwargs["config"].max_tokens

    async def test_by_default_it_uses_the_normal_llm_default(self):
        from continuum.config import settings

        assert await self._max_tokens() == settings.default_llm_max_tokens

    async def test_a_cap_can_still_be_set(self):
        assert await self._max_tokens(reflection_max_tokens=400) == 400

    def test_the_default_is_none(self):
        from continuum.agent.config import ReflectionConfig

        assert ReflectionConfig().reflection_max_tokens is None


# ---------------------------------------------------------------------------
# 2. the critic sees the request
# ---------------------------------------------------------------------------


class TestTheCriticSeesTheRequest:
    async def test_execute_sends_the_original_request_before_the_draft(self):
        from continuum.agent.workflow.reflection import ReflectionAgent

        llm = _recording_llm("PASS")
        await ReflectionAgent(name="r", agent=_agent()).execute(
            "What is the capital of France?",
            _runner("Paris is the capital of France."),
            RunContext(run_id="r"),
            llm_client=llm,
        )
        messages = llm.chat.await_args.kwargs["messages"]
        joined = [m["content"] for m in messages]
        request_at = next(i for i, c in enumerate(joined) if "What is the capital of France?" in c)
        draft_at = next(i for i, c in enumerate(joined) if "Paris is the capital of France." in c)
        prompt_at = next(
            i for i, c in enumerate(joined) if "PASS" in c and "NEEDS IMPROVEMENT" in c
        )
        assert request_at < draft_at < prompt_at

    async def test_a_retry_still_critiques_against_the_original_request(self):
        """Attempt 2's input carries the previous draft and feedback; the critic
        must still be shown what the user asked, not that refinement input."""
        from continuum.agent.workflow.reflection import ReflectionAgent

        llm = MagicMock()
        llm.chat = AsyncMock(
            side_effect=[
                MagicMock(content="NEEDS IMPROVEMENT: add the country", usage=None),
                MagicMock(content="PASS", usage=None),
            ]
        )
        await ReflectionAgent(name="r", agent=_agent()).execute(
            "What is the capital of France?",
            _runner("Paris.", "Paris, France."),
            RunContext(run_id="r"),
            llm_client=llm,
        )
        second = [m["content"] for m in llm.chat.await_args_list[1].kwargs["messages"]]
        assert second[0].endswith("What is the capital of France?")
        assert not any("Previous attempt" in c for c in second)

    async def test_without_a_request_the_critique_is_unchanged(self):
        """Direct callers of _critique that pass no request keep today's messages."""
        from continuum.agent.workflow.reflection import ReflectionAgent

        llm = _recording_llm("PASS")
        await ReflectionAgent(name="r", agent=_agent())._critique(
            response_content="draft", llm_client=llm
        )
        messages = llm.chat.await_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "draft"


# ---------------------------------------------------------------------------
# 3. the same cap bug in the router and loop decision calls
# ---------------------------------------------------------------------------
# Measured live on gemini/gemini-2.5-flash (2026-09-29), 5 calls each:
# * LoopAgent's completion check (max_tokens=20): 5/5 empty replies
#   (finish_reason=length, 17 hidden reasoning tokens), so every check read as
#   CONTINUE -- an LLM_DECISION loop could never stop before max_iterations.
# * RouterAgent's LLM route (max_tokens=50): 5/5 routed correctly but ended
#   finish_reason=length with 43 of 50 tokens spent on hidden reasoning.


class TestTheRoutingCallIsNotStarvedOfTokens:
    async def _max_tokens(self, **config):
        from continuum.agent.config import RouterConfig
        from continuum.agent.types import Route
        from continuum.agent.workflow.router import RouterAgent

        router = RouterAgent(
            name="r",
            instructions="route",
            routes=[Route(agent_name="billing-agent", description="Billing")],
            router_config=RouterConfig(**config),
        )
        llm = _recording_llm("billing-agent")
        assert await router._llm_route("refund me", llm) == "billing-agent"
        return llm.chat.await_args.kwargs["config"].max_tokens

    async def test_by_default_it_uses_the_normal_llm_default(self):
        from continuum.config import settings

        assert await self._max_tokens() == settings.default_llm_max_tokens

    async def test_a_cap_can_still_be_set(self):
        assert await self._max_tokens(routing_max_tokens=64) == 64

    def test_the_setting_is_serialised(self):
        from continuum.agent.config import RouterConfig

        assert RouterConfig(routing_max_tokens=64).to_dict()["routing_max_tokens"] == 64
        assert RouterConfig().routing_max_tokens is None


class TestTheLoopCompletionCheckIsNotStarvedOfTokens:
    async def _max_tokens(self, **config):
        from continuum.agent.types import TerminationConfig
        from continuum.agent.workflow.loop import LoopAgent

        loop = LoopAgent(name="l", agent=_agent(), termination=TerminationConfig(**config))
        llm = _recording_llm("COMPLETE")
        done = await loop._llm_termination_check(
            AgentResponse(content="Paris."), [{"iteration": 1, "output": "Paris."}], llm
        )
        assert done is True
        return llm.chat.await_args.kwargs["config"].max_tokens

    async def test_by_default_it_uses_the_normal_llm_default(self):
        from continuum.config import settings

        assert await self._max_tokens() == settings.default_llm_max_tokens

    async def test_a_cap_can_still_be_set(self):
        assert await self._max_tokens(decision_max_tokens=32) == 32

    def test_the_default_is_none(self):
        from continuum.agent.types import TerminationConfig

        assert TerminationConfig().decision_max_tokens is None


# ---------------------------------------------------------------------------
# 4. the loop's completion check sees the task
# ---------------------------------------------------------------------------
# Once the cap was fixed, the check answered -- but judged "Paris is the capital
# of France." CONTINUE 5 times in 10 (gemini-2.5-flash, 2026-09-29): its prompt
# carried the recent outputs and never the task, so "is the task complete?" had
# nothing to be complete against. The same defect the critic had (section 2).


def _loop(**termination):
    from continuum.agent.types import TerminationConfig
    from continuum.agent.workflow.loop import LoopAgent

    return LoopAgent(name="l", agent=_agent(), termination=TerminationConfig(**termination))


def _prompt(llm, call=-1):
    return llm.chat.await_args_list[call].kwargs["messages"][0]["content"]


class TestTheLoopCompletionCheckSeesTheTask:
    async def test_execute_sends_the_original_task(self):
        llm = _recording_llm("COMPLETE")
        await _loop().execute(
            "What is the capital of France?",
            _runner("Paris is the capital of France."),
            RunContext(run_id="r"),
            llm_client=llm,
        )
        prompt = _prompt(llm)
        assert "What is the capital of France?" in prompt
        assert prompt.index("What is the capital of France?") < prompt.index("Current output")

    async def test_a_later_iteration_still_judges_against_the_original_task(self):
        """Iteration 2's input is the refinement prompt; the check must be shown
        what the user asked, not that."""
        llm = MagicMock()
        llm.chat = AsyncMock(
            side_effect=[
                MagicMock(content="CONTINUE", usage=None),
                MagicMock(content="COMPLETE", usage=None),
            ]
        )
        await _loop().execute(
            "What is the capital of France?",
            _runner("Paris.", "Paris is the capital of France."),
            RunContext(run_id="r"),
            llm_client=llm,
        )
        task_block = _prompt(llm, 1).split("Recent iterations")[0]
        assert "What is the capital of France?" in task_block
        assert "Previous output" not in task_block

    @pytest.mark.parametrize("path", ["kill_switch", "backend_error"])
    async def test_the_system_one_fallbacks_send_the_task_too(self, path):
        from continuum.agent.types import TerminationType
        from continuum.system_one import SystemOneError

        loop = _loop(type=TerminationType.SYSTEM_ONE_CLASSIFIER)
        llm = _recording_llm("COMPLETE")
        with (
            patch(
                "continuum.agent.workflow.loop.system_one_disabled",
                return_value=path == "kill_switch",
            ),
            patch.object(
                type(loop),
                "_system_one_termination_check",
                AsyncMock(side_effect=SystemOneError("backend down")),
            ),
        ):
            done = await loop._check_termination(
                response=AgentResponse(content="Paris."),
                iteration=1,
                history=[{"iteration": 1, "output": "Paris."}],
                llm_client=llm,
                original_input="What is the capital of France?",
            )
        assert done is True
        assert "What is the capital of France?" in _prompt(llm)

    async def test_without_a_task_the_prompt_is_unchanged(self):
        llm = _recording_llm("COMPLETE")
        await _loop()._llm_termination_check(
            AgentResponse(content="Paris."), [{"iteration": 1, "output": "Paris."}], llm
        )
        prompt = _prompt(llm)
        assert "Original task" not in prompt
        assert prompt.startswith(
            "Is the task complete? Respond with 'COMPLETE' if done, or 'CONTINUE'"
        )


# ---------------------------------------------------------------------------
# 5. an empty routing reply is not a route
# ---------------------------------------------------------------------------
# _llm_route matched with `result in route.agent_name.lower()`; for an empty
# reply ("" is a substring of every string) that is true for the first route, so
# a blank or truncated reply silently dispatched to whichever agent was listed
# first instead of falling back.


def _router():
    from continuum.agent.types import Route
    from continuum.agent.workflow.router import RouterAgent

    return RouterAgent(
        name="r",
        instructions="route",
        routes=[
            Route(agent_name="billing-agent", description="Billing"),
            Route(agent_name="technical-agent", description="Technical issues"),
        ],
    )


class TestAnEmptyRoutingReplyIsNotARoute:
    @pytest.mark.parametrize("reply", ["", "   \n", None], ids=["empty", "whitespace", "none"])
    async def test_it_selects_no_route(self, reply):
        assert await _router()._llm_route("my app crashes", _recording_llm(reply)) is None

    async def test_it_is_logged(self):
        from tests.unit.agent.test_workflow_check_failures import _Logs

        with _Logs("continuum.agent.workflow.router") as logs:
            await _router()._llm_route("my app crashes", _recording_llm(""))
        assert any("empty" in m for m in logs.messages(logging.WARNING))

    @pytest.mark.parametrize(
        ("reply", "route"),
        [
            ("technical-agent", "technical-agent"),
            ("billing-agent", "billing-agent"),
            ("none", None),
        ],
    )
    async def test_real_answers_still_route(self, reply, route):
        assert await _router()._llm_route("x", _recording_llm(reply)) == route
