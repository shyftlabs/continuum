"""
Tests for the consecutive-handoff loop guard (executor).

A routing agent under the default return_to_parent=True can keep re-handing-off
to the same target every turn — the recipient is popped from the stack, so the
cycle/depth checks never fire, and the run silently burns turns until
MaxTurnsExceededError. The guard catches this fast with a clear HandoffLoopError.

These tests drive the REAL Executor.execute_loop with a fake LLM that always
requests a handoff, and a stub handoff executor that mimics the real one
(pushes the target and appends to run_state.handoff_chain).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig
from continuum.agent.exceptions import HandoffLoopError
from continuum.agent.execution.executor import Executor
from continuum.agent.types import (
    AgentResponse,
    Handoff,
    HandoffResult,
    ResponseStatus,
    RunContext,
    RunState,
)


def _usage():
    return SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFn(name, arguments)

    def to_dict(self):
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def _handoff_call_response(target):
    return SimpleNamespace(
        content="",
        tool_calls=[_FakeToolCall("tc", f"handoff_to_{target}", "{}")],
        usage=_usage(),
        model="m",
    )


class _StubHandoff:
    """Mimics HandoffExecutor: pushes target + records the handoff, returns success."""

    def __init__(self, return_to_parent=True):
        self._executor = None

    async def execute_handoff(self, agent, target_name, tool_call, messages, context, run_state):
        run_state.push_agent(target_name)
        run_state.handoff_chain.append({"to_agent": target_name})
        return HandoffResult(
            handoff_id="h",
            from_agent=agent.name,
            to_agent=target_name,
            success=True,
            response=AgentResponse(
                content="specialist answer", agent_name=target_name, status=ResponseStatus.SUCCESS
            ),
        )


def _triage_agent(targets=("billing",), return_to_parent=True):
    return BaseAgent(
        name="triage",
        instructions="Route customer requests.",
        config=AgentConfig(),
        handoffs=[
            Handoff(target_agent=t, description=f"route to {t}", return_to_parent=return_to_parent)
            for t in targets
        ],
    )


def _ctx(max_turns=10):
    return RunContext(run_id="run-loop", max_turns=max_turns)


def _run_state():
    rs = RunState(run_id="run-loop")
    rs.push_agent("triage")
    return rs


class TestHandoffLoopGuard:
    async def test_repeated_same_target_raises_handoff_loop_error(self):
        # LLM always asks to hand off to billing → return_to_parent=True keeps the
        # triage loop alive → guard must fire (well before max_turns=10).
        llm = SimpleNamespace(chat=AsyncMock(return_value=_handoff_call_response("billing")))
        ex = Executor(llm_client=llm, handoff_executor=_StubHandoff())

        with pytest.raises(HandoffLoopError) as exc:
            await ex.execute_loop(
                _triage_agent(("billing",)),
                [{"role": "user", "content": "refund please"}],
                _ctx(max_turns=10),
                _run_state(),
            )

        # Clear, actionable message — not a generic MaxTurnsExceededError.
        assert "billing" in str(exc.value)
        assert "return_to_parent=False" in str(exc.value)
        assert exc.value.count >= 3

    async def test_guard_fires_before_max_turns(self):
        # With a high max_turns, the guard (3) should still stop the loop early.
        llm = SimpleNamespace(chat=AsyncMock(return_value=_handoff_call_response("billing")))
        ex = Executor(llm_client=llm, handoff_executor=_StubHandoff())
        with pytest.raises(HandoffLoopError):
            await ex.execute_loop(
                _triage_agent(("billing",)),
                [{"role": "user", "content": "q"}],
                _ctx(max_turns=50),
                _run_state(),
            )
        # The stub recorded handoffs; the loop stopped at the guard threshold, far
        # below 50 turns.
        assert True


def _multi_target_llm():
    # Alternates billing, technical, billing, technical … — never 3 of the same
    # in a row, so the guard must NOT fire (it would just continue to max_turns).
    seq = ["billing", "technical"]
    calls = {"i": 0}

    async def chat(*a, **k):
        target = seq[calls["i"] % 2]
        calls["i"] += 1
        return _handoff_call_response(target)

    return SimpleNamespace(chat=chat)


class TestGuardDoesNotMisfire:
    async def test_alternating_targets_do_not_trip_guard(self):
        # Alternating targets never reach 3-in-a-row, so HandoffLoopError must NOT
        # be raised; the run ends via MaxTurnsExceededError instead (the normal cap).
        from continuum.agent.exceptions import MaxTurnsExceededError

        ex = Executor(llm_client=_multi_target_llm(), handoff_executor=_StubHandoff())
        agent = _triage_agent(("billing", "technical"))
        with pytest.raises(MaxTurnsExceededError):
            await ex.execute_loop(
                agent,
                [{"role": "user", "content": "q"}],
                _ctx(max_turns=8),
                _run_state(),
            )

    async def test_return_to_parent_false_does_not_trip_guard(self):
        # A return_to_parent=False handoff returns its result directly and ends the
        # loop, so it can never re-route and must NOT be subject to the loop guard.
        # The run completes normally with the specialist's answer — no
        # HandoffLoopError — proving the guard is scoped to return_to_parent=True.
        llm = SimpleNamespace(chat=AsyncMock(return_value=_handoff_call_response("billing")))
        ex = Executor(llm_client=llm, handoff_executor=_StubHandoff())
        agent = _triage_agent(("billing",), return_to_parent=False)

        response = await ex.execute_loop(
            agent,
            [{"role": "user", "content": "refund please"}],
            _ctx(max_turns=6),
            _run_state(),
        )

        # Returned directly from the specialist; no exception raised.
        assert response.content == "specialist answer"
