"""
Adversarial integration tests for agent_stack cleanup on handoff FAILURE.

Issue under test:
    "agent_stack not popped on handoff failure → unbounded growth on repeated
     failures."

Background — who owns the push and the pop:
    * ``HandoffExecutor.execute_handoff`` PUSHES the target onto
      ``run_state.agent_stack`` right before it runs the target agent.
    * The caller, ``Executor.execute_loop``, owns the POP. It pops in three of
      the four post-push outcomes:
        - success + return_to_parent   → pops
        - success + no return_to_parent → returns up (run ends; harmless)
        - failure + return_to_parent   → pops
        - failure + NO return_to_parent → *** does NOT pop *** ← the leak

So the failing path is specifically: a handoff whose target raises during
execution, on a Handoff configured with ``return_to_parent=False``. The target
name is left stuck on the stack. The observable consequences:
    1. The stack does not return to its clean state after the failed hop.
    2. A retry of the SAME target is then rejected as a false "cycle", because
       cycle detection sees the stuck name still in the stack.

These tests drive the REAL ``Executor.execute_loop`` (not a stub) with a scripted
LLM client, so they exercise the true production push/pop path end to end.

No real LLM, Redis, or network — the LLM client is a MagicMock with a scripted
``chat`` coroutine.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.execution.executor import Executor
from continuum.agent.execution.handoff_executor import HandoffExecutor
from continuum.agent.handoff.manager import HandoffManager
from continuum.agent.types import Handoff, RunContext, RunState, generate_run_id

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_agent(name: str, handoffs: list[Handoff] | None = None) -> BaseAgent:
    return BaseAgent(name=name, instructions=f"I am {name}.", handoffs=handoffs or [])


class _FakeToolCall:
    """Mirrors a real LLM ToolCall: attribute access (.function.name, .id) AND
    .to_dict() so the executor can serialize it into the assistant message the
    same way it does in production."""

    def __init__(self, name: str, arguments: str, tc_id: str):
        self.function = SimpleNamespace(name=name, arguments=arguments)
        self.id = tc_id
        self.type = "function"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def _handoff_tool_call(target: str, tc_id: str = "tc-1") -> Any:
    """A scripted LLM tool call requesting handoff_to_<target>."""
    return _FakeToolCall(f"handoff_to_{target}", '{"reason": "test"}', tc_id)


def _llm_response(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    """Minimal object matching what Executor reads off an LLM response."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="test-model",
    )


def _build_executor(chat_side_effect: Any) -> tuple[Executor, HandoffExecutor]:
    """Wire a real Executor + HandoffExecutor with a scripted LLM client."""
    manager = HandoffManager(llm_client=None, max_depth=10)
    handoff_executor = HandoffExecutor(handoff_manager=manager, agent_registry={})

    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=chat_side_effect)

    # Executor.__init__ back-wires handoff_executor._executor = self.
    executor = Executor(
        llm_client=llm,
        tool_handler=None,
        handoff_executor=handoff_executor,
    )
    return executor, handoff_executor


def _run_state_with_root(root: str) -> RunState:
    """Mirror create_run_state: the root agent starts on the stack."""
    state = RunState(run_id=generate_run_id())
    state.push_agent(root)
    state.current_agent = root
    state.entry_agent = root
    return state


def _context() -> RunContext:
    return RunContext(run_id=generate_run_id(), max_turns=10)


# --------------------------------------------------------------------------- #
# The leak: failure + return_to_parent=False
# --------------------------------------------------------------------------- #


class TestHandoffFailureStackCleanup:
    async def test_failed_handoff_no_return_to_parent_leaks_stack(self) -> None:
        """
        PRIMARY repro. src hands off to dst (return_to_parent=False). dst raises
        mid-execution, so the handoff fails. After the run completes, the stack
        should be back to its clean state (['src']) — dst must not be stuck on it.
        """
        call_n = {"i": 0}

        async def chat(*args: Any, **kwargs: Any) -> Any:
            call_n["i"] += 1
            i = call_n["i"]
            if i == 1:
                # src turn 1 → request the handoff
                return _llm_response(tool_calls=[_handoff_tool_call("dst")])
            if i == 2:
                # dst turn 1 → blow up so the handoff fails AFTER the push
                raise RuntimeError("dst execution boom")
            # src turn 2 → recover with a plain answer, no more tool calls
            return _llm_response(content="recovered without the handoff")

        executor, he = _build_executor(chat)
        src = _make_agent(
            "src",
            handoffs=[Handoff(target_agent="dst", description="", return_to_parent=False)],
        )
        dst = _make_agent("dst")
        he.register_agent(src)
        he.register_agent(dst)

        state = _run_state_with_root("src")
        ctx = _context()

        response = await executor.execute_loop(
            agent=src,
            messages=[{"role": "user", "content": "hi"}],
            context=ctx,
            run_state=state,
        )

        # The run itself recovers and returns a normal response...
        assert response.content == "recovered without the handoff"
        # ...but the stack must be clean. THIS is the bug: dst is left behind.
        assert "dst" not in state.agent_stack, (
            f"dst leaked onto the stack after a failed handoff: {state.agent_stack}"
        )
        assert state.agent_stack == ["src"]

    async def test_failed_handoff_with_return_to_parent_cleans_stack(self) -> None:
        """
        CONTROL. Same failure, but return_to_parent=True. The executor's failure
        branch pops here, so the stack IS cleaned. This passing while the test
        above fails pinpoints the missing pop to the return_to_parent=False path.
        """
        call_n = {"i": 0}

        async def chat(*args: Any, **kwargs: Any) -> Any:
            call_n["i"] += 1
            i = call_n["i"]
            if i == 1:
                return _llm_response(tool_calls=[_handoff_tool_call("dst")])
            if i == 2:
                raise RuntimeError("dst execution boom")
            return _llm_response(content="recovered")

        executor, he = _build_executor(chat)
        src = _make_agent(
            "src",
            handoffs=[Handoff(target_agent="dst", description="", return_to_parent=True)],
        )
        dst = _make_agent("dst")
        he.register_agent(src)
        he.register_agent(dst)

        state = _run_state_with_root("src")
        ctx = _context()

        await executor.execute_loop(
            agent=src,
            messages=[{"role": "user", "content": "hi"}],
            context=ctx,
            run_state=state,
        )

        assert state.agent_stack == ["src"]

    async def test_repeated_failed_handoff_same_target_allows_retry(self) -> None:
        """
        CONSEQUENCE (fixed behavior). After the first failed handoff, dst is
        popped cleanly (return_to_parent=False). src then retries the SAME
        target. Because the stack is clean, the retry is NOT rejected as a
        false cycle — it is allowed to run, and this time dst succeeds.

        Before the fix, the leaked dst entry caused cycle detection to block
        this retry with a false 'cycle' error.
        """
        call_n = {"i": 0}

        async def chat(*args: Any, **kwargs: Any) -> Any:
            call_n["i"] += 1
            i = call_n["i"]
            if i == 1:
                return _llm_response(tool_calls=[_handoff_tool_call("dst", "tc-1")])
            if i == 2:
                raise RuntimeError("dst execution boom")  # first attempt fails
            if i == 3:
                return _llm_response(tool_calls=[_handoff_tool_call("dst", "tc-2")])  # retry
            return _llm_response(content="dst recovered on retry")

        executor, he = _build_executor(chat)
        src = _make_agent(
            "src",
            handoffs=[Handoff(target_agent="dst", description="", return_to_parent=False)],
        )
        dst = _make_agent("dst")
        he.register_agent(src)
        he.register_agent(dst)

        state = _run_state_with_root("src")
        ctx = _context()

        response = await executor.execute_loop(
            agent=src,
            messages=[{"role": "user", "content": "hi"}],
            context=ctx,
            run_state=state,
        )

        # No tool result should mention a cycle — the leaked-entry false positive
        # is gone now that the failed hop pops cleanly.
        tool_msgs = [m.get("content", "") for m in state.messages if m.get("role") == "tool"]
        joined = " ".join(tool_msgs).lower()
        assert "cycle" not in joined, (
            "retry was wrongly rejected as a cycle — the stack was not cleaned "
            f"after the first failure; tool messages were: {tool_msgs}"
        )
        # The retry was allowed and actually ran dst this time (proving it was
        # not blocked by a leftover stack entry). Note: a successful
        # return_to_parent=False handoff ends the run and returns dst's answer
        # directly, so the terminal stack state is not asserted here — the leak
        # itself is covered by test_failed_handoff_no_return_to_parent_leaks_stack.
        assert response.content == "dst recovered on retry"
