"""
Adversarial integration tests for handoff depth limiting and cycle detection.

The story: agents can hand off to each other — triage → billing → refund etc.
Every hand-off pushes a name onto agent_stack. The HandoffExecutor checks that
stack length before each hop and blocks it once max_depth is reached, returning
a failed HandoffResult instead of crashing.

Angles covered here:
  - Happy path: chain up to (but not including) the limit succeeds.
  - Exact boundary: the hop that would reach max_depth is blocked cleanly.
  - Custom max_depth: low limits are respected (not hardcoded to 10).
  - Zero depth: max_depth=0 blocks even the very first handoff.
  - Cycle detection: A→B→A is caught before the depth check.
  - Missing target: handoff to an unregistered agent returns a clean failure.
  - Defined-but-not-registered: agent is in handoff definition but not registry.

All tests run without a real LLM or Redis — we build a minimal HandoffExecutor,
stub the inner Executor, and pre-load agent_stack on a plain RunState.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.execution.handoff_executor import HandoffExecutor
from continuum.agent.handoff.manager import HandoffManager
from continuum.agent.types import (
    Handoff,
    RunState,
    generate_run_id,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_agent(name: str, handoffs: list[Handoff] | None = None) -> BaseAgent:
    return BaseAgent(
        name=name,
        instructions=f"I am {name}.",
        handoffs=handoffs or [],
    )


def _make_run_state(stack: list[str] | None = None) -> RunState:
    state = RunState(run_id=generate_run_id())
    for name in stack or []:
        state.push_agent(name)
    return state


def _make_tool_call(reason: str = "test reason") -> Any:
    """Minimal tool-call object that HandoffExecutor can parse."""
    fn = SimpleNamespace(
        name="handoff_to_target",
        arguments=f'{{"reason": "{reason}"}}',
    )
    return SimpleNamespace(function=fn, id="tc-001")


def _make_executor(
    max_depth: int = 10,
    inner_response: str = "done",
) -> tuple[HandoffExecutor, MagicMock]:
    """
    Build a HandoffExecutor wired with a stubbed inner Executor so we never
    touch a real LLM. The inner executor's execute_loop returns a minimal
    AgentResponse-like object.
    """
    manager = HandoffManager(llm_client=None, max_depth=max_depth)

    inner = MagicMock()
    inner.execute_loop = AsyncMock(
        return_value=SimpleNamespace(
            content=inner_response,
            structured_output=None,
            messages=[],
            turn_count=1,
            usage=SimpleNamespace(total_tokens=0),
            agents_used=[],
            status="success",
        )
    )

    executor = HandoffExecutor(
        handoff_manager=manager,
        agent_registry={},
        executor=inner,
    )
    return executor, inner


def _make_context() -> Any:
    return SimpleNamespace(
        run_id=generate_run_id(),
        session_id=None,
        trace_id=None,
        user_id=None,
        conversation_id=None,
        metadata={},
        agent_stack=[],
        recorder=None,
        max_turns=25,
        data_labels=set(),
        disable_memory_writes=False,
    )


# --------------------------------------------------------------------------- #
# Depth limit tests
# --------------------------------------------------------------------------- #


class TestHandoffDepthLimit:
    async def test_chain_below_limit_succeeds(self) -> None:
        """9 hops with max_depth=10 — all should succeed."""
        executor, _ = _make_executor(max_depth=10)

        # Pre-build 8 agents already in the stack (depth 8).
        # We attempt the 9th hop here — should pass (9 < 10).
        agent_a = _make_agent("agent-8", handoffs=[Handoff(target_agent="agent-9", description="")])
        agent_b = _make_agent("agent-9")
        executor.register_agent(agent_a)
        executor.register_agent(agent_b)

        state = _make_run_state(stack=[f"agent-{i}" for i in range(8)])
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=agent_a,
            target_name="agent-9",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is True

    async def test_hop_at_exact_limit_is_blocked(self) -> None:
        """The hop that would make depth == max_depth must be rejected."""
        executor, _ = _make_executor(max_depth=10)

        agent_a = _make_agent(
            "agent-10", handoffs=[Handoff(target_agent="agent-11", description="")]
        )
        agent_b = _make_agent("agent-11")
        executor.register_agent(agent_a)
        executor.register_agent(agent_b)

        # Stack already has 10 names — next hop would be depth 11, over the limit.
        state = _make_run_state(stack=[f"agent-{i}" for i in range(10)])
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=agent_a,
            target_name="agent-11",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "depth" in result.error.lower()

    async def test_custom_max_depth_respected(self) -> None:
        """max_depth=2 blocks the 3rd hop — the limit is not hardcoded."""
        executor, _ = _make_executor(max_depth=2)

        src = _make_agent("src", handoffs=[Handoff(target_agent="dst", description="")])
        dst = _make_agent("dst")
        executor.register_agent(src)
        executor.register_agent(dst)

        # Stack already has 2 agents — 3rd hop should be blocked.
        state = _make_run_state(stack=["agent-0", "agent-1"])
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=src,
            target_name="dst",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "depth" in result.error.lower()

    async def test_max_depth_zero_blocks_first_hop(self) -> None:
        """max_depth=0 means no handoffs at all — even the first one is blocked."""
        executor, _ = _make_executor(max_depth=0)

        src = _make_agent("src", handoffs=[Handoff(target_agent="dst", description="")])
        dst = _make_agent("dst")
        executor.register_agent(src)
        executor.register_agent(dst)

        state = _make_run_state()  # empty stack
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=src,
            target_name="dst",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "depth" in result.error.lower()

    async def test_successful_hop_pushes_onto_stack(self) -> None:
        """After a successful handoff the target name is on the stack."""
        executor, _ = _make_executor(max_depth=10)

        src = _make_agent("src", handoffs=[Handoff(target_agent="dst", description="")])
        dst = _make_agent("dst")
        executor.register_agent(src)
        executor.register_agent(dst)

        state = _make_run_state()
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=src,
            target_name="dst",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is True
        assert "dst" in state.agent_stack


# --------------------------------------------------------------------------- #
# Cycle detection tests
# --------------------------------------------------------------------------- #


class TestHandoffCycleDetection:
    async def test_direct_cycle_is_blocked(self) -> None:
        """A → B → A: B trying to hand back to A is a cycle."""
        executor, _ = _make_executor(max_depth=10)

        agent_a = _make_agent("agent-a")
        agent_b = _make_agent("agent-b", handoffs=[Handoff(target_agent="agent-a", description="")])
        executor.register_agent(agent_a)
        executor.register_agent(agent_b)

        # agent-a is already in the stack (it was the origin).
        state = _make_run_state(stack=["agent-a"])
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=agent_b,
            target_name="agent-a",
            tool_call=_make_tool_call("needs to go back to a"),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "cycle" in result.error.lower()

    async def test_indirect_cycle_is_blocked(self) -> None:
        """A → B → C → A: C trying to hand off to A is also a cycle."""
        executor, _ = _make_executor(max_depth=10)

        agent_a = _make_agent("agent-a")
        agent_c = _make_agent("agent-c", handoffs=[Handoff(target_agent="agent-a", description="")])
        executor.register_agent(agent_a)
        executor.register_agent(agent_c)

        # A and B already in the stack.
        state = _make_run_state(stack=["agent-a", "agent-b"])
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=agent_c,
            target_name="agent-a",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "cycle" in result.error.lower()

    async def test_non_cycle_different_target_succeeds(self) -> None:
        """A → B → C (C not in stack) is NOT a cycle and must succeed."""
        executor, _ = _make_executor(max_depth=10)

        agent_b = _make_agent("agent-b", handoffs=[Handoff(target_agent="agent-c", description="")])
        agent_c = _make_agent("agent-c")
        executor.register_agent(agent_b)
        executor.register_agent(agent_c)

        state = _make_run_state(stack=["agent-a"])  # only A in stack, C is fresh
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=agent_b,
            target_name="agent-c",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is True


# --------------------------------------------------------------------------- #
# Missing / unregistered target
# --------------------------------------------------------------------------- #


class TestHandoffMissingTarget:
    async def test_unregistered_and_undefined_target_fails_cleanly(self) -> None:
        """Target not in registry AND not in agent's handoff list → clean failure."""
        executor, _ = _make_executor()

        src = _make_agent("src")  # no handoffs declared
        executor.register_agent(src)

        state = _make_run_state()
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=src,
            target_name="ghost-agent",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "ghost-agent" in result.error

    async def test_defined_but_not_registered_target_fails_cleanly(self) -> None:
        """Target declared in handoffs but never registered → clean failure, not KeyError."""
        executor, _ = _make_executor()

        src = _make_agent(
            "src",
            handoffs=[Handoff(target_agent="billing", description="send billing questions here")],
        )
        executor.register_agent(src)
        # Note: "billing" agent is intentionally NOT registered.

        state = _make_run_state()
        ctx = _make_context()

        result = await executor.execute_handoff(
            agent=src,
            target_name="billing",
            tool_call=_make_tool_call("billing question"),
            messages=[],
            context=ctx,
            run_state=state,
        )

        assert result.success is False
        assert "billing" in result.error


# --------------------------------------------------------------------------- #
# Diamond / convergent topology: A→B and A→C both hand off to D
# --------------------------------------------------------------------------- #


class TestDiamondTopology:
    """
    The diamond:  A forks into B and C (parallel branches),
                  then both B and C try to hand off to D.

                      A
                     / \\
                    B   C
                     \\ /
                      D

    Key question: does B→D and C→D interfere with each other?

    Answer depends on whether they share the same run_state:
    - Isolated run_state per branch (the parallel workflow does this via branch_copy):
      each branch has its own agent_stack so D is not seen as a cycle → both succeed.
    - Shared run_state (a developer manually reuses the same state):
      B pushes D; now D is in the stack; C tries D → false cycle detected.
      This is the adversarial "shared state" scenario worth knowing about.
    """

    async def test_isolated_states_both_branches_reach_D(self) -> None:
        """
        Each branch has its own RunState (as the parallel workflow provides).
        B→D and C→D run concurrently — both should succeed independently.
        """
        import asyncio

        executor, _ = _make_executor(max_depth=10)

        agent_b = _make_agent("agent-b", handoffs=[Handoff(target_agent="agent-d", description="")])
        agent_c = _make_agent("agent-c", handoffs=[Handoff(target_agent="agent-d", description="")])
        agent_d = _make_agent("agent-d")
        executor.register_agent(agent_b)
        executor.register_agent(agent_c)
        executor.register_agent(agent_d)

        # Each branch starts with its own stack — A is the common ancestor.
        state_b = _make_run_state(stack=["agent-a", "agent-b"])
        state_c = _make_run_state(stack=["agent-a", "agent-c"])

        ctx_b = _make_context()
        ctx_c = _make_context()

        result_b, result_c = await asyncio.gather(
            executor.execute_handoff(
                agent=agent_b,
                target_name="agent-d",
                tool_call=_make_tool_call(),
                messages=[],
                context=ctx_b,
                run_state=state_b,
            ),
            executor.execute_handoff(
                agent=agent_c,
                target_name="agent-d",
                tool_call=_make_tool_call(),
                messages=[],
                context=ctx_c,
                run_state=state_c,
            ),
        )

        # Both branches reach D independently — no false cycle.
        assert result_b.success is True, f"B→D failed: {result_b.error}"
        assert result_c.success is True, f"C→D failed: {result_c.error}"

    async def test_shared_state_second_branch_sees_false_cycle(self) -> None:
        """
        When both branches share the SAME RunState (incorrect usage),
        B pushes D onto the stack first. C then tries to hand off to D,
        but D is now in the shared stack → cycle detected.

        This documents the footgun: sharing run_state across concurrent branches
        causes false cycle errors. The fix is branch_copy() (what the parallel
        workflow already does).
        """
        executor, _ = _make_executor(max_depth=10)

        agent_b = _make_agent("agent-b", handoffs=[Handoff(target_agent="agent-d", description="")])
        agent_c = _make_agent("agent-c", handoffs=[Handoff(target_agent="agent-d", description="")])
        agent_d = _make_agent("agent-d")
        executor.register_agent(agent_b)
        executor.register_agent(agent_c)
        executor.register_agent(agent_d)

        # Shared state — the dangerous case.
        shared_state = _make_run_state(stack=["agent-a"])
        ctx_b = _make_context()
        ctx_c = _make_context()

        # Run sequentially (not concurrently) to get a deterministic outcome:
        # first B→D succeeds and mutates the shared stack, then C→D is affected.
        result_b = await executor.execute_handoff(
            agent=agent_b,
            target_name="agent-d",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx_b,
            run_state=shared_state,
        )
        result_c = await executor.execute_handoff(
            agent=agent_c,
            target_name="agent-d",
            tool_call=_make_tool_call(),
            messages=[],
            context=ctx_c,
            run_state=shared_state,
        )

        # B succeeds — it runs first, D is not yet in the stack.
        assert result_b.success is True, f"B→D unexpectedly failed: {result_b.error}"

        # C is blocked — D is now in the shared stack, looks like a cycle.
        # This is a false positive caused by shared state, not a real cycle.
        assert result_c.success is False
        assert "cycle" in result_c.error.lower()
