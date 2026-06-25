"""Streaming parity for the parallel-tool pre-taint hardening.

The non-streaming batch path (execute_tools_batch) pre-taints from every batched
tool's DECLARED labels before gating any, so the tool gate is order-independent.
run_stream executes tools sequentially via execute_tool_call, so without the same
pre-taint a producer+exfil pair in one streamed turn could bypass the gate by
listing the exfil tool first (it'd be gated before the producer taints the run).

This drives run_stream with a fake LLM that emits [exfil, producer] (exfil FIRST)
in one turn and asserts the exfil tool's executor was invoked against an
already-tainted context — proving the pre-taint ran before any tool was gated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.types import EventType, PrepareRunResult, RunContext, RunState
from continuum.llm.types import FunctionCall, StreamChunk, ToolCall


def _build_stream_runner(llm, ctx):
    from continuum.agent.runner import AgentRunner
    from continuum.agent.utils.circuit_breaker import CircuitBreaker

    runner = AgentRunner.__new__(AgentRunner)
    runner._llm_client = llm
    runner._memory_client = None
    runner._session_client = None
    runner._tool_executor = None
    runner._tracing_manager = None
    runner._state_manager = None
    runner._agent_registry = {}
    runner._config = MagicMock()
    runner._circuit_breaker = CircuitBreaker(threshold=5, cooldown=60)
    runner._handoff_executor = None
    runner._tool_service = MagicMock()
    runner._finalizer = MagicMock()
    runner._finalizer.finalize = AsyncMock()
    runner._finalizer.handle_error = AsyncMock()

    rs = RunState(run_id="run-test")
    rs.messages = [{"role": "user", "content": "look up P-123 and email it"}]
    runner._prepare_run = AsyncMock(
        return_value=PrepareRunResult(success=True, context=ctx, run_state=rs, user_message_index=0)
    )
    return runner


def _tc(call_id: str, name: str) -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name=name, arguments="{}"))


def _tools_then_done_stream(tool_calls):
    """Turn 1 emits the tool calls; turn 2 emits a final answer to end the run."""
    state = {"turn": 0}

    async def _stream(*args, **kwargs):
        state["turn"] += 1
        if state["turn"] == 1:
            yield StreamChunk(tool_calls=tool_calls, is_finished=True)
        else:
            yield StreamChunk(content="done", is_finished=True)

    return _stream


async def test_exfil_tool_first_is_gated_against_pretaint_in_streaming():
    # lookup_patient is the declared PHI producer; send_referral_email is the exfil
    # tool listed FIRST in the same streamed turn.
    agent = BaseAgent(
        name="clinic",
        instructions="x",
        model="model-x",
        config=AgentConfig(tool_data_labels={"lookup_patient": {"phi"}}),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )
    ctx = RunContext(run_id="run-test", max_turns=3)

    seen: list[tuple[str, frozenset[str]]] = []

    async def _record(agent_, tc, ctx_):
        name = tc.function.name
        seen.append((name, frozenset(ctx_.data_labels)))
        return ({"role": "tool", "tool_call_id": tc.id, "content": "ok"}, None)

    runner = _build_stream_runner(MagicMock(), ctx)
    runner._llm_client.chat_stream = _tools_then_done_stream(
        [_tc("c1", "send_referral_email"), _tc("c2", "lookup_patient")]
    )
    runner._tool_service.execute_tool_call = AsyncMock(side_effect=_record)

    async for ev in runner.run_stream(agent, "look up P-123 and email it"):
        if ev.type == EventType.RUN_END:
            break

    # Both tools were executed, exfil first.
    assert [n for n, _ in seen] == ["send_referral_email", "lookup_patient"]
    # The exfil tool — though listed FIRST — was reached with the run already
    # tainted "phi" (pre-taint folded in the producer's declared label up front).
    exfil_labels = seen[0][1]
    assert "phi" in exfil_labels, (
        f"exfil tool was gated against {set(exfil_labels)} — pre-taint did not run "
        "before the sequential tool loop (streaming order-dependence bug)"
    )
