"""after_tool_round: a product rule steers the agent once its tools have answered.

Measured with a finance research agent: asked to compare two companies on
leverage, the agent called SQL two or three times, got nothing, and answered
"no records found" in 2 of 3 runs, though document search had the answer. The
product's own loop sent the model back once to try the unused tool and answered
3 of 3. This hook lets the product supply that rule without forking the loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.execution.tool_round import ToolRound, ToolRoundResult, ToolRoundTracker
from continuum.agent.types import EventType, PrepareRunResult, RunContext, RunState
from continuum.llm.types import FunctionCall, StreamChunk, ToolCall

EMPTY = '{"rows": [], "result_status": "no_results_found"}'


def _tc(call_id: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name=name, arguments=args))


def _agent(hook=None, max_nudges=1) -> BaseAgent:
    return BaseAgent(
        name="ask",
        instructions="x",
        model="m",
        config=AgentConfig(after_tool_round=hook, after_tool_round_max_nudges=max_nudges),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


def dead_end(round_: ToolRound) -> str | None:
    if all(r.is_empty(["no_results_found"]) for r in round_.history):
        unused = {"vector_search", "sql_query"} - round_.tools_used
        if unused:
            return f"Those lookups returned nothing. Call {', '.join(sorted(unused))} now."
    return None


# --------------------------------------------------------------------- tracker


async def test_hook_sees_results_and_its_guidance_is_a_system_message():
    seen: list[ToolRound] = []

    def hook(r):
        seen.append(r)
        return dead_end(r)

    tracker = ToolRoundTracker(_agent(hook))
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": EMPTY},
    ]
    nudge = await tracker.after_round(1, [_tc("c1", "sql_query", '{"q": "lev"}')], messages)

    assert nudge == "Those lookups returned nothing. Call vector_search now."
    assert messages[-1] == {"role": "system", "content": nudge}
    [round_] = seen
    assert round_.results == [ToolRoundResult("sql_query", {"q": "lev"}, EMPTY)]
    assert round_.tools_used == {"sql_query"}


async def test_at_most_max_nudges_per_run():
    tracker = ToolRoundTracker(_agent(lambda r: "try again", max_nudges=1))
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": EMPTY}]
    assert await tracker.after_round(1, [_tc("c1", "sql_query")], msgs) == "try again"
    msgs.append({"role": "tool", "tool_call_id": "c2", "content": EMPTY})
    assert await tracker.after_round(2, [_tc("c2", "sql_query")], msgs) is None


async def test_async_hook_and_failing_hook():
    async def ahook(r):
        return "async nudge"

    msgs = [{"role": "tool", "tool_call_id": "c1", "content": "x"}]
    assert await ToolRoundTracker(_agent(ahook)).after_round(1, [_tc("c1", "t")], msgs)

    def boom(r):
        raise RuntimeError("bad rule")

    msgs = [{"role": "tool", "tool_call_id": "c1", "content": "x"}]
    assert await ToolRoundTracker(_agent(boom)).after_round(1, [_tc("c1", "t")], msgs) is None
    assert msgs[-1]["role"] == "tool"


async def test_no_hook_is_a_no_op():
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": EMPTY}]
    assert await ToolRoundTracker(_agent()).after_round(1, [_tc("c1", "t")], msgs) is None
    assert len(msgs) == 1


def test_is_empty():
    assert ToolRoundResult("t", {}, " [] ").is_empty()
    assert ToolRoundResult("t", {}, EMPTY).is_empty(["no_results_found"])
    assert not ToolRoundResult("t", {}, '{"rows": [1]}').is_empty(["no_results_found"])


# ------------------------------------------------------------------ run_stream


def _runner(stream):
    from continuum.agent.runner import AgentRunner
    from continuum.agent.utils.circuit_breaker import CircuitBreaker

    runner = AgentRunner.__new__(AgentRunner)
    runner._llm_client = MagicMock()
    runner._llm_client.chat_stream = stream
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
    rs.messages = [{"role": "user", "content": "compare leverage"}]
    runner._prepare_run = AsyncMock(
        return_value=PrepareRunResult(
            success=True,
            context=RunContext(run_id="run-test", max_turns=5),
            run_state=rs,
            user_message_index=0,
        )
    )
    return runner


async def test_run_stream_sends_the_nudge_before_the_next_model_call():
    sent: list[list[dict]] = []

    async def stream(*args, messages, **kwargs):
        sent.append([dict(m) for m in messages])
        if len(sent) == 1:
            yield StreamChunk(tool_calls=[_tc("c1", "sql_query")], is_finished=True)
        elif len(sent) == 2:
            yield StreamChunk(tool_calls=[_tc("c2", "vector_search")], is_finished=True)
        else:
            yield StreamChunk(content="Halvorsen 2.1x, Brightwater 3.4x", is_finished=True)

    async def run_tool(agent_, tc, ctx_):
        content = EMPTY if tc.function.name == "sql_query" else '{"chunks": ["2.1x", "3.4x"]}'
        return ({"role": "tool", "tool_call_id": tc.id, "content": content}, None)

    runner = _runner(stream)
    runner._tool_service.execute_tool_call = AsyncMock(side_effect=run_tool)
    events = [e async for e in runner.run_stream(_agent(dead_end), "compare leverage")]

    second_call = sent[1]
    assert second_call[-1] == {
        "role": "system",
        "content": "Those lookups returned nothing. Call vector_search now.",
    }
    # vector_search returned rows, so no second nudge before the answer
    assert all(m.get("role") != "system" for m in sent[2][len(second_call) :])
    [run_end] = [e for e in events if e.type == EventType.RUN_END]
    assert "3.4x" in run_end.data["content"]


async def test_run_stream_without_hook_is_unchanged():
    sent = []

    async def stream(*args, messages, **kwargs):
        sent.append(list(messages))
        if len(sent) == 1:
            yield StreamChunk(tool_calls=[_tc("c1", "sql_query")], is_finished=True)
        else:
            yield StreamChunk(content="no records found", is_finished=True)

    runner = _runner(stream)
    runner._tool_service.execute_tool_call = AsyncMock(
        return_value=({"role": "tool", "tool_call_id": "c1", "content": EMPTY}, None)
    )
    [e async for e in runner.run_stream(_agent(), "compare leverage")]
    assert all(m.get("role") != "system" for m in sent[1])
