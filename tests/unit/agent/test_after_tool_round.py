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
from continuum.llm.types import FunctionCall, LLMResponse, StreamChunk, ToolCall

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


# ---------------------------------------------------------- run / run_stream parity
#
# run's executor drops think, Headroom-retrieve and handoff calls before the hook;
# run_stream used to pass every call, so a react_mode agent that thought first
# never reached the dead_end nudge when streaming. Same script, both paths.


def _think_then_empty_sql():
    return [
        [_tc("c0", "think", '{"thought": "check SQL first"}')],
        [_tc("c1", "sql_query", '{"q": "lev"}')],
        None,  # answer
    ]


def _recording_hook(rounds):
    def hook(r: ToolRound):
        rounds.append((r.turn, [x.name for x in r.results], sorted(r.tools_used)))
        return dead_end(r)

    return hook


NUDGE = "Those lookups returned nothing. Call vector_search now."


async def _via_execute_loop():
    from unittest.mock import patch

    from continuum.agent.execution.executor import Executor

    script = _think_then_empty_sql()
    sent: list[list[dict]] = []

    async def chat(*args, messages, **kwargs):
        sent.append([dict(m) for m in messages])
        calls = script[len(sent) - 1]
        return LLMResponse(model="m", content=None if calls else "no records", tool_calls=calls)

    async def execute_tools_batch(agent, tool_calls, context, tool_summary):
        return [{"role": "tool", "tool_call_id": tc.id, "content": EMPTY} for tc in tool_calls]

    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=chat)
    handler = MagicMock()
    handler.execute_tools_batch = AsyncMock(side_effect=execute_tools_batch)
    rounds: list = []
    agent = _agent(_recording_hook(rounds))
    messages = [{"role": "user", "content": "compare leverage"}]
    run_state = RunState(run_id="run-test")
    run_state.messages = list(messages)
    with patch("continuum.agent.execution.executor.LLMConfig") as cfg:
        cfg.from_agent_config.return_value = MagicMock()
        await Executor(llm_client=llm, tool_handler=handler).execute_loop(
            agent, messages, RunContext(run_id="run-test", max_turns=5), run_state
        )
    return rounds, sent


async def _via_run_stream():
    script = _think_then_empty_sql()
    sent: list[list[dict]] = []

    async def stream(*args, messages, **kwargs):
        sent.append([dict(m) for m in messages])
        calls = script[len(sent) - 1]
        if calls:
            yield StreamChunk(tool_calls=calls, is_finished=True)
        else:
            yield StreamChunk(content="no records", is_finished=True)

    async def run_tool(agent_, tc, ctx_):
        content = "Thought recorded." if tc.function.name == "think" else EMPTY
        return ({"role": "tool", "tool_call_id": tc.id, "content": content}, None)

    runner = _runner(stream)
    runner._tool_service.execute_tool_call = AsyncMock(side_effect=run_tool)
    rounds: list = []
    [e async for e in runner.run_stream(_agent(_recording_hook(rounds)), "compare leverage")]
    return rounds, sent


async def test_run_and_run_stream_hand_the_hook_the_same_rounds():
    run_rounds, run_sent = await _via_execute_loop()
    stream_rounds, stream_sent = await _via_run_stream()

    # The think-only turn does not call the hook; the empty sql_query turn does,
    # and think is not in the history it sees.
    assert run_rounds == [(2, ["sql_query"], ["sql_query"])]
    assert stream_rounds == run_rounds
    # and both send the nudge before the third model call
    for sent in (run_sent, stream_sent):
        assert len(sent) == 3
        assert sent[2][-1] == {"role": "system", "content": NUDGE}


async def test_internal_calls_are_left_out_of_a_round():
    from continuum.llm.headroom.compressor import RETRIEVE_TOOL_NAME

    seen: list[ToolRound] = []
    agent = _agent(lambda r: seen.append(r))
    agent.is_handoff_tool_call = lambda name: (name == "transfer_to_b", "b")
    msgs = [{"role": "tool", "tool_call_id": i, "content": EMPTY} for i in ("c1", "c2", "c3", "c4")]
    calls = [
        _tc("c1", "think"),
        _tc("c2", RETRIEVE_TOOL_NAME),
        _tc("c3", "transfer_to_b"),
    ]
    tracker = ToolRoundTracker(agent)
    assert await tracker.after_round(1, calls, msgs) is None
    assert seen == []  # a round of only internal calls does not call the hook

    await tracker.after_round(2, [*calls, _tc("c4", "sql_query")], msgs)
    [round_] = seen
    assert [r.name for r in round_.results] == ["sql_query"]


def test_bad_max_nudges_only_matters_when_a_hook_is_set():
    import pytest

    ToolRoundTracker(_agent(None, max_nudges="one"))  # no hook: never read
    with pytest.raises(ValueError):
        ToolRoundTracker(_agent(dead_end, max_nudges="one"))


async def test_hook_keeps_being_called_after_the_cap_but_its_nudge_is_ignored():
    seen: list[ToolRound] = []

    def hook(r):
        seen.append(r)
        return "try again"

    tracker = ToolRoundTracker(_agent(hook, max_nudges=1))
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": EMPTY}]
    assert await tracker.after_round(1, [_tc("c1", "sql_query")], msgs) == "try again"
    for turn, call_id in ((2, "c2"), (3, "c3")):
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": EMPTY})
        assert await tracker.after_round(turn, [_tc(call_id, "sql_query")], msgs) is None
    assert [r.turn for r in seen] == [1, 2, 3]
    assert [r.nudges_sent for r in seen] == [0, 1, 1]
    assert sum(1 for m in msgs if m["role"] == "system") == 1

    # max_nudges=0: the rule still observes every round; nothing goes in.
    seen.clear()
    tracker = ToolRoundTracker(_agent(hook, max_nudges=0))
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": EMPTY}]
    assert await tracker.after_round(1, [_tc("c1", "sql_query")], msgs) is None
    assert len(seen) == 1 and msgs[-1]["role"] == "tool"


async def test_non_serialisable_tool_result_does_not_fail_the_run():
    seen: list[ToolRound] = []
    tracker = ToolRoundTracker(_agent(lambda r: seen.append(r)))
    circular: list = []
    circular.append(circular)
    msgs = [
        {"role": "tool", "tool_call_id": "c1", "content": {"at": object()}},
        {"role": "tool", "tool_call_id": "c2", "content": circular},
    ]

    await tracker.after_round(1, [_tc("c1", "a"), _tc("c2", "b")], msgs)

    [round_] = seen
    assert [r.name for r in round_.results] == ["a", "b"]
    assert "object at" in round_.results[0].content
    assert all(isinstance(r.content, str) for r in round_.results)


async def test_calls_with_the_same_or_no_id_each_get_their_own_result():
    seen: list[ToolRound] = []
    tracker = ToolRoundTracker(_agent(lambda r: seen.append(r)))
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "dup", "content": "from an earlier round"},
        {"role": "assistant", "tool_calls": [{"id": "dup"}, {"id": "dup"}, {"id": ""}, {}]},
        {"role": "tool", "tool_call_id": "dup", "content": "first"},
        {"role": "tool", "tool_call_id": "dup", "content": "second"},
        {"role": "tool", "tool_call_id": "", "content": "third"},
        {"role": "tool", "content": "fourth"},
    ]
    calls = [_tc("dup", "a"), _tc("dup", "b"), _tc("", "c"), _tc("", "d")]

    await tracker.after_round(2, calls, msgs)

    [round_] = seen
    assert [(r.name, r.content) for r in round_.results] == [
        ("a", "first"),
        ("b", "second"),
        ("c", "third"),
        ("d", "fourth"),
    ]
