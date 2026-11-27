"""EvalCase.from_run: score an answer against what the agent's tools actually returned.

EvalCase.context had to be filled by hand, so evaluating a document-search answer
meant re-running the search outside the agent, which can return different chunks
from the ones the agent saw.
"""

from __future__ import annotations

import pytest

from continuum.evaluation import EvalCase, contexts_from_messages
from continuum.llm.types import ChatMessage, FunctionCall, ToolCall

MESSAGES = [
    {"role": "user", "content": "Compare leverage"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "docs__sql_query", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "docs__vector_search", "arguments": "{}"},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"rows": []}'},
    {"role": "tool", "tool_call_id": "c2", "content": "Net debt / EBITDA is 2.1x."},
    {"role": "tool", "tool_call_id": "c3", "content": "   "},
    {"role": "assistant", "content": "Halvorsen 2.1x"},
]


def test_all_tool_results_in_order():
    assert contexts_from_messages(MESSAGES) == ['{"rows": []}', "Net debt / EBITDA is 2.1x."]


def test_filter_by_tool_name_or_its_last_segment():
    assert contexts_from_messages(MESSAGES, tools=["vector_search"]) == [
        "Net debt / EBITDA is 2.1x."
    ]
    assert contexts_from_messages(MESSAGES, tools=["docs__sql_query"]) == ['{"rows": []}']


def test_from_run_builds_a_case():
    case = EvalCase.from_run(
        "Compare leverage",
        MESSAGES,
        tools=["vector_search"],
        expected_output="2.1x",
        metadata={"q": "q6"},
    )
    assert case.context == ["Net debt / EBITDA is 2.1x."]
    assert case.expected_output == "2.1x" and case.metadata == {"q": "q6"}


def test_accepts_chat_message_models():
    messages = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="c1", function=FunctionCall(name="vector_search", arguments="{}"))
            ],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="chunk"),
    ]
    assert contexts_from_messages(messages, tools=["vector_search"]) == ["chunk"]


def _call(id: str, name: str) -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": "{}"}}


NOISY = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _call("t1", "think"),
            _call("h1", "handoff_to_specialist"),
            _call("v1", "vector_search"),
            _call("v2", "vector_search"),
            _call("v3", "vector_search"),
        ],
    },
    {"role": "tool", "tool_call_id": "t1", "content": "Thought recorded."},
    {"role": "tool", "tool_call_id": "h1", "content": "Handoff failed: no such agent"},
    {
        "role": "tool",
        "tool_call_id": "v1",
        "content": '{"error": "boom", "error_type": "TimeoutError"}',
    },
    {"role": "tool", "tool_call_id": "v2", "content": '{"error": "not an envelope"}'},
    {"role": "tool", "tool_call_id": "v3", "content": "Net debt / EBITDA is 2.1x."},
]


def test_think_handoff_and_error_envelopes_are_skipped_whatever_the_filter():
    kept = ['{"error": "not an envelope"}', "Net debt / EBITDA is 2.1x."]
    assert contexts_from_messages(NOISY) == kept
    assert contexts_from_messages(NOISY, tools=["vector_search"]) == kept
    assert contexts_from_messages(NOISY, tools=["think", "handoff_to_specialist"]) == []


def test_a_record_with_error_type_and_other_fields_is_kept():
    # Only the SDK's own envelope keys mark a tool error; a log or incident record
    # that happens to carry error_type is retrieved data.
    record = '{"error": "E1042", "error_type": "SensorFault", "description": "Pump 3 tripped"}'
    envelope = '{"error": "boom", "error_type": "TimeoutError", "is_error": true}'
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("l1", "log_search"), _call("l2", "log_search")],
        },
        {"role": "tool", "tool_call_id": "l1", "content": record},
        {"role": "tool", "tool_call_id": "l2", "content": envelope},
    ]
    assert contexts_from_messages(messages) == [record]


def test_tools_none_means_all_and_empty_means_none():
    assert len(contexts_from_messages(MESSAGES, tools=None)) == 2
    assert contexts_from_messages(MESSAGES, tools=[]) == []


def test_a_str_for_tools_is_rejected():
    with pytest.raises(TypeError):
        contexts_from_messages(MESSAGES, tools="vector_search")
    with pytest.raises(TypeError):
        EvalCase.from_run("q", MESSAGES, tools="vector_search")


def test_accepts_an_agent_response_and_its_handoff_child():
    from continuum.agent.types import AgentResponse, HandoffResult

    child = AgentResponse(
        content="2.1x",
        agent_name="specialist",
        messages=[
            {"role": "assistant", "tool_calls": [_call("c9", "vector_search")]},
            {"role": "tool", "tool_call_id": "c9", "content": "child chunk"},
        ],
    )
    parent = AgentResponse(
        content="2.1x",
        agent_name="triage",
        messages=[
            {"role": "assistant", "tool_calls": [_call("c1", "vector_search")]},
            {"role": "tool", "tool_call_id": "c1", "content": "parent chunk"},
            {"role": "assistant", "tool_calls": [_call("h1", "handoff_to_specialist")]},
            {"role": "tool", "tool_call_id": "h1", "content": "handoff accepted"},
        ],
        handoff_result=HandoffResult(
            handoff_id="h",
            from_agent="triage",
            to_agent="specialist",
            success=True,
            response=child,
        ),
    )
    case = EvalCase.from_run("q", parent, tools=["vector_search"])
    assert case.context == ["parent chunk", "child chunk"]
    assert contexts_from_messages(parent) == ["parent chunk", "child chunk"]


def test_a_handoff_child_repeating_the_parents_history_is_not_counted_twice():
    """``return_to_parent=False`` with shared tool names: each chunk appears once."""
    from continuum.agent.types import AgentResponse, HandoffResult

    parent_history = [
        {"role": "assistant", "tool_calls": [_call("c1", "vector_search")]},
        {"role": "tool", "tool_call_id": "c1", "content": "CHUNK-1"},
        {"role": "assistant", "tool_calls": [_call("h1", "handoff_to_specialist")]},
        {"role": "tool", "tool_call_id": "h1", "content": "handoff accepted"},
    ]
    child = AgentResponse(
        content="2.1x",
        agent_name="specialist",
        messages=[
            *parent_history,
            {"role": "assistant", "tool_calls": [_call("c2", "vector_search")]},
            {"role": "tool", "tool_call_id": "c2", "content": "CHUNK-2"},
        ],
    )
    parent = AgentResponse(
        content="2.1x",
        agent_name="triage",
        messages=parent_history,
        handoff_result=HandoffResult(
            handoff_id="h",
            from_agent="triage",
            to_agent="specialist",
            success=True,
            response=child,
        ),
    )
    assert EvalCase.from_run("q", parent).context == ["CHUNK-1", "CHUNK-2"]
    # Two calls that returned the same text are still two results.
    same_text = [
        {
            "role": "assistant",
            "tool_calls": [_call("a", "vector_search"), _call("b", "vector_search")],
        },
        {"role": "tool", "tool_call_id": "a", "content": "CHUNK-1"},
        {"role": "tool", "tool_call_id": "b", "content": "CHUNK-1"},
    ]
    assert contexts_from_messages(same_text) == ["CHUNK-1", "CHUNK-1"]


def _messages_gen():
    yield from MESSAGES


@pytest.mark.parametrize(
    "unsupported",
    [
        pytest.param(lambda: {"messages": MESSAGES}, id="to_dict"),
        pytest.param(_messages_gen, id="generator"),
        pytest.param(lambda: "Net debt / EBITDA is 2.1x.", id="str"),
        pytest.param(lambda: None, id="none"),
    ],
)
def test_unsupported_input_is_rejected(unsupported):
    with pytest.raises(TypeError, match="list of messages or an AgentResponse"):
        contexts_from_messages(unsupported())
    with pytest.raises(TypeError):
        EvalCase.from_run("q", unsupported())
