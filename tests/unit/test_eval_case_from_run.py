"""EvalCase.from_run: score an answer against what the agent's tools actually returned.

EvalCase.context had to be filled by hand, so evaluating a document-search answer
meant re-running the search outside the agent, which can return different chunks
from the ones the agent saw.
"""

from __future__ import annotations

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
