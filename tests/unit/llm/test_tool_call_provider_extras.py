"""
Tool calls must round-trip provider-specific fields.

Gemini 3.x attaches a `thought_signature` to every function call and REQUIRES it
back on the next turn; without it the whole request is rejected with a 400. The
Smart Gateway already carries the field in both directions (it surfaces Google's
`thoughtSignature` as `tool_calls[].function.thought_signature`, and replays an
incoming one back to Google) — but Continuum parsed tool calls field-by-field
into a closed two-field model, so the signature was dropped on arrival and
absent from the assistant message replayed on turn 2.

Turn 1 therefore succeeded and turn 2 always failed, which is why single-shot
schema tests never caught it.

The fix is deliberately generic rather than a `thought_signature` field: any
provider may attach state to a tool call, and a closed shape loses all of it.
"""

from __future__ import annotations

from types import SimpleNamespace

from continuum.llm.types import ChatMessage, FunctionCall, LLMResponse, StreamChunk, ToolCall

SIGNATURE = "opaque-google-signature"


def _openai_completion(**function_fields):
    """An OpenAI-shaped completion carrying one tool call, as the gateway emits it."""
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "google/gemini-3.5-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "sql_execute",
                                    "arguments": '{"q": "select 1"}',
                                    **function_fields,
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )


class TestNonStreamingRoundTrip:
    def test_signature_survives_parsing(self):
        response = LLMResponse.from_openai_response(_openai_completion(thought_signature=SIGNATURE))

        assert response.tool_calls[0].function.thought_signature == SIGNATURE

    def test_signature_is_replayed_on_the_next_turn(self):
        # to_dict() is what executor.py rebuilds the assistant message from, so
        # this is the exact payload that goes back to the gateway on turn 2.
        response = LLMResponse.from_openai_response(_openai_completion(thought_signature=SIGNATURE))

        replayed = response.tool_calls[0].to_dict()

        assert replayed["function"]["thought_signature"] == SIGNATURE
        assert replayed["function"]["name"] == "sql_execute"
        assert replayed["id"] == "call_1"

    def test_assistant_message_carries_it_through_chatmessage(self):
        response = LLMResponse.from_openai_response(_openai_completion(thought_signature=SIGNATURE))

        message = ChatMessage(role="assistant", tool_calls=response.tool_calls).to_dict()

        assert message["tool_calls"][0]["function"]["thought_signature"] == SIGNATURE

    def test_absent_extras_add_no_keys(self):
        # A provider that sends nothing extra must produce exactly the OpenAI
        # shape — no `thought_signature: None` echoed back at OpenAI itself.
        response = LLMResponse.from_openai_response(_openai_completion())

        assert response.tool_calls[0].to_dict()["function"] == {
            "name": "sql_execute",
            "arguments": '{"q": "select 1"}',
        }

    def test_multiple_extras_are_all_preserved(self):
        response = LLMResponse.from_openai_response(
            _openai_completion(thought_signature=SIGNATURE, some_future_field="x")
        )

        function = response.tool_calls[0].to_dict()["function"]
        assert function["thought_signature"] == SIGNATURE
        assert function["some_future_field"] == "x"


class TestStreamingRoundTrip:
    def test_stream_chunk_preserves_the_signature(self):
        chunk = SimpleNamespace(
            id="chatcmpl-1",
            model="google/gemini-3.5-flash",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=_function_delta(
                                    "sql_execute", "{}", thought_signature=SIGNATURE
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        parsed = StreamChunk.from_openai_chunk(chunk)

        assert parsed.tool_calls[0].function.thought_signature == SIGNATURE

    def test_openai_provider_accumulator_preserves_the_signature(self):
        from continuum.llm.providers.openai_provider import OpenAIProvider

        acc: dict = {}
        OpenAIProvider._accumulate_tool_call(
            acc,
            SimpleNamespace(
                index=0,
                id="call_1",
                function=_function_delta("sql_", "{}", thought_signature=SIGNATURE),
            ),
        )
        # Signature arrives on the first delta; later deltas carry only argument text.
        OpenAIProvider._accumulate_tool_call(
            acc,
            SimpleNamespace(index=0, id=None, function=_function_delta("execute", '{"q":1}')),
        )

        built = OpenAIProvider._build_tool_calls_from_acc(acc)

        assert built[0].function.name == "sql_execute"
        assert built[0].function.thought_signature == SIGNATURE

    def test_gemini_provider_accumulator_preserves_the_signature(self):
        from continuum.llm.providers.gemini_provider import GeminiProvider

        acc: dict = {}
        GeminiProvider._accumulate_tool_call(
            acc,
            SimpleNamespace(
                index=0,
                id="call_1",
                function=_function_delta("sql_execute", "{}", thought_signature=SIGNATURE),
            ),
        )

        built = GeminiProvider._build_tool_calls_from_acc(acc)

        assert built[0].function.thought_signature == SIGNATURE


class TestConstructionIsUnaffected:
    def test_plain_construction_still_works(self):
        call = ToolCall(id="1", function=FunctionCall(name="f", arguments="{}"))

        assert call.to_dict() == {
            "id": "1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }

    def test_extras_can_be_set_explicitly(self):
        call = FunctionCall(name="f", arguments="{}", thought_signature=SIGNATURE)

        assert call.to_dict()["thought_signature"] == SIGNATURE


def _function_delta(name, arguments, **extras):
    """A streaming function delta as the openai SDK models it (extras allowed)."""
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction

    return ChoiceDeltaToolCallFunction.model_validate(
        {"name": name, "arguments": arguments, **extras}
    )
