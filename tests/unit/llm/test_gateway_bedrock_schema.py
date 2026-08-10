"""
Through the Smart Gateway, a schema must be phrased the way Bedrock can hear it.

The gateway translates OpenAI-format requests into each provider's dialect using
a fixed lookup table. Bedrock's table (verified against
`src/providers/bedrock/chatComplete.ts`) has entries for `tools` and
`tool_choice` — which it turns into Converse `toolConfig` + a forced
`toolChoice` — but **no entry for `response_format`**. A parameter with no entry
is not translated and not reported; it is simply dropped, for every Bedrock
model. So Continuum could ask for a schema all day and AWS would never hear it.

Both phrasings express the same requirement, and one of them survives the trip:
send the schema as a forced tool and the gateway carries it through untouched.
No gateway change is involved — Continuum was choosing the one wording that
cannot cross.

This only applies to gateway providers known to drop `response_format`. OpenAI
through the gateway honours it natively and must keep it: a forced tool there
would be a downgrade, not a fix.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from continuum.llm.config import LLMConfig
from continuum.llm.providers.gateway_provider import GatewayProvider
from continuum.llm.structured_output import (
    STRUCTURED_OUTPUT_TOOL,
    to_openai_response_format,
)


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


SCHEMA_FORMAT = to_openai_response_format(Review)
SCHEMA = Review.model_json_schema()
MESSAGES = [{"role": "user", "content": "The hotel was fantastic but expensive."}]
CALLER_TOOL = {
    "type": "function",
    "function": {"name": "search", "description": "search", "parameters": {"type": "object"}},
}


def _gateway() -> GatewayProvider:
    return GatewayProvider(
        gateway_url="http://localhost:8787/v1", api_key="ck-test", router_mode="modest"
    )


def _kwargs(model: str, *, tools=None, enforce_schema: bool = True, **cfg: Any) -> dict[str, Any]:
    config = LLMConfig(model=model, response_format=SCHEMA_FORMAT, **cfg)
    return _gateway()._build_kwargs(config, tools, None, enforce_schema=enforce_schema)


class TestBedrockGetsTheSchemaAsAForcedTool:
    MODEL = "bedrock/anthropic.claude-sonnet-4-5-v1:0"

    def test_response_format_is_not_sent(self) -> None:
        """Leaving it in would be dead weight — the gateway drops it anyway."""
        assert "response_format" not in _kwargs(self.MODEL)

    def test_the_schema_travels_as_a_tool(self) -> None:
        tools = _kwargs(self.MODEL)["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == STRUCTURED_OUTPUT_TOOL
        # The gateway maps function.parameters → toolSpec.inputSchema.json,
        # so this is the field that reaches AWS as the schema.
        assert tools[0]["function"]["parameters"] == SCHEMA

    def test_the_call_is_forced(self) -> None:
        """Offered but not forced, the model may still answer in prose — which
        is the unreliability being fixed."""
        assert _kwargs(self.MODEL)["tool_choice"] == {
            "type": "function",
            "function": {"name": STRUCTURED_OUTPUT_TOOL},
        }

    def test_bare_model_ids_under_bedrock_are_recognised(self) -> None:
        """Nova and Llama go through the same Converse path as Claude."""
        assert "tools" in _kwargs("bedrock/amazon.nova-pro-v1:0")


class TestProvidersThatHonourResponseFormatAreUntouched:
    """A forced tool is a workaround for a provider that cannot express the
    request. Applying it where `response_format` works would trade constrained
    decoding for a tool call — strictly worse."""

    @pytest.mark.parametrize(
        "model",
        ["openai/gpt-4o", "anthropic/claude-opus-4-8", "google/gemini-2.5-flash"],
    )
    def test_response_format_is_preserved(self, model: str) -> None:
        kwargs = _kwargs(model)
        assert kwargs["response_format"] == SCHEMA_FORMAT
        assert "tools" not in kwargs

    def test_tier_routing_keeps_response_format(self) -> None:
        """With `auto/<tier>` the gateway picks the model AFTER this request is
        built, so the target provider is unknowable here. Guessing "might be
        Bedrock" would degrade every tier request."""
        kwargs = _kwargs("auto/mid")
        assert kwargs["response_format"] == SCHEMA_FORMAT
        assert "tools" not in kwargs


class TestTheCallersOwnToolsWin:
    """Forcing the synthetic tool would take the caller's tools off the table,
    so a tool-using turn keeps the prompt-and-salvage floor instead."""

    MODEL = "bedrock/anthropic.claude-sonnet-4-5-v1:0"

    def test_caller_tools_survive(self) -> None:
        tools = _kwargs(self.MODEL, tools=[CALLER_TOOL])["tools"]
        assert [t["function"]["name"] for t in tools] == ["search"]

    def test_no_forced_tool_choice(self) -> None:
        assert "tool_choice" not in _kwargs(self.MODEL, tools=[CALLER_TOOL])


class TestStreamingKeepsTheOldShape:
    """A forced tool streams `tool_calls` deltas and no text, so the runner
    would accumulate an empty string. enforce_schema is off for stream paths."""

    def test_no_forced_tool_when_streaming(self) -> None:
        kwargs = _kwargs("bedrock/anthropic.claude-sonnet-4-5-v1:0", enforce_schema=False)
        assert "tools" not in kwargs


class TestBareJsonModeIsLeftAlone:
    """`json_object` names no shape, so there is no schema to build a tool from.
    It is dropped by the gateway for Bedrock too, but inventing an empty tool
    would not help and would break the answer's form."""

    def test_no_tool_invented(self) -> None:
        config = LLMConfig(model="bedrock/amazon.nova-pro-v1:0", json_mode=True)
        kwargs = _gateway()._build_kwargs(config, None, None, enforce_schema=True)
        assert "tools" not in kwargs
        assert kwargs["response_format"] == {"type": "json_object"}


def _openai_reply(tool_name: str | None, arguments: str, content: str | None = None) -> Any:
    tool_calls = (
        [
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name=tool_name, arguments=arguments),
            )
        ]
        if tool_name
        else None
    )
    message = SimpleNamespace(
        content=content, role="assistant", tool_calls=tool_calls, refusal=None
    )
    return SimpleNamespace(
        id="chatcmpl-1",
        model="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls", index=0)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


class TestTheAnswerComesBackAsContent:
    """The reply is a tool call. Handed on as-is, the executor would try to
    execute a tool that does not exist; the arguments have to be moved into
    `content`, where coerce_and_validate and AgentResponse.content look."""

    PAYLOAD = {"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}
    MODEL = "bedrock/anthropic.claude-sonnet-4-5-v1:0"

    def _complete(self, monkeypatch: pytest.MonkeyPatch, reply: Any) -> Any:
        provider = _gateway()
        monkeypatch.setattr(
            provider._client.chat.completions, "create", lambda **kw: reply, raising=False
        )
        config = LLMConfig(model=self.MODEL, response_format=SCHEMA_FORMAT)
        return provider.complete(MESSAGES, config)

    def test_arguments_become_the_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reply = _openai_reply(STRUCTURED_OUTPUT_TOOL, json.dumps(self.PAYLOAD))
        assert json.loads(self._complete(monkeypatch, reply).content) == self.PAYLOAD

    def test_the_synthetic_call_is_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reply = _openai_reply(STRUCTURED_OUTPUT_TOOL, json.dumps(self.PAYLOAD))
        assert not self._complete(monkeypatch, reply).tool_calls

    def test_finish_reason_reads_as_a_finished_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reply = _openai_reply(STRUCTURED_OUTPUT_TOOL, json.dumps(self.PAYLOAD))
        assert self._complete(monkeypatch, reply).finish_reason == "stop"

    def test_a_real_tool_call_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keyed on the synthetic tool's name, not on "a tool was called"."""
        provider = _gateway()
        reply = _openai_reply("search", '{"q": "hotels"}')
        monkeypatch.setattr(
            provider._client.chat.completions, "create", lambda **kw: reply, raising=False
        )
        config = LLMConfig(model=self.MODEL)
        response = provider.complete(MESSAGES, config, [CALLER_TOOL], "auto")
        assert [tc.function.name for tc in response.tool_calls or []] == ["search"]

    async def test_async_path_unwraps_identically(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _gateway()
        reply = _openai_reply(STRUCTURED_OUTPUT_TOOL, json.dumps(self.PAYLOAD))

        async def _create(**kw: Any) -> Any:
            return reply

        monkeypatch.setattr(
            provider._async_client.chat.completions, "create", _create, raising=False
        )
        config = LLMConfig(model=self.MODEL, response_format=SCHEMA_FORMAT)
        response = await provider.acomplete(MESSAGES, config)
        assert json.loads(response.content) == self.PAYLOAD
        assert not response.tool_calls


class TestDirectOpenAIIsUnaffected:
    """GatewayProvider subclasses OpenAIProvider, so the swap must not leak into
    the direct provider — there, `bedrock/…` is not even a valid model."""

    def test_direct_openai_never_swaps(self) -> None:
        from continuum.llm.providers.openai_provider import OpenAIProvider

        config = LLMConfig(model="gpt-4o", response_format=SCHEMA_FORMAT)
        kwargs = OpenAIProvider(api_key="sk-test")._build_kwargs(
            config, None, None, enforce_schema=True
        )
        assert kwargs["response_format"] == SCHEMA_FORMAT
        assert "tools" not in kwargs
