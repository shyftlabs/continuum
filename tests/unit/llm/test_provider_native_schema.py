"""
Each provider must enforce an output schema in ITS OWN dialect.

Continuum's ``LLMConfig`` is OpenAI's parameter set, so ``response_format`` was
only ever honoured by the OpenAI provider. Everyone else fell back to the
prompt-and-salvage floor in ``llm/structured_output.py``:

  * Anthropic replaced the whole schema with one system line, "Respond with
    valid JSON only." — the model was never told the field names by that line.
  * Gemini dropped the ``json_schema`` dict entirely and forwarded only the
    weak ``{"type": "json_object"}`` form, if anything.

Both providers CAN enforce a schema; Continuum just never asked. Anthropic does
it through forced tool use (a tool's ``input_schema`` is a JSON schema, and a
forced ``tool_choice`` makes emitting matching arguments the model's only legal
move). Gemini's OpenAI-compatible endpoint takes ``response_format`` directly.

These tests pin the translation at the kwargs level — what would go on the wire —
plus the response unwrapping, which is where forced tool use can silently break
the caller (a schema answer arrives as a ``tool_use`` block, not text; left
alone, the executor would try to *execute* the synthetic tool).

The prompt floor stays as the fallback and is asserted here too: it is what
still runs whenever native enforcement is unavailable.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from continuum.llm.config import LLMConfig
from continuum.llm.providers.anthropic_provider import AnthropicProvider
from continuum.llm.providers.gemini_provider import GeminiProvider
from continuum.llm.structured_output import (
    STRUCTURED_OUTPUT_TOOL,
    to_openai_response_format,
)


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


SCHEMA_FORMAT = to_openai_response_format(Review)
MESSAGES = [{"role": "user", "content": "The hotel was fantastic but expensive."}]
CALLER_TOOL = {
    "type": "function",
    "function": {"name": "search", "description": "search", "parameters": {"type": "object"}},
}

_JSON_ONLY_LINE = "Respond with valid JSON only."


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-test")


def _gemini() -> GeminiProvider:
    return GeminiProvider(api_key="sk-test")


# ---------------------------------------------------------------------------
# Anthropic — forced tool use
# ---------------------------------------------------------------------------


class TestAnthropicForcedTool:
    def _kwargs(self, **overrides: Any) -> dict[str, Any]:
        cfg = LLMConfig(model="claude-sonnet-4-5", response_format=SCHEMA_FORMAT)
        tools = overrides.pop("tools", None)
        return _anthropic()._build_kwargs(
            MESSAGES, cfg, tools, None, enforce_schema=overrides.pop("enforce_schema", True)
        )

    def test_declares_a_tool_whose_input_schema_is_the_output_schema(self) -> None:
        kwargs = self._kwargs()
        tools = kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == STRUCTURED_OUTPUT_TOOL
        assert tools[0]["input_schema"] == Review.model_json_schema()

    def test_forces_the_model_to_call_it(self) -> None:
        """Without a forced tool_choice the model may still reply with prose,
        which is exactly the unreliability being fixed."""
        assert self._kwargs()["tool_choice"] == {
            "type": "tool",
            "name": STRUCTURED_OUTPUT_TOOL,
        }

    def test_drops_the_prompt_only_fallback_line(self) -> None:
        """The schema is enforced now; the vague line would only add tokens."""
        assert _JSON_ONLY_LINE not in (self._kwargs().get("system") or "")

    def test_json_mode_without_a_schema_still_uses_the_prompt_floor(self) -> None:
        """`json_object` names no shape, so there is nothing to build a tool from."""
        cfg = LLMConfig(model="claude-sonnet-4-5", json_mode=True)
        kwargs = _anthropic()._build_kwargs(MESSAGES, cfg, None, None, enforce_schema=True)
        assert "tools" not in kwargs
        assert _JSON_ONLY_LINE in kwargs["system"]

    def test_prompt_floor_survives_an_agent_with_no_instructions(self) -> None:
        """Regression: the fallback appended to `system` without checking it was
        set, so JSON mode on an agent carrying no system message raised
        `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'`
        before the request was ever sent."""
        cfg = LLMConfig(model="claude-sonnet-4-5", json_mode=True)
        no_system = [{"role": "user", "content": "hi"}]
        kwargs = _anthropic()._build_kwargs(no_system, cfg, None, None)
        assert kwargs["system"] == _JSON_ONLY_LINE


class TestAnthropicLeavesRealToolCallingAlone:
    """Forcing a synthetic tool would make real tool use impossible: the model
    could no longer choose the caller's tools. Tools present → prompt floor."""

    def _kwargs_with_tools(self) -> dict[str, Any]:
        cfg = LLMConfig(model="claude-sonnet-4-5", response_format=SCHEMA_FORMAT)
        return _anthropic()._build_kwargs(MESSAGES, cfg, [CALLER_TOOL], "auto", enforce_schema=True)

    def test_callers_tools_survive_untouched(self) -> None:
        tools = self._kwargs_with_tools()["tools"]
        assert [t["name"] for t in tools] == ["search"]

    def test_tool_choice_is_not_forced_to_the_synthetic_tool(self) -> None:
        assert self._kwargs_with_tools()["tool_choice"] == {"type": "auto"}

    def test_falls_back_to_the_prompt_line(self) -> None:
        assert _JSON_ONLY_LINE in self._kwargs_with_tools()["system"]


class TestAnthropicStreamingKeepsThePromptFloor:
    """A forced tool emits `input_json_delta`, not text, so `text_stream` would
    yield nothing at all — a streaming run would go silently empty."""

    def _stream_kwargs(self) -> dict[str, Any]:
        cfg = LLMConfig(model="claude-sonnet-4-5", response_format=SCHEMA_FORMAT)
        # enforce_schema defaults to False; stream()/astream() rely on that default.
        return _anthropic()._build_kwargs(MESSAGES, cfg, None, None)

    def test_no_synthetic_tool_on_the_streaming_path(self) -> None:
        assert "tools" not in self._stream_kwargs()

    def test_prompt_floor_still_applies(self) -> None:
        assert _JSON_ONLY_LINE in self._stream_kwargs()["system"]


def _anthropic_reply(blocks: list[Any], stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_1",
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )


def _tool_use_block(name: str, payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="toolu_1", name=name, input=payload)


class TestAnthropicUnwrapsTheStructuredAnswer:
    """The schema answer comes back as tool arguments. It must be handed to the
    caller as ``content`` — otherwise the executor sees a tool call and tries to
    execute a tool that does not exist."""

    PAYLOAD = {"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}

    def _complete(self, monkeypatch: pytest.MonkeyPatch, reply: SimpleNamespace) -> Any:
        provider = _anthropic()
        monkeypatch.setattr(
            provider._client.messages, "create", lambda **kwargs: reply, raising=False
        )
        cfg = LLMConfig(model="claude-sonnet-4-5", response_format=SCHEMA_FORMAT)
        return provider.complete(MESSAGES, cfg)

    def test_tool_arguments_become_the_response_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reply = _anthropic_reply([_tool_use_block(STRUCTURED_OUTPUT_TOOL, self.PAYLOAD)])
        response = self._complete(monkeypatch, reply)
        assert json.loads(response.content) == self.PAYLOAD

    def test_the_synthetic_call_is_not_surfaced_as_a_tool_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reply = _anthropic_reply([_tool_use_block(STRUCTURED_OUTPUT_TOOL, self.PAYLOAD)])
        assert not self._complete(monkeypatch, reply).tool_calls

    def test_finish_reason_reads_as_a_completed_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Left as 'tool_calls' it would look like a turn that still owes work."""
        reply = _anthropic_reply([_tool_use_block(STRUCTURED_OUTPUT_TOOL, self.PAYLOAD)])
        assert self._complete(monkeypatch, reply).finish_reason == "stop"

    def test_a_genuine_tool_call_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard: unwrapping must key on the synthetic tool's name,
        not on 'the response contains a tool_use block'."""
        provider = _anthropic()
        reply = _anthropic_reply([_tool_use_block("search", {"q": "hotels"})])
        monkeypatch.setattr(
            provider._client.messages, "create", lambda **kwargs: reply, raising=False
        )
        cfg = LLMConfig(model="claude-sonnet-4-5")
        response = provider.complete(MESSAGES, cfg, [CALLER_TOOL], "auto")
        assert [tc.function.name for tc in response.tool_calls or []] == ["search"]

    async def test_async_path_unwraps_identically(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _anthropic()
        reply = _anthropic_reply([_tool_use_block(STRUCTURED_OUTPUT_TOOL, self.PAYLOAD)])

        async def _create(**kwargs: Any) -> SimpleNamespace:
            return reply

        monkeypatch.setattr(provider._async_client.messages, "create", _create, raising=False)
        cfg = LLMConfig(model="claude-sonnet-4-5", response_format=SCHEMA_FORMAT)
        response = await provider.acomplete(MESSAGES, cfg)
        assert json.loads(response.content) == self.PAYLOAD
        assert not response.tool_calls


# ---------------------------------------------------------------------------
# Gemini — forward the schema instead of discarding it
# ---------------------------------------------------------------------------


class TestGeminiForwardsTheSchema:
    def test_json_schema_reaches_the_request(self) -> None:
        cfg = LLMConfig(model="gemini/gemini-2.5-flash", response_format=SCHEMA_FORMAT)
        kwargs = _gemini()._build_kwargs(cfg, None, None)
        assert kwargs["response_format"] == SCHEMA_FORMAT

    def test_json_mode_still_maps_to_json_object(self) -> None:
        cfg = LLMConfig(model="gemini/gemini-2.5-flash", json_mode=True)
        kwargs = _gemini()._build_kwargs(cfg, None, None)
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_schema_wins_over_bare_json_mode(self) -> None:
        cfg = LLMConfig(
            model="gemini/gemini-2.5-flash", json_mode=True, response_format=SCHEMA_FORMAT
        )
        assert _gemini()._build_kwargs(cfg, None, None)["response_format"] == SCHEMA_FORMAT

    def test_nothing_sent_when_no_format_requested(self) -> None:
        cfg = LLMConfig(model="gemini/gemini-2.5-flash")
        assert "response_format" not in _gemini()._build_kwargs(cfg, None, None)


class TestAnthropicStructuredOutputEndToEnd:
    """The provider-level unwrap only matters if the object actually lands in
    ``AgentResponse.structured_output``. This runs the real Executor and the real
    LLMClient against a mocked Anthropic SDK, so a mismatch anywhere along the
    chain — provider, client, executor — fails here rather than in production."""

    async def test_agent_gets_a_validated_object_from_the_forced_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig
        from continuum.agent.execution.executor import Executor
        from continuum.agent.types import RunContext, RunState
        from continuum.llm.client import LLMClient

        payload = {"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}
        provider = _anthropic()
        seen: dict[str, Any] = {}

        async def _create(**kwargs: Any) -> SimpleNamespace:
            seen.update(kwargs)
            return _anthropic_reply([_tool_use_block(STRUCTURED_OUTPUT_TOOL, payload)])

        monkeypatch.setattr(provider._async_client.messages, "create", _create, raising=False)
        monkeypatch.setattr(
            "continuum.llm.client.get_provider", lambda config: provider, raising=True
        )

        agent = BaseAgent(
            name="reviewer",
            model="claude-sonnet-4-5",
            instructions="Analyze the review.",
            config=AgentConfig(log_to_session=False, input_sanitization=False),
            output_schema=Review,
        )
        run_state = RunState(run_id="run-native")
        run_state.push_agent("reviewer")
        response = await Executor(llm_client=LLMClient(enable_langfuse=False)).execute_loop(
            agent,
            MESSAGES,
            RunContext(run_id="run-native", max_turns=3),
            run_state,
        )

        # The schema went out as a forced tool, not as a vague prompt line...
        assert seen["tool_choice"]["name"] == STRUCTURED_OUTPUT_TOOL
        # ...and came back as a validated instance rather than salvaged prose.
        assert isinstance(response.structured_output, Review)
        assert response.structured_output.model_dump() == payload
        assert response.structured_output_error is None
        # content must carry the JSON too — callers read it directly.
        assert json.loads(response.content) == payload


# ---------------------------------------------------------------------------
# Provider capability — reported, not guessed
# ---------------------------------------------------------------------------


class TestProvidersDeclareTheirCapability:
    """`llm/utils.py` used to answer this from a stale hardcoded allowlist that
    nothing consulted. The provider itself is the only honest source."""

    def test_anthropic_can_enforce_a_schema(self) -> None:
        assert _anthropic().supports_native_schema() is True

    def test_gemini_can_enforce_a_schema(self) -> None:
        assert _gemini().supports_native_schema() is True

    def test_openai_can_enforce_a_schema(self) -> None:
        from continuum.llm.providers.openai_provider import OpenAIProvider

        assert OpenAIProvider(api_key="sk-test").supports_native_schema() is True

    def test_the_base_default_is_no(self) -> None:
        """A third-party provider that has not implemented enforcement must not
        be reported as enforcing one."""
        from continuum.llm.providers.base import BaseProvider

        assert BaseProvider.supports_native_schema() is False
