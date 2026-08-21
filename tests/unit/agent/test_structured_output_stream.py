"""
Streaming (run_stream) structured-output tests — runner + mocked LLM.

Verifies parity with the non-streaming path:
  - correct JSON streamed → RUN_END event carries structured_output (validates as Review)
  - wrong-shaped stream → blocking formatting call produces structured_output
  - no output_schema → inert (RUN_END has no structured_output)

Streaming events are JSON-serialized for transport, so structured_output arrives
on the event as a DICT (model_dump); we rehydrate it via Review.model_validate.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from continuum.agent.runner import AgentRunner
from continuum.agent.types import EventType, PrepareRunResult

GOOD = '{"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}'


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


def _make_runner(mock_llm) -> AgentRunner:
    with patch("continuum.agent.runner.get_container") as mock_gc:
        mock_gc.return_value = MagicMock()
        return AgentRunner(llm_client=mock_llm)


def _make_agent(*, with_schema=True):
    agent = MagicMock()
    agent.name = "reviewer"
    agent.model = "gpt-4o-mini"
    agent.temperature = 0.7
    agent.max_tokens = 1024
    agent.gateway_mode = None
    agent.extra_body = None
    agent.enable_json_mode = False
    agent.json_schema = None
    agent.json_strict = False
    agent.metadata = {}
    agent.on_end = None
    agent.output_schema = Review if with_schema else None
    agent.output_schema_strict = False
    agent.config.output_scanners = None  # no scanners → simple streaming path
    agent.is_handoff_tool_call.return_value = (False, None)
    agent.get_tools_for_llm.return_value = []  # no tools → constrained final call
    return agent


def _make_ctx():
    ctx = MagicMock()
    ctx.max_turns = 5
    ctx.session_id = None
    ctx.priority = 5
    ctx.metadata = {}
    ctx.run_id = "run-1"
    ctx.trace_id = "trace-1"
    ctx.recorder = None  # skip decision-trace capture
    return ctx


def _stream_of(content: str):
    async def fake_chat_stream(**kwargs):
        yield SimpleNamespace(content=content, tool_calls=None, model="m")

    return fake_chat_stream


async def _collect(runner, agent):
    return [e async for e in runner.run_stream(agent, "The hotel was fantastic but expensive.")]


def _run_end_structured(events):
    for e in reversed(events):
        if e.type == EventType.RUN_END:
            return (e.data or {}).get("structured_output")
    return None


@pytest.fixture
def _wire():
    """Returns a helper that wires prepare_run/finalize and runs the stream."""

    async def go(mock_llm, agent):
        runner = _make_runner(mock_llm)
        runner._prepare_run = AsyncMock(
            return_value=PrepareRunResult(
                success=True, context=_make_ctx(), run_state=MagicMock(messages=[])
            )
        )
        runner._finalizer.finalize = AsyncMock()
        return await _collect(runner, agent)

    return go


class TestStreamingStructuredOutput:
    async def test_correct_json_in_run_end(self, _wire):
        mock_llm = MagicMock()
        mock_llm.chat_stream = _stream_of(GOOD)
        events = await _wire(mock_llm, _make_agent())

        so = _run_end_structured(events)
        assert isinstance(so, dict)  # serialized for transport
        assert Review.model_validate(so).score == 0.75  # validates against the schema

    async def test_wrong_then_blocking_formatting_call(self, _wire):
        # Streamed content is wrong-shaped; the blocking formatting call returns GOOD.
        mock_llm = MagicMock()
        mock_llm.chat_stream = _stream_of('{"review_summary": {"x": 1}}')
        mock_llm.chat = AsyncMock(
            return_value=SimpleNamespace(content=GOOD, tool_calls=[], model="m")
        )
        events = await _wire(mock_llm, _make_agent())

        so = _run_end_structured(events)
        assert isinstance(so, dict)
        assert Review.model_validate(so).sentiment == "mixed"
        mock_llm.chat.assert_awaited()  # the formatting retry happened

    async def test_no_schema_is_inert(self, _wire):
        mock_llm = MagicMock()
        mock_llm.chat_stream = _stream_of("just a normal answer")
        mock_llm.chat = AsyncMock()
        events = await _wire(mock_llm, _make_agent(with_schema=False))

        assert _run_end_structured(events) is None
        mock_llm.chat.assert_not_awaited()  # no formatting call
