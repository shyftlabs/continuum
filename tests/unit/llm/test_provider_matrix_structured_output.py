"""
Provider-matrix contract test for structured output (mocked — no API keys, runs
on every PR).

The original bug report's second issue: "a prompt that works on gpt-4o-mini may
produce differently-shaped JSON on claude-haiku or gemini-flash, so structured
output becomes unreliable across the 100+ models." Continuum's defense is the
cross-provider floor in ``llm/structured_output.py`` — a schema prompt plus a
tolerant parser that recovers each provider's characteristic JSON quirk.

This file pins that defense into CI. It does NOT call real providers (the
unit suite never does); instead it parametrizes over the *shapes* each provider
realistically returns and asserts the framework recovers a valid object —
  * at the pure-parse level (coerce_and_validate), and
  * end-to-end through the real Executor (run) and AgentRunner (run_stream).

A real-provider matrix (actually hitting gpt-4o-mini / claude-haiku /
gemini-flash) is a separate, key-gated, nightly job — see the standalone
acceptance script. This mocked matrix is the free, every-PR layer: if a provider
quirk regresses in the parsing/retry logic, CI goes red here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig
from continuum.agent.execution.executor import Executor
from continuum.agent.runner import AgentRunner
from continuum.agent.types import EventType, PrepareRunResult, RunContext, RunState
from continuum.llm.structured_output import coerce_and_validate


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


# The one valid object every shape below must decode to.
EXPECTED = {"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}
_INNER = '{"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}'

# Each entry is one provider's *characteristic* way of returning the same object.
# These are the exact divergences the bug report called out — gpt returns clean
# JSON, Claude tends to wrap it in prose or fences, Gemini tends to nest it under
# the schema name. All must recover to EXPECTED.
PROVIDER_SHAPES: list[tuple[str, str]] = [
    ("openai_clean_json", _INNER),
    ("anthropic_prose_wrapped", f"Sure! Here is the analysis you asked for:\n{_INNER}"),
    ("anthropic_markdown_fenced", f"```json\n{_INNER}\n```"),
    ("gemini_schema_name_wrapper", f'{{"Review": {_INNER}}}'),
    ("gemini_fenced_wrapper", f'```json\n{{"Review": {_INNER}}}\n```'),
    ("anthropic_trailing_prose", f"{_INNER}\nLet me know if you'd like anything else."),
]


# ---------------------------------------------------------------------------
# Layer 1 — pure parse: the universal recovery floor, per provider quirk.
# ---------------------------------------------------------------------------


class TestParseRecoversEveryProviderShape:
    @pytest.mark.parametrize("label,content", PROVIDER_SHAPES, ids=[s[0] for s in PROVIDER_SHAPES])
    def test_coerce_recovers(self, label: str, content: str):
        obj, err = coerce_and_validate(content, Review)
        assert err is None, f"{label}: expected recovery, got error {err!r}"
        assert isinstance(obj, Review)
        assert obj.model_dump() == EXPECTED


# ---------------------------------------------------------------------------
# Layer 2 — end-to-end through the real Executor (non-streaming run()).
# ---------------------------------------------------------------------------


def _usage():
    return SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


class _OneShotLLM:
    """Fake LLM returning a single canned content (a provider's shaped answer)."""

    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self._content, tool_calls=[], usage=_usage(), model="m")


def _agent():
    return BaseAgent(
        name="reviewer",
        instructions="Analyze the review.",
        config=AgentConfig(log_to_session=False, input_sanitization=False),
        output_schema=Review,
    )


async def _run(content: str) -> tuple[object, int]:
    llm = _OneShotLLM(content)
    ex = Executor(llm_client=llm)
    rs = RunState(run_id="run-pm")
    rs.push_agent("reviewer")
    resp = await ex.execute_loop(
        _agent(),
        [{"role": "user", "content": "The hotel was fantastic but expensive."}],
        RunContext(run_id="run-pm", max_turns=5),
        rs,
    )
    return resp, llm.calls


class TestExecutorRecoversEveryProviderShape:
    @pytest.mark.parametrize("label,content", PROVIDER_SHAPES, ids=[s[0] for s in PROVIDER_SHAPES])
    async def test_structured_output_populated(self, label: str, content: str):
        resp, calls = await _run(content)
        assert isinstance(resp.structured_output, Review), f"{label}: structured_output not built"
        assert resp.structured_output.model_dump() == EXPECTED
        assert resp.structured_output_error is None
        # A no-tool agent's constrained answer already parses, so no retry is
        # needed for any of these shapes: exactly one LLM call.
        assert calls == 1, f"{label}: expected no retry, made {calls} calls"


# ---------------------------------------------------------------------------
# Layer 3 — streaming parity (run_stream): RUN_END carries the same object.
# ---------------------------------------------------------------------------


def _make_stream_runner(content: str) -> AgentRunner:
    mock_llm = MagicMock()

    async def fake_chat_stream(**kwargs):
        yield SimpleNamespace(content=content, tool_calls=None, model="m")

    mock_llm.chat_stream = fake_chat_stream
    with patch("continuum.agent.runner.get_container") as mock_gc:
        mock_gc.return_value = MagicMock()
        runner = AgentRunner(llm_client=mock_llm)
    runner._prepare_run = AsyncMock(
        return_value=PrepareRunResult(
            success=True, context=_stream_ctx(), run_state=MagicMock(messages=[])
        )
    )
    runner._finalizer.finalize = AsyncMock()
    return runner


def _stream_agent():
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
    agent.output_schema = Review
    agent.output_schema_strict = False
    agent.config.output_scanners = None
    agent.is_handoff_tool_call.return_value = (False, None)
    agent.get_tools_for_llm.return_value = []
    return agent


def _stream_ctx():
    ctx = MagicMock()
    ctx.max_turns = 5
    ctx.session_id = None
    ctx.priority = 5
    ctx.metadata = {}
    ctx.run_id = "run-pm"
    ctx.trace_id = "trace-pm"
    ctx.recorder = None
    return ctx


def _run_end_structured(events):
    for e in reversed(events):
        if e.type == EventType.RUN_END:
            return (e.data or {}).get("structured_output")
    return None


class TestStreamingRecoversEveryProviderShape:
    @pytest.mark.parametrize("label,content", PROVIDER_SHAPES, ids=[s[0] for s in PROVIDER_SHAPES])
    async def test_run_end_carries_structured_output(self, label: str, content: str):
        runner = _make_stream_runner(content)
        events = [
            e async for e in runner.run_stream(_stream_agent(), "The hotel was great but pricey.")
        ]
        so = _run_end_structured(events)
        assert isinstance(so, dict), f"{label}: RUN_END missing structured_output"
        # Serialized for transport; rehydrate and validate against the schema.
        assert Review.model_validate(so).model_dump() == EXPECTED
