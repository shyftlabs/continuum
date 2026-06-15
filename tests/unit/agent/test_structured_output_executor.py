"""
Executor-level tests for structured output (the real Executor + a fake LLM).

Covers the fix:
  - no-tool agent: schema-constrained call → structured_output populated
  - retry: first attempt wrong-shaped → formatting retry succeeds
  - soft failure: always wrong → structured_output=None + structured_output_error (no raise)
  - strict: output_schema_strict=True → raises StructuredOutputError
  - tool agent: loop unconstrained → separate formatting call produces structured_output
  - no output_schema: inert (no structured output, no extra calls)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig
from continuum.agent.exceptions import StructuredOutputError
from continuum.agent.execution.executor import Executor
from continuum.agent.types import RunContext, RunState

GOOD = '{"sentiment": "mixed", "score": 0.75, "summary": "nice but pricey"}'


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


def _usage():
    return SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)


def _resp(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [], usage=_usage(), model="m")


class _SeqLLM:
    """Fake LLM that returns a programmed sequence of contents (repeats the last)."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        content = self._contents.pop(0) if self._contents else None
        if content is None and self.calls > 1:
            content = "still not json"
        return _resp(content)


def _agent(*, strict=False, with_schema=True):
    return BaseAgent(
        name="reviewer",
        instructions="Analyze the review.",
        config=AgentConfig(log_to_session=False, input_sanitization=False),
        output_schema=Review if with_schema else None,
        output_schema_strict=strict,
    )


def _ctx():
    return RunContext(run_id="run-so", max_turns=5)


def _run_state():
    rs = RunState(run_id="run-so")
    rs.push_agent("reviewer")
    return rs


async def _run(llm, agent):
    ex = Executor(llm_client=llm)
    return await ex.execute_loop(
        agent,
        [{"role": "user", "content": "The hotel was fantastic but expensive."}],
        _ctx(),
        _run_state(),
    )


class TestNoToolAgent:
    async def test_correct_json_populates_structured_output(self):
        llm = _SeqLLM([GOOD])
        resp = await _run(llm, _agent())
        assert isinstance(resp.structured_output, Review)
        assert resp.structured_output.score == 0.75
        assert resp.structured_output_error is None
        assert llm.calls == 1  # single constrained call, no retry

    async def test_wrong_then_retry_succeeds(self):
        # First (constrained) call returns wrong shape; the formatting retry fixes it.
        llm = _SeqLLM(['{"review_summary": {"x": 1}}', GOOD])
        resp = await _run(llm, _agent())
        assert isinstance(resp.structured_output, Review)
        assert resp.structured_output_error is None
        assert llm.calls == 2  # main + 1 formatting retry

    async def test_always_wrong_soft_fail(self):
        llm = _SeqLLM(["not json", "still not", "nope"])
        resp = await _run(llm, _agent(strict=False))
        assert resp.structured_output is None
        assert resp.structured_output_error is not None  # visible, not silent
        assert resp.content is not None  # text preserved
        # Budget: 1 constrained call (counts as primary) + 1 retry = 2 total.
        # The constrained inline attempt must not be re-spent as an extra retry.
        assert llm.calls == 2

    async def test_always_wrong_strict_raises(self):
        llm = _SeqLLM(["not json", "still not", "nope"])
        with pytest.raises(StructuredOutputError) as exc:
            await _run(llm, _agent(strict=True))
        assert "Review" in str(exc.value)


class TestToolAgent:
    async def test_separate_formatting_call(self, monkeypatch):
        agent = _agent()
        # Pretend the agent has tools → main loop is NOT schema-constrained.
        monkeypatch.setattr(
            agent,
            "get_tools_for_llm",
            lambda: [{"type": "function", "function": {"name": "dummy"}}],
        )
        # Main call → plain text answer (no tool calls); formatting call → JSON.
        llm = _SeqLLM(["Here is your order summary in prose.", GOOD])
        resp = await _run(llm, agent)
        assert isinstance(resp.structured_output, Review)
        assert llm.calls == 2  # unconstrained answer + separate formatting call

    async def test_tool_agent_keeps_full_retry_budget(self, monkeypatch):
        # Tool agents' unconstrained prose is not a structured attempt, so the full
        # 1 + _MAX_STRUCTURED_OUTPUT_RETRIES formatting budget applies: when every
        # attempt fails we expect main loop (1) + 2 formatting calls = 3 total.
        agent = _agent()
        monkeypatch.setattr(
            agent,
            "get_tools_for_llm",
            lambda: [{"type": "function", "function": {"name": "dummy"}}],
        )
        llm = _SeqLLM(["prose answer", "still prose", "nope"])
        resp = await _run(llm, agent)
        assert resp.structured_output is None
        assert resp.structured_output_error is not None
        assert llm.calls == 3  # unconstrained answer + 2 formatting attempts


class TestNoSchemaInert:
    async def test_no_output_schema_is_inert(self):
        llm = _SeqLLM(["just a normal answer"])
        resp = await _run(llm, _agent(with_schema=False))
        assert resp.structured_output is None
        assert resp.structured_output_error is None
        assert resp.content == "just a normal answer"
        assert llm.calls == 1  # no formatting calls
