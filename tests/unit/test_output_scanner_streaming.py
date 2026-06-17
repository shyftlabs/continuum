"""
Tests for output_scanner behaviour, especially the streaming path.

Covers the fix that makes `output_scanners` apply BEFORE tokens reach the
client in `run_stream`: when scanners are configured, raw per-token
CONTENT_DELTA events are suppressed and a single redacted CONTENT_COMPLETE is
emitted instead. Without scanners, token streaming is unchanged.

Also unit-tests the shared `apply_output_scanners` helper (chaining + fail-open).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.types import EventType, PrepareRunResult, RunContext, RunState
from continuum.agent.utils.validation_utils import apply_output_scanners
from continuum.llm.types import StreamChunk

EMAIL = "alice@example.com"
REDACTED = "[REDACTED]"


def _redactor(prompt: str, output: str) -> tuple[str, bool, str | None]:
    return output.replace(EMAIL, REDACTED), True, None


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def _agent(scanners=None) -> BaseAgent:
    return BaseAgent(
        name="a",
        instructions="x",
        model="model-x",
        config=AgentConfig(output_scanners=scanners or []),
        memory_config=AgentMemoryConfig(),
    )


class TestApplyOutputScanners:
    def test_redacts(self):
        out = apply_output_scanners(_agent([_redactor]), "p", f"mail {EMAIL}")
        assert out == f"mail {REDACTED}"

    def test_no_scanners_is_passthrough(self):
        assert apply_output_scanners(_agent([]), "p", "hello") == "hello"

    def test_chained_scanners_apply_in_order(self):
        def upper(prompt, output):
            return output.upper(), True, None

        out = apply_output_scanners(_agent([_redactor, upper]), "p", f"x {EMAIL}")
        assert out == f"X {REDACTED}".upper()  # redact then uppercase

    def test_fail_open_on_scanner_exception(self):
        def boom(prompt, output):
            raise RuntimeError("scanner crashed")

        # Crashing scanner is skipped; content survives unmodified.
        assert apply_output_scanners(_agent([boom]), "p", "keepme") == "keepme"


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


def _build_stream_runner(llm):
    from continuum.agent.runner import AgentRunner
    from continuum.agent.utils.circuit_breaker import CircuitBreaker

    runner = AgentRunner.__new__(AgentRunner)
    runner._llm_client = llm
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

    # Skip the heavy prepare; provide a run_state with a user message.
    rs = RunState(run_id="run-test")
    rs.messages = [{"role": "user", "content": "what is my email?"}]
    runner._prepare_run = AsyncMock(
        return_value=PrepareRunResult(
            success=True,
            context=RunContext(run_id="run-test", max_turns=3),
            run_state=rs,
            user_message_index=0,
        )
    )
    return runner


def _content_stream():
    async def _stream(*args, **kwargs):
        yield StreamChunk(content="My email is ")
        yield StreamChunk(content=EMAIL, is_finished=True)

    return _stream


async def _collect(runner, agent):
    return [e async for e in runner.run_stream(agent, "what is my email?")]


class TestStreamingOutputScanner:
    async def test_scanners_suppress_raw_deltas_and_redact_complete(self):
        agent = _agent([_redactor])
        runner = _build_stream_runner(MagicMock())
        runner._llm_client.chat_stream = _content_stream()

        events = await _collect(runner, agent)

        deltas = [e for e in events if e.type == EventType.CONTENT_DELTA]
        completes = [e for e in events if e.type == EventType.CONTENT_COMPLETE]

        # No raw token ever streamed out.
        assert deltas == []
        # The only content the client sees is the redacted complete.
        assert len(completes) == 1
        assert completes[0].data["content"] == "My email is [REDACTED]"
        assert EMAIL not in completes[0].data["content"]

    async def test_no_scanners_streams_raw_deltas(self):
        agent = _agent([])  # no scanners → normal streaming
        runner = _build_stream_runner(MagicMock())
        runner._llm_client.chat_stream = _content_stream()

        events = await _collect(runner, agent)

        deltas = [e for e in events if e.type == EventType.CONTENT_DELTA]
        # Tokens streamed live, including the raw email.
        assert deltas, "expected CONTENT_DELTA events when no scanners configured"
        streamed = "".join(e.data["content"] for e in deltas)
        assert EMAIL in streamed
