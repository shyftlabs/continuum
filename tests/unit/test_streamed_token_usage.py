"""Streamed runs report token usage.

OpenAI-compatible servers (OpenAI, the Smart Gateway) send usage in a last chunk
when asked with stream_options.include_usage. The provider asks for it and puts
it on the closing StreamChunk; run_stream sums it per turn into the response,
the decision trace and the RUN_END event, as the non-streaming path already does.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig, AgentMemoryConfig
from continuum.agent.types import EventType, PrepareRunResult, RunContext, RunState
from continuum.llm.config import LLMConfig
from continuum.llm.providers.openai_provider import OpenAIProvider
from continuum.llm.types import StreamChunk, Usage


def _chunk(content=None, finish=None, usage=None, model="gpt-4o-mini-2024-07-18"):
    choices = []
    if content is not None or finish is not None:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content, role=None, tool_calls=None),
                finish_reason=finish,
            )
        ]
    return SimpleNamespace(id="c1", model=model, choices=choices, usage=usage)


def _aiter(items):
    async def gen():
        for item in items:
            yield item

    return gen()


class TestProviderStreamUsage:
    async def test_requests_usage_and_puts_it_on_the_closing_chunk(self):
        provider = OpenAIProvider(api_key="test-key")
        create = AsyncMock(
            return_value=_aiter(
                [
                    _chunk("Hel"),
                    _chunk("lo", finish="stop"),
                    _chunk(
                        usage=SimpleNamespace(
                            prompt_tokens=11, completion_tokens=2, total_tokens=13
                        )
                    ),
                ]
            )
        )
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = create

        chunks = [
            c
            async for c in provider.astream(
                [{"role": "user", "content": "hi"}], LLMConfig(model="gpt-4o-mini")
            )
        ]

        assert create.call_args.kwargs["stream_options"] == {"include_usage": True}
        assert "".join(c.content or "" for c in chunks) == "Hello"
        last = chunks[-1]
        assert last.is_finished and last.finish_reason == "stop"
        assert last.usage == Usage(prompt_tokens=11, completion_tokens=2, total_tokens=13)
        assert last.model == "gpt-4o-mini-2024-07-18"
        assert all(c.usage is None for c in chunks[:-1])

    async def test_server_without_usage_still_streams(self):
        provider = OpenAIProvider(api_key="test-key")
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = AsyncMock(
            return_value=_aiter([_chunk("ok", finish="stop")])
        )
        chunks = [
            c
            async for c in provider.astream(
                [{"role": "user", "content": "hi"}], LLMConfig(model="m")
            )
        ]
        assert chunks[-1].is_finished and chunks[-1].usage is None

    async def test_rejected_stream_options_is_dropped_and_remembered(self):
        provider = OpenAIProvider(api_key="test-key")
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        rejected = openai.BadRequestError(
            "unsupported",
            response=httpx.Response(400, request=request),
            body={"code": "unsupported_parameter", "param": "stream_options"},
        )
        calls = []

        async def create(**kwargs):
            calls.append(dict(kwargs))
            if "stream_options" in kwargs:
                raise rejected
            return _aiter([_chunk("ok", finish="stop")])

        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = create
        cfg = LLMConfig(model="old-compat-model")

        [c async for c in provider.astream([{"role": "user", "content": "hi"}], cfg)]
        [c async for c in provider.astream([{"role": "user", "content": "hi"}], cfg)]

        assert ["stream_options" in c for c in calls] == [True, False, False]


# ---------------------------------------------------------------------------
# run_stream
# ---------------------------------------------------------------------------


def _agent() -> BaseAgent:
    return BaseAgent(
        name="a",
        instructions="x",
        model="model-x",
        config=AgentConfig(),
        memory_config=AgentMemoryConfig(),
    )


def _runner(chat_stream):
    from continuum.agent.runner import AgentRunner
    from continuum.agent.utils.circuit_breaker import CircuitBreaker

    runner = AgentRunner.__new__(AgentRunner)
    runner._llm_client = MagicMock()
    runner._llm_client.chat_stream = chat_stream
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
    rs.messages = [{"role": "user", "content": "hi"}]
    runner._prepare_run = AsyncMock(
        return_value=PrepareRunResult(
            success=True,
            context=RunContext(run_id="run-test", max_turns=3),
            run_state=rs,
            user_message_index=0,
        )
    )
    return runner


async def test_run_stream_reports_usage():
    async def chat_stream(*args, **kwargs):
        yield StreamChunk(content="Hello", model="gpt-4o-mini-2024-07-18")
        yield StreamChunk(
            model="gpt-4o-mini-2024-07-18",
            finish_reason="stop",
            is_finished=True,
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

    runner = _runner(chat_stream)
    events = [e async for e in runner.run_stream(_agent(), "hi")]

    [run_end] = [e for e in events if e.type == EventType.RUN_END]
    assert run_end.data["usage"]["total_tokens"] == 120
    assert run_end.data["usage"]["model_usage"] == {
        "gpt-4o-mini-2024-07-18": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
    }
    response = runner._finalizer.finalize.await_args.args[3]
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (100, 20)


@pytest.mark.parametrize("usage", [None])
async def test_run_stream_without_provider_usage_is_zero(usage):
    async def chat_stream(*args, **kwargs):
        yield StreamChunk(content="Hello", finish_reason="stop", is_finished=True, usage=usage)

    runner = _runner(chat_stream)
    events = [e async for e in runner.run_stream(_agent(), "hi")]
    [run_end] = [e for e in events if e.type == EventType.RUN_END]
    assert run_end.data["usage"]["total_tokens"] == 0
