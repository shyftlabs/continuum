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
from continuum.agent.types import (
    EventType,
    PrepareRunResult,
    RunContext,
    RunState,
    TokenUsage,
)
from continuum.llm.config import LLMConfig
from continuum.llm.exceptions import LLMContextLengthError
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


@pytest.fixture(autouse=True)
def _clear_unsupported_params():
    # Rejections are remembered on the class; keep tests independent.
    OpenAIProvider._unsupported_params.clear()
    OpenAIProvider._stream_options_rejected.clear()
    yield
    OpenAIProvider._unsupported_params.clear()
    OpenAIProvider._stream_options_rejected.clear()


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

    @pytest.mark.parametrize(
        ("status", "error_cls", "body", "text"),
        [
            # OpenAI's own shape.
            (
                400,
                openai.BadRequestError,
                {"code": "unsupported_parameter", "param": "stream_options"},
                None,
            ),
            # OpenAI's shape, naming the nested field.
            (
                400,
                openai.BadRequestError,
                {"code": "unsupported_parameter", "param": "stream_options.include_usage"},
                None,
            ),
            # Compatible server: 400 that names the field only in the message.
            (
                400,
                openai.BadRequestError,
                {
                    "message": "Unrecognized request argument supplied: stream_options",
                    "type": "invalid_request_error",
                    "code": None,
                    "param": None,
                },
                None,
            ),
            # Strict proxy (pydantic extra="forbid"): 422 extra_forbidden.
            (
                422,
                openai.UnprocessableEntityError,
                {
                    "detail": [
                        {
                            "type": "extra_forbidden",
                            "loc": ["body", "stream_options"],
                            "msg": "Extra inputs are not permitted",
                            "input": {"include_usage": True},
                        }
                    ]
                },
                None,
            ),
            # marshmallow/webargs 422: the field is only a dict key.
            (
                422,
                openai.UnprocessableEntityError,
                {"errors": {"json": {"stream_options": ["Unknown field."]}}},
                None,
            ),
            # Joi/Boom or NestJS 400: the SDK keeps only "Bad Request" as the
            # body, so the field name is only in the raw response text.
            (
                400,
                openai.BadRequestError,
                "Bad Request",
                '{"statusCode": 400, "error": "Bad Request",'
                ' "message": "\\"stream_options\\" is not allowed"}',
            ),
        ],
        ids=[
            "openai-400",
            "openai-400-nested-param",
            "400-null-code",
            "422-extra-forbidden",
            "422-marshmallow-key",
            "400-joi-text-only",
        ],
    )
    async def test_rejected_stream_options_is_dropped_and_remembered(
        self, status, error_cls, body, text
    ):
        provider = OpenAIProvider(api_key="test-key")
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        # The message is what openai 3.x gives a 422: no hint of the field.
        rejected = error_cls(
            f"Error code: {status}",
            response=httpx.Response(status, request=request, text=text),
            body=body,
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

    def test_sync_stream_drops_stream_options_named_only_in_response_text(self):
        provider = OpenAIProvider(api_key="test-key")
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        rejected = openai.BadRequestError(
            "Error code: 400",
            response=httpx.Response(
                400,
                request=request,
                text='{"statusCode": 400, "message": "\\"stream_options\\" is not allowed"}',
            ),
            body="Bad Request",
        )
        calls = []

        def create(**kwargs):
            calls.append(dict(kwargs))
            if "stream_options" in kwargs:
                raise rejected
            return iter([_chunk("ok", finish="stop")])

        provider._client = MagicMock()
        provider._client.chat.completions.create = create

        chunks = list(
            provider.stream([{"role": "user", "content": "hi"}], LLMConfig(model="joi-model"))
        )

        assert chunks[-1].is_finished
        assert ["stream_options" in c for c in calls] == [True, False]

    async def _assert_not_adapted(self, rejected, model="strict-model"):
        provider = OpenAIProvider(api_key="test-key")
        create = AsyncMock(side_effect=rejected)
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = create

        with pytest.raises(Exception):
            [
                c
                async for c in provider.astream(
                    [{"role": "user", "content": "hi"}], LLMConfig(model=model)
                )
            ]
        assert create.await_count == 1
        assert OpenAIProvider._unsupported_params == {}
        assert OpenAIProvider._stream_options_rejected == set()

    async def test_unrelated_422_whose_echoed_input_quotes_stream_options_is_not_adapted(self):
        # A 422 for another field echoes the whole request under "input". Here
        # the echo names stream_options in a plain string (the user's own
        # message), so only the skip of "input" keeps it from looking like a
        # rejection of stream_options.
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        await self._assert_not_adapted(
            openai.UnprocessableEntityError(
                "Error code: 422",
                response=httpx.Response(422, request=request),
                body={
                    "detail": [
                        {
                            "type": "missing",
                            "loc": ["body", "model"],
                            "msg": "Field required",
                            "input": {
                                "messages": [
                                    {"role": "user", "content": "what does stream_options do?"}
                                ]
                            },
                        }
                    ]
                },
            )
        )

    @pytest.mark.parametrize(
        "message",
        [
            # str(pydantic.ValidationError), as vLLM returns it.
            "1 validation error for ChatCompletionRequest\n  Value error, When using "
            "`tool_choice`, `tools` must be set. [type=value_error, input_value="
            "{'model': 'm', 'messages': [], 'stream': True, 'stream_options': "
            "{'include_usage': True}, 'tool_choice': 'auto'}, input_type=dict]",
            # repr of ValidationError.errors(), the other form vLLM has used.
            "[{'type': 'value_error', 'loc': ('body',), 'msg': 'Value error, When "
            "using `tool_choice`, `tools` must be set.', 'input': {'model': 'm', "
            "'stream': True, 'stream_options': {'include_usage': True}}}]",
        ],
        ids=["pydantic-str", "errors-repr"],
    )
    async def test_vllm_400_that_echoes_the_request_is_not_adapted(self, message):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        body = {"object": "error", "message": message, "type": "BadRequestError", "code": 400}
        await self._assert_not_adapted(
            openai.BadRequestError(
                "Error code: 400",
                response=httpx.Response(400, request=request, json=body),
                body=body,
            )
        )

    async def test_rejection_is_remembered_per_server_not_per_model_name(self):
        request = httpx.Request("POST", "https://strict.test/v1/chat/completions")
        rejected = openai.BadRequestError(
            "Error code: 400",
            response=httpx.Response(400, request=request),
            body={"message": "Unrecognized request argument supplied: stream_options"},
        )
        calls: dict[str, list[bool]] = {"strict": [], "open": []}

        def make_create(name, reject):
            async def create(**kwargs):
                calls[name].append("stream_options" in kwargs)
                if reject and "stream_options" in kwargs:
                    raise rejected
                return _aiter([_chunk("ok", finish="stop")])

            return create

        strict = OpenAIProvider(api_key="test-key", api_base="https://strict.test/v1")
        strict._async_client = MagicMock()
        strict._async_client.chat.completions.create = make_create("strict", True)
        other = OpenAIProvider(api_key="test-key", api_base="https://open.test/v1")
        other._async_client = MagicMock()
        other._async_client.chat.completions.create = make_create("open", False)
        cfg = LLMConfig(model="shared-model")
        msgs = [{"role": "user", "content": "hi"}]

        [c async for c in strict.astream(msgs, cfg)]
        [c async for c in other.astream(msgs, cfg)]
        [c async for c in strict.astream(msgs, cfg)]

        assert calls == {"strict": [True, False, False], "open": [True]}

    async def test_context_length_400_quoting_stream_options_stays_a_context_error(self):
        provider = OpenAIProvider(api_key="test-key")
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        body = {
            "message": "This model's maximum context length is 4096 tokens, but the "
            "request (with stream_options) needs 5000 tokens.",
            "code": None,
            "param": None,
        }
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = AsyncMock(
            side_effect=openai.BadRequestError(
                f"Error code: 400 - {body}",
                response=httpx.Response(400, request=request),
                body=body,
            )
        )

        with pytest.raises(LLMContextLengthError):
            await provider.acomplete([{"role": "user", "content": "hi"}], LLMConfig(model="m"))


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


async def test_run_stream_adds_handoff_child_usage():
    from continuum.agent.types import AgentResponse, HandoffResult
    from continuum.llm.types import FunctionCall, ToolCall

    turns = iter(
        [
            [
                StreamChunk(
                    model="parent-m",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            function=FunctionCall(name="transfer_to_child", arguments="{}"),
                        )
                    ],
                    finish_reason="tool_calls",
                    is_finished=True,
                    usage=Usage(prompt_tokens=30, completion_tokens=8, total_tokens=38),
                )
            ],
            [StreamChunk(content="done", finish_reason="stop", is_finished=True)],
        ]
    )

    async def chat_stream(*args, **kwargs):
        for chunk in next(turns):
            yield chunk

    runner = _runner(chat_stream)
    child_usage = TokenUsage(
        prompt_tokens=400,
        completion_tokens=100,
        total_tokens=500,
        model_usage={
            "child-m": {"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500}
        },
    )
    runner._handoff_executor = MagicMock()
    runner._handoff_executor.execute_handoff = AsyncMock(
        return_value=HandoffResult(
            handoff_id="h1",
            from_agent="a",
            to_agent="child",
            success=True,
            response=AgentResponse(content="child answer", usage=child_usage),
        )
    )
    agent = _agent()
    agent.is_handoff_tool_call = lambda name: (name == "transfer_to_child", "child")

    events = [e async for e in runner.run_stream(agent, "hi")]

    [run_end] = [e for e in events if e.type == EventType.RUN_END]
    assert run_end.data["usage"]["total_tokens"] == 538
    assert set(run_end.data["usage"]["model_usage"]) == {"parent-m", "child-m"}


async def test_every_run_end_carries_usage():
    # Error exit: the stream fails mid-turn.
    async def chat_stream(*args, **kwargs):
        yield StreamChunk(content="x", model="m")
        raise RuntimeError("boom")

    runner = _runner(chat_stream)
    events = []
    with pytest.raises(Exception):
        async for e in runner.run_stream(_agent(), "hi"):
            events.append(e)
    [run_end] = [e for e in events if e.type == EventType.RUN_END]
    assert run_end.data["usage"]["total_tokens"] == 0

    # Validation-failure exit.
    runner = _runner(chat_stream)
    runner._prepare_run = AsyncMock(return_value=PrepareRunResult(success=False))
    events = [e async for e in runner.run_stream(_agent(), "hi")]
    [run_end] = [e for e in events if e.type == EventType.RUN_END]
    assert run_end.data["usage"] == TokenUsage().to_dict()
