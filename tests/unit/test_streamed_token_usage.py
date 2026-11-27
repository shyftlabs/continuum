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

    async def _assert_not_remembered(self, rejected, model="strict-model"):
        # A 400/422 that is not about stream_options is retried once without it,
        # fails again, and leaves nothing remembered.
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
        assert ["stream_options" in c.kwargs for c in create.await_args_list] == [True, False]
        assert OpenAIProvider._unsupported_params == {}
        assert OpenAIProvider._stream_options_rejected == set()

    async def test_unrelated_422_whose_echoed_input_quotes_stream_options_is_not_remembered(self):
        # A 422 for another field echoes the whole request under "input", here
        # with a user message that names stream_options. The retry fails too.
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        await self._assert_not_remembered(
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
    async def test_vllm_400_that_echoes_the_request_is_not_remembered(self, message):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        body = {"object": "error", "message": message, "type": "BadRequestError", "code": 400}
        await self._assert_not_remembered(
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


def _api_error(status, body, text=None):
    """The openai SDK error for `status`, as the client raises it."""
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = (
        httpx.Response(status, request=request, text=text)
        if text is not None
        else httpx.Response(status, request=request, json=body)
    )
    cls = {
        400: openai.BadRequestError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
    }.get(status, openai.InternalServerError)
    return cls(f"Error code: {status}", response=response, body=body)


class TestRound3:
    """Opt-out, 500/501 rejections, a missing total, and errors that only mention the field."""

    async def _stream_calls(self, rejected, model="shim-model", stream_twice=True):
        provider = OpenAIProvider(api_key="test-key")
        calls = []

        async def create(**kwargs):
            calls.append("stream_options" in kwargs)
            if "stream_options" in kwargs:
                raise rejected
            return _aiter([_chunk("ok", finish="stop")])

        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = create
        cfg = LLMConfig(model=model)
        msgs = [{"role": "user", "content": "hi"}]
        for _ in range(2 if stream_twice else 1):
            [c async for c in provider.astream(msgs, cfg)]
        return calls

    @pytest.mark.parametrize("via", ["setting", "config"])
    def test_opt_out_never_sends_stream_options(self, via, monkeypatch):
        from continuum.config import settings

        if via == "setting":
            monkeypatch.setattr(settings, "llm_stream_include_usage", False)
            cfg = LLMConfig(model="m")
        else:
            cfg = LLMConfig(model="m", stream_include_usage=False)
        provider = OpenAIProvider(api_key="test-key")
        provider._client = MagicMock()
        provider._client.chat.completions.create = MagicMock(
            return_value=iter([_chunk("ok", finish="stop")])
        )

        list(provider.stream([{"role": "user", "content": "hi"}], cfg))

        assert "stream_options" not in provider._client.chat.completions.create.call_args.kwargs

    def test_include_usage_defaults_on(self):
        assert LLMConfig(model="m").stream_include_usage is True

    @pytest.mark.parametrize(
        ("status", "body", "text"),
        [
            (500, {"detail": "create() got an unexpected keyword argument 'stream_options'"}, None),
            (501, {"error": {"message": "property stream_options should not exist"}}, None),
            # Body reduced to the status text; the field is named only in the raw text.
            (
                500,
                "Internal Server Error",
                "TypeError: unexpected keyword argument 'stream_options'",
            ),
        ],
        ids=["500-kwarg", "501-nest", "500-text-only"],
    )
    async def test_500_or_501_naming_stream_options_is_dropped_and_remembered(
        self, status, body, text
    ):
        calls = await self._stream_calls(_api_error(status, body, text))
        assert calls == [True, False, False]

    def test_sync_500_naming_stream_options_is_dropped(self):
        provider = OpenAIProvider(api_key="test-key")
        rejected = _api_error(500, {"detail": "unexpected keyword argument 'stream_options'"})
        calls = []

        def create(**kwargs):
            calls.append("stream_options" in kwargs)
            if "stream_options" in kwargs:
                raise rejected
            return iter([_chunk("ok", finish="stop")])

        provider._client = MagicMock()
        provider._client.chat.completions.create = create
        chunks = list(provider.stream([{"role": "user", "content": "hi"}], LLMConfig(model="m")))
        assert chunks[-1].is_finished
        assert calls == [True, False]

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (500, {"message": "Internal Server Error"}),
            (502, {"message": "unexpected keyword argument 'stream_options'"}),
            (503, {"message": "Service Unavailable"}),
            (429, {"message": "Rate limit reached"}),
        ],
        ids=["500-plain", "502-names-field", "503", "429"],
    )
    async def test_other_errors_are_not_retried_without_stream_options(self, status, body):
        rejected = _api_error(status, body)
        provider = OpenAIProvider(api_key="test-key")
        create = AsyncMock(side_effect=rejected)
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = create

        with pytest.raises(Exception):
            [
                c
                async for c in provider.astream(
                    [{"role": "user", "content": "hi"}], LLMConfig(model="m")
                )
            ]
        assert create.await_count == 1
        assert OpenAIProvider._stream_options_rejected == set()

    @pytest.mark.parametrize(
        "message",
        [
            # FastAPI/vLLM: a validation error on another field echoes the request,
            # whose user message names the field.
            "[{'type': 'value_error', 'loc': ('body',), 'msg': 'Value error, When "
            "using `tool_choice`, `tools` must be set.', 'input': {'model': 'm', "
            "'messages': [{'role': 'user', 'content': 'How do I set stream_options "
            "in the OpenAI API?'}], 'stream': True, 'stream_options': "
            "{'include_usage': True}, 'tool_choice': 'required'}}]",
            # serde: another unknown field, stream_options among the accepted ones.
            "unknown field `foo`, expected one of `model`, `messages`, `stream_options`",
            # Context-length error quoting the prompt.
            "This model's maximum context length is 4096 tokens. Your messages "
            "('How do I set stream_options?') resulted in 5000 tokens.",
        ],
        ids=["echoed-user-message", "serde-expected-list", "context-length-quote"],
    )
    async def test_message_that_only_mentions_stream_options_is_not_remembered(self, message):
        rejected = _api_error(400, {"object": "error", "message": message, "code": 400})
        provider = OpenAIProvider(api_key="test-key")
        create = AsyncMock(side_effect=rejected)
        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = create

        with pytest.raises(Exception):
            [
                c
                async for c in provider.astream(
                    [{"role": "user", "content": "hi"}], LLMConfig(model="m")
                )
            ]
        # Retried once without the field; that fails too, so nothing is remembered.
        assert create.await_count == 2
        assert OpenAIProvider._stream_options_rejected == set()

    @pytest.mark.parametrize(
        "message",
        [
            "unknown field `stream_options`, expected one of `model`, `messages`",
            "property stream_options should not exist",
            "Unrecognized request arguments supplied: foo, stream_options",
        ],
        ids=["serde", "nest", "openai-plural"],
    )
    async def test_rejection_wording_in_free_text_is_adapted(self, message):
        calls = await self._stream_calls(_api_error(400, {"message": message}))
        assert calls == [True, False, False]

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    @pytest.mark.parametrize(
        ("status", "body"),
        [
            # Go encoding/json DisallowUnknownFields / grpc-gateway.
            (400, {"code": 3, "message": 'json: unknown field "stream_options"'}),
            # Jackson FAIL_ON_UNKNOWN_PROPERTIES.
            (
                400,
                {
                    "message": 'JSON parse error: Unrecognized field "stream_options" '
                    "(class com.example.ChatRequest), not marked as ignorable"
                },
            ),
            # Zod .strict(): message plus the structured keys list.
            (
                400,
                {
                    "error": [
                        {
                            "code": "unrecognized_keys",
                            "keys": ["stream_options"],
                            "path": [],
                            "message": "Unrecognized key(s) in object: 'stream_options'",
                        }
                    ]
                },
            ),
            # AJV additionalProperties: false.
            (
                400,
                {
                    "message": "body must NOT have additional properties, "
                    'params.additionalProperty: "stream_options"',
                    "errors": [
                        {
                            "keyword": "additionalProperties",
                            "params": {"additionalProperty": "stream_options"},
                            "message": "must NOT have additional properties",
                        }
                    ],
                },
            ),
            # Python jsonschema / Connexion.
            (
                400,
                {
                    "detail": "Additional properties are not allowed "
                    "('stream_options' was unexpected)",
                    "status": 400,
                    "title": "Bad Request",
                },
            ),
            # str(pydantic.ValidationError) in a 400 message.
            (
                400,
                {
                    "object": "error",
                    "message": "1 validation error for ChatCompletionRequest\n"
                    "stream_options\n  Extra inputs are not permitted "
                    "[type=extra_forbidden, input_value={'include_usage': True}, "
                    "input_type=dict]",
                    "code": 400,
                },
            ),
            # Groq.
            (
                400,
                {
                    "error": {
                        "message": "property 'stream_options' is unsupported",
                        "type": "invalid_request_error",
                    }
                },
            ),
        ],
        ids=["go-json", "jackson", "zod", "ajv", "jsonschema", "pydantic-str", "groq"],
    )
    async def test_strict_server_rejection_shapes_are_adapted(self, status, body, sync):
        rejected = _api_error(status, body)
        if not sync:
            assert await self._stream_calls(rejected) == [True, False, False]
            return
        provider = OpenAIProvider(api_key="test-key")
        calls = []

        def create(**kwargs):
            calls.append("stream_options" in kwargs)
            if "stream_options" in kwargs:
                raise rejected
            return iter([_chunk("ok", finish="stop")])

        provider._client = MagicMock()
        provider._client.chat.completions.create = create
        cfg = LLMConfig(model="shim-model")
        for _ in range(2):
            chunks = list(provider.stream([{"role": "user", "content": "hi"}], cfg))
            assert chunks[-1].is_finished
        assert calls == [True, False, False]

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    async def test_missing_total_is_prompt_plus_completion(self, sync):
        chunks_in = [
            _chunk("ok", finish="stop"),
            _chunk(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=None)),
        ]
        provider = OpenAIProvider(api_key="test-key")
        msgs = [{"role": "user", "content": "hi"}]
        cfg = LLMConfig(model="m")
        if sync:
            provider._client = MagicMock()
            provider._client.chat.completions.create = MagicMock(return_value=iter(chunks_in))
            chunks = list(provider.stream(msgs, cfg))
        else:
            provider._async_client = MagicMock()
            provider._async_client.chat.completions.create = AsyncMock(
                return_value=_aiter(chunks_in)
            )
            chunks = [c async for c in provider.astream(msgs, cfg)]

        assert chunks[-1].usage == Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10)


def _stub_provider(sync, create_impl):
    """A provider whose (a)sync create runs `create_impl(kwargs)`; returns (provider, calls)."""
    provider = OpenAIProvider(api_key="test-key")
    calls: list[dict] = []
    if sync:

        def create(**kwargs):
            calls.append(dict(kwargs))
            return iter(create_impl(kwargs))

        provider._client = MagicMock()
        provider._client.chat.completions.create = create
    else:

        async def acreate(**kwargs):
            calls.append(dict(kwargs))
            return _aiter(create_impl(kwargs))

        provider._async_client = MagicMock()
        provider._async_client.chat.completions.create = acreate
    return provider, calls


async def _consume(provider, sync, content="hi", model="shim-model"):
    msgs = [{"role": "user", "content": content}]
    cfg = LLMConfig(model=model)
    if sync:
        return list(provider.stream(msgs, cfg))
    return [c async for c in provider.astream(msgs, cfg)]


class TestRound6:
    """Any 400/422/501 at stream open earns one retry without stream_options;
    the rejection is remembered only if that retry opens."""

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    @pytest.mark.parametrize(
        ("status", "body", "text"),
        [
            # OpenAI's own unknown_parameter code, naming the nested field.
            (
                400,
                {
                    "error": {
                        "message": "Unknown parameter: 'stream_options.include_usage'.",
                        "type": "invalid_request_error",
                        "param": "stream_options.include_usage",
                        "code": "unknown_parameter",
                    }
                },
                None,
            ),
            (400, {"error": {"message": "Unknown parameter: 'stream_options'."}}, None),
            # Zod 4.
            (
                400,
                {
                    "error": [
                        {
                            "code": "unrecognized_keys",
                            "message": 'Unrecognized key: "stream_options"',
                        }
                    ]
                },
                None,
            ),
            # pydantic v1.
            (422, {"detail": "stream_options\n  extra fields not permitted"}, None),
            # Unquoted, as JSON detail and as plain text.
            (400, {"detail": "unknown field stream_options"}, None),
            (400, "Bad Request", "unknown field stream_options"),
            # Bodies that never name the field.
            (400, {"message": "body must NOT have additional properties"}, None),
            (400, "Bad Request", "Bad Request"),
            # 501.
            (501, {"error": {"message": "stream_options is not implemented"}}, None),
            (501, "Not Implemented", "Not Implemented"),
        ],
        ids=[
            "openai-unknown-parameter",
            "unknown-parameter-no-param",
            "zod4",
            "pydantic-v1",
            "unquoted-json",
            "unquoted-text",
            "ajv-default",
            "plain-bad-request",
            "501-not-implemented",
            "501-plain",
        ],
    )
    async def test_rejection_shapes_are_adapted_and_remembered(self, status, body, text, sync):
        rejected = _api_error(status, body, text)

        def impl(kwargs):
            if "stream_options" in kwargs:
                raise rejected
            return [_chunk("ok", finish="stop")]

        provider, calls = _stub_provider(sync, impl)
        for _ in range(2):
            chunks = await _consume(provider, sync)
            assert chunks[-1].is_finished
        assert ["stream_options" in c for c in calls] == [True, False, False]

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    @pytest.mark.parametrize("status", [400, 500])
    async def test_failure_echoing_rejection_wording_does_not_poison_the_memo(self, status, sync):
        # The prompt carries rejection wording; the server fails on it with or
        # without stream_options, echoing it each time.
        prompt = 'what does the error: unknown field "stream_options" mean?'
        healthy = {"on": False}

        def impl(kwargs):
            if not healthy["on"]:
                raise _api_error(status, {"message": f"moderation failed for: {prompt}"})
            return [
                _chunk("ok", finish="stop"),
                _chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4)),
            ]

        provider, calls = _stub_provider(sync, impl)
        with pytest.raises(Exception):
            await _consume(provider, sync, content=prompt)
        assert ["stream_options" in c for c in calls] == [True, False]
        assert OpenAIProvider._stream_options_rejected == set()

        healthy["on"] = True
        chunks = await _consume(provider, sync)
        assert calls[-1]["stream_options"] == {"include_usage": True}
        assert chunks[-1].usage == Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4)

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    async def test_failed_retry_raises_the_retry_error(self, sync):
        first = _api_error(400, {"message": "first"})
        second = _api_error(400, {"message": "second"})

        def impl(kwargs):
            raise first if "stream_options" in kwargs else second

        provider, _ = _stub_provider(sync, impl)
        with pytest.raises(Exception) as info:
            await _consume(provider, sync)
        err = info.value
        assert err is second or err.__cause__ is second
        assert OpenAIProvider._stream_options_rejected == set()

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    @pytest.mark.parametrize(
        "make_error",
        [
            lambda: _api_error(429, {"message": "stream_options rate limited"}),
            lambda: _api_error(500, {"message": "Internal Server Error"}),
            lambda: _api_error(502, {"message": "unknown field stream_options"}),
            lambda: _api_error(503, {"message": "Service Unavailable"}),
            lambda: openai.AuthenticationError(
                "Error code: 401",
                response=httpx.Response(
                    401, request=httpx.Request("POST", "https://example.test/v1")
                ),
                body={"message": "stream_options: invalid api key"},
            ),
            lambda: openai.APITimeoutError(
                request=httpx.Request("POST", "https://example.test/v1")
            ),
            lambda: openai.APIConnectionError(
                request=httpx.Request("POST", "https://example.test/v1")
            ),
        ],
        ids=["429", "500-plain", "502", "503", "401", "timeout", "connection"],
    )
    async def test_other_errors_are_not_retried(self, make_error, sync):
        error = make_error()

        def impl(kwargs):
            raise error

        provider, calls = _stub_provider(sync, impl)
        with pytest.raises(Exception):
            await _consume(provider, sync)
        assert len(calls) == 1
        assert OpenAIProvider._stream_options_rejected == set()

    @pytest.mark.parametrize("sync", [False, True], ids=["astream", "stream"])
    async def test_coded_rejection_of_another_param_keeps_stream_options(self, sync):
        rejected = _api_error(
            400, {"code": "unsupported_parameter", "param": "temperature", "message": "no"}
        )

        def impl(kwargs):
            if "temperature" in kwargs:
                raise rejected
            return [_chunk("ok", finish="stop")]

        provider, calls = _stub_provider(sync, impl)
        await _consume(provider, sync)
        assert ["stream_options" in c for c in calls] == [True, True]
        assert "temperature" not in calls[-1]
        assert OpenAIProvider._stream_options_rejected == set()


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
