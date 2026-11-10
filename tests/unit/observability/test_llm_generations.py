"""Model calls are traced as generations with model and usage, so Langfuse can price them.

Before this, every LLM call was a plain span: Langfuse showed the call but no
model, no token counts and therefore no cost.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from continuum.llm.types import StreamChunk
from continuum.observability.decorators import observe
from continuum.observability.trace_context import restore_trace_context, set_trace_context


def _trace():
    trace = MagicMock(name="trace")
    obs = MagicMock(name="observation")
    trace.generation.return_value = obs
    trace.span.return_value = obs
    token = set_trace_context(trace_id="t1", trace_client=trace)
    return trace, obs, token


def _response():
    return SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        content="ok",
    )


async def test_async_call_is_a_generation_with_model_and_usage():
    @observe(name="llm_chat", as_type="generation")
    async def chat():
        return _response()

    trace, obs, token = _trace()
    try:
        await chat()
    finally:
        restore_trace_context(token)

    trace.generation.assert_called_once()
    trace.span.assert_not_called()
    end = obs.end.call_args.kwargs
    assert end["model"] == "gpt-4o-mini-2024-07-18"
    assert end["usage_details"] == {"input": 120, "output": 30, "total": 150}
    # Langfuse v2 servers read only `usage`
    assert end["usage"] == {"input": 120, "output": 30, "total": 150, "unit": "TOKENS"}


def test_sync_call_is_a_generation():
    @observe(name="llm_chat_sync", as_type="generation")
    def chat():
        return _response()

    trace, obs, token = _trace()
    try:
        chat()
    finally:
        restore_trace_context(token)
    assert obs.end.call_args.kwargs["usage_details"]["total"] == 150


async def test_stream_records_usage_from_closing_chunk_and_first_token_time():
    # The closing chunk's usage comes with the streamed-usage fix (StreamChunk.usage);
    # any chunk object carrying .usage / .model is read the same way.
    @observe(name="llm_chat_stream", capture_output=False, as_type="generation")
    async def stream():
        yield SimpleNamespace(content="He", model="gpt-4o-mini-2024-07-18", usage=None)
        yield SimpleNamespace(content="llo", model=None, usage=None)
        yield SimpleNamespace(
            content=None,
            model=None,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    trace, obs, token = _trace()
    try:
        assert [c.content async for c in stream()] == ["He", "llo", None]
    finally:
        restore_trace_context(token)

    end = obs.end.call_args.kwargs
    assert end["model"] == "gpt-4o-mini-2024-07-18"
    assert end["usage_details"] == {"input": 10, "output": 2, "total": 12}
    assert end["completion_start_time"] is not None
    assert "output" in end and end["output"] is None  # stream content still not captured


async def test_stream_without_usage_still_names_the_model():
    @observe(name="llm_chat_stream", capture_output=False, as_type="generation")
    async def stream():
        yield StreamChunk(content="x", model="m-1")

    trace, obs, token = _trace()
    try:
        [c async for c in stream()]
    finally:
        restore_trace_context(token)
    end = obs.end.call_args.kwargs
    assert end["model"] == "m-1" and "usage_details" not in end


async def test_plain_spans_are_unchanged():
    @observe(name="tool")
    async def tool():
        return _response()

    trace, obs, token = _trace()
    try:
        await tool()
    finally:
        restore_trace_context(token)
    trace.span.assert_called_once()
    trace.generation.assert_not_called()
    end = obs.end.call_args.kwargs
    assert "model" not in end and "usage_details" not in end


def test_llm_client_calls_are_generations():
    import inspect

    from continuum.llm import client

    source = inspect.getsource(client)
    for name in ("llm_chat", "llm_chat_sync", "llm_chat_stream", "llm_chat_stream_sync"):
        assert f'name="{name}"' in source
    assert source.count('as_type="generation"') == 4


def test_tracing_failure_never_breaks_the_call():
    @observe(name="llm_chat_sync", as_type="generation")
    def chat():
        return SimpleNamespace(model=None, usage=object())

    trace, obs, token = _trace()
    try:
        assert chat().model is None
    finally:
        restore_trace_context(token)
