"""The gateway's routing decision reaches the run's response.

The Smart Gateway reports which model it ran (``x-aura-router-*``), cache
status, timing and request id in response headers. The OpenAI SDK drops them
and only the opt-in timing log read them, so an application could not tell
which model answered.

Covers:
- through the real OpenAI SDK (mock transport), a gateway reply's headers are
  filed under the current run, for normal and streamed calls, with the
  answering model read from a non-streamed body
- a reply without gateway headers, or a call outside a run, records nothing
- GatewayProvider adds the hook; OpenAIProvider does not
- the finalizer moves the run's entries to run_artifacts["gateway_routing"]
  (and drops them on error); the store stays bounded
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from continuum.agent.types import AgentResponse, ResponseStatus, RunState
from continuum.agent.utils.context_utils import create_run_context
from continuum.llm import gateway_routing
from continuum.observability.trace_context import clear_trace_context, set_trace_context

HEADERS = {
    "x-aura-router-mode": "modest",
    "x-aura-router-complexity": "low",
    "x-aura-router-attempts": "gpt-4o-mini@openai:ok",
    "x-aura-cache-status": "MISS",
    "x-aura-timing": json.dumps({"gateway_total_ms": 812.5, "upstream_ms": 790}),
    "x-aura-ratelimit-rpm-remaining": "247",
    "x-request-id": "req-123",
}
COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
}
STREAM = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture(autouse=True)
def _clean():
    gateway_routing._by_run.clear()
    clear_trace_context()
    yield
    gateway_routing._by_run.clear()
    clear_trace_context()


def _client(headers: dict, stream: bool = False) -> AsyncOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        if stream:
            return httpx.Response(
                200, headers={**headers, "content-type": "text/event-stream"}, text=STREAM
            )
        return httpx.Response(200, headers=headers, json=COMPLETION)

    http = gateway_routing.with_async_hook(
        httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return AsyncOpenAI(api_key="test", base_url="http://gateway.test/v1", http_client=http)


async def _ask(client: AsyncOpenAI, stream: bool = False) -> None:
    resp = await client.chat.completions.create(
        model="auto/mid", messages=[{"role": "user", "content": "q"}], stream=stream
    )
    if stream:
        async for _ in resp:
            pass


async def test_gateway_reply_is_filed_under_the_run():
    set_trace_context(run_id="run-a")
    await _ask(_client(HEADERS))
    await _ask(_client(HEADERS, stream=True), stream=True)

    entries = gateway_routing.pop_run("run-a")
    assert len(entries) == 2
    first = entries[0]
    assert first["router_attempts"] == "gpt-4o-mini@openai:ok"
    assert first["router_mode"] == "modest"
    assert first["cache_status"] == "MISS"
    assert first["request_id"] == "req-123"
    assert first["timing"]["gateway_total_ms"] == 812.5
    assert first["status"] == 200
    assert first["ratelimit"] == {"rpm_remaining": "247"}
    assert first["served_model"] == "gpt-4o-mini"  # read from the reply body
    assert "served_model" not in entries[1]  # a streamed body is not read
    assert gateway_routing.pop_run("run-a") is None


async def test_non_gateway_reply_or_no_run_records_nothing():
    set_trace_context(run_id="run-b")
    await _ask(_client({}))
    assert gateway_routing.pop_run("run-b") is None

    clear_trace_context()
    await _ask(_client(HEADERS))
    assert not gateway_routing._by_run


def test_only_the_gateway_provider_adds_the_hook():
    from continuum.llm.providers.gateway_provider import GatewayProvider
    from continuum.llm.providers.openai_provider import OpenAIProvider

    gateway = GatewayProvider(gateway_url="http://gateway.test/v1", api_key="k", router_mode=None)
    direct = OpenAIProvider(api_key="k")

    assert gateway_routing._async_hook in gateway._async_client._client.event_hooks["response"]
    assert gateway_routing._sync_hook in gateway._client._client.event_hooks["response"]
    assert gateway_routing._async_hook not in direct._async_client._client.event_hooks.get(
        "response", []
    )


def test_store_is_bounded(monkeypatch):
    monkeypatch.setattr(gateway_routing, "_MAX_RUNS", 3)
    response = httpx.Response(200, headers=HEADERS)
    for i in range(5):
        set_trace_context(run_id=f"run-{i}")
        gateway_routing.record(response)
    assert list(gateway_routing._by_run) == ["run-2", "run-3", "run-4"]


def _finalizer():
    from continuum.agent.execution.run_finalizer import RunFinalizer

    lifecycle = MagicMock(
        report_metrics=AsyncMock(), end_trace=AsyncMock(), report_error=AsyncMock()
    )
    return RunFinalizer(
        session_service=MagicMock(save_messages=AsyncMock(), save_tool_context_state=AsyncMock()),
        context_service=MagicMock(save_run_state=AsyncMock()),
        lifecycle=lifecycle,
        session_client=MagicMock(is_enabled=False),
    )


async def test_finalizer_puts_routing_on_the_response():
    from continuum.agent.base import BaseAgent

    ctx = create_run_context()
    set_trace_context(run_id=ctx.run_id)
    gateway_routing.record(httpx.Response(200, headers=HEADERS))
    response = AgentResponse(content="a", agent_name="x", status=ResponseStatus.SUCCESS)

    await _finalizer().finalize(
        BaseAgent(name="x"), ctx, RunState(run_id=ctx.run_id), response, 0, None, time.time(), []
    )

    routing = response.run_artifacts["gateway_routing"]
    assert routing[0]["router_attempts"] == "gpt-4o-mini@openai:ok"
    assert gateway_routing.pop_run(ctx.run_id) is None


async def test_failed_run_drops_its_entries():
    from continuum.agent.base import BaseAgent

    ctx = create_run_context()
    set_trace_context(run_id=ctx.run_id)
    gateway_routing.record(httpx.Response(200, headers=HEADERS))

    await _finalizer().handle_error(
        BaseAgent(name="x"), ctx, RunState(run_id=ctx.run_id), RuntimeError("x"), time.time()
    )

    assert gateway_routing.pop_run(ctx.run_id) is None
