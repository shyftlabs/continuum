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
- a body that fails mid-read raises from the hook, so SDK retries still run
  and the error type is the same as without the hook (timing probe too)
- a workflow collects its steps' entries and its own, on its response and its
  streamed RUN_END; the steps' finalizers leave them in place
- non-finite timing values are dropped; a streamed run's model comes from its
  chunks
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# A body that fails mid-read must still be retried by the SDK
# ---------------------------------------------------------------------------


class _BrokenBody(httpx.AsyncByteStream):
    """Sends the start of a JSON body, then the connection drops."""

    async def __aiter__(self):
        yield b'{"id": "c1", "model": '
        raise httpx.ReadError("connection dropped mid-body")


async def _mid_body_failure(provider_cls, monkeypatch, **kwargs):
    """Requests sent and the error raised by ``provider_cls.acomplete``."""
    from continuum.llm.config import LLMConfig
    from continuum.llm.providers import openai_provider

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200, headers={**HEADERS, "content-type": "application/json"}, stream=_BrokenBody()
        )

    monkeypatch.setattr(
        openai_provider,
        "build_async_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    # No back-off sleeps between the SDK's retries.
    monkeypatch.setattr(
        "openai._base_client.AsyncAPIClient._calculate_retry_timeout", lambda *a, **k: 0
    )
    provider = provider_cls(max_retries=2, **kwargs)
    set_trace_context(run_id="run-mid")
    with pytest.raises(Exception) as raised:
        await provider.acomplete([{"role": "user", "content": "q"}], LLMConfig(model="gpt-4o-mini"))
    return len(sent), type(raised.value)


async def test_mid_body_failure_is_retried_like_without_the_hook(monkeypatch):
    from continuum.llm.providers.gateway_provider import GatewayProvider
    from continuum.llm.providers.openai_provider import OpenAIProvider

    with_hook = await _mid_body_failure(
        GatewayProvider,
        monkeypatch,
        gateway_url="http://gateway.test/v1",
        api_key="k",
        router_mode=None,
    )
    without_hook = await _mid_body_failure(
        OpenAIProvider, monkeypatch, api_key="k", api_base="http://gateway.test/v1"
    )

    assert with_hook[0] == 3  # first attempt + 2 retries
    assert with_hook[1].__name__ == "LLMServiceUnavailableError"
    assert with_hook == without_hook


@pytest.mark.parametrize(
    "hook",
    [
        gateway_routing._async_hook,
        pytest.param("timing_probe", id="timing_probe"),
    ],
)
async def test_async_hooks_let_a_failed_read_raise(hook):
    from continuum.llm import timing_probe

    if hook == "timing_probe":
        hook = timing_probe._async_response_hook
    set_trace_context(run_id="run-mid")
    request = httpx.Request("POST", "http://gateway.test/v1/chat/completions")
    response = httpx.Response(200, headers=HEADERS, stream=_BrokenBody(), request=request)

    with pytest.raises(httpx.ReadError):
        await hook(response)


def test_sync_hooks_let_a_failed_read_raise():
    from continuum.llm import timing_probe

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b"{"
            raise httpx.ReadError("connection dropped mid-body")

    set_trace_context(run_id="run-mid")
    request = httpx.Request("POST", "http://gateway.test/v1/chat/completions")
    for hook in (gateway_routing._sync_hook, timing_probe._sync_response_hook):
        response = httpx.Response(200, headers=HEADERS, stream=Broken(), request=request)
        with pytest.raises(httpx.ReadError):
            hook(response)


# ---------------------------------------------------------------------------
# Workflows collect their steps' entries and their own
# ---------------------------------------------------------------------------


def _routing_runner():
    """A runner whose every LLM call files a gateway entry, as the HTTP hook does."""
    from continuum.agent.runner import AgentRunner

    calls = {"n": 0}

    async def chat(*args, **kwargs):
        calls["n"] += 1
        gateway_routing.record(
            httpx.Response(200, headers={**HEADERS, "x-request-id": f"step-{calls['n']}"})
        )
        return SimpleNamespace(
            content=f"out{calls['n']}",
            tool_calls=None,
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="gpt-4o-mini",
        )

    llm = MagicMock()
    llm.is_enabled = True
    llm.chat = AsyncMock(side_effect=chat)
    container = MagicMock(
        llm_client=llm, memory_client=None, session_client=None, tool_executor=None
    )
    with patch("continuum.agent.runner.get_container", return_value=container):
        return AgentRunner(
            llm_client=llm, memory_client=None, session_client=None, tool_executor=None
        )


def _merging_workflow():
    from continuum.agent.base import BaseAgent

    @dataclass
    class MergingWorkflow(BaseAgent):
        """Runs two steps in parallel, then makes its own call (a merge)."""

        steps: list = field(default_factory=list)

        async def execute(self, input_text, runner, context, llm_client=None):
            import asyncio

            outs = await asyncio.gather(
                *(runner.run(step, input_text, context=context) for step in self.steps)
            )
            gateway_routing.record(
                httpx.Response(200, headers={**HEADERS, "x-request-id": "merge"})
            )
            return AgentResponse(content="|".join(o.content for o in outs), agent_name=self.name)

    return MergingWorkflow(
        name="merge", steps=[BaseAgent(name="a", instructions="x"), BaseAgent(name="b")]
    )


async def test_sequential_workflow_collects_its_steps_entries():
    from continuum.agent.base import BaseAgent
    from continuum.agent.workflow.sequential import SequentialAgent

    runner = _routing_runner()
    workflow = SequentialAgent(
        name="seq",
        agents=[BaseAgent(name="a", instructions="x"), BaseAgent(name="b", instructions="y")],
    )

    response = await runner.run(workflow, "q")

    routing = response.run_artifacts["gateway_routing"]
    assert [e["request_id"] for e in routing] == ["step-1", "step-2"]
    assert not gateway_routing._by_run  # nothing left behind


async def test_workflow_collects_steps_and_its_own_calls():
    runner = _routing_runner()

    response = await runner.run(_merging_workflow(), "q")

    ids = [e["request_id"] for e in response.run_artifacts["gateway_routing"]]
    assert sorted(ids) == ["merge", "step-1", "step-2"]
    assert ids[-1] == "merge"
    assert not gateway_routing._by_run


async def test_workflow_run_end_carries_the_entries():
    from continuum.agent.types import EventType

    runner = _routing_runner()

    events = [e async for e in runner.run_stream(_merging_workflow(), "q")]

    run_end = events[-1]
    assert run_end.type == EventType.RUN_END
    assert sorted(e["request_id"] for e in run_end.data["gateway_routing"]) == [
        "merge",
        "step-1",
        "step-2",
    ]


async def test_failed_workflow_drops_its_entries():
    from continuum.agent.base import BaseAgent

    @dataclass
    class Failing(BaseAgent):
        async def execute(self, input_text, runner, context, llm_client=None):
            gateway_routing.record(httpx.Response(200, headers=HEADERS))
            raise RuntimeError("broke")

    runner = _routing_runner()
    with pytest.raises(Exception, match="broke"):
        await runner.run(Failing(name="f"), "q")

    assert not gateway_routing._by_run


def test_step_inside_a_workflow_leaves_its_entries():
    set_trace_context(run_id="run-w")
    gateway_routing.record(httpx.Response(200, headers=HEADERS))

    with gateway_routing.collect_for_workflow("run-w") as collects:
        assert collects
        assert gateway_routing.claim_run("run-w") is None  # a step's finalizer
        with gateway_routing.collect_for_workflow("run-w") as nested:
            assert not nested  # the outer workflow collects
    assert len(gateway_routing.claim_run("run-w")) == 1


# ---------------------------------------------------------------------------
# Small follow-ups
# ---------------------------------------------------------------------------


def test_non_finite_timing_values_are_dropped():
    set_trace_context(run_id="run-t")
    timing = '{"gateway_total_ms": NaN, "upstream_ms": Infinity, "queue_ms": 3}'
    gateway_routing.record(httpx.Response(200, headers={**HEADERS, "x-aura-timing": timing}))

    (entry,) = gateway_routing.pop_run("run-t")
    assert entry["timing"] == {"queue_ms": 3}
    json.dumps(entry, allow_nan=False)  # strict encoders accept it


def test_streamed_model_fills_the_last_entry():
    set_trace_context(run_id="run-s")
    gateway_routing.record(httpx.Response(503, headers=HEADERS))
    gateway_routing.record(httpx.Response(200, headers=HEADERS))

    gateway_routing.note_streamed_model("run-s", "gpt-4o-mini")

    failed, served = gateway_routing.pop_run("run-s")
    assert "served_model" not in failed
    assert served["served_model"] == "gpt-4o-mini"


async def test_retries_give_one_entry_per_attempt():
    replies = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(replies)
        if status == 503:
            return httpx.Response(503, headers=HEADERS, json={"error": {"message": "busy"}})
        return httpx.Response(200, headers=HEADERS, json=COMPLETION)

    http = gateway_routing.with_async_hook(
        httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client = AsyncOpenAI(
        api_key="test", base_url="http://gateway.test/v1", http_client=http, max_retries=1
    )
    set_trace_context(run_id="run-r")
    with patch("openai._base_client.AsyncAPIClient._calculate_retry_timeout", lambda *a, **k: 0):
        await _ask(client)

    assert [e["status"] for e in gateway_routing.pop_run("run-r")] == [503, 200]
