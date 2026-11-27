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
  chunks, tool-call turns and the model-tier router stream included
- a custom workflow may return anything; only an AgentResponse gets entries,
  on a copy; a cancelled workflow leaves none behind
- runner.fork of a workflow, and a workflow running Parallel.execute on a new
  context, collect their steps' entries
- a forked child gets a fresh lock and an empty store
- a streamed workflow whose run_artifacts is not a dict still ends with RUN_END
- calls sent through either dispatcher are filed under their own run, started
  lazily or at startup; entries per run are capped
- a cancelled plain run and an early-closed plain stream leave no entries
"""

from __future__ import annotations

import json
import os
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
        assert len(gateway_routing.pop_run("run-w")) == 1


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


# ---------------------------------------------------------------------------
# Round 2: custom results, shared responses, fork, new contexts, cancellation
# ---------------------------------------------------------------------------


def _returning_workflow(value):
    from continuum.agent.base import BaseAgent

    @dataclass
    class Returning(BaseAgent):
        async def execute(self, input_text, runner, context, llm_client=None):
            gateway_routing.record(httpx.Response(200, headers=HEADERS))
            return value

    return Returning(name="custom")


@pytest.mark.parametrize("value", ["plain text", {"answer": 1}, None, SimpleNamespace(x=1)])
async def test_custom_workflow_result_is_returned_as_is(value):
    runner = _routing_runner()

    result = await runner.run(_returning_workflow(value), "q")

    assert result is value
    assert not gateway_routing._by_run


async def test_shared_response_is_not_changed():
    shared = AgentResponse(content="cached", agent_name="custom")
    runner = _routing_runner()
    workflow = _returning_workflow(shared)

    first = await runner.run(workflow, "q")
    second = await runner.run(workflow, "q")

    assert shared.run_artifacts is None
    assert len(first.run_artifacts["gateway_routing"]) == 1
    assert len(second.run_artifacts["gateway_routing"]) == 1


async def test_cancelled_workflow_leaves_no_entries():
    import asyncio

    from continuum.agent.base import BaseAgent

    started = asyncio.Event()

    @dataclass
    class Hanging(BaseAgent):
        async def execute(self, input_text, runner, context, llm_client=None):
            gateway_routing.record(httpx.Response(200, headers=HEADERS))
            started.set()
            await asyncio.Event().wait()

    runner = _routing_runner()
    task = asyncio.create_task(runner.run(Hanging(name="h"), "q"))
    await started.wait()
    assert gateway_routing._by_run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not gateway_routing._by_run


async def test_fork_of_a_workflow_collects_its_steps_entries():
    from continuum.agent.base import BaseAgent

    @dataclass
    class Resumable(BaseAgent):
        steps: list = field(default_factory=list)

        async def resume_from(self, *, parent_trace, from_step, override, runner, context):
            outs = [await runner.run(step, "q", context=context) for step in self.steps]
            return AgentResponse(content="|".join(o.content for o in outs), agent_name=self.name)

    parent = SimpleNamespace(steps=[SimpleNamespace(step_id="s1", data_labels=[])], root_agent="wf")
    store = MagicMock()
    store.get = AsyncMock(return_value=parent)
    runner = _routing_runner()
    workflow = Resumable(
        name="wf",
        steps=[BaseAgent(name="a", instructions="x"), BaseAgent(name="b", instructions="y")],
    )

    with patch("continuum.agent.trace.config.get_trace_store", return_value=store):
        response = await runner.fork("run-parent", "s1", agent=workflow)

    ids = [e["request_id"] for e in response.run_artifacts["gateway_routing"]]
    assert ids == ["step-1", "step-2"]
    assert not gateway_routing._by_run


async def test_parallel_execute_on_a_new_context_collects_its_workers():
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import ParallelConfig
    from continuum.agent.types import MergeStrategy
    from continuum.agent.workflow.parallel import ParallelAgent

    parallel = ParallelAgent(
        name="par",
        agents=[BaseAgent(name="a", instructions="x"), BaseAgent(name="b", instructions="y")],
        parallel_config=ParallelConfig(merge_strategy=MergeStrategy.CONCATENATE),
    )

    @dataclass
    class Outer(BaseAgent):
        async def execute(self, input_text, runner, context, llm_client=None):
            return await parallel.execute(input_text, runner, create_run_context())

    runner = _routing_runner()
    response = await runner.run(Outer(name="outer"), "q")

    ids = sorted(e["request_id"] for e in response.run_artifacts["gateway_routing"])
    assert ids == ["step-1", "step-2"]
    assert not gateway_routing._by_run


# ---------------------------------------------------------------------------
# Streamed model for tool-call turns and the model-tier router
# ---------------------------------------------------------------------------

TOOL_STREAM = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function",'
    '"function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    "data: [DONE]\n\n"
)


async def test_tool_call_stream_names_the_model():
    from continuum.llm.config import LLMConfig
    from continuum.llm.providers.gateway_provider import GatewayProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={**HEADERS, "content-type": "text/event-stream"}, text=TOOL_STREAM
        )

    provider = GatewayProvider("http://gateway.test", "test", None)
    provider._async_client = AsyncOpenAI(
        api_key="test",
        base_url="http://gateway.test/v1",
        http_client=gateway_routing.with_async_hook(
            httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {}}}

    chunks = [
        c
        async for c in provider.astream(
            [{"role": "user", "content": "q"}], LLMConfig(model="auto/mid"), tools=[tool]
        )
    ]

    (tool_chunk,) = [c for c in chunks if c.tool_calls]
    assert tool_chunk.model == "gpt-4o-mini"


async def test_model_tier_stream_names_the_model():
    from continuum.agent.config import RouterConfig
    from continuum.agent.smart_layer import runner_facade
    from continuum.agent.smart_layer.types import ProductTier
    from continuum.agent.types import Route
    from continuum.agent.workflow.router import RouterAgent
    from continuum.llm.types import StreamChunk

    async def chat_stream(messages, config, **kwargs):
        gateway_routing.record(
            httpx.Response(200, headers={**HEADERS, "content-type": "text/event-stream"})
        )
        yield StreamChunk(model="gpt-4o-mini", content="Paris.")

    llm = MagicMock()
    llm.chat_stream = chat_stream
    router = RouterAgent(
        name="tier",
        routes=[Route(agent_name="geo", description="geo", condition="geo")],
        router_config=RouterConfig(routing_strategy="model_tier"),
    )
    ctx = create_run_context()
    set_trace_context(run_id=ctx.run_id)
    outcome = SimpleNamespace(
        tier=ProductTier.fast, classify_ms=0.0, skipped_classifier=True, skip_reason="test"
    )

    with patch.object(runner_facade, "classify_product_tier", AsyncMock(return_value=outcome)):
        async for _ in runner_facade.stream_model_tier_turn(
            router, llm, user_text="Capital of France?", ctx=ctx
        ):
            pass

    (entry,) = gateway_routing.pop_run(ctx.run_id)
    assert entry["served_model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# A forked child does not inherit a held lock
# ---------------------------------------------------------------------------


def test_after_fork_reset_gives_a_free_lock_and_empty_store():
    set_trace_context(run_id="run-f")
    gateway_routing.record(httpx.Response(200, headers=HEADERS))
    held = gateway_routing._lock
    held.acquire()
    try:
        gateway_routing._reset_after_fork()
        assert gateway_routing._lock is not held
        assert not gateway_routing._by_run
        gateway_routing.record(httpx.Response(200, headers=HEADERS))  # does not block
        assert len(gateway_routing.pop_run("run-f")) == 1
    finally:
        held.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_fork_while_another_thread_holds_the_lock_does_not_hang_the_child():
    import threading
    import warnings

    holding, release = threading.Event(), threading.Event()

    def hold():
        with gateway_routing._lock:
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait(5)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads
            pid = os.fork()
        if pid == 0:  # child: record a gateway response, then leave
            code = 1
            try:
                set_trace_context(run_id="run-child")
                gateway_routing.record(httpx.Response(200, headers=HEADERS))
                code = 0 if gateway_routing.pop_run("run-child") else 2
            finally:
                os._exit(code)
        deadline = time.monotonic() + 10
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > deadline:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                pytest.fail("the forked child hung on the gateway routing lock")
            time.sleep(0.02)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        holder.join()


# ---------------------------------------------------------------------------
# Round 3: non-dict run_artifacts, dispatchers, per-run cap, cancelled runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gateway_on", [True, False])
@pytest.mark.parametrize(
    "artifacts", [["keep"], "keep", ("keep",), {"gateway_routing": "not a list"}]
)
async def test_stream_run_end_with_non_dict_run_artifacts(artifacts, gateway_on):
    from continuum.agent.base import BaseAgent
    from continuum.agent.types import EventType

    @dataclass
    class Odd(BaseAgent):
        async def execute(self, input_text, runner, context, llm_client=None):
            if gateway_on:
                gateway_routing.record(httpx.Response(200, headers=HEADERS))
            return AgentResponse(content="ok", agent_name=self.name, run_artifacts=artifacts)

    events = [e async for e in _routing_runner().run_stream(Odd(name="odd"), "q")]

    run_end = events[-1]
    assert run_end.type == EventType.RUN_END
    assert run_end.data["content"] == "ok"
    if gateway_on and isinstance(artifacts, dict):
        # A dict gets the run's entries; its non-list value is not kept.
        assert len(run_end.data["gateway_routing"]) == 1
    else:
        assert "gateway_routing" not in run_end.data
    assert not gateway_routing._by_run


def _dispatching_runner(dispatcher):
    """``_routing_runner`` whose LLM calls go through ``dispatcher``, as LLMClient does."""
    runner = _routing_runner()
    chat = runner.llm_client.chat.side_effect

    async def dispatched_chat(*args, **kwargs):
        if hasattr(dispatcher, "max_workers"):
            return await dispatcher.dispatch(
                lambda: chat(*args, **kwargs), stage_priority=5, request_priority=5
            )
        return await dispatcher.dispatch(lambda: chat(*args, **kwargs), priority=5)

    runner.llm_client.chat.side_effect = dispatched_chat
    return runner


@pytest.mark.parametrize("started_at_startup", [False, True])
@pytest.mark.parametrize("kind", ["priority", "two_level"])
async def test_dispatched_calls_are_filed_under_their_own_run(kind, started_at_startup):
    from continuum.agent.base import BaseAgent
    from continuum.llm.dispatcher import PriorityDispatcher, TwoLevelDispatcher

    dispatcher = (
        PriorityDispatcher(max_concurrent=2) if kind == "priority" else TwoLevelDispatcher(2)
    )
    if started_at_startup:
        await dispatcher.start()  # outside any run, as the module docstring shows
    runner = _dispatching_runner(dispatcher)
    try:
        # Lazily started: run 1's first call starts the workers inside run 1.
        responses = [await runner.run(BaseAgent(name="a", instructions="x"), "q") for _ in range(3)]
    finally:
        await dispatcher.stop()

    ids = [[e["request_id"] for e in r.run_artifacts["gateway_routing"]] for r in responses]
    assert ids == [["step-1"], ["step-2"], ["step-3"]]
    assert not gateway_routing._by_run


def test_entries_per_run_are_capped(monkeypatch):
    monkeypatch.setattr(gateway_routing, "_MAX_ENTRIES_PER_RUN", 3)
    set_trace_context(run_id="run-long")
    for i in range(5):
        gateway_routing.record(httpx.Response(200, headers={**HEADERS, "x-request-id": f"r{i}"}))

    assert [e["request_id"] for e in gateway_routing._by_run["run-long"]] == ["r2", "r3", "r4"]


async def test_cancelled_plain_run_leaves_no_entries():
    import asyncio

    from continuum.agent.base import BaseAgent

    started = asyncio.Event()
    runner = _routing_runner()

    async def hanging_chat(*args, **kwargs):
        gateway_routing.record(httpx.Response(200, headers=HEADERS))
        started.set()
        await asyncio.Event().wait()

    runner.llm_client.chat.side_effect = hanging_chat
    task = asyncio.create_task(runner.run(BaseAgent(name="a", instructions="x"), "q"))
    await started.wait()
    assert gateway_routing._by_run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not gateway_routing._by_run


async def test_early_closed_plain_stream_leaves_no_entries():
    import asyncio

    from continuum.agent.base import BaseAgent
    from continuum.agent.types import EventType

    runner = _routing_runner()

    async def chat_stream(*args, **kwargs):
        gateway_routing.record(httpx.Response(200, headers=HEADERS))
        yield SimpleNamespace(content="hi", tool_calls=None, model="gpt-4o-mini", usage=None)
        await asyncio.Event().wait()

    runner.llm_client.chat_stream = chat_stream
    stream = runner.run_stream(BaseAgent(name="a", instructions="x"), "q")
    async for event in stream:
        if event.data and event.data.get("content") == "hi" and event.type != EventType.RUN_END:
            break
    assert gateway_routing._by_run
    await stream.aclose()

    assert not gateway_routing._by_run
