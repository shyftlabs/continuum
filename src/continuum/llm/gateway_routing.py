"""Which model the Smart Gateway actually ran, per agent run.

The gateway decides the model (``auto/...`` aliases, effort variants, failover)
and says what it did in response headers (``x-aura-router-*``,
``x-aura-cache-status``, ``x-aura-timing``, rate limits, ``x-request-id``). The
OpenAI SDK drops headers, and the only reader was the opt-in timing log, so an
application could not tell which model answered a question.

Gateway clients get an httpx response hook that reads those headers and files
them under the current run's id. On a non-streamed reply it also reads the
body's ``model`` (the model that answered); a streamed body is never touched,
and the provider fills the model in from the stream's chunks instead, on the
entry of that stream's own HTTP response.
The run finalizer then puts the list on
``AgentResponse.run_artifacts["gateway_routing"]``.

There is one entry per HTTP attempt, not per model call: when the SDK retries
(say a 503, then a 200), each attempt the gateway answered has its own entry.

Steps of a workflow share the workflow's run id. While a workflow runs, its
steps leave their entries in place and the workflow collects them all after
``execute()``. For a workflow the runner dispatches (``runner.run`` /
``run_stream``) that includes the workflow's own calls (a merge, a completion
check). For ``runner.fork`` of a workflow, and for a built-in workflow whose
``execute()`` is called directly on a new context, only the steps' calls are
listed: the workflow's own calls may run without the run id, or after a step
cleared it. Entries are only attached to an ``AgentResponse``, on a copy. A
failed or cancelled run returns no response, and its entries are removed when
it ends; a call whose HTTP response arrives after that (a step of a cancelled
``ParallelAgent`` still in flight), or a call made after the run under its
still-current run id, is filed under that run and stays until evicted.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx

from continuum.llm.timing_probe import _parse_routing, _parse_timing
from continuum.logging import get_logger
from continuum.observability.trace_context import get_current_run_id

logger = get_logger(__name__)

# Runs whose calls have not been collected yet (a run that crashed before its
# finalizer, a call outside any run that still had a run id). Oldest go first.
_MAX_RUNS = 1024
# Entries kept per run; past this the oldest go first. A long run (or calls
# filed under a run id that is never collected) cannot grow without bound.
_MAX_ENTRIES_PER_RUN = 1024

_by_run: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
# The entry filed for each streamed HTTP response, so the model named in that
# stream's chunks fills its own entry, not another call's under the same run.
_by_response: weakref.WeakKeyDictionary[httpx.Response, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_lock = threading.Lock()


def _reset_after_fork() -> None:
    """In a forked child: a fresh lock and an empty store.

    A fork copies the lock as it was; if another thread held it, the child's
    first gateway response would wait on it forever. The parent's runs do not
    continue in the child, so their entries go too.
    """
    global _lock
    _lock = threading.Lock()
    _by_run.clear()
    _by_response.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)

# Run ids whose entries an enclosing workflow will collect.
_workflow_runs: ContextVar[frozenset[str]] = ContextVar(
    "continuum_gateway_routing_workflow_runs", default=frozenset()
)


_RATELIMIT_PREFIX = "x-aura-ratelimit-"


def _served_model(response: httpx.Response) -> str | None:
    """The ``model`` of a JSON reply whose body is already read; None otherwise."""
    try:
        body = json.loads(response.content)
    except Exception:  # noqa: BLE001 - not read, not JSON
        return None
    model = body.get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) else None


def _entry(response: httpx.Response, served_model: str | None = None) -> dict[str, Any] | None:
    headers = response.headers
    routing = _parse_routing(headers)
    timing = _parse_timing(headers.get("x-aura-timing"))
    if timing:
        # NaN / Infinity parse as floats but break strict JSON encoders.
        timing = {k: v for k, v in timing.items() if math.isfinite(v)} or None
    ratelimit = {
        k[len(_RATELIMIT_PREFIX) :].replace("-", "_"): v
        for k, v in headers.items()
        if k.startswith(_RATELIMIT_PREFIX)
    }
    if not (routing or timing or ratelimit):
        return None  # not a gateway response
    entry: dict[str, Any] = dict(routing or {})
    if served_model:
        entry["served_model"] = served_model
    if timing:
        entry["timing"] = timing
    if ratelimit:
        entry["ratelimit"] = ratelimit
    if request_id := headers.get("x-request-id"):
        entry["request_id"] = request_id
    entry["status"] = response.status_code
    return entry


def record(response: httpx.Response, served_model: str | None = None) -> None:
    """File a gateway response's routing under the current run. Never raises."""
    try:
        run_id = get_current_run_id()
        if not run_id:
            return
        entry = _entry(response, served_model)
        if entry is None:
            return
        with _lock:
            entries = _by_run.setdefault(run_id, [])
            entries.append(entry)
            if _is_streaming(response):
                _by_response[response] = entry
            if len(entries) > _MAX_ENTRIES_PER_RUN:
                del entries[: len(entries) - _MAX_ENTRIES_PER_RUN]
            _by_run.move_to_end(run_id)
            while len(_by_run) > _MAX_RUNS:
                _by_run.popitem(last=False)
    except Exception:  # noqa: BLE001 - reading headers must never break the call
        logger.debug("gateway routing: failed to read a response", exc_info=True)


def pop_run(run_id: str | None) -> list[dict[str, Any]] | None:
    """The routing entries recorded for ``run_id``, removing them."""
    if not run_id:
        return None
    with _lock:
        return _by_run.pop(run_id, None)


def note_streamed_model(response: httpx.Response | None, model: str | None) -> None:
    """Set ``served_model`` from a streamed reply's chunks on that reply's entry.

    A streamed body is not read by the hook, so its entry has no model. The
    stream's own chunks name it; the provider passes the stream's HTTP response
    and the model once the stream ends. Only the entry filed for that response
    is filled, so two streams overlapping under one run id each get their own
    model. Never raises.
    """
    if response is None or not model:
        return
    try:
        with _lock:
            entry = _by_response.pop(response, None)
            if entry is not None and "served_model" not in entry:
                entry["served_model"] = model
    except Exception:  # noqa: BLE001 - not a weakref-able response
        logger.debug("gateway routing: failed to note a streamed model", exc_info=True)


def claim_run(run_id: str | None) -> list[dict[str, Any]] | None:
    """Like :func:`pop_run`, but None for a step whose workflow collects the entries."""
    if run_id and run_id in _workflow_runs.get():
        return None
    return pop_run(run_id)


@contextmanager
def collect_for_workflow(run_id: str | None) -> Iterator[bool]:
    """Keep the entries of ``run_id`` for the workflow running inside this block.

    Yields True when this workflow is the one that collects them, False when an
    enclosing workflow on the same run id already does (a nested workflow).
    """
    current = _workflow_runs.get()
    if not run_id or run_id in current:
        yield False
        return
    token = _workflow_runs.set(current | {run_id})
    try:
        yield True
    finally:
        _workflow_runs.reset(token)
        # Whatever was not collected (the workflow failed or was cancelled).
        pop_run(run_id)


def attach(result: Any, routing: list[dict[str, Any]] | None) -> Any:
    """``result`` with ``routing`` added to its ``run_artifacts["gateway_routing"]``.

    Only an ``AgentResponse`` whose ``run_artifacts`` is a dict (or None) gets
    them, and on a copy: a workflow may return a response object it shares
    across runs. Anything else (a custom workflow's str, dict, None) is
    returned unchanged.
    """
    from continuum.agent.types import AgentResponse

    if not routing or not isinstance(result, AgentResponse):
        return result
    artifacts = result.run_artifacts if result.run_artifacts is not None else {}
    if not isinstance(artifacts, dict):
        return result
    earlier = artifacts.get("gateway_routing")
    result = copy.copy(result)
    result.run_artifacts = {
        **artifacts,
        "gateway_routing": [*(earlier if isinstance(earlier, list) else []), *routing],
    }
    return result


async def collect_from(run_id: str | None, call: Awaitable[Any]) -> Any:
    """Await a workflow ``call`` and attach the entries of ``run_id`` to its result.

    Inside an enclosing workflow on the same run id, the result is returned as
    is and the enclosing workflow collects.
    """
    with collect_for_workflow(run_id) as collects:
        result = await call
        return attach(result, pop_run(run_id)) if collects else result


def collects_routing(method: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator for a workflow's ``execute()``: collect its run's entries.

    The runner already collects for the workflows it dispatches; this covers a
    workflow called directly, say a custom workflow running ``Parallel.execute``
    on a new context. The ``RunContext`` is found among the call args, as
    ``publish_active_policy`` does.
    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        from continuum.agent.types import RunContext

        ctx = kwargs.get("context")
        if ctx is None:
            ctx = next((a for a in args if isinstance(a, RunContext)), None)
        return await collect_from(getattr(ctx, "run_id", None), method(self, *args, **kwargs))

    return wrapper


def _is_streaming(response: httpx.Response) -> bool:
    return "text/event-stream" in (response.headers.get("content-type") or "")


def _sync_hook(response: httpx.Response) -> None:
    served = None
    if get_current_run_id() and not _is_streaming(response):
        # httpx keeps the body; the SDK reads it again for free. A failed read
        # (connection dropped or timed out mid-body) must raise here, where the
        # SDK retries it; swallowed, the SDK later hits StreamConsumed instead.
        response.read()
        served = _served_model(response)
    record(response, served)


async def _async_hook(response: httpx.Response) -> None:
    served = None
    if get_current_run_id() and not _is_streaming(response):
        # httpx keeps the body; the SDK reads it again for free. A failed read
        # (connection dropped or timed out mid-body) must raise here, where the
        # SDK retries it; swallowed, the SDK later hits StreamConsumed instead.
        await response.aread()
        served = _served_model(response)
    record(response, served)


def with_sync_hook(client: httpx.Client | None) -> httpx.Client:
    """``client`` (or an SDK-default client) with the routing hook added."""
    if client is None:
        from openai import DefaultHttpxClient

        return DefaultHttpxClient(event_hooks={"response": [_sync_hook]})
    hooks = client.event_hooks
    client.event_hooks = {**hooks, "response": [*hooks.get("response", []), _sync_hook]}
    return client


def with_async_hook(client: httpx.AsyncClient | None) -> httpx.AsyncClient:
    """Async counterpart of :func:`with_sync_hook`."""
    if client is None:
        from openai import DefaultAsyncHttpxClient

        return DefaultAsyncHttpxClient(event_hooks={"response": [_async_hook]})
    hooks = client.event_hooks
    client.event_hooks = {**hooks, "response": [*hooks.get("response", []), _async_hook]}
    return client
