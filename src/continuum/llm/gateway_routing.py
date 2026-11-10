"""Which model the Smart Gateway actually ran, per agent run.

The gateway decides the model (``auto/...`` aliases, effort variants, failover)
and says what it did in response headers (``x-aura-router-*``,
``x-aura-cache-status``, ``x-aura-timing``, rate limits, ``x-request-id``). The
OpenAI SDK drops headers, and the only reader was the opt-in timing log, so an
application could not tell which model answered a question.

Gateway clients get an httpx response hook that reads those headers and files
them under the current run's id. On a non-streamed reply it also reads the
body's ``model`` (the model that answered); a streamed body is never touched. The run finalizer then puts the list on
``AgentResponse.run_artifacts["gateway_routing"]``, one entry per model call.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

import httpx

from continuum.llm.timing_probe import _parse_routing, _parse_timing
from continuum.logging import get_logger
from continuum.observability.trace_context import get_current_run_id

logger = get_logger(__name__)

# Runs whose calls have not been collected yet (a run that crashed before its
# finalizer, a call outside any run that still had a run id). Oldest go first.
_MAX_RUNS = 1024

_by_run: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_lock = threading.Lock()


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
            _by_run.setdefault(run_id, []).append(entry)
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


def _is_streaming(response: httpx.Response) -> bool:
    return "text/event-stream" in (response.headers.get("content-type") or "")


def _sync_hook(response: httpx.Response) -> None:
    served = None
    if get_current_run_id() and not _is_streaming(response):
        try:
            response.read()  # httpx keeps it; the SDK reads it again for free
            served = _served_model(response)
        except Exception:  # noqa: BLE001
            pass
    record(response, served)


async def _async_hook(response: httpx.Response) -> None:
    served = None
    if get_current_run_id() and not _is_streaming(response):
        try:
            await response.aread()  # httpx keeps it; the SDK reads it again for free
            served = _served_model(response)
        except Exception:  # noqa: BLE001
            pass
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
