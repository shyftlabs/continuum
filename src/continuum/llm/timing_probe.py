"""Client-side latency probe for LLM HTTP calls.

Phase 1a of the gateway latency measurement plan. Exists because the OpenAI SDK
discards response headers: the gateway stamps every response with
``x-aura-timing`` carrying its per-phase breakdown, and until now nothing on the
Continuum side ever read it.

Rather than rewrite every call site to use ``with_raw_response``, the probe
installs an httpx event hook on the client the SDK already uses. Call sites are
untouched and return types are unchanged.

Enabled by ``CONTINUUM_TIMING_LOG=1``. When unset, the ``build_*_http_client``
functions return ``None`` and the provider constructs its clients exactly as it
did before — no hook, no file, no import cost beyond this module.

One JSON object per HTTP call is appended to ``CONTINUUM_TIMING_LOG_PATH``
(default ``continuum-timing.jsonl`` in the working directory):

    ts            wall-clock ISO timestamp — a point in time, so Date-based
    turn_id       set by the caller via `timing_turn`; None outside a turn
    seq           per-turn call ordinal, so a turn's calls stay ordered even if
                  two finish in the same millisecond
    host, path    which endpoint; direct-OpenAI calls are logged too, which is
                  what makes the Phase 3 A/B control possible
    status        HTTP status
    client_ms     round trip as the CALLER sees it — see the note below
    timing        parsed x-aura-timing, or None when absent
    request_id    x-request-id, for joining against gateway logs and Langfuse

`client_ms` is measured with ``time.perf_counter()``, not ``time.time()``, for
the same reason the gateway moved to ``performance.now()``: the wall clock is
coarse and can step backwards, and a sub-millisecond span must not floor to zero.

WHAT `client_ms` INCLUDES that the gateway's own `gateway_total_ms` cannot:
connection setup, TLS, network transit both ways, and the five middleware
wrappers mounted outside `requestTiming`. The difference between the two is the
only view we have of the edge, so both are recorded rather than one.

STREAMING CAVEAT: httpx fires the response hook when the response *headers*
arrive, before the body is consumed. On a streaming call `client_ms` is
therefore time-to-headers, not total time — and `timing` will usually be None,
because the gateway cannot stamp a header onto a body that is already flowing.
Measure with streaming off; a streaming-aware probe is a separate exercise.

The probe never raises into the request path. Every hook body is wrapped: a
broken probe degrades to missing measurements, never to a failed LLM call.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx

from continuum.logging import get_logger

logger = get_logger(__name__)

_ENV_ENABLED = "CONTINUUM_TIMING_LOG"
_ENV_PATH = "CONTINUUM_TIMING_LOG_PATH"
_DEFAULT_PATH = "continuum-timing.jsonl"

_TIMING_HEADER = "x-aura-timing"
_REQUEST_ID_HEADER = "x-request-id"

# Stamped onto the outgoing request by the request hook and read back by the
# response hook. httpx carries `extensions` through to `response.request`, and
# it is the documented place for per-request user data — unlike setattr, which
# is not part of the Request contract.
_START_KEY = "continuum_probe_start"

# One turn is one user query, which fans out into several LLM calls. Phase 1b
# sets this from the agent loop; until then every record carries turn_id=None,
# which is still a usable per-call log.
_turn_id: ContextVar[str | None] = ContextVar("continuum_timing_turn_id", default=None)
_turn_seq: ContextVar[int] = ContextVar("continuum_timing_turn_seq", default=0)

# The sync client may be driven from a worker thread, so the append is locked.
# Held only around the write itself, never around the request.
_write_lock = threading.Lock()


def is_enabled() -> bool:
    """Whether the probe should install itself.

    Read on every client construction rather than cached at import, so a test or
    a REPL can toggle it without reimporting the module.
    """
    return os.environ.get(_ENV_ENABLED, "").strip().lower() in {"1", "true", "yes"}


def log_path() -> str:
    return os.environ.get(_ENV_PATH, "").strip() or _DEFAULT_PATH


@contextmanager
def timing_turn(turn_id: str) -> Iterator[None]:
    """Tag every LLM call made inside this block with `turn_id`.

    A contextvar rather than a parameter because the call sites are several
    frames below the agent loop and none of them should have to know the probe
    exists. Also restores the previous value on exit, so nested turns (a memory
    extraction call made while handling a user turn) cannot leak their id
    outward.
    """
    token_id = _turn_id.set(turn_id)
    token_seq = _turn_seq.set(0)
    try:
        yield
    finally:
        _turn_id.reset(token_id)
        _turn_seq.reset(token_seq)


def _next_seq() -> int:
    seq = _turn_seq.get() + 1
    _turn_seq.set(seq)
    return seq


def _parse_timing(raw: str | None) -> dict[str, Any] | None:
    """Parse the gateway's timing header, keeping only numeric phases.

    Deliberately unfiltered against a known-phase list: the gateway's registry is
    the authority on what phases exist, and a probe that dropped unrecognised
    keys would silently hide any phase added after this file was written — which
    is exactly the failure mode the gateway-side registry was introduced to end.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {k: v for k, v in parsed.items() if isinstance(v, (int, float))}


def _write(record: dict[str, Any]) -> None:
    path = log_path()
    line = json.dumps(record, sort_keys=True)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _record(response: httpx.Response) -> None:
    """Build and append one record. Never raises."""
    try:
        request = response.request
        start = request.extensions.get(_START_KEY)
        client_ms = (
            round((time.perf_counter() - start) * 1000, 3) if start is not None else None
        )

        _write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "turn_id": _turn_id.get(),
                "seq": _next_seq(),
                "host": request.url.host,
                "path": request.url.path,
                "status": response.status_code,
                "client_ms": client_ms,
                "timing": _parse_timing(response.headers.get(_TIMING_HEADER)),
                "request_id": response.headers.get(_REQUEST_ID_HEADER),
            }
        )
    except Exception:  # noqa: BLE001 — a probe must never break the call it measures
        logger.debug("timing probe: failed to record a response", exc_info=True)


def _stamp(request: httpx.Request) -> None:
    """Mark the send time. Never raises."""
    try:
        request.extensions[_START_KEY] = time.perf_counter()
    except Exception:  # noqa: BLE001
        logger.debug("timing probe: failed to stamp a request", exc_info=True)


def _sync_request_hook(request: httpx.Request) -> None:
    _stamp(request)


def _sync_response_hook(response: httpx.Response) -> None:
    _record(response)


async def _async_request_hook(request: httpx.Request) -> None:
    _stamp(request)


async def _async_response_hook(response: httpx.Response) -> None:
    _record(response)


def build_sync_http_client() -> httpx.Client | None:
    """An SDK-default httpx client with the probe attached, or None if disabled.

    Built from ``openai.DefaultHttpxClient`` rather than a bare ``httpx.Client``
    on purpose. The SDK's client applies its own timeout and connection-pool
    limits; a bare client would silently replace both, changing the very latency
    this probe exists to measure. ``DefaultHttpxClient`` applies them with
    ``setdefault`` and passes everything else — including ``event_hooks`` —
    through untouched.
    """
    if not is_enabled():
        return None
    from openai import DefaultHttpxClient

    return DefaultHttpxClient(
        event_hooks={
            "request": [_sync_request_hook],
            "response": [_sync_response_hook],
        }
    )


def build_async_http_client() -> httpx.AsyncClient | None:
    """Async counterpart of `build_sync_http_client`."""
    if not is_enabled():
        return None
    from openai import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(
        event_hooks={
            "request": [_async_request_hook],
            "response": [_async_response_hook],
        }
    )
