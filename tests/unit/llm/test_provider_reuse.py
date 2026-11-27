"""Built-in providers are reused across calls instead of rebuilt each call.

get_provider() built a new provider, with new SDK clients and connection pools,
for every model call, so each call paid a fresh TCP + TLS handshake. Through the
gateway that measured ~0.5 s of time-to-first-token per call (reused client
0.96 s median, new client per call 1.58 s, Continuum 1.45 s).

Covers:
- same settings in one event loop -> the same provider (gateway and direct)
- another event loop gets its own (async clients are loop-bound); calls outside
  a loop get a new provider each time, so sync code before asyncio.run() works
- stores of closed loops are dropped (asyncio.run per request must not leak)
- stores of finished but unclosed loops are dropped too (new_event_loop() +
  run_until_complete() per request without close() must not run out of files)
- a loop running in another thread keeps its providers; stores are capped
- each loop keeps at most _MAX_PROVIDERS_PER_LOOP, least recently used dropped
- a forked child starts with no providers and a fresh lock
- a different key, model routing argument or retry count -> a different provider
- custom registered factories are called every time, as before
- clear_provider_cache() forgets them
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from continuum.llm import providers
from continuum.llm.config import LLMConfig
from continuum.llm.providers import clear_provider_cache, get_provider, register_provider


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    # Another test module swaps the default factory; use the built-in one here.
    monkeypatch.setattr(providers, "_default_factory", providers._make_openai)
    clear_provider_cache()
    yield
    clear_provider_cache()


def _settings(**over):
    from continuum.config import settings

    values = {"smart_gateway_url": "", "openai_api_key": "sk-test", **over}
    return patch.multiple(settings, **values)


async def test_gateway_provider_is_reused_in_a_loop():
    with _settings(smart_gateway_url="http://gateway.test/v1", smart_gateway_api_key="k"):
        a = get_provider(LLMConfig(model="openai/gpt-4o-mini"))
        b = get_provider(LLMConfig(model="openai/gpt-4o-mini"))
        other_retries = get_provider(LLMConfig(model="openai/gpt-4o-mini", max_retries=5))
    assert a is b
    assert other_retries is not a


async def test_direct_provider_is_reused_and_keyed_by_its_arguments():
    with _settings():
        a = get_provider(LLMConfig(model="gpt-4o-mini"))
        b = get_provider(LLMConfig(model="gpt-4o"))
        other_key = get_provider(LLMConfig(model="gpt-4o-mini", api_key="sk-other"))
    assert a is b  # same client settings, model is per call
    assert other_key is not a


def test_each_event_loop_and_sync_calls_get_their_own():
    async def in_loop():
        return get_provider(LLMConfig(model="gpt-4o-mini"))

    with _settings():
        first = asyncio.run(in_loop())
        second = asyncio.run(in_loop())
        sync_a = get_provider(LLMConfig(model="gpt-4o-mini"))
        sync_b = get_provider(LLMConfig(model="gpt-4o-mini"))
    assert first is not second
    assert sync_a is not sync_b  # outside a loop: new each time, as before
    assert sync_a is not first


def test_custom_factories_are_still_called_every_time():
    made = []

    def factory(config, settings):
        made.append(object())
        return made[-1]

    register_provider("myco/", factory)
    try:
        with _settings():
            get_provider(LLMConfig(model="myco/model"))
            get_provider(LLMConfig(model="myco/model"))
    finally:
        providers._PROVIDERS.pop("myco/", None)
    assert len(made) == 2


async def test_clear_forgets_reused_providers():
    with _settings():
        a = get_provider(LLMConfig(model="gpt-4o-mini"))
        clear_provider_cache()
        b = get_provider(LLMConfig(model="gpt-4o-mini"))
    assert a is not b


class _HoldsLoop:  # stands in for a provider whose pooled connection holds its loop
    def __init__(self):
        self.loop = asyncio.get_running_loop()


def test_closed_loops_are_dropped():
    # A provider's pooled connections hold its loop, so without the purge every
    # asyncio.run() left a closed loop, a provider and a socket behind.
    async def in_loop():
        providers._reuse(_HoldsLoop)
        return len(providers._loop_providers)

    counts = [asyncio.run(in_loop()) for _ in range(50)]
    assert max(counts) <= 2  # this loop's store, plus at most the last closed one


def test_unclosed_loops_are_dropped():
    # new_event_loop() + run_until_complete() per request, never close(): the
    # loop is not closed, and the provider keeps it alive, so only "is it
    # running" can tell that it is finished.
    async def in_loop():
        providers._reuse(_HoldsLoop)
        return len(providers._loop_providers)

    loops = []
    try:
        for _ in range(50):
            loops.append(asyncio.new_event_loop())
            count = loops[-1].run_until_complete(in_loop())
            assert count == 1  # only the loop running now
    finally:
        for loop in loops:
            loop.close()


@pytest.fixture
def background_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def _in(loop, fn):
    async def call():
        return fn()

    return asyncio.run_coroutine_threadsafe(call(), loop).result(5)


async def test_a_loop_running_in_another_thread_keeps_its_providers(background_loop):
    first = _in(background_loop, lambda: providers._reuse(_HoldsLoop))
    mine = providers._reuse(_HoldsLoop)
    assert _in(background_loop, lambda: providers._reuse(_HoldsLoop)) is first
    assert providers._reuse(_HoldsLoop) is mine
    assert mine is not first


async def test_stores_are_capped_to_the_most_recently_used_loops(background_loop, monkeypatch):
    monkeypatch.setattr(providers, "_MAX_LOOPS", 1)
    first = _in(background_loop, lambda: providers._reuse(_HoldsLoop))
    providers._reuse(_HoldsLoop)  # this loop's store pushes the other one out
    assert len(providers._loop_providers) == 1
    assert _in(background_loop, lambda: providers._reuse(_HoldsLoop)) is not first


_UNCLOSED_LOOPS_SCRIPT = textwrap.dedent(
    """
    import asyncio, resource, sys, threading
    from http.server import ThreadingHTTPServer
    from unittest.mock import patch

    from continuum.config import settings
    from continuum.llm.config import LLMConfig
    from continuum.llm.providers import get_provider
    from test_provider_reuse import _ChatHandler

    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    config = LLMConfig(model="gpt-4o-mini", api_base=base, max_retries=0)
    messages = [{"role": "user", "content": "hi"}]

    async def call():
        return (await get_provider(config).acomplete(messages, config)).content

    with patch.multiple(settings, smart_gateway_url="", openai_api_key="sk-test"):
        for n in range(1, 401):
            try:
                loop = asyncio.new_event_loop()  # never closed, as some callers do
                assert loop.run_until_complete(call()) == "ok"
            except OSError as exc:
                print(f"failed after {n - 1} requests: {exc}")
                sys.exit(1)
    print("ok")
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="needs the resource module")
def test_unclosed_loop_per_request_does_not_run_out_of_files():
    # Reported in review: on the earlier fix this failed with "Too many open
    # files" after about 60 requests under a 256 file limit; dev made 400.
    here = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([here, *sys.path])}
    done = subprocess.run(
        [sys.executable, "-c", _UNCLOSED_LOOPS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr[-2000:]


async def test_each_loop_keeps_only_the_most_recently_used_providers():
    cap = providers._MAX_PROVIDERS_PER_LOOP
    with _settings():
        first = get_provider(LLMConfig(model="gpt-4o-mini", api_key="sk-0"))
        second = get_provider(LLMConfig(model="gpt-4o-mini", api_key="sk-1"))
        for i in range(2, cap + 1):
            get_provider(LLMConfig(model="gpt-4o-mini", api_key=f"sk-{i}"))
            # Keep the first one recently used, so the second is the oldest.
            assert get_provider(LLMConfig(model="gpt-4o-mini", api_key="sk-0")) is first
        store = providers._loop_providers[id(asyncio.get_running_loop())][1]
        assert len(store) == cap
        assert get_provider(LLMConfig(model="gpt-4o-mini", api_key="sk-0")) is first
        assert get_provider(LLMConfig(model="gpt-4o-mini", api_key="sk-1")) is not second


async def test_fork_hook_rebinds_without_taking_the_old_lock():
    with _settings():
        get_provider(LLMConfig(model="gpt-4o-mini"))
    old_lock = providers._providers_lock
    old_lock.acquire()  # another thread holds it at the moment of the fork
    try:
        providers._forget_providers_after_fork()
        assert providers._providers_lock is not old_lock
        assert len(providers._loop_providers) == 0
        with _settings():
            get_provider(LLMConfig(model="gpt-4o-mini"))  # does not deadlock
    finally:
        old_lock.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
async def test_forked_child_gets_its_own_provider():
    with _settings():
        parent = get_provider(LLMConfig(model="gpt-4o-mini"))
    pid = os.fork()
    if pid == 0:  # child: exit status carries the result, never return to pytest
        try:
            ok = len(providers._loop_providers) == 0 and providers._providers_lock.acquire(
                timeout=5
            )
            os._exit(0 if ok else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with _settings():
        assert get_provider(LLMConfig(model="gpt-4o-mini")) is parent


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the client pools the socket

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(
            {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_provider_got_outside_a_loop_works_in_each_asyncio_run():
    # A provider cached outside any loop had its async client bound to the first
    # asyncio.run(), so every second call failed with "Event loop is closed".
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    config = LLMConfig(model="gpt-4o-mini", api_base=base, max_retries=0)
    messages = [{"role": "user", "content": "hi"}]
    try:
        with _settings():
            for _ in range(6):
                p = get_provider(config)
                assert asyncio.run(p.acomplete(messages, config)).content == "ok"
    finally:
        server.shutdown()
        server.server_close()
