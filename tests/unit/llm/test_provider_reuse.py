"""Built-in providers are reused across calls instead of rebuilt each call.

get_provider() built a new provider, with new SDK clients and connection pools,
for every model call, so each call paid a fresh TCP + TLS handshake. Through the
gateway that measured ~0.5 s of time-to-first-token per call (reused client
0.96 s median, new client per call 1.58 s, Continuum 1.45 s).

Covers:
- same settings in one event loop -> the same provider (gateway and direct)
- another event loop, or no loop, gets its own (async clients are loop-bound)
- a different key, model routing argument or retry count -> a different provider
- custom registered factories are called every time, as before
- clear_provider_cache() forgets them
"""

from __future__ import annotations

import asyncio
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
    assert sync_a is sync_b
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


def test_clear_forgets_reused_providers():
    with _settings():
        a = get_provider(LLMConfig(model="gpt-4o-mini"))
        clear_provider_cache()
        b = get_provider(LLMConfig(model="gpt-4o-mini"))
    assert a is not b
