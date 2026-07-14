"""Unit tests for LocalHeadroomClient — the in-process Headroom backend
(``headroom_mode=local``).

All tests stub the ``headroom`` package in sys.modules — the library does not
need to be installed. Contract stubbed to headroom-ai v0.29.0:

  headroom.compress(messages, model=...) -> CompressResult(messages,
      tokens_before, tokens_after, tokens_saved,
      compression_ratio,  # FRACTION SAVED — inverted vs the sidecar's field
      transforms_applied)
  headroom.cache.compression_store.get_compression_store().retrieve(hash, query)
      -> CompressionEntry(original_content=...) | None
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from continuum.config import settings
from continuum.llm.headroom.client import CompressionStats, HeadroomClient
from continuum.llm.headroom.compressor import (
    HeadroomCompressor,
    get_headroom_client,
    reset_headroom_compressor,
)

MESSAGES = [
    {"role": "user", "content": "Which users are active?"},
    {"role": "tool", "tool_call_id": "c1", "content": '[{"id": 1}]'},
]

MARKER_HASH = "7e443033ad1ff3f9ca0b8c49"

COMPRESSED_MESSAGES = [
    {"role": "user", "content": "Which users are active?"},
    {
        "role": "tool",
        "tool_call_id": "c1",
        "content": f"[2501 lines compressed to 7. Retrieve more: hash={MARKER_HASH}]",
    },
]


def _compress_result(**overrides) -> SimpleNamespace:
    result = SimpleNamespace(
        messages=COMPRESSED_MESSAGES,
        tokens_before=5000,
        tokens_after=1000,
        tokens_saved=4000,
        compression_ratio=0.8,  # library semantics: 80% SAVED
        transforms_applied=["router:smart_crusher:0.20"],
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def _install_stub_headroom(monkeypatch, compress_fn=None, retrieve_fn=None):
    """Register a minimal fake ``headroom`` package in sys.modules.

    Returns the stub router config so tests can assert the client's
    pipeline alignment (relevance_split) took effect.
    """
    headroom_mod = types.ModuleType("headroom")
    headroom_mod.compress = compress_fn or (lambda messages, **kw: _compress_result())

    store = SimpleNamespace(retrieve=retrieve_fn or (lambda h, q=None: None))
    store_mod = types.ModuleType("headroom.cache.compression_store")
    store_mod.get_compression_store = lambda: store
    cache_mod = types.ModuleType("headroom.cache")
    cache_mod.compression_store = store_mod
    headroom_mod.cache = cache_mod

    # headroom.compress the MODULE (for `from headroom.compress import
    # _get_pipeline`); the attribute above still wins for `from headroom
    # import compress`.
    router_config = SimpleNamespace(relevance_split=True)
    pipeline = SimpleNamespace(transforms=[SimpleNamespace(config=router_config)])
    compress_mod = types.ModuleType("headroom.compress")
    compress_mod._get_pipeline = lambda: pipeline

    monkeypatch.setitem(sys.modules, "headroom", headroom_mod)
    monkeypatch.setitem(sys.modules, "headroom.cache", cache_mod)
    monkeypatch.setitem(sys.modules, "headroom.cache.compression_store", store_mod)
    monkeypatch.setitem(sys.modules, "headroom.compress", compress_mod)
    return router_config


def _local_client(monkeypatch, **stub_kwargs):
    _install_stub_headroom(monkeypatch, **stub_kwargs)
    from continuum.llm.headroom.local_client import LocalHeadroomClient

    return LocalHeadroomClient(timeout=5.0)


class TestCompress:
    async def test_returns_same_tuple_shape_as_http_client(self, monkeypatch):
        seen: dict = {}

        def fake_compress(messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return _compress_result()

        client = _local_client(monkeypatch, compress_fn=fake_compress)
        messages, stats, ccr_hashes = await client.compress(MESSAGES, model="gpt-4o")

        assert seen["messages"] == MESSAGES
        assert seen["kwargs"] == {
            "model": "gpt-4o",
            "compress_system_messages": False,
            "compress_user_messages": False,
        }
        assert messages == COMPRESSED_MESSAGES
        assert isinstance(stats, CompressionStats)
        assert stats.tokens_before == 5000
        assert stats.tokens_after == 1000
        assert stats.tokens_saved == 4000
        assert stats.transforms_applied == ["router:smart_crusher:0.20"]
        assert ccr_hashes == []

    async def test_compression_ratio_is_after_over_before_not_library_semantics(
        self, monkeypatch
    ):
        # Library reports 0.8 (= 80% saved); Continuum's field means
        # after/before, so it must come out as 0.2 — NOT be passed through.
        client = _local_client(monkeypatch)
        _, stats, _ = await client.compress(MESSAGES, model="gpt-4o")
        assert stats.compression_ratio == pytest.approx(0.2)

    async def test_zero_tokens_before_yields_ratio_one(self, monkeypatch):
        client = _local_client(
            monkeypatch,
            compress_fn=lambda messages, **kw: _compress_result(
                tokens_before=0, tokens_after=0, tokens_saved=0
            ),
        )
        _, stats, _ = await client.compress(MESSAGES, model="gpt-4o")
        assert stats.compression_ratio == 1.0

    async def test_disables_relevance_split_at_construction(self, monkeypatch):
        # Sidecar parity: with the library's relevance_split ON and Kompress
        # resident, LOG/SEARCH content is fed whole to the ML model (14.5%
        # instead of the crusher's 99%). Constructing the client must flip the
        # router config off BEFORE any compress runs (results are cached).
        router_config = _install_stub_headroom(monkeypatch)
        assert router_config.relevance_split is True
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        LocalHeadroomClient(timeout=5.0)
        assert router_config.relevance_split is False

    async def test_protects_system_and_user_messages(self, monkeypatch):
        # SAFETY: library default compresses system messages; we must pin the
        # sidecar's protect-system/protect-user defaults so local mode can't
        # corrupt instructions.
        seen: dict = {}

        def fake_compress(messages, **kwargs):
            seen.update(kwargs)
            return _compress_result()

        client = _local_client(monkeypatch, compress_fn=fake_compress)
        await client.compress(MESSAGES, model="gpt-4o")
        assert seen.get("compress_system_messages") is False
        assert seen.get("compress_user_messages") is False

    async def test_none_model_omits_the_model_kwarg_but_keeps_protection(self, monkeypatch):
        seen: dict = {}

        def fake_compress(messages, **kwargs):
            seen.update(kwargs)
            return _compress_result()

        client = _local_client(monkeypatch, compress_fn=fake_compress)
        await client.compress(MESSAGES, model=None)
        assert "model" not in seen
        assert seen.get("compress_system_messages") is False

    async def test_slow_compress_raises_timeout_for_fail_open(self, monkeypatch):
        # Parity with the sidecar's HTTP timeout: a hung/cold path (e.g. first
        # Kompress ONNX load) must surface as TimeoutError so the compressor
        # fail-opens instead of stalling the run.
        def slow_compress(messages, **kwargs):
            time.sleep(0.5)
            return _compress_result()

        _install_stub_headroom(monkeypatch, compress_fn=slow_compress)
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        client = LocalHeadroomClient(timeout=0.05)
        with pytest.raises(TimeoutError):
            await client.compress(MESSAGES, model="gpt-4o")


class TestRetrieve:
    async def test_returns_original_content(self, monkeypatch):
        entry = SimpleNamespace(original_content="the original 2501 log lines")
        client = _local_client(monkeypatch, retrieve_fn=lambda h, q=None: entry)
        assert await client.retrieve(MARKER_HASH) == "the original 2501 log lines"

    async def test_miss_raises_like_the_sidecar_404(self, monkeypatch):
        client = _local_client(monkeypatch, retrieve_fn=lambda h, q=None: None)
        with pytest.raises(KeyError):
            await client.retrieve("f" * 24)


class TestMissingLibrary:
    async def test_constructor_raises_actionable_error(self, monkeypatch):
        # None in sys.modules makes `import headroom` raise ImportError.
        monkeypatch.setitem(sys.modules, "headroom", None)
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        with pytest.raises(RuntimeError, match="headroom-local"):
            LocalHeadroomClient()


class TestCompressorIntegration:
    """The compressor's guarantees must hold identically over the local backend."""

    async def test_marker_hashes_flow_into_issued_hashes(self, monkeypatch):
        # ccr_hashes is [] from the local client — the marker scan alone must
        # authorize the retrieve (decision #6: union of field + markers).
        client = _local_client(monkeypatch)
        comp = HeadroomCompressor(client=client, fail_open=True)
        compressed = await comp.apply(MESSAGES, model="gpt-4o")
        assert compressed == COMPRESSED_MESSAGES
        assert MARKER_HASH in comp.issued_hashes

    async def test_resolve_retrieve_round_trip(self, monkeypatch):
        entry = SimpleNamespace(original_content="the original 2501 log lines")
        client = _local_client(monkeypatch, retrieve_fn=lambda h, q=None: entry)
        comp = HeadroomCompressor(client=client, fail_open=True)
        await comp.apply(MESSAGES, model="gpt-4o")
        assert await comp.resolve_retrieve(MARKER_HASH) == "the original 2501 log lines"

    async def test_forged_hash_still_rejected(self, monkeypatch):
        client = _local_client(monkeypatch)
        comp = HeadroomCompressor(client=client, fail_open=True)
        out = await comp.resolve_retrieve("f" * 24)
        assert "not issued" in out

    async def test_expired_entry_fail_opens(self, monkeypatch):
        client = _local_client(monkeypatch, retrieve_fn=lambda h, q=None: None)
        comp = HeadroomCompressor(client=client, fail_open=True)
        await comp.apply(MESSAGES, model="gpt-4o")  # issues MARKER_HASH
        out = await comp.resolve_retrieve(MARKER_HASH)
        assert "retrieval failed" in out


class TestFactory:
    @pytest.fixture(autouse=True)
    def _fresh_globals(self):
        reset_headroom_compressor()
        yield
        reset_headroom_compressor()

    async def test_local_mode_builds_local_client(self, monkeypatch):
        _install_stub_headroom(monkeypatch)
        monkeypatch.setattr(settings, "headroom_mode", "local")
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        client = get_headroom_client()
        assert isinstance(client, LocalHeadroomClient)
        assert client._timeout == settings.headroom_timeout_seconds

    async def test_endpoint_mode_builds_http_client(self, monkeypatch):
        monkeypatch.setattr(settings, "headroom_mode", "endpoint")
        client = get_headroom_client()
        assert isinstance(client, HeadroomClient)
        await client.aclose()

    async def test_default_mode_is_local(self):
        assert type(settings).model_fields["headroom_mode"].default == "local"

    async def test_missing_library_fail_open_returns_passthrough_backend(self, monkeypatch):
        # headroom not importable → construction fails. Fail-open must give a
        # no-op backend (compression disabled) instead of crashing the call.
        monkeypatch.setattr(settings, "headroom_mode", "local")
        monkeypatch.setattr(settings, "headroom_fail_open", True)
        monkeypatch.setitem(sys.modules, "headroom", None)

        client = get_headroom_client()
        msgs = [{"role": "user", "content": "hi"}]
        out, stats, hashes = await client.compress(msgs, model="gpt-4o")
        assert out == msgs  # unchanged — no compression
        assert stats.tokens_saved == 0
        assert hashes == []

    async def test_missing_library_fail_closed_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "headroom_mode", "local")
        monkeypatch.setattr(settings, "headroom_fail_open", False)
        monkeypatch.setitem(sys.modules, "headroom", None)

        with pytest.raises(RuntimeError, match="headroom-local"):
            get_headroom_client()


class TestKompressPrewarm:
    """Option-1 prose enablement: pre-warm the model + raise the execution
    budget so in-process Kompress actually fires."""

    @pytest.fixture(autouse=True)
    def _isolate_kompress_env(self):
        # _enable_prose_kompress sets a real env var (os.environ.setdefault),
        # which monkeypatch does not track — clean it around every test so
        # these cases can't leak into each other regardless of order.
        os.environ.pop("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", None)
        yield
        os.environ.pop("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", None)

    def _stub_kompress(self, monkeypatch):
        """Stub headroom.transforms.kompress_compressor.KompressCompressor;
        return an Event set when preload() is called (it runs on a daemon
        thread)."""
        called = threading.Event()

        class _StubKompress:
            def preload(self, *, allow_download=True):
                called.set()
                return "onnx"

        mod = types.ModuleType("headroom.transforms.kompress_compressor")
        mod.KompressCompressor = _StubKompress
        monkeypatch.setitem(sys.modules, "headroom.transforms.kompress_compressor", mod)
        return called

    async def test_prewarm_sets_budget_and_loads_model(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", raising=False)
        _install_stub_headroom(monkeypatch)
        called = self._stub_kompress(monkeypatch)
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        LocalHeadroomClient(kompress_prewarm=True, kompress_execution_timeout_ms=7000)

        # (1) execution budget raised from the 25ms default so calls WAIT for a slot
        assert os.environ["HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS"] == "7000"
        # (2) model preloaded on the background daemon thread
        assert called.wait(timeout=5.0), "preload() was not called"

    async def test_prewarm_off_by_default_does_not_load(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", raising=False)
        _install_stub_headroom(monkeypatch)
        called = self._stub_kompress(monkeypatch)
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        LocalHeadroomClient()  # prewarm defaults off

        assert "HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS" not in os.environ
        assert not called.wait(timeout=0.5), "preload() must not run when prewarm is off"

    async def test_explicit_env_budget_is_not_overridden(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", "1234")
        _install_stub_headroom(monkeypatch)
        self._stub_kompress(monkeypatch)
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        LocalHeadroomClient(kompress_prewarm=True, kompress_execution_timeout_ms=7000)

        # setdefault — an operator's explicit env wins over our default
        assert os.environ["HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS"] == "1234"

    async def test_first_prose_call_waits_for_warmup_not_races_it(self, monkeypatch):
        # The fix for the incident-desk cold-start: an immediate first query
        # must WAIT for the model instead of skipping to passthrough.
        monkeypatch.delenv("HEADROOM_KOMPRESS_EXECUTION_TIMEOUT_MS", raising=False)
        _install_stub_headroom(monkeypatch)
        release = threading.Event()

        class _SlowKompress:
            def preload(self, *, allow_download=True):
                release.wait(5.0)  # hold the warmup open until the test releases
                return "onnx"

        mod = types.ModuleType("headroom.transforms.kompress_compressor")
        mod.KompressCompressor = _SlowKompress
        monkeypatch.setitem(sys.modules, "headroom.transforms.kompress_compressor", mod)
        from continuum.llm.headroom.local_client import LocalHeadroomClient

        client = LocalHeadroomClient(timeout=5.0, kompress_prewarm=True)
        task = asyncio.create_task(client.compress([{"role": "user", "content": "hi"}], "gpt-4o"))
        await asyncio.sleep(0.3)
        assert not task.done(), "first prose compress must wait for warmup, not race it"

        release.set()  # model becomes ready
        out, _stats, _hashes = await task
        assert out == COMPRESSED_MESSAGES  # compressed once the model was ready
