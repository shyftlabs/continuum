"""Phase 2 (CCR retrieval) unit tests.

Covers: resolve_retrieve (anti-forgery + fail-open), startup tool
registration in get_tools_for_llm, and tool_attention always-promotion.
The tool-loop interception itself is exercised via the executor/runner
code paths mirroring think/handoff handling.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx

from continuum.config import settings
from continuum.llm.headroom.compressor import (
    RETRIEVE_TOOL,
    RETRIEVE_TOOL_NAME,
    HeadroomCompressor,
    enter_run_compressor,
    get_headroom_compressor,
    reset_headroom_compressor,
    use_run_compressor_if_enabled,
)

HASH = "7e443033ad1ff3f9ca0b8c49"


def _compressor(retrieve_result="original content", error=None) -> HeadroomCompressor:
    client = AsyncMock()
    if error is not None:
        client.retrieve.side_effect = error
    else:
        client.retrieve.return_value = retrieve_result
    return HeadroomCompressor(client=client, fail_open=True)


class TestResolveRetrieve:
    async def test_authorized_hash_returns_original(self):
        compressor = _compressor()
        compressor._issued_hashes.add(HASH)
        result = await compressor.resolve_retrieve(HASH, query="the error")
        assert result == "original content"
        compressor._client.retrieve.assert_awaited_once_with(HASH, "the error")

    async def test_unissued_hash_rejected_without_sidecar_call(self):
        """SECURITY: fabricated/replayed hashes must never reach the store."""
        compressor = _compressor()
        result = await compressor.resolve_retrieve("f" * 24)
        assert "not issued" in result
        compressor._client.retrieve.assert_not_awaited()

    async def test_sidecar_error_fails_open_with_guidance(self):
        compressor = _compressor(error=httpx.ConnectError("down"))
        compressor._issued_hashes.add(HASH)
        result = await compressor.resolve_retrieve(HASH)
        assert "retrieval failed" in result
        assert "re-run" in result  # tells the model the recovery path

    async def test_never_raises(self):
        compressor = _compressor(error=RuntimeError("boom"))
        compressor._issued_hashes.add(HASH)
        result = await compressor.resolve_retrieve(HASH)  # must not raise
        assert isinstance(result, str)


DUMMY_TOOL = {
    "type": "function",
    "function": {"name": "search", "description": "d", "parameters": {"type": "object"}},
}


class TestStartupRegistration:
    def test_tool_registered_when_enabled_and_agent_has_tools(self, monkeypatch):
        monkeypatch.setattr(settings, "headroom_enabled", True)
        from continuum.agent.base import BaseAgent

        agent = BaseAgent(name="t", instructions="x", tools=[DUMMY_TOOL])
        names = [t.get("function", {}).get("name") for t in agent.get_tools_for_llm()]
        assert RETRIEVE_TOOL_NAME in names

    def test_tool_absent_for_toolless_agent_even_when_enabled(self, monkeypatch):
        """CCR markers only appear on tool outputs — a no-tool agent can never
        see one, and adding a tool would change its provider call shape
        (e.g. structured-output constrained mode)."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        from continuum.agent.base import BaseAgent

        agent = BaseAgent(name="t", instructions="x")
        names = [t.get("function", {}).get("name") for t in agent.get_tools_for_llm()]
        assert RETRIEVE_TOOL_NAME not in names

    def test_tool_absent_when_headroom_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "headroom_enabled", False)
        from continuum.agent.base import BaseAgent

        agent = BaseAgent(name="t", instructions="x", tools=[DUMMY_TOOL])
        names = [t.get("function", {}).get("name") for t in agent.get_tools_for_llm()]
        assert RETRIEVE_TOOL_NAME not in names

    def test_tool_attention_always_promotes_retrieve(self):
        from continuum.tools.tool_attention.router import _BUILTIN_ALWAYS_PROMOTE

        assert RETRIEVE_TOOL_NAME in _BUILTIN_ALWAYS_PROMOTE

    def test_retrieve_tool_schema_shape(self):
        fn = RETRIEVE_TOOL["function"]
        assert fn["name"] == RETRIEVE_TOOL_NAME
        assert "hash" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["hash"]


class TestRetrieveResultProtection:
    """Anti-doom-loop: a continuum_headroom_retrieve result must never be re-compressed
    (observed live: retrieve -> recompress -> the original vanishes again)."""

    ORIGINAL_LOG = "line1\n" * 5000 + "NEEDLE-XYZ"

    def _messages(self):
        return [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "r1", "type": "function",
                 "function": {"name": RETRIEVE_TOOL_NAME, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "r1", "content": self.ORIGINAL_LOG},
        ]

    async def test_retrieve_result_content_restored_after_compression(self):
        original = self._messages()
        crushed = [dict(m) for m in original]
        crushed[2] = {**crushed[2], "content": "[5001 lines compressed to 2]"}
        client = AsyncMock()
        from continuum.llm.headroom.client import CompressionStats
        client.compress.return_value = (
            crushed,
            CompressionStats(100, 1, 99, 0.01, ["router:log:0.00"]),
            [],
        )
        compressor = HeadroomCompressor(client=client, fail_open=True)
        out = await compressor.apply(original, model="gpt-4o")
        assert out[2]["content"] == self.ORIGINAL_LOG  # protected
        assert "NEEDLE-XYZ" in out[2]["content"]

    async def test_restore_is_tool_call_id_keyed_not_positional(self):
        """Robustness: if the sidecar returns messages reordered (or a different
        count), the restore must still land on the RIGHT message — matched by
        tool_call_id, never by list position."""
        original = self._messages()  # [user, assistant(retrieve r1), tool r1]
        # Sidecar crushes the retrieve result AND returns the list reordered:
        # the tool message is no longer at the same index it occupied in input.
        crushed_tool = {"role": "tool", "tool_call_id": "r1",
                        "content": "[5001 lines compressed to 2]"}
        reordered = [
            {**original[0]},
            crushed_tool,          # moved earlier than its original index (was last)
            {**original[1]},
        ]
        client = AsyncMock()
        from continuum.llm.headroom.client import CompressionStats
        client.compress.return_value = (
            reordered, CompressionStats(100, 1, 99, 0.01, []), [],
        )
        compressor = HeadroomCompressor(client=client, fail_open=True)
        out = await compressor.apply(original, model="gpt-4o")
        restored = next(m for m in out if m.get("tool_call_id") == "r1")
        assert restored["content"] == self.ORIGINAL_LOG  # right message, despite reorder
        assert "NEEDLE-XYZ" in restored["content"]

    async def test_normal_tool_results_still_compressed(self):
        original = self._messages()
        original[1]["tool_calls"][0]["function"]["name"] = "fetch_logs"  # not retrieve
        crushed = [dict(m) for m in original]
        crushed[2] = {**crushed[2], "content": "[compressed]"}
        client = AsyncMock()
        from continuum.llm.headroom.client import CompressionStats
        client.compress.return_value = (
            crushed, CompressionStats(100, 1, 99, 0.01, []), [],
        )
        compressor = HeadroomCompressor(client=client, fail_open=True)
        out = await compressor.apply(original, model="gpt-4o")
        assert out[2]["content"] == "[compressed]"  # NOT protected


class TestPerRunIsolation:
    """The retrieve anti-forgery boundary (``issued_hashes``) must be per-run,
    not process-wide: one agent cannot authorize a retrieve of another agent's
    cached originals from the shared sidecar store."""

    def _mock_client(self, monkeypatch):
        """new_run_compressor() builds a real httpx client by default — swap it
        for a mock so these tests exercise only the scoping, not the network."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(
            "continuum.llm.headroom.compressor.get_headroom_client",
            lambda: AsyncMock(),
        )
        reset_headroom_compressor()

    def test_sequential_runs_get_distinct_compressors(self, monkeypatch):
        self._mock_client(monkeypatch)
        with use_run_compressor_if_enabled():
            c1 = get_headroom_compressor()
        with use_run_compressor_if_enabled():
            c2 = get_headroom_compressor()
        assert c1 is not c2

    def test_nested_scope_inherits_same_compressor(self, monkeypatch):
        """A handoff/executor re-entry must KEEP the run's compressor so
        pre-handoff issued hashes stay retrievable — it neither rebinds nor
        resets."""
        self._mock_client(monkeypatch)
        with use_run_compressor_if_enabled():
            outer = get_headroom_compressor()
            with use_run_compressor_if_enabled():  # nested handoff/executor
                inner = get_headroom_compressor()
                inner._issued_hashes.add(HASH)
            # nested exit must not drop the run's compressor or its hashes
            assert get_headroom_compressor() is outer
            assert HASH in outer._issued_hashes
        assert inner is outer

    def test_outside_run_uses_stable_global_fallback(self, monkeypatch):
        self._mock_client(monkeypatch)
        assert get_headroom_compressor() is get_headroom_compressor()

    def test_finished_run_is_readable_as_global_for_observability(self, monkeypatch):
        """Post-run inspectors (the rig glassbox) read get_headroom_compressor()
        AFTER the run returns; it must reflect the run that just finished, not an
        empty global — else the run-scoped isolation would blank the glassbox."""
        self._mock_client(monkeypatch)
        with use_run_compressor_if_enabled():
            run_comp = get_headroom_compressor()
            run_comp._issued_hashes.add(HASH)
        # Outside the scope now — the fallback must be the finished run's compressor
        assert get_headroom_compressor() is run_comp
        assert HASH in get_headroom_compressor().issued_hashes

    def test_disabled_headroom_does_not_bind(self, monkeypatch):
        monkeypatch.setattr(settings, "headroom_enabled", False)
        reset_headroom_compressor()
        assert enter_run_compressor() is None

    def test_bind_failure_is_fail_safe_not_raising(self, monkeypatch):
        """A binding failure must return None (never raise) so it can't skip the
        caller's policy-context teardown — which would leak data-label
        enforcement state across runs."""
        monkeypatch.setattr(settings, "headroom_enabled", True)
        reset_headroom_compressor()

        def _boom() -> object:
            raise RuntimeError("misconfigured client")

        monkeypatch.setattr(
            "continuum.llm.headroom.compressor.new_run_compressor", _boom
        )
        assert enter_run_compressor() is None  # fail-safe, no exception

    async def test_concurrent_runs_have_isolated_issued_hashes(self, monkeypatch):
        """Two parallel agents (separate async tasks) each get their own
        compressor; neither sees the other's issued hashes."""
        self._mock_client(monkeypatch)
        a, b = "a" * 24, "b" * 24
        barrier = asyncio.Barrier(2)
        results: dict[str, tuple[bool, bool]] = {}

        async def one(tag: str, mine: str, theirs: str) -> None:
            with use_run_compressor_if_enabled():
                comp = get_headroom_compressor()
                comp._issued_hashes.add(mine)
                await barrier.wait()  # both have added before anyone checks
                results[tag] = (mine in comp._issued_hashes, theirs in comp._issued_hashes)

        await asyncio.gather(one("A", a, b), one("B", b, a))
        assert results["A"] == (True, False)  # sees own, never the peer's
        assert results["B"] == (True, False)

    def test_global_fallback_does_not_deadlock(self, monkeypatch):
        """Regression: the fallback must build the compressor OUTSIDE _global_lock.
        new_run_compressor() -> get_headroom_client() re-acquires the same
        non-reentrant lock; nesting it self-deadlocks (hung 'thinking' live).
        Runs the REAL lock path (only the httpx client is stubbed) in a worker
        thread so a regression fails on timeout instead of hanging forever."""
        import threading as _threading

        monkeypatch.setattr(settings, "headroom_enabled", True)
        monkeypatch.setattr(
            "continuum.llm.headroom.compressor.HeadroomClient", lambda **kw: object()
        )
        reset_headroom_compressor()
        try:
            out: dict[str, object] = {}

            def work() -> None:
                c1 = get_headroom_compressor()  # fallback: contextvar is None here
                out["stable"] = get_headroom_compressor() is c1

            t = _threading.Thread(target=work)
            t.start()
            t.join(timeout=5)
            assert not t.is_alive(), "get_headroom_compressor() deadlocked"
            assert out.get("stable") is True
        finally:
            reset_headroom_compressor()

    async def test_hash_from_one_run_rejected_in_another(self, monkeypatch):
        """The end-to-end guarantee: a hash issued in run 1 is un-issued in a
        fresh run 2, so resolve_retrieve rejects it without hitting the store."""
        self._mock_client(monkeypatch)
        with use_run_compressor_if_enabled():
            get_headroom_compressor()._issued_hashes.add(HASH)
        with use_run_compressor_if_enabled():
            comp = get_headroom_compressor()
            result = await comp.resolve_retrieve(HASH)
        assert "not issued" in result
        comp._client.retrieve.assert_not_awaited()
