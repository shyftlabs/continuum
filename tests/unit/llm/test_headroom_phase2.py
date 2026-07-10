"""Phase 2 (CCR retrieval) unit tests.

Covers: resolve_retrieve (anti-forgery + fail-open), startup tool
registration in get_tools_for_llm, and tool_attention always-promotion.
The tool-loop interception itself is exercised via the executor/runner
code paths mirroring think/handoff handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx

from continuum.config import settings
from continuum.llm.headroom.compressor import (
    RETRIEVE_TOOL,
    RETRIEVE_TOOL_NAME,
    HeadroomCompressor,
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
