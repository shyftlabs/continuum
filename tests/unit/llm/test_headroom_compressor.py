"""Unit tests for HeadroomCompressor — Phase-1 orchestration (compress-only).

Phase-1 scope: pre-call compression with fail-open/fail-closed policy and
per-run hash bookkeeping from the response's ``ccr_hashes`` field.
No `continuum_retrieve` tool injection (Phase 2, evidence-gated).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from continuum.llm.headroom.client import CompressionStats
from continuum.llm.headroom.compressor import HeadroomCompressor

ORIGINAL = [
    {"role": "user", "content": "Which users are active?"},
    {"role": "tool", "tool_call_id": "c1", "content": '[{"id": 1}, {"id": 2}]'},
]
COMPRESSED = [
    {"role": "user", "content": "Which users are active?"},
    {"role": "tool", "tool_call_id": "c1", "content": "[compressed]"},
]
STATS = CompressionStats(
    tokens_before=5000,
    tokens_after=1000,
    tokens_saved=4000,
    compression_ratio=0.2,
    transforms_applied=["router:smart_crusher:0.20"],
)


def _mock_client(result=None, error: Exception | None = None) -> AsyncMock:
    client = AsyncMock()
    if error is not None:
        client.compress.side_effect = error
    else:
        client.compress.return_value = result or (COMPRESSED, STATS, [])
    return client


class TestApply:
    async def test_returns_compressed_messages(self):
        compressor = HeadroomCompressor(client=_mock_client(), fail_open=True)
        messages = await compressor.apply(ORIGINAL, model="gpt-4o")
        assert messages == COMPRESSED

    async def test_returns_new_list_not_input_mutation(self):
        """The seam contract: rebind, don't mutate — the caller's dicts stay pristine."""
        compressor = HeadroomCompressor(client=_mock_client(), fail_open=True)
        original_snapshot = [dict(m) for m in ORIGINAL]
        result = await compressor.apply(ORIGINAL, model="gpt-4o")
        assert result is not ORIGINAL
        assert ORIGINAL == original_snapshot

    async def test_records_issued_hashes_from_ccr_hashes_field(self):
        client = _mock_client(result=(COMPRESSED, STATS, ["aaa111", "bbb222"]))
        compressor = HeadroomCompressor(client=client, fail_open=True)
        await compressor.apply(ORIGINAL, model="gpt-4o")
        assert compressor.issued_hashes == {"aaa111", "bbb222"}

    async def test_issued_hashes_accumulate_across_turns(self):
        client = _mock_client(result=(COMPRESSED, STATS, ["aaa111"]))
        compressor = HeadroomCompressor(client=client, fail_open=True)
        await compressor.apply(ORIGINAL, model="gpt-4o")
        client.compress.return_value = (COMPRESSED, STATS, ["bbb222"])
        await compressor.apply(ORIGINAL, model="gpt-4o")
        assert compressor.issued_hashes == {"aaa111", "bbb222"}

    async def test_last_stats_exposed_for_observability(self):
        compressor = HeadroomCompressor(client=_mock_client(), fail_open=True)
        await compressor.apply(ORIGINAL, model="gpt-4o")
        assert compressor.last_stats is STATS


class TestFailurePolicy:
    async def test_fail_open_returns_original_messages(self):
        client = _mock_client(error=httpx.ConnectError("sidecar down"))
        compressor = HeadroomCompressor(client=client, fail_open=True)
        messages = await compressor.apply(ORIGINAL, model="gpt-4o")
        assert messages is ORIGINAL

    async def test_fail_closed_reraises(self):
        client = _mock_client(error=httpx.ConnectError("sidecar down"))
        compressor = HeadroomCompressor(client=client, fail_open=False)
        with pytest.raises(httpx.ConnectError):
            await compressor.apply(ORIGINAL, model="gpt-4o")

    async def test_fail_open_covers_http_status_errors(self):
        error = httpx.HTTPStatusError(
            "502",
            request=httpx.Request("POST", "http://x/v1/compress"),
            response=httpx.Response(502),
        )
        compressor = HeadroomCompressor(client=_mock_client(error=error), fail_open=True)
        messages = await compressor.apply(ORIGINAL, model="gpt-4o")
        assert messages is ORIGINAL

    async def test_failure_does_not_record_hashes_or_stats(self):
        client = _mock_client(error=httpx.ConnectError("down"))
        compressor = HeadroomCompressor(client=client, fail_open=True)
        await compressor.apply(ORIGINAL, model="gpt-4o")
        assert compressor.issued_hashes == set()
        assert compressor.last_stats is None
