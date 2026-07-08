"""Unit tests for HeadroomClient — the thin HTTP client for the Headroom sidecar.

Contract verified 2026-07-08 against a live `headroom proxy` (v0.29.0):
  POST /v1/compress  {messages, model} ->
      {messages, tokens_before, tokens_after, tokens_saved,
       compression_ratio, transforms_applied, ccr_hashes}
  GET  /v1/retrieve/{hash}?query=... -> {original_content, ...}

All tests mock the transport — no real sidecar needed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from continuum.llm.headroom.client import CompressionStats, HeadroomClient

MESSAGES = [
    {"role": "user", "content": "Which users are active?"},
    {"role": "tool", "tool_call_id": "c1", "content": '[{"id": 1}]'},
]

COMPRESS_RESPONSE = {
    "messages": [
        {"role": "user", "content": "Which users are active?"},
        {"role": "tool", "tool_call_id": "c1", "content": "[compressed]"},
    ],
    "tokens_before": 5000,
    "tokens_after": 1000,
    "tokens_saved": 4000,
    "compression_ratio": 0.2,
    "transforms_applied": ["router:smart_crusher:0.20"],
    "ccr_hashes": ["abc123def456abc123def456"],
}


def _client_with(handler) -> HeadroomClient:
    """Build a HeadroomClient whose httpx client uses a mock transport."""
    client = HeadroomClient(api_base="http://127.0.0.1:8787", api_key=None)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


class TestCompress:
    async def test_returns_compressed_messages_stats_and_hashes(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/compress"
            body = json.loads(request.content)
            assert body["messages"] == MESSAGES
            assert body["model"] == "gpt-4o"
            return httpx.Response(200, json=COMPRESS_RESPONSE)

        client = _client_with(handler)
        messages, stats, ccr_hashes = await client.compress(MESSAGES, model="gpt-4o")

        assert messages == COMPRESS_RESPONSE["messages"]
        assert isinstance(stats, CompressionStats)
        assert stats.tokens_before == 5000
        assert stats.tokens_after == 1000
        assert stats.tokens_saved == 4000
        assert stats.compression_ratio == 0.2
        assert stats.transforms_applied == ["router:smart_crusher:0.20"]
        assert ccr_hashes == ["abc123def456abc123def456"]

    async def test_ccr_hashes_defaults_to_empty_list(self):
        """The common (lossless-JSON) case: no hashes issued."""
        resp = {**COMPRESS_RESPONSE, "ccr_hashes": []}

        client = _client_with(lambda req: httpx.Response(200, json=resp))
        _, _, ccr_hashes = await client.compress(MESSAGES, model="gpt-4o")
        assert ccr_hashes == []

    async def test_missing_ccr_hashes_field_tolerated(self):
        """Older sidecar without the field → empty list, not KeyError."""
        resp = {k: v for k, v in COMPRESS_RESPONSE.items() if k != "ccr_hashes"}

        client = _client_with(lambda req: httpx.Response(200, json=resp))
        _, _, ccr_hashes = await client.compress(MESSAGES, model="gpt-4o")
        assert ccr_hashes == []

    async def test_model_omitted_from_payload_when_none(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=COMPRESS_RESPONSE)

        client = _client_with(handler)
        await client.compress(MESSAGES, model=None)
        assert "model" not in captured

    async def test_http_error_raises_for_caller_policy(self):
        """Transport/HTTP errors must propagate — fail-open/closed is the
        compressor's decision, not the client's."""
        client = _client_with(lambda req: httpx.Response(502))
        with pytest.raises(httpx.HTTPStatusError):
            await client.compress(MESSAGES, model="gpt-4o")

    async def test_connect_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("sidecar down")

        client = _client_with(handler)
        with pytest.raises(httpx.ConnectError):
            await client.compress(MESSAGES, model="gpt-4o")

    async def test_api_key_sent_as_bearer(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=COMPRESS_RESPONSE)

        client = HeadroomClient(api_base="http://127.0.0.1:8787", api_key="sk-test")
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await client.compress(MESSAGES, model="gpt-4o")
        assert seen["auth"] == "Bearer sk-test"


class TestRetrieve:
    async def test_returns_original_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/retrieve/abc123def456abc123def456"
            assert request.url.params.get("query") == "auth rows"
            return httpx.Response(200, json={"original_content": "<full 5000 rows>"})

        client = _client_with(handler)
        content = await client.retrieve("abc123def456abc123def456", query="auth rows")
        assert content == "<full 5000 rows>"

    async def test_falls_back_to_content_field(self):
        client = _client_with(
            lambda req: httpx.Response(200, json={"content": "fallback body"})
        )
        assert await client.retrieve("abc123def456abc123def456") == "fallback body"

    async def test_query_omitted_when_none(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"original_content": "x"})

        client = _client_with(handler)
        await client.retrieve("abc123def456abc123def456")
        assert seen["params"] == {}

    async def test_http_error_raises(self):
        client = _client_with(lambda req: httpx.Response(404))
        with pytest.raises(httpx.HTTPStatusError):
            await client.retrieve("ffffffffffffffffffffffff")


class TestHealthAndLifecycle:
    async def test_health_check_true_when_healthy(self):
        client = _client_with(lambda req: httpx.Response(200, json={"status": "healthy"}))
        assert await client.health_check() is True

    async def test_health_check_false_when_unreachable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = _client_with(handler)
        assert await client.health_check() is False

    async def test_aclose_closes_transport(self):
        client = _client_with(lambda req: httpx.Response(200, json={}))
        await client.aclose()
        with pytest.raises(RuntimeError):
            await client._client.get("http://127.0.0.1:8787/health")

    def test_api_base_trailing_slash_stripped(self):
        client = HeadroomClient(api_base="http://127.0.0.1:8787/", api_key=None)
        assert client._base == "http://127.0.0.1:8787"
