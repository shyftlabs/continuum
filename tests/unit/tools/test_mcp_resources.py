"""
Unit tests for MCP resource support.

Covers:
- MCPServerFunction.list_resources() / read_resource() stubs
- _MCPServerWithClientSession.list_resources() / read_resource() via mocked session
- Error cases: not connected, empty contents, BlobResourceContents ignored
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import AnyUrl

from orchestrator.tools.mcp import MCPServerFunction, MCPServerStreamableHttp, function_tool
from orchestrator.tools.exceptions import MCPError


# ---------------------------------------------------------------------------
# MCPServerFunction stubs
# ---------------------------------------------------------------------------


class TestMCPServerFunctionResourceStubs:
    @pytest.mark.asyncio
    async def test_list_resources_returns_empty_list(self):
        server = MCPServerFunction("math", [])
        resources = await server.list_resources()
        assert resources == []

    @pytest.mark.asyncio
    async def test_read_resource_returns_empty_string(self):
        server = MCPServerFunction("math", [])
        result = await server.read_resource("shop://catalogue")
        assert result == ""

    @pytest.mark.asyncio
    async def test_list_resources_does_not_raise(self):
        @function_tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        server = MCPServerFunction("calc", [add])
        resources = await server.list_resources()
        assert isinstance(resources, list)

    @pytest.mark.asyncio
    async def test_read_resource_any_uri_returns_empty(self):
        server = MCPServerFunction("x", [])
        assert await server.read_resource("file:///etc/hosts") == ""
        assert await server.read_resource("shop://products/p1") == ""


# ---------------------------------------------------------------------------
# _MCPServerWithClientSession — list_resources via mocked session
# ---------------------------------------------------------------------------


def _make_streamable_http_server() -> MCPServerStreamableHttp:
    return MCPServerStreamableHttp(
        params={"url": "http://localhost:8888/mcp"},
        cache_tools_list=False,
    )


class TestMCPServerListResources:
    @pytest.mark.asyncio
    async def test_list_resources_returns_resources_from_session(self):
        from mcp.types import Resource

        server = _make_streamable_http_server()
        mock_resource = Resource(
            uri=AnyUrl("shop://catalogue"),
            name="catalogue",
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.resources = [mock_resource]
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        server.session = mock_session

        resources = await server.list_resources()
        assert len(resources) == 1
        assert str(resources[0].uri) == "shop://catalogue"

    @pytest.mark.asyncio
    async def test_list_resources_returns_multiple(self):
        from mcp.types import Resource

        server = _make_streamable_http_server()
        mock_resources = [
            Resource(uri=AnyUrl("shop://catalogue"), name="catalogue"),
            Resource(uri=AnyUrl("shop://categories"), name="categories"),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.resources = mock_resources
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        server.session = mock_session

        resources = await server.list_resources()
        assert len(resources) == 2

    @pytest.mark.asyncio
    async def test_list_resources_raises_when_not_connected(self):
        server = _make_streamable_http_server()
        server.session = None

        with pytest.raises(MCPError):
            await server.list_resources()

    @pytest.mark.asyncio
    async def test_list_resources_returns_empty_when_server_has_none(self):
        server = _make_streamable_http_server()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.resources = []
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        server.session = mock_session

        resources = await server.list_resources()
        assert resources == []


# ---------------------------------------------------------------------------
# _MCPServerWithClientSession — read_resource via mocked session
# ---------------------------------------------------------------------------


class TestMCPServerReadResource:
    @pytest.mark.asyncio
    async def test_read_resource_returns_text_content(self):
        from mcp.types import TextResourceContents

        server = _make_streamable_http_server()
        payload = json.dumps([{"id": "p1", "name": "Dog Food"}])

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.contents = [
            TextResourceContents(uri=AnyUrl("shop://catalogue"), text=payload)
        ]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        server.session = mock_session

        text = await server.read_resource("shop://catalogue")
        assert text == payload

    @pytest.mark.asyncio
    async def test_read_resource_returns_first_text_item(self):
        from mcp.types import TextResourceContents

        server = _make_streamable_http_server()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.contents = [
            TextResourceContents(uri=AnyUrl("shop://catalogue"), text="first"),
            TextResourceContents(uri=AnyUrl("shop://catalogue"), text="second"),
        ]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        server.session = mock_session

        text = await server.read_resource("shop://catalogue")
        assert text == "first"

    @pytest.mark.asyncio
    async def test_read_resource_returns_empty_when_no_contents(self):
        server = _make_streamable_http_server()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.contents = []
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        server.session = mock_session

        text = await server.read_resource("shop://catalogue")
        assert text == ""

    @pytest.mark.asyncio
    async def test_read_resource_skips_blob_returns_empty(self):
        from mcp.types import BlobResourceContents
        import base64

        server = _make_streamable_http_server()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.contents = [
            BlobResourceContents(
                uri=AnyUrl("shop://image"),
                blob=base64.b64encode(b"binary data").decode(),
            )
        ]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        server.session = mock_session

        # BlobResourceContents is not TextResourceContents — should return ""
        text = await server.read_resource("shop://image")
        assert text == ""

    @pytest.mark.asyncio
    async def test_read_resource_raises_when_not_connected(self):
        server = _make_streamable_http_server()
        server.session = None

        with pytest.raises(MCPError):
            await server.read_resource("shop://catalogue")

    @pytest.mark.asyncio
    async def test_read_resource_passes_uri_to_session(self):
        from mcp.types import TextResourceContents

        server = _make_streamable_http_server()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.contents = [
            TextResourceContents(uri=AnyUrl("shop://categories"), text="{}")
        ]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        server.session = mock_session

        await server.read_resource("shop://categories")

        called_uri = mock_session.read_resource.call_args[0][0]
        assert "categories" in str(called_uri)
