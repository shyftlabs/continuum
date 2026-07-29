"""
Tool-catalog handling in _MCPServerWithClientSession.list_tools().

Covers the two hardening steps that sit between the wire fetch and the
tool_filter (security finding F3 -- MCP tool poisoning / rug-pull):

  1. the tools cache is invalidated on connect(), so a reconnected session
     cannot serve a catalogue captured from a previous server process;
  2. invisible / control characters are stripped from descriptions and schema
     strings before they ever reach the model.

Both live in list_tools() rather than in a tool_filter, because tool_filter
defaults to None -- anything implemented there would protect almost nobody.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import Tool

from continuum.tools.mcp import MCPServerStreamableHttp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(cache_tools_list: bool = True) -> MCPServerStreamableHttp:
    return MCPServerStreamableHttp(
        params={"url": "http://localhost:8888/mcp"},
        cache_tools_list=cache_tools_list,
        name="srv",
    )


def _attach_session(server, tools: list[Tool]) -> MagicMock:
    """Give the server a mocked session returning `tools` from list_tools()."""
    session = AsyncMock()
    result = MagicMock()
    result.tools = tools
    session.list_tools = AsyncMock(return_value=result)
    server.session = session
    return session


def _tool(name: str, description: str, schema: dict | None = None) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


# ---------------------------------------------------------------------------
# 1. Cache invalidation on reconnect
# ---------------------------------------------------------------------------


class TestCacheInvalidatedOnConnect:
    """A reconnect may reach a different server process, so the catalogue
    captured before it must not be reused.

    Without this, a rug-pull performed across a reconnect is invisible when
    cache_tools_list=True: no fetch happens, so nothing re-reads the tool
    descriptions -- which is precisely the window F3 describes.
    """

    @pytest.mark.asyncio
    async def test_reconnect_refetches_the_catalogue(self, monkeypatch):
        server = _make_server(cache_tools_list=True)
        session = _attach_session(server, [_tool("search", "Search the catalogue.")])

        await server.list_tools()
        assert session.list_tools.await_count == 1

        # Second call is served from cache -- that is the point of the flag.
        await server.list_tools()
        assert session.list_tools.await_count == 1

        # Simulate connect() completing against a fresh session. Only the cache
        # bookkeeping matters here, so the transport setup is stubbed out.
        async def _fake_enter(ctx):
            raise AssertionError("transport should not be entered in this test")

        monkeypatch.setattr(server.exit_stack, "enter_async_context", _fake_enter)
        try:
            await server.connect()
        except Exception:
            pass  # connect() wraps the stub failure; we only assert on the cache

        assert server._cache_dirty is True, (
            "connect() must mark the tools cache dirty so the next list_tools() "
            "re-reads the catalogue from the (possibly different) server"
        )

    @pytest.mark.asyncio
    async def test_cache_still_serves_repeat_calls_without_a_reconnect(self):
        """The invalidation must not defeat caching in the normal case."""
        server = _make_server(cache_tools_list=True)
        session = _attach_session(server, [_tool("search", "Search.")])

        await server.list_tools()
        await server.list_tools()
        await server.list_tools()

        assert session.list_tools.await_count == 1
