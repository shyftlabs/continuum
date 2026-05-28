"""
Unit tests for MCPUtil.

Covers:
- Fix 1: use_structured_content flag is respected in invoke_mcp_tool_with_artifact
- Fix 3: namespace_tools in get_all_function_tools deduplicates across servers
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent, Tool

from orchestrator.tools.types import ToolContextConfig
from orchestrator.tools.util import MCPUtil

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(use_structured_content: bool = False, name: str = "test-server"):
    """Minimal fake MCPServer for MCPUtil tests."""
    server = MagicMock()
    server.name = name
    server.use_structured_content = use_structured_content
    server.context_config = ToolContextConfig()
    return server


def _make_call_tool_result(structured_content: dict | None, content_text: str):
    """Fake CallToolResult with both fields populated."""
    result = MagicMock()
    result.structuredContent = structured_content
    result.content = [TextContent(type="text", text=content_text)]
    result.meta = None
    result.isError = False
    return result


def _fake_tool() -> Tool:
    return Tool(
        name="get_data",
        description="Get some data.",
        inputSchema={"type": "object", "properties": {}},
    )


def _disabled_provider_manager():
    mgr = MagicMock()
    mgr.is_enabled = False
    return mgr


# ---------------------------------------------------------------------------
# Fix 1: use_structured_content flag
# ---------------------------------------------------------------------------


class TestUseStructuredContentFlag:
    @pytest.mark.asyncio
    @patch(
        "orchestrator.observability.provider_manager.get_provider_manager",
        return_value=_disabled_provider_manager(),
    )
    async def test_flag_false_uses_content_not_structured(self, _mock_pm):
        """When use_structured_content=False, output must come from content, not structuredContent."""
        server = _make_server(use_structured_content=False)
        server.call_tool = AsyncMock(
            return_value=_make_call_tool_result(
                structured_content={"value": "from_structured"},
                content_text="from_content",
            )
        )

        text, artifact = await MCPUtil.invoke_mcp_tool_with_artifact(server, _fake_tool(), "{}")

        assert "from_content" in text
        assert "from_structured" not in text

    @pytest.mark.asyncio
    @patch(
        "orchestrator.observability.provider_manager.get_provider_manager",
        return_value=_disabled_provider_manager(),
    )
    async def test_flag_true_uses_structured_content(self, _mock_pm):
        """When use_structured_content=True, output must come from structuredContent."""
        server = _make_server(use_structured_content=True)
        server.call_tool = AsyncMock(
            return_value=_make_call_tool_result(
                structured_content={"value": "from_structured"},
                content_text="from_content",
            )
        )

        text, artifact = await MCPUtil.invoke_mcp_tool_with_artifact(server, _fake_tool(), "{}")

        assert "from_structured" in text
        assert "from_content" not in text

    @pytest.mark.asyncio
    @patch(
        "orchestrator.observability.provider_manager.get_provider_manager",
        return_value=_disabled_provider_manager(),
    )
    async def test_flag_true_no_structured_content_falls_back_to_content(self, _mock_pm):
        """When use_structured_content=True but structuredContent is absent, falls back to content."""
        server = _make_server(use_structured_content=True)
        server.call_tool = AsyncMock(
            return_value=_make_call_tool_result(
                structured_content=None,
                content_text="only_content",
            )
        )

        text, artifact = await MCPUtil.invoke_mcp_tool_with_artifact(server, _fake_tool(), "{}")

        assert "only_content" in text

    @pytest.mark.asyncio
    @patch(
        "orchestrator.observability.provider_manager.get_provider_manager",
        return_value=_disabled_provider_manager(),
    )
    async def test_artifact_always_captures_structured_content(self, _mock_pm):
        """Artifact stores structuredContent regardless of the flag."""
        server = _make_server(use_structured_content=False)
        server.call_tool = AsyncMock(
            return_value=_make_call_tool_result(
                structured_content={"key": "val"},
                content_text="text",
            )
        )

        _, artifact = await MCPUtil.invoke_mcp_tool_with_artifact(server, _fake_tool(), "{}")

        assert artifact.structured_content == {"key": "val"}


# ---------------------------------------------------------------------------
# Fix 3: namespace_tools
# ---------------------------------------------------------------------------


def _make_list_tools_server(name: str, tool_names: list[str]):
    """Fake server whose list_tools returns tools with the given names."""
    server = MagicMock()
    server.name = name
    server.context_config = ToolContextConfig()
    tools = [
        Tool(
            name=tn,
            description=f"Tool {tn}",
            inputSchema={"type": "object", "properties": {}},
        )
        for tn in tool_names
    ]
    server.list_tools = AsyncMock(return_value=tools)
    return server


class TestNamespaceTools:
    @pytest.mark.asyncio
    async def test_namespace_true_prefixes_tool_names(self):
        """With namespace_tools=True, each tool name is prefixed with server name."""
        server_a = _make_list_tools_server("server-a", ["search", "get"])
        server_b = _make_list_tools_server("server-b", ["list"])

        tools = await MCPUtil.get_all_function_tools([server_a, server_b], namespace_tools=True)

        names = {t.function.name for t in tools}
        assert "server-a__search" in names
        assert "server-a__get" in names
        assert "server-b__list" in names

    @pytest.mark.asyncio
    async def test_namespace_true_no_collision_error_on_duplicate_names(self):
        """With namespace_tools=True, duplicate tool names across servers do not raise."""
        server_a = _make_list_tools_server("server-a", ["search"])
        server_b = _make_list_tools_server("server-b", ["search"])

        tools = await MCPUtil.get_all_function_tools([server_a, server_b], namespace_tools=True)

        names = {t.function.name for t in tools}
        assert "server-a__search" in names
        assert "server-b__search" in names
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_namespace_false_raises_on_duplicate_names(self):
        """With namespace_tools=False (default), duplicate tool names raise MCPError."""
        from orchestrator.tools.exceptions import MCPError

        server_a = _make_list_tools_server("server-a", ["search"])
        server_b = _make_list_tools_server("server-b", ["search"])

        with pytest.raises(MCPError):
            await MCPUtil.get_all_function_tools([server_a, server_b])

    @pytest.mark.asyncio
    async def test_namespace_false_unique_names_no_error(self):
        """With namespace_tools=False, unique tool names across servers work fine."""
        server_a = _make_list_tools_server("server-a", ["search"])
        server_b = _make_list_tools_server("server-b", ["list"])

        tools = await MCPUtil.get_all_function_tools([server_a, server_b])

        names = {t.function.name for t in tools}
        assert "search" in names
        assert "list" in names

    @pytest.mark.asyncio
    async def test_executor_namespace_true_stores_prefixed_keys(self):
        """ToolExecutor with namespace_tools=True stores registry keys with server prefix."""
        from orchestrator.tools.executor import ToolExecutor
        from orchestrator.tools.mcp import MCPServerFunction

        server = MCPServerFunction(
            name="my-server",
            tools=[{"name": "echo", "fn": lambda args: "ok", "description": "Echo"}],
        )
        await server.connect()

        executor = ToolExecutor(
            tool_registry={server: None},
            namespace_tools=True,
        )
        await executor.initialize()

        assert "my-server__echo" in executor.tool_registry
        assert "echo" not in executor.tool_registry

    @pytest.mark.asyncio
    async def test_executor_namespace_false_stores_plain_keys(self):
        """ToolExecutor default (namespace_tools=False) stores plain tool names."""
        from orchestrator.tools.executor import ToolExecutor
        from orchestrator.tools.mcp import MCPServerFunction

        server = MCPServerFunction(
            name="my-server",
            tools=[{"name": "echo", "fn": lambda args: "ok", "description": "Echo"}],
        )
        await server.connect()

        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        assert "echo" in executor.tool_registry
        assert "my-server__echo" not in executor.tool_registry
