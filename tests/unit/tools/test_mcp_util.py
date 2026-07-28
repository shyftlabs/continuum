"""
Unit tests for MCPUtil.

Covers:
- Fix 1: use_structured_content flag is respected in invoke_mcp_tool_with_artifact
- Fix 3: namespace_tools in get_all_function_tools deduplicates across servers
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent, Tool

from continuum.tools.types import ToolContextConfig
from continuum.tools.util import MCPUtil, build_namespaced_tool_name

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
        "continuum.observability.provider_manager.get_provider_manager",
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
        "continuum.observability.provider_manager.get_provider_manager",
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
        "continuum.observability.provider_manager.get_provider_manager",
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
        "continuum.observability.provider_manager.get_provider_manager",
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
    async def test_namespace_defaults_to_true(self):
        """Namespacing is the default, so duplicate names across servers coexist."""
        server_a = _make_list_tools_server("server-a", ["search"])
        server_b = _make_list_tools_server("server-b", ["search"])

        tools = await MCPUtil.get_all_function_tools([server_a, server_b])

        names = {t.function.name for t in tools}
        assert names == {"server-a__search", "server-b__search"}

    @pytest.mark.asyncio
    async def test_namespace_false_raises_on_duplicate_names(self):
        """With namespace_tools=False, duplicate tool names raise MCPError."""
        from continuum.tools.exceptions import MCPError

        server_a = _make_list_tools_server("server-a", ["search"])
        server_b = _make_list_tools_server("server-b", ["search"])

        with pytest.raises(MCPError):
            await MCPUtil.get_all_function_tools([server_a, server_b], namespace_tools=False)

    @pytest.mark.asyncio
    async def test_namespace_false_unique_names_no_error(self):
        """With namespace_tools=False, unique tool names across servers work fine."""
        server_a = _make_list_tools_server("server-a", ["search"])
        server_b = _make_list_tools_server("server-b", ["list"])

        tools = await MCPUtil.get_all_function_tools([server_a, server_b], namespace_tools=False)

        names = {t.function.name for t in tools}
        assert "search" in names
        assert "list" in names

    @pytest.mark.asyncio
    async def test_executor_namespace_true_stores_prefixed_keys(self):
        """ToolExecutor with namespace_tools=True stores registry keys with server prefix."""
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

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
    async def test_executor_defaults_to_namespaced_keys(self):
        """ToolExecutor namespaces registry keys by default."""
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

        server = MCPServerFunction(
            name="my-server",
            tools=[{"name": "echo", "fn": lambda args: "ok", "description": "Echo"}],
        )
        await server.connect()

        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        assert "my-server__echo" in executor.tool_registry
        assert "echo" not in executor.tool_registry

    @pytest.mark.asyncio
    async def test_executor_namespace_false_stores_plain_keys(self):
        """With namespace_tools=False, ToolExecutor stores plain tool names."""
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

        server = MCPServerFunction(
            name="my-server",
            tools=[{"name": "echo", "fn": lambda args: "ok", "description": "Echo"}],
        )
        await server.connect()

        executor = ToolExecutor(tool_registry={server: None}, namespace_tools=False)
        await executor.initialize()

        assert "echo" in executor.tool_registry
        assert "my-server__echo" not in executor.tool_registry


# ---------------------------------------------------------------------------
# Namespaced name construction (provider charset + length budget)
# ---------------------------------------------------------------------------

# OpenAI and Anthropic both enforce this on function names.
PROVIDER_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class TestBuildNamespacedToolName:
    @pytest.mark.parametrize(
        "server_name",
        [
            "stdio: npx",
            "sse: https://api.example.com/mcp",
            "streamable_http: https://mcp.internal.corp:8443/v1/mcp",
            "sse: https://mcp-gateway.prod.us-east-1.internal.example-corp.com/v2/mcp",
            "weather",
        ],
    )
    def test_auto_derived_server_names_produce_valid_keys(self, server_name):
        """Auto-derived names carry ': ', '/', '.' and ':' -- all provider-invalid."""
        key = build_namespaced_tool_name(server_name, "read_file")
        assert PROVIDER_NAME_RE.match(key), key

    def test_long_server_name_is_truncated_within_budget(self):
        server = "sse: https://mcp-gateway.prod.us-east-1.internal.example-corp.com/v2/mcp"
        key = build_namespaced_tool_name(server, "list_directory_contents")
        assert len(key) <= 64
        assert PROVIDER_NAME_RE.match(key)

    def test_tool_name_survives_truncation_intact(self):
        """The tool name carries the semantics -- only the prefix may be shortened."""
        server = "sse: https://mcp-gateway.prod.us-east-1.internal.example-corp.com/v2/mcp"
        key = build_namespaced_tool_name(server, "list_directory_contents")
        assert key.endswith("__list_directory_contents")

    def test_truncated_prefixes_stay_distinct(self):
        """Two URLs differing only late would collide under naive truncation."""
        a = "sse: https://mcp-gateway.prod.us-east-1.internal.example-corp.com/v2/mcp"
        b = "sse: https://mcp-gateway.prod.us-east-1.internal.example-corp.com/v3/mcp"
        tool = "list_directory_contents"
        assert build_namespaced_tool_name(a, tool) != build_namespaced_tool_name(b, tool)

    def test_is_deterministic(self):
        """Stable across calls so provider prompt caches are not invalidated."""
        server = "sse: https://api.example.com/mcp"
        assert build_namespaced_tool_name(server, "read_file") == build_namespaced_tool_name(
            server, "read_file"
        )

    def test_clean_server_name_is_untouched(self):
        assert build_namespaced_tool_name("weather", "read_file") == "weather__read_file"


class TestNamespaceAppliedExactlyOnce:
    """get_function_tools namespaces by default, and get_all_function_tools must
    not prefix a second time on top of it."""

    @pytest.mark.asyncio
    async def test_get_function_tools_namespaces_by_default(self):
        """Must match ToolExecutor's default: a bare tool list paired with a
        namespacing registry leaves the model calling unresolvable names."""
        server = _make_list_tools_server("weather", ["get_forecast"])

        tools = await MCPUtil.get_function_tools(server)

        assert {t.function.name for t in tools} == {"weather__get_forecast"}

    @pytest.mark.asyncio
    async def test_get_function_tools_can_opt_out(self):
        server = _make_list_tools_server("weather", ["get_forecast"])

        tools = await MCPUtil.get_function_tools(server, namespace_tools=False)

        assert {t.function.name for t in tools} == {"get_forecast"}

    @pytest.mark.asyncio
    async def test_get_all_function_tools_does_not_double_prefix(self):
        """get_all_function_tools delegates to get_function_tools, which now
        namespaces on its own -- so the prefix must be suppressed there and
        applied once at this level."""
        server = _make_list_tools_server("weather", ["get_forecast"])

        tools = await MCPUtil.get_all_function_tools([server])

        names = {t.function.name for t in tools}
        assert names == {"weather__get_forecast"}
        assert "weather__weather__get_forecast" not in names


class TestDuplicateRegistryKeyRaises:
    """_build_registry must fail closed: a shadowed tool would inherit the
    original's PolicyStore grant, since policy resources are keyed by name."""

    @pytest.mark.asyncio
    async def test_duplicate_bare_names_raise(self):
        from continuum.tools.exceptions import MCPError
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

        def _srv(name):
            return MCPServerFunction(
                name=name,
                tools=[{"name": "read_file", "fn": lambda args: "ok", "description": "Read"}],
            )

        server_a, server_b = _srv("server-a"), _srv("server-b")
        await server_a.connect()
        await server_b.connect()

        executor = ToolExecutor(
            tool_registry={server_a: None, server_b: None}, namespace_tools=False
        )
        with pytest.raises(MCPError, match="Duplicate tool name"):
            await executor.initialize()

    @pytest.mark.asyncio
    async def test_same_server_name_collides_even_when_namespaced(self):
        """Namespacing does not save you when two servers share a name -- which is
        easy, since names are auto-derived from the URL."""
        from continuum.tools.exceptions import MCPError
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

        def _srv():
            return MCPServerFunction(
                name="dup-server",
                tools=[{"name": "read_file", "fn": lambda args: "ok", "description": "Read"}],
            )

        server_a, server_b = _srv(), _srv()
        await server_a.connect()
        await server_b.connect()

        executor = ToolExecutor(tool_registry={server_a: None, server_b: None})
        with pytest.raises(MCPError, match="Duplicate tool name"):
            await executor.initialize()

    @pytest.mark.asyncio
    async def test_distinct_servers_coexist_when_namespaced(self):
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

        def _srv(name):
            return MCPServerFunction(
                name=name,
                tools=[{"name": "read_file", "fn": lambda args: "ok", "description": "Read"}],
            )

        server_a, server_b = _srv("server-a"), _srv("server-b")
        await server_a.connect()
        await server_b.connect()

        executor = ToolExecutor(tool_registry={server_a: None, server_b: None})
        await executor.initialize()

        assert set(executor.tool_registry) == {"server-a__read_file", "server-b__read_file"}
