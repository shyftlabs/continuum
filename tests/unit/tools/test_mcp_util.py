"""
Unit tests for MCPUtil.

Covers:
- Fix 1: use_structured_content flag is respected in invoke_mcp_tool_with_artifact
- Fix 3: namespace_tools in get_all_function_tools deduplicates across servers
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent, Tool

from continuum.tools.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp
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


class TestExecutorForwardsMetadata:
    """ToolExecutor must be able to pass metadata to list_tools().

    Without it, a dynamic tool_filter -- whose predicate reads
    ToolFilterContext.metadata -- receives None. The documented `admin_only`
    pattern then raises AttributeError per tool, which _apply_dynamic_tool_filter
    catches and treats as "exclude for safety", so the agent silently ends up
    with zero tools. That gap is the only reason MCPUtil.get_all_function_tools
    still has to exist for agent-building.
    """

    @pytest.mark.asyncio
    async def test_initialize_forwards_metadata_to_list_tools(self):
        from continuum.tools.executor import ToolExecutor

        server = _make_list_tools_server("srv", ["search"])

        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize(metadata={"role": "admin"})

        server.list_tools.assert_awaited_with(metadata={"role": "admin"})

    @pytest.mark.asyncio
    async def test_initialize_without_metadata_passes_none(self):
        from continuum.tools.executor import ToolExecutor

        server = _make_list_tools_server("srv", ["search"])

        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        server.list_tools.assert_awaited_with(metadata=None)

    @pytest.mark.asyncio
    async def test_refresh_registry_forwards_metadata(self):
        from continuum.tools.executor import ToolExecutor

        server = _make_list_tools_server("srv", ["search"])

        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()
        await executor.refresh_registry({server: None}, metadata={"role": "ops"})

        server.list_tools.assert_awaited_with(metadata={"role": "ops"})

    @pytest.mark.asyncio
    async def test_metadata_reaches_a_filtering_server_through_executor(self):
        """End-to-end: a server that filters on metadata must see it via ToolExecutor,
        not only via MCPUtil.get_all_function_tools. Mirrors what
        _apply_dynamic_tool_filter does with ToolFilterContext.metadata.
        """
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.mcp import MCPServerFunction

        class _AdminOnlyServer(MCPServerFunction):
            async def list_tools(self, metadata=None):
                tools = await super().list_tools(metadata)
                if not (metadata and metadata.get("role") == "admin"):
                    return []
                return tools

        def _server():
            return _AdminOnlyServer(
                name="srv",
                tools=[{"name": "echo", "fn": lambda args: "ok", "description": "Echo"}],
            )

        admin_exec = ToolExecutor(tool_registry={_server(): None})
        await admin_exec.initialize(metadata={"role": "admin"})
        assert "srv__echo" in admin_exec.tool_registry

        # Without forwarding, both cases would look like this one.
        guest_exec = ToolExecutor(tool_registry={_server(): None})
        await guest_exec.initialize(metadata={"role": "guest"})
        assert guest_exec.tool_registry == {}


class TestContextVariableNamesAreValidated:
    """capture_from / inject_into name tools by exact string.

    A name that matches nothing is a silent no-op: the variable is never
    captured, so the later injection has nothing to inject and the tool runs
    without it -- no error, no log. Same failure shape as the always_promote bug
    (a5e15a0) and the PolicyStore deny that stopped denying (cd993fa).
    """

    @staticmethod
    def _config(**kwargs):
        from continuum.tools.types import ToolContextConfig, ToolContextVariable

        return ToolContextConfig(
            variables=[ToolContextVariable(name="session_id", **kwargs)],
            auto_capture_common=False,
        )

    @staticmethod
    def _server(name: str, tool_names: list[str], config):
        from continuum.tools.mcp import MCPServerFunction

        return MCPServerFunction(
            name=name,
            tools=[
                {"name": tn, "fn": lambda args: "ok", "description": f"Tool {tn}"}
                for tn in tool_names
            ],
            context_config=config,
        )

    @pytest.mark.asyncio
    async def test_warns_when_capture_from_matches_no_tool(self):
        from continuum.tools.executor import ToolExecutor

        server = self._server("srv", ["do_work"], self._config(capture_from=["create_sesion"]))
        await server.connect()

        with _captured_tool_warnings() as warnings:
            await ToolExecutor(tool_registry={server: None}).initialize()

        assert any("create_sesion" in m for m in warnings), warnings

    @pytest.mark.asyncio
    async def test_warns_when_inject_into_matches_no_tool(self):
        from continuum.tools.executor import ToolExecutor

        server = self._server("srv", ["do_work"], self._config(inject_into=["do_wrok"]))
        await server.connect()

        with _captured_tool_warnings() as warnings:
            await ToolExecutor(tool_registry={server: None}).initialize()

        assert any("do_wrok" in m for m in warnings), warnings

    @pytest.mark.asyncio
    async def test_no_warning_when_every_name_matches(self):
        from continuum.tools.executor import ToolExecutor

        server = self._server(
            "srv",
            ["create_session", "do_work"],
            self._config(capture_from=["create_session"], inject_into=["do_work"]),
        )
        await server.connect()

        with _captured_tool_warnings() as warnings:
            await ToolExecutor(tool_registry={server: None}).initialize()

        assert warnings == []

    @pytest.mark.asyncio
    async def test_none_means_all_tools_and_must_not_warn(self):
        """capture_from=None is 'capture from any tool' (see
        ToolContextConfig.should_capture), not an empty list of names."""
        from continuum.tools.executor import ToolExecutor

        server = self._server("srv", ["do_work"], self._config())
        await server.connect()

        with _captured_tool_warnings() as warnings:
            await ToolExecutor(tool_registry={server: None}).initialize()

        assert warnings == []

    @pytest.mark.asyncio
    async def test_matches_the_raw_tool_name_not_the_namespaced_key(self):
        """ToolContextConfig is attached to one server, so its names are that
        server's own -- writing 'srv__create_session' inside a config already
        scoped to srv would be redundant. Mirrors the capture fix in 10e7b79."""
        from continuum.tools.executor import ToolExecutor

        server = self._server(
            "srv", ["create_session"], self._config(capture_from=["create_session"])
        )
        await server.connect()

        # namespace_tools=True (default) => registry key is "srv__create_session"
        with _captured_tool_warnings() as warnings:
            executor = ToolExecutor(tool_registry={server: None})
            await executor.initialize()

        assert "srv__create_session" in executor.tool_registry
        assert warnings == [], "the raw name matched, so there is nothing to warn about"


@contextmanager
def _captured_tool_warnings():
    """Collect WARNINGs from the executor's logger.

    caplog cannot see them: the "continuum" parent logger sets propagate=False
    and owns its handler, so records never reach root.
    """
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Collector()
    logger = logging.getLogger("continuum.tools.executor")
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Auto-derived server names under namespacing
#
# The transports fall back to f"streamable_http: {url}" / f"stdio: {command}"
# when no name= is given. That was written as a display label for error
# messages. With namespace_tools defaulting to True it became the prefix on
# every LLM-facing tool name -- so tool identity now carries the host and port,
# and moving the server to another port silently renames every tool, breaking
# policies, digest pins, always_promote and capture/inject by exact-string
# match. It also eats 39 of the 64-character provider budget.
# ---------------------------------------------------------------------------


@contextmanager
def _captured_util_warnings():
    """Collect WARNINGs from continuum.tools.util (caplog cannot: propagate=False)."""
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Collector()
    logger = logging.getLogger("continuum.tools.util")
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


class TestDerivedServerNameFlag:
    """Each transport must report whether its name was supplied or invented."""

    def test_streamable_http_without_name_is_derived(self):
        s = MCPServerStreamableHttp(params={"url": "http://localhost:8890/mcp"})
        assert s.name_is_derived is True

    def test_streamable_http_with_name_is_not_derived(self):
        s = MCPServerStreamableHttp(params={"url": "http://localhost:8890/mcp"}, name="shop")
        assert s.name_is_derived is False

    def test_sse_without_name_is_derived(self):
        s = MCPServerSse(params={"url": "http://localhost:8889/sse"})
        assert s.name_is_derived is True

    def test_stdio_without_name_is_derived(self):
        s = MCPServerStdio(params={"command": "python", "args": ["x.py"]})
        assert s.name_is_derived is True

    def test_stdio_with_name_is_not_derived(self):
        s = MCPServerStdio(params={"command": "python", "args": ["x.py"]}, name="files")
        assert s.name_is_derived is False


class TestDerivedNameWarning:
    def _server(self, derived: bool, name: str = "streamable_http: http://localhost:8890/mcp"):
        server = _make_server(name=name)
        server.name_is_derived = derived
        server.list_tools = AsyncMock(return_value=[_fake_tool()])
        return server

    @pytest.mark.asyncio
    async def test_warns_when_namespacing_an_auto_derived_name(self):
        server = self._server(derived=True)
        with _captured_util_warnings() as warnings:
            await MCPUtil.get_function_tools(server, namespace_tools=True)
        assert any("name=" in m for m in warnings), warnings

    @pytest.mark.asyncio
    async def test_no_warning_when_the_name_was_supplied(self):
        server = self._server(derived=False, name="shop")
        with _captured_util_warnings() as warnings:
            await MCPUtil.get_function_tools(server, namespace_tools=True)
        assert not warnings, warnings

    @pytest.mark.asyncio
    async def test_no_warning_when_namespacing_is_off(self):
        """Without namespacing the derived name stays a display label, which is
        what it was designed to be -- nothing to warn about."""
        server = self._server(derived=True)
        with _captured_util_warnings() as warnings:
            await MCPUtil.get_function_tools(server, namespace_tools=False)
        assert not warnings, warnings

    @pytest.mark.asyncio
    async def test_warns_only_once_per_server(self):
        """Registry rebuilds re-enter this path; the advice does not change."""
        server = self._server(derived=True)
        with _captured_util_warnings() as warnings:
            await MCPUtil.get_function_tools(server, namespace_tools=True)
            await MCPUtil.get_function_tools(server, namespace_tools=True)
        assert len(warnings) == 1, warnings

    @pytest.mark.asyncio
    async def test_warning_names_the_prefix_the_tools_actually_got(self):
        server = self._server(derived=True)
        with _captured_util_warnings() as warnings:
            tools = await MCPUtil.get_function_tools(server, namespace_tools=True)
        prefix = tools[0].function.name.rsplit("__", 1)[0]
        assert any(prefix in m for m in warnings), (prefix, warnings)

    @pytest.mark.asyncio
    async def test_warns_via_get_all_function_tools(self):
        """get_all_function_tools applies the prefix itself and passes
        namespace_tools=False downward, so it needs its own check -- this is the
        aggregating entry point most multi-server code uses."""
        server = self._server(derived=True)
        with _captured_util_warnings() as warnings:
            await MCPUtil.get_all_function_tools([server], namespace_tools=True)
        assert any("name=" in m for m in warnings), warnings

    @pytest.mark.asyncio
    async def test_no_warning_via_get_all_function_tools_when_namespacing_off(self):
        server = self._server(derived=True)
        with _captured_util_warnings() as warnings:
            await MCPUtil.get_all_function_tools([server], namespace_tools=False)
        assert not warnings, warnings

    @pytest.mark.asyncio
    async def test_warns_when_the_executor_builds_a_namespaced_registry(self):
        """ToolExecutor namespaces registry keys in _build_registry without going
        through MCPUtil, so it is a third independent site."""
        from continuum.tools.executor import ToolExecutor

        server = self._server(derived=True)
        server.connect = AsyncMock()
        executor = ToolExecutor(tool_registry={server: None}, namespace_tools=True)
        with _captured_util_warnings() as warnings:
            await executor.initialize()
        assert any("name=" in m for m in warnings), warnings
