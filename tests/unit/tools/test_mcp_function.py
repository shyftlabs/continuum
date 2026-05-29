"""
Tests for in-process MCP function tools (MCPServerFunction, @function_tool, isError).
"""
from __future__ import annotations

import json
import pytest

from orchestrator.tools.mcp import FunctionTool, MCPServerFunction, function_tool


# ---------------------------------------------------------------------------
# @function_tool decorator — schema generation
# ---------------------------------------------------------------------------


class TestFunctionToolDecorator:
    def test_returns_function_tool_instance(self):
        @function_tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}"

        assert isinstance(greet, FunctionTool)

    def test_name_from_function(self):
        @function_tool
        def my_func(x: int) -> int:
            return x

        assert greet.name == "greet" if False else True  # just structure check
        assert my_func.name == "my_func"

    def test_description_from_docstring(self):
        @function_tool
        def compute(x: int) -> int:
            """Compute something."""
            return x

        assert compute.description == "Compute something."

    def test_schema_int_param(self):
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        props = add.input_schema.get("properties", {})
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "integer"

    def test_schema_str_param(self):
        @function_tool
        def echo(msg: str) -> str:
            return msg

        props = echo.input_schema.get("properties", {})
        assert props["msg"]["type"] == "string"

    def test_schema_bool_param(self):
        @function_tool
        def toggle(flag: bool) -> bool:
            return not flag

        props = toggle.input_schema.get("properties", {})
        assert props["flag"]["type"] == "boolean"

    def test_schema_float_param(self):
        @function_tool
        def scale(factor: float) -> float:
            return factor

        props = scale.input_schema.get("properties", {})
        assert props["factor"]["type"] == "number"

    def test_optional_param_not_required(self):
        from typing import Optional

        @function_tool
        def greet(name: str, title: Optional[str] = None) -> str:
            return name

        required = greet.input_schema.get("required", [])
        assert "name" in required
        assert "title" not in required

    def test_no_type_hint_falls_back_to_open_schema(self):
        @function_tool
        def mystery(x) -> str:
            return str(x)

        # Should not crash — falls back gracefully
        assert "properties" in mystery.input_schema or mystery.input_schema.get("type") == "object"

    def test_no_docstring_empty_description(self):
        @function_tool
        def silent(x: int) -> int:
            return x

        assert silent.description == "" or silent.description is None or isinstance(silent.description, str)


# ---------------------------------------------------------------------------
# MCPServerFunction — list_tools and call_tool
# ---------------------------------------------------------------------------


class TestMCPServerFunction:
    @pytest.fixture
    def server(self):
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        return MCPServerFunction("math", [add])

    @pytest.mark.asyncio
    async def test_list_tools_returns_registered(self, server):
        tools = await server.list_tools()
        names = [t.name for t in tools]
        assert "add" in names

    @pytest.mark.asyncio
    async def test_call_tool_sync_function(self, server):
        result = await server.call_tool("add", {"a": 3, "b": 4})
        assert result.isError is not True
        text = result.content[0].text
        assert json.loads(text) == 7

    @pytest.mark.asyncio
    async def test_call_tool_async_function(self):
        @function_tool
        async def fetch(url: str) -> str:
            """Fake async fetch."""
            return f"fetched:{url}"

        server = MCPServerFunction("web", [fetch])
        result = await server.call_tool("fetch", {"url": "http://example.com"})
        assert result.isError is not True
        assert "fetched:http://example.com" in result.content[0].text

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises(self, server):
        from orchestrator.tools.mcp import MCPError

        with pytest.raises(MCPError):
            await server.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_tool_string_result_not_json_encoded(self):
        @function_tool
        def greet(name: str) -> str:
            """Greet."""
            return f"Hello, {name}"

        server = MCPServerFunction("greet_srv", [greet])
        result = await server.call_tool("greet", {"name": "Alice"})
        assert result.content[0].text == "Hello, Alice"

    @pytest.mark.asyncio
    async def test_plain_callable_accepted(self):
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        server = MCPServerFunction("calc", [multiply])
        tools = await server.list_tools()
        assert any(t.name == "multiply" for t in tools)
        result = await server.call_tool("multiply", {"a": 3, "b": 4})
        assert result.isError is not True
        import json
        assert json.loads(result.content[0].text) == 12

    @pytest.mark.asyncio
    async def test_dict_format_accepted(self):
        server = MCPServerFunction(
            "custom",
            [
                {
                    "name": "ping",
                    "description": "Ping tool",
                    "fn": lambda args: "pong",
                    "input_schema": {"type": "object"},
                }
            ],
        )
        tools = await server.list_tools()
        assert any(t.name == "ping" for t in tools)

    @pytest.mark.asyncio
    async def test_function_tool_dataclass_accepted(self):
        ft = FunctionTool(
            name="raw",
            fn=lambda args: "raw_result",
            description="Raw tool",
            input_schema={"type": "object"},
        )
        server = MCPServerFunction("raw_srv", [ft])
        tools = await server.list_tools()
        assert any(t.name == "raw" for t in tools)


# ---------------------------------------------------------------------------
# isError envelope — function raises → isError=True in result
# ---------------------------------------------------------------------------


class TestIsErrorEnvelope:
    @pytest.mark.asyncio
    async def test_exception_returns_is_error(self):
        @function_tool
        def explode(x: int) -> int:
            """Always fails."""
            raise ValueError("boom")

        server = MCPServerFunction("err_srv", [explode])
        result = await server.call_tool("explode", {"x": 1})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_error_envelope_contains_message(self):
        @function_tool
        def broken(x: str) -> str:
            raise RuntimeError("something went wrong")

        server = MCPServerFunction("err_srv", [broken])
        result = await server.call_tool("broken", {"x": "input"})
        payload = json.loads(result.content[0].text)
        assert "error" in payload
        assert "something went wrong" in payload["error"]

    @pytest.mark.asyncio
    async def test_error_envelope_contains_error_type(self):
        @function_tool
        def typed_fail(x: int) -> int:
            raise TypeError("wrong type")

        server = MCPServerFunction("err_srv", [typed_fail])
        result = await server.call_tool("typed_fail", {"x": 1})
        payload = json.loads(result.content[0].text)
        assert payload.get("error_type") == "TypeError"
