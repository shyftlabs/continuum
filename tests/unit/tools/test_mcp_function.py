"""
Tests for in-process MCP function tools (MCPServerFunction, @function_tool, isError).
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Optional

import pytest

from continuum.tools.mcp import FunctionTool, MCPServerFunction, function_tool

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
        @function_tool
        def greet(name: str, title: str | None = None) -> str:
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

        assert (
            silent.description == ""
            or silent.description is None
            or isinstance(silent.description, str)
        )


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
        from continuum.tools.mcp import MCPError

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


# ---------------------------------------------------------------------------
# Argument validation at the in-process boundary (security finding F5)
#
# An in-process tool's ``input_schema`` was advertised to the model and then
# discarded: ``call_tool`` passed ``arguments`` straight into the callable, so a
# parameter declared ``{"type": "string"}`` would happily receive a dict. The
# schema was a hint to the model, never a gate on the way back in -- and the
# model's arguments are attacker-influenced the moment any upstream tool result
# is. This is the one server class where the framework can validate: for stdio
# and HTTP servers the remote owns its own input checking.
#
# Rejection is enveloped (isError=True), not raised: a wrong argument is
# something the model can correct on the next turn.
# ---------------------------------------------------------------------------


class TestArgumentValidationRejectsWrongTypes:
    @pytest.fixture
    def sql_server(self):
        """A tool whose single parameter is declared a string."""
        calls: list[object] = []

        @function_tool
        def run_sql(query: str) -> str:
            """Run a query."""
            calls.append(query)
            return "ok"

        server = MCPServerFunction("db", [run_sql])
        server._test_calls = calls  # type: ignore[attr-defined]
        return server

    @pytest.mark.asyncio
    async def test_dict_for_declared_string_is_rejected(self, sql_server):
        result = await sql_server.call_tool("run_sql", {"query": {"not": "a string"}})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_rejected_call_never_reaches_the_function_body(self, sql_server):
        await sql_server.call_tool("run_sql", {"query": {"not": "a string"}})
        assert sql_server._test_calls == [], "the tool body must not observe invalid arguments"

    @pytest.mark.asyncio
    async def test_valid_string_still_reaches_the_function_body(self, sql_server):
        result = await sql_server.call_tool("run_sql", {"query": "SELECT 1"})
        assert result.isError is not True
        assert sql_server._test_calls == ["SELECT 1"]

    @pytest.mark.asyncio
    async def test_error_envelope_names_the_offending_parameter(self, sql_server):
        result = await sql_server.call_tool("run_sql", {"query": 123})
        payload = json.loads(result.content[0].text)
        assert "query" in payload["error"]

    @pytest.mark.asyncio
    async def test_error_type_identifies_an_argument_error(self, sql_server):
        result = await sql_server.call_tool("run_sql", {"query": 123})
        payload = json.loads(result.content[0].text)
        assert payload["error_type"] == "ToolArgumentError"

    @pytest.mark.asyncio
    async def test_string_for_declared_integer_is_rejected(self):
        """No silent coercion: "3" is not 3, and guessing hides a real mismatch."""

        @function_tool
        def repeat(times: int) -> int:
            return times

        server = MCPServerFunction("s", [repeat])
        result = await server.call_tool("repeat", {"times": "3"})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_bool_for_declared_integer_is_rejected(self):
        """``bool`` is a Python ``int`` subclass; JSON Schema treats them apart."""

        @function_tool
        def repeat(times: int) -> int:
            return times

        server = MCPServerFunction("s", [repeat])
        result = await server.call_tool("repeat", {"times": True})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_integer_for_declared_number_is_accepted(self):
        """JSON has no int/float split -- 3 is a valid ``number``."""

        @function_tool
        def scale(factor: float) -> float:
            return factor

        server = MCPServerFunction("s", [scale])
        result = await server.call_tool("scale", {"factor": 3})
        assert result.isError is not True

    @pytest.mark.asyncio
    async def test_integer_for_declared_boolean_is_rejected(self):
        @function_tool
        def toggle(flag: bool) -> bool:
            return flag

        server = MCPServerFunction("s", [toggle])
        result = await server.call_tool("toggle", {"flag": 1})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_object_for_declared_array_is_rejected(self):
        @function_tool
        def pick(items: list) -> int:
            return len(items)

        server = MCPServerFunction("s", [pick])
        result = await server.call_tool("pick", {"items": {"a": 1}})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_array_for_declared_object_is_rejected(self):
        @function_tool
        def store(payload: dict) -> int:
            return len(payload)

        server = MCPServerFunction("s", [store])
        result = await server.call_tool("store", {"payload": [1, 2]})
        assert result.isError is True


class TestArgumentValidationRequiredAndUnknownKeys:
    @pytest.mark.asyncio
    async def test_missing_required_parameter_is_rejected(self):
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        server = MCPServerFunction("math", [add])
        result = await server.call_tool("add", {"a": 1})
        assert result.isError is True
        assert "b" in json.loads(result.content[0].text)["error"]

    @pytest.mark.asyncio
    async def test_unknown_parameter_is_rejected(self):
        """An invented argument is a signal, not noise -- surface it by name."""

        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        server = MCPServerFunction("math", [add])
        result = await server.call_tool("add", {"a": 1, "b": 2, "sudo": True})
        assert result.isError is True
        assert "sudo" in json.loads(result.content[0].text)["error"]

    @pytest.mark.asyncio
    async def test_omitted_optional_parameter_is_accepted(self):
        @function_tool
        def greet(name: str, title: str | None = None) -> str:
            return f"{title or ''}{name}"

        server = MCPServerFunction("s", [greet])
        result = await server.call_tool("greet", {"name": "Alice"})
        assert result.isError is not True

    @pytest.mark.asyncio
    async def test_none_for_optional_parameter_is_accepted(self):
        @function_tool
        def greet(name: str, title: str | None = None) -> str:
            return f"{title or ''}{name}"

        server = MCPServerFunction("s", [greet])
        result = await server.call_tool("greet", {"name": "Alice", "title": None})
        assert result.isError is not True

    @pytest.mark.asyncio
    async def test_wrong_type_for_optional_parameter_is_rejected(self):
        @function_tool
        def greet(name: str, title: str | None = None) -> str:
            return f"{title or ''}{name}"

        server = MCPServerFunction("s", [greet])
        result = await server.call_tool("greet", {"name": "Alice", "title": 7})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_var_keyword_function_still_accepts_extra_keys(self):
        """``**kwargs`` means the developer opted into open arguments."""

        @function_tool
        def flexible(a: int, **rest) -> int:
            return a + len(rest)

        server = MCPServerFunction("s", [flexible])
        result = await server.call_tool("flexible", {"a": 1, "anything": "goes"})
        assert result.isError is not True

    @pytest.mark.asyncio
    async def test_unhintable_parameter_accepts_any_value(self):
        """A ``{}`` property declares no constraint, so validation must not invent one."""

        @function_tool
        def mystery(x) -> str:
            return str(x)

        server = MCPServerFunction("s", [mystery])
        for value in ({"a": 1}, [1], "s", 3, True, None):
            result = await server.call_tool("mystery", {"x": value})
            assert result.isError is not True, f"{value!r} should pass an open schema"


class TestArgumentValidationHonoursHandWrittenSchemas:
    """Explicit ``FunctionTool``/dict tools receive a raw dict, not kwargs -- so
    validation keys off the declared schema rather than the callable."""

    @pytest.mark.asyncio
    async def test_hand_written_schema_is_enforced(self):
        ft = FunctionTool(
            name="format_currency",
            fn=lambda args: f"${args['amount']:,.2f}",
            description="Format USD",
            input_schema={
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
            },
        )
        server = MCPServerFunction("money", [ft])
        assert (await server.call_tool("format_currency", {"amount": "lots"})).isError is True
        assert (await server.call_tool("format_currency", {})).isError is True
        assert (await server.call_tool("format_currency", {"amount": 5})).isError is not True

    @pytest.mark.asyncio
    async def test_bare_object_schema_constrains_nothing(self):
        """``{"type": "object"}`` with no ``properties`` is an open contract."""
        server = MCPServerFunction(
            "custom",
            [
                {
                    "name": "ping",
                    "description": "Ping",
                    "fn": lambda args: "pong",
                    "input_schema": {"type": "object"},
                }
            ],
        )
        result = await server.call_tool("ping", {"whatever": [1, 2, 3]})
        assert result.isError is not True

    @pytest.mark.asyncio
    async def test_additional_properties_true_allows_unknown_keys(self):
        ft = FunctionTool(
            name="open_tool",
            fn=lambda args: "ok",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "additionalProperties": True,
            },
        )
        server = MCPServerFunction("s", [ft])
        result = await server.call_tool("open_tool", {"a": 1, "extra": "fine"})
        assert result.isError is not True

    @pytest.mark.asyncio
    async def test_enum_membership_is_enforced(self):
        """A hand-written ``enum`` is a real constraint, not documentation."""
        ft = FunctionTool(
            name="set_mode",
            fn=lambda args: args["mode"],
            input_schema={
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["read", "write"]}},
                "required": ["mode"],
            },
        )
        server = MCPServerFunction("s", [ft])
        assert (await server.call_tool("set_mode", {"mode": "admin"})).isError is True
        assert (await server.call_tool("set_mode", {"mode": "read"})).isError is not True

    @pytest.mark.asyncio
    async def test_none_arguments_treated_as_empty(self):
        @function_tool
        def nullary() -> str:
            return "ran"

        server = MCPServerFunction("s", [nullary])
        result = await server.call_tool("nullary", None)
        assert result.isError is not True


# ---------------------------------------------------------------------------
# Loud degradation for parameters that cannot be expressed to the model
#
# ``_type_to_schema`` falls back to ``{}`` for anything it cannot map. The
# fallback itself is correct -- the defect was that it happened silently, and a
# parameter like ``conn: Any`` then reaches the model as a *required* argument
# with no constraint, so the model is obliged to fabricate a database handle.
#
# The schema stays truthful (the parameter really is required in Python); it is
# the developer who is told, at registration, that this parameter belongs in
# context injection rather than in the model's hands.
# ---------------------------------------------------------------------------


@contextmanager
def _captured_mcp_warnings():
    """Collect WARNINGs from continuum.tools.mcp (caplog cannot: propagate=False)."""
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Collector()
    logger = logging.getLogger("continuum.tools.mcp")
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


class TestUnhintableParameterWarning:
    def test_warns_naming_tool_and_parameter(self):
        from typing import Any

        with _captured_mcp_warnings() as warnings:

            @function_tool
            def run_sql(query: str, conn: Any) -> str:
                """Run a query."""
                return "ok"

        joined = "\n".join(warnings)
        assert "run_sql" in joined
        assert "conn" in joined

    def test_warning_does_not_name_the_hintable_parameter(self):
        from typing import Any

        with _captured_mcp_warnings() as warnings:

            @function_tool
            def run_sql(query: str, conn: Any) -> str:
                return "ok"

        joined = "\n".join(warnings)
        assert "query" not in joined

    def test_fully_hinted_function_warns_nothing(self):
        with _captured_mcp_warnings() as warnings:

            @function_tool
            def add(a: int, b: int) -> int:
                return a + b

        assert warnings == []

    def test_unhinted_parameter_also_warns(self):
        with _captured_mcp_warnings() as warnings:

            @function_tool
            def mystery(x) -> str:
                return str(x)

        assert any("mystery" in m and "x" in m for m in warnings)

    def test_schema_still_reports_the_parameter_as_required(self):
        """Truthfulness beats convenience: Python really does require it, and
        lying to the model guarantees a TypeError instead of a fabrication."""
        from typing import Any

        @function_tool
        def run_sql(query: str, conn: Any) -> str:
            return "ok"

        assert run_sql.input_schema["properties"]["conn"] == {}
        assert "conn" in run_sql.input_schema["required"]


# ---------------------------------------------------------------------------
# Schema generation must not degrade more than it has to
#
# Both of these surfaced from the F5 validation tests above: an argument
# validator is only as good as the schema it validates against, and these two
# paths were quietly emitting {} for parameters that are perfectly expressible.
# ---------------------------------------------------------------------------


class TestPep604UnionsAreExpressed:
    """``str | None`` is the same type as ``Optional[str]`` and must produce the
    same schema. It did not: the generator tested only ``typing.Union``, so the
    modern spelling -- the one used throughout this codebase -- fell through to
    an open ``{}`` and the model was told the parameter was unconstrained."""

    def test_pep604_optional_gets_the_underlying_type(self):
        @function_tool
        def greet(name: str, title: str | None = None) -> str:
            return name

        assert greet.input_schema["properties"]["title"] == {"type": "string"}

    def test_pep604_optional_is_not_required(self):
        @function_tool
        def greet(name: str, title: str | None) -> str:
            return name

        assert "title" not in greet.input_schema.get("required", [])

    def test_pep604_matches_typing_optional(self):
        @function_tool
        def modern(x: int | None = None) -> int:
            return x or 0

        @function_tool
        def legacy(x: Optional[int] = None) -> int:  # noqa: UP045
            return x or 0

        assert modern.input_schema == legacy.input_schema

    def test_pep604_multi_member_union_stays_open(self):
        """``int | str`` has no single JSON Schema type here, so ``{}`` is honest."""

        @function_tool
        def either(x: int | str) -> str:
            return str(x)

        assert either.input_schema["properties"]["x"] == {}

    @pytest.mark.asyncio
    async def test_pep604_optional_is_then_enforced(self):
        @function_tool
        def greet(name: str, title: str | None = None) -> str:
            return f"{title or ''}{name}"

        server = MCPServerFunction("s", [greet])
        assert (await server.call_tool("greet", {"name": "A", "title": 7})).isError is True
        assert (await server.call_tool("greet", {"name": "A", "title": "Dr"})).isError is not True
        assert (await server.call_tool("greet", {"name": "A", "title": None})).isError is not True


class TestOneUnresolvableAnnotationCostsOnlyItself:
    """``typing.get_type_hints`` is all-or-nothing: under ``from __future__
    import annotations`` every annotation is a string resolved against the
    function's *module* globals, so one name that only exists in a local scope
    raised NameError and the whole hint dict was discarded. Every parameter then
    became an open, required ``{}`` -- turning a cosmetic import detail into the
    schema-loss this module's validation depends on not happening."""

    @staticmethod
    def _locally_annotated():
        from datetime import datetime

        def report(title: str, count: int, when: datetime) -> str:
            return f"{title}:{count}:{when}"

        return report

    def test_resolvable_parameters_keep_their_types(self):
        tool = function_tool(self._locally_annotated())
        props = tool.input_schema["properties"]
        assert props["title"] == {"type": "string"}
        assert props["count"] == {"type": "integer"}

    def test_unresolvable_parameter_falls_back_alone(self):
        tool = function_tool(self._locally_annotated())
        assert tool.input_schema["properties"]["when"] == {}

    def test_warning_names_only_the_unresolvable_parameter(self):
        with _captured_mcp_warnings() as warnings:
            function_tool(self._locally_annotated())

        joined = "\n".join(warnings)
        assert "when" in joined
        assert "title" not in joined
        assert "count" not in joined

    @pytest.mark.asyncio
    async def test_recovered_types_are_enforced(self):
        server = MCPServerFunction("s", [function_tool(self._locally_annotated())])
        bad = await server.call_tool("report", {"title": 1, "count": 2, "when": "now"})
        assert bad.isError is True
        assert "title" in json.loads(bad.content[0].text)["error"]
