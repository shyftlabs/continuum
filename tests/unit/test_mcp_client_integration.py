"""
Unit tests verifying that client MCP servers integrate with all Continuum features.

Uses MCPServerFunction throughout — no external server, no network, no LLM needed.

Covers:
- Tool execution via ToolExecutor
- get_tool_definitions() fix (no double list_tools)
- AgentConfigurationError when mcp_servers set without tool_executor
- Policy enforcement on MCP tool calls
- ToolContextConfig capture and injection
- get_tool_definitions() guard before initialize()
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(name: str = "test-server"):
    """In-process MCP server with two simple tools."""
    from orchestrator.tools.mcp import MCPServerFunction

    def greet(user: str) -> str:
        """Greet a user."""
        return f"Hello, {user}!"

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return MCPServerFunction(name, [greet, add])


async def _make_executor(server, namespace_tools: bool = False):
    from orchestrator.tools.executor import ToolExecutor

    executor = ToolExecutor(
        tool_registry={server: None},
        namespace_tools=namespace_tools,
    )
    await executor.initialize()
    return executor


# ---------------------------------------------------------------------------
# 1. Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    """ToolExecutor correctly calls an MCPServerFunction tool."""

    @pytest.mark.asyncio
    async def test_call_tool_returns_result(self):
        from orchestrator.llm.types import FunctionCall, ToolCall

        server = _make_server()
        executor = await _make_executor(server)

        tc = ToolCall(
            id="tc-1",
            type="function",
            function=FunctionCall(name="greet", arguments=json.dumps({"user": "Alice"})),
        )
        result = await executor.execute_tool_call(tc)
        assert "Alice" in result.content

    @pytest.mark.asyncio
    async def test_call_add_tool(self):
        from orchestrator.llm.types import FunctionCall, ToolCall

        server = _make_server()
        executor = await _make_executor(server)

        tc = ToolCall(
            id="tc-2",
            type="function",
            function=FunctionCall(name="add", arguments=json.dumps({"a": 3, "b": 4})),
        )
        result = await executor.execute_tool_call(tc)
        assert "7" in result.content

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        from orchestrator.llm.types import FunctionCall, ToolCall
        from orchestrator.tools.exceptions import MCPToolError

        server = _make_server()
        executor = await _make_executor(server)

        tc = ToolCall(
            id="tc-3",
            type="function",
            function=FunctionCall(name="nonexistent", arguments="{}"),
        )
        with pytest.raises(MCPToolError):
            await executor.execute_tool_call(tc)


# ---------------------------------------------------------------------------
# 2. get_tool_definitions() — our fix
# ---------------------------------------------------------------------------


class TestGetToolDefinitions:
    """get_tool_definitions() derives schemas from registry — no second list_tools call."""

    @pytest.mark.asyncio
    async def test_returns_correct_tool_names(self):
        server = _make_server()
        executor = await _make_executor(server)

        defs = executor.get_tool_definitions()
        names = {d.function.name for d in defs}
        assert names == {"greet", "add"}

    @pytest.mark.asyncio
    async def test_tool_has_description(self):
        server = _make_server()
        executor = await _make_executor(server)

        defs = executor.get_tool_definitions()
        greet_def = next(d for d in defs if d.function.name == "greet")
        assert greet_def.function.description == "Greet a user."

    @pytest.mark.asyncio
    async def test_namespaced_names_match_registry_keys(self):
        """When namespace_tools=True, tool def names must match registry keys."""
        server = _make_server("my-server")
        executor = await _make_executor(server, namespace_tools=True)

        defs = executor.get_tool_definitions()
        names = {d.function.name for d in defs}

        assert "my-server__greet" in names
        assert "my-server__add" in names

    @pytest.mark.asyncio
    async def test_namespaced_names_match_what_executor_can_route(self):
        """Tool def names must exactly match executor registry keys so routing works."""
        server = _make_server("my-server")
        executor = await _make_executor(server, namespace_tools=True)

        defs = executor.get_tool_definitions()
        for d in defs:
            assert d.function.name in executor.tool_registry


# ---------------------------------------------------------------------------
# 3. AgentConfigurationError — our Option C fix
# ---------------------------------------------------------------------------


class TestMcpServersDeadField:
    """Runner raises AgentConfigurationError when mcp_servers set without tool_executor."""

    @pytest.mark.asyncio
    async def test_raises_on_run(self):
        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.exceptions import AgentConfigurationError
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.tools.mcp import MCPServerFunction

        server = MCPServerFunction("s", [])
        agent = BaseAgent(name="bad-agent", mcp_servers=[server])

        runner = AgentRunner()
        with pytest.raises(AgentConfigurationError) as exc_info:
            await runner.run(agent, "hello")

        assert "mcp_servers" in str(exc_info.value)
        assert "tool_executor" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_error_when_tool_executor_also_set(self):
        """If both mcp_servers and tool_executor are set, no AgentConfigurationError for mcp_servers."""
        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.exceptions import AgentConfigurationError
        from orchestrator.agent.runner import AgentRunner

        server = _make_server()
        executor = await _make_executor(server)
        tools = executor.get_tool_definitions()

        agent = BaseAgent(
            name="good-agent",
            mcp_servers=[server],
            tool_executor=executor,
            tools=tools,
        )

        runner = AgentRunner()
        try:
            await runner.run(agent, "hello")
        except AgentConfigurationError as e:
            if "mcp_servers" in str(e):
                pytest.fail("AgentConfigurationError raised for mcp_servers — should not happen")
        except Exception:
            pass  # other errors (LLM not set up) are acceptable


# ---------------------------------------------------------------------------
# 4. Policy enforcement
# ---------------------------------------------------------------------------


class TestPolicyEnforcement:
    """PolicyStore deny/allow applies uniformly to MCP tool calls."""

    @pytest.mark.asyncio
    async def test_deny_policy_blocks_mcp_tool(self):
        from orchestrator.agent.exceptions import ToolAccessDeniedError
        from orchestrator.llm.types import FunctionCall, ToolCall
        from orchestrator.security.policy import AccessPolicy, PolicyStore

        server = _make_server()
        executor = await _make_executor(server)

        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="block-greet",
                subjects=["agent-x"],
                resources=["tool:greet"],
                effect="deny",
            )
        )

        tc = ToolCall(
            id="tc-p1",
            type="function",
            function=FunctionCall(name="greet", arguments=json.dumps({"user": "Bob"})),
        )
        with pytest.raises(ToolAccessDeniedError):
            await executor.execute_tool_call(tc, policy_store=store, subject="agent-x")

    @pytest.mark.asyncio
    async def test_allow_policy_passes_mcp_tool(self):
        from orchestrator.llm.types import FunctionCall, ToolCall
        from orchestrator.security.policy import AccessPolicy, PolicyStore

        server = _make_server()
        executor = await _make_executor(server)

        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="allow-add",
                subjects=["agent-x"],
                resources=["tool:add"],
                effect="allow",
            )
        )

        tc = ToolCall(
            id="tc-p2",
            type="function",
            function=FunctionCall(name="add", arguments=json.dumps({"a": 1, "b": 2})),
        )
        result = await executor.execute_tool_call(tc, policy_store=store, subject="agent-x")
        assert "3" in result.content


# ---------------------------------------------------------------------------
# 5. ToolContextConfig — capture and injection
# ---------------------------------------------------------------------------


class TestToolContextConfig:
    """ToolContextConfig captures a value from one tool and injects it into the next."""

    @pytest.mark.asyncio
    async def test_captures_session_id_from_tool_result(self):
        from orchestrator.llm.types import FunctionCall, ToolCall
        from orchestrator.tools.executor import ToolExecutor
        from orchestrator.tools.mcp import MCPServerFunction
        from orchestrator.tools.types import ToolContextConfig, ToolContextVariable

        def create_session() -> dict:
            """Create a session and return session_id."""
            return {"session_id": "abc-123", "status": "created"}

        config = ToolContextConfig(
            variables=[ToolContextVariable(name="session_id", capture_from=["create_session"])],
            auto_capture_common=False,
        )
        server = MCPServerFunction("session-server", [create_session], context_config=config)
        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        tc = ToolCall(
            id="tc-s1",
            type="function",
            function=FunctionCall(name="create_session", arguments="{}"),
        )
        await executor.execute_tool_call(tc)

        namespace = server.name
        captured = executor.context_state.get(namespace, "session_id")
        assert captured == "abc-123"

    @pytest.mark.asyncio
    async def test_injects_captured_value_into_next_call(self):
        from orchestrator.llm.types import FunctionCall, ToolCall
        from orchestrator.tools.executor import ToolExecutor
        from orchestrator.tools.mcp import MCPServerFunction
        from orchestrator.tools.types import ToolContextConfig, ToolContextVariable

        received_args = {}

        def create_session() -> dict:
            """Create a session."""
            return {"session_id": "xyz-999"}

        def do_work(session_id: str) -> str:
            """Do work using session."""
            received_args["session_id"] = session_id
            return f"done with {session_id}"

        config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="session_id",
                    capture_from=["create_session"],
                    inject_into=["do_work"],
                )
            ],
            auto_capture_common=False,
        )
        server = MCPServerFunction("work-server", [create_session, do_work], context_config=config)
        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        await executor.execute_tool_call(
            ToolCall(
                id="tc-w1",
                type="function",
                function=FunctionCall(name="create_session", arguments="{}"),
            )
        )
        await executor.execute_tool_call(
            ToolCall(
                id="tc-w2",
                type="function",
                function=FunctionCall(name="do_work", arguments="{}"),
            )
        )

        assert received_args.get("session_id") == "xyz-999"


# ---------------------------------------------------------------------------
# 6. get_tool_definitions() guard before initialize()
# ---------------------------------------------------------------------------


class TestGetToolDefinitionsBeforeInit:
    """get_tool_definitions() raises RuntimeError if called before initialize()."""

    def test_raises_before_initialize(self):
        from orchestrator.tools.executor import ToolExecutor
        from orchestrator.tools.mcp import MCPServerFunction

        server = MCPServerFunction("s", [])
        executor = ToolExecutor(tool_registry={server: None})

        with pytest.raises(RuntimeError, match="initialize"):
            executor.get_tool_definitions()

    @pytest.mark.asyncio
    async def test_no_error_after_initialize(self):
        server = _make_server()
        executor = await _make_executor(server)

        defs = executor.get_tool_definitions()
        assert len(defs) == 2

    def test_no_error_when_no_registry_config(self):
        """Empty executor with no servers — empty list is valid."""
        from orchestrator.tools.executor import ToolExecutor

        executor = ToolExecutor()
        defs = executor.get_tool_definitions()
        assert defs == []
