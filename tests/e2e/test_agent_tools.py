"""
E2E tests — Agent with real tool execution via FakeMCPServer.

Tests tool calling, multi-tool usage, tool error handling, and
agent reasoning with tool results through real LLM calls.
"""

from __future__ import annotations

import json
import os

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from orchestrator.tools.types import ToolContextConfig

pytestmark = pytest.mark.e2e


from tests.e2e.conftest import skip_if_no_api_key as _skip_if_no_api_key
from tests.e2e.conftest import skip_on_api_error as _skip_on_api_error


# ---------------------------------------------------------------------------
# Fake MCP servers for different tool scenarios
# ---------------------------------------------------------------------------


class CalculatorMCPServer:
    """Fake MCP server with arithmetic tools."""

    def __init__(self):
        self._name = "calculator-server"
        self.context_config = ToolContextConfig()
        self._tools = [
            Tool(
                name="calculate",
                description="Perform arithmetic. Operations: add, subtract, multiply, divide.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "One of: add, subtract, multiply, divide",
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            ),
        ]
        self._connected = False
        self.call_log: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    async def connect(self):
        self._connected = True

    async def cleanup(self):
        self._connected = False

    async def list_tools(self, metadata=None):
        return self._tools

    async def call_tool(self, tool_name, arguments):
        arguments = arguments or {}
        self.call_log.append({"tool": tool_name, "args": arguments})

        if tool_name == "calculate":
            op = arguments.get("operation", "add")
            a = float(arguments.get("a", 0))
            b = float(arguments.get("b", 0))
            try:
                if op == "add":
                    result = a + b
                elif op == "subtract":
                    result = a - b
                elif op == "multiply":
                    result = a * b
                elif op == "divide":
                    if b == 0:
                        return CallToolResult(
                            content=[TextContent(type="text", text="Error: Division by zero")],
                            isError=True,
                        )
                    result = a / b
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown operation: {op}")],
                        isError=True,
                    )
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result))],
                    isError=False,
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {e}")],
                    isError=True,
                )

        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True,
        )

    async def list_prompts(self):
        from mcp.types import ListPromptsResult
        return ListPromptsResult(prompts=[])

    async def get_prompt(self, name, arguments=None):
        from mcp.types import GetPromptResult
        return GetPromptResult(messages=[])


class WeatherAndTimeMCPServer:
    """Fake MCP server with multiple tools for multi-tool testing."""

    def __init__(self):
        self._name = "weather-time-server"
        self.context_config = ToolContextConfig()
        self._tools = [
            Tool(
                name="get_weather",
                description="Get current weather for a city. Returns temperature and conditions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            ),
            Tool(
                name="get_time",
                description="Get current time in a timezone.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone like UTC, EST, PST"},
                    },
                    "required": ["timezone"],
                },
            ),
        ]
        self._connected = False
        self.call_count = 0

    @property
    def name(self):
        return self._name

    async def connect(self):
        self._connected = True

    async def cleanup(self):
        self._connected = False

    async def list_tools(self, metadata=None):
        return self._tools

    async def call_tool(self, tool_name, arguments):
        self.call_count += 1
        arguments = arguments or {}

        if tool_name == "get_weather":
            city = arguments.get("city", "Unknown")
            # Return deterministic fake weather
            weather_data = {
                "new york": {"temp": 72, "condition": "sunny", "humidity": 45},
                "london": {"temp": 58, "condition": "cloudy", "humidity": 78},
                "tokyo": {"temp": 85, "condition": "humid", "humidity": 90},
            }
            data = weather_data.get(city.lower(), {"temp": 65, "condition": "partly cloudy", "humidity": 50})
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Weather in {city}: {data['temp']}°F, {data['condition']}, humidity {data['humidity']}%"
                )],
                isError=False,
            )

        if tool_name == "get_time":
            tz = arguments.get("timezone", "UTC")
            times = {"UTC": "14:30", "EST": "09:30", "PST": "06:30", "JST": "23:30"}
            t = times.get(tz.upper(), "12:00")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Current time in {tz}: {t}")],
                isError=False,
            )

        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True,
        )

    async def list_prompts(self):
        from mcp.types import ListPromptsResult
        return ListPromptsResult(prompts=[])

    async def get_prompt(self, name, arguments=None):
        from mcp.types import GetPromptResult
        return GetPromptResult(messages=[])


class FailingToolMCPServer:
    """Fake MCP server where tools always fail — for error handling tests."""

    def __init__(self):
        self._name = "failing-server"
        self.context_config = ToolContextConfig()
        self._tools = [
            Tool(
                name="unstable_api",
                description="Call an unstable external API. May fail.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string", "description": "API endpoint to call"},
                    },
                    "required": ["endpoint"],
                },
            ),
        ]
        self._connected = False
        self.call_count = 0

    @property
    def name(self):
        return self._name

    async def connect(self):
        self._connected = True

    async def cleanup(self):
        self._connected = False

    async def list_tools(self, metadata=None):
        return self._tools

    async def call_tool(self, tool_name, arguments):
        self.call_count += 1
        return CallToolResult(
            content=[TextContent(type="text", text="Error: Service unavailable (503)")],
            isError=True,
        )

    async def list_prompts(self):
        from mcp.types import ListPromptsResult
        return ListPromptsResult(prompts=[])

    async def get_prompt(self, name, arguments=None):
        from mcp.types import GetPromptResult
        return GetPromptResult(messages=[])


class DataLookupMCPServer:
    """Fake MCP server for data retrieval — tests multi-step reasoning."""

    def __init__(self):
        self._name = "data-lookup-server"
        self.context_config = ToolContextConfig()
        self._tools = [
            Tool(
                name="lookup_employee",
                description="Look up employee info by name. Returns role, department, salary.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Employee name"},
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="lookup_department",
                description="Look up department info. Returns head count, budget, manager.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "description": "Department name"},
                    },
                    "required": ["department"],
                },
            ),
        ]
        self._connected = False

    @property
    def name(self):
        return self._name

    async def connect(self):
        self._connected = True

    async def cleanup(self):
        self._connected = False

    async def list_tools(self, metadata=None):
        return self._tools

    async def call_tool(self, tool_name, arguments):
        arguments = arguments or {}

        if tool_name == "lookup_employee":
            name = arguments.get("name", "").lower()
            employees = {
                "alice": {"role": "Engineer", "department": "Engineering", "salary": 150000},
                "bob": {"role": "Manager", "department": "Engineering", "salary": 180000},
                "carol": {"role": "Designer", "department": "Design", "salary": 130000},
            }
            emp = employees.get(name)
            if emp:
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(emp))],
                    isError=False,
                )
            return CallToolResult(
                content=[TextContent(type="text", text=f"Employee '{name}' not found")],
                isError=False,
            )

        if tool_name == "lookup_department":
            dept = arguments.get("department", "").lower()
            departments = {
                "engineering": {"head_count": 45, "budget": 5000000, "manager": "Bob"},
                "design": {"head_count": 12, "budget": 1200000, "manager": "Eve"},
            }
            d = departments.get(dept)
            if d:
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(d))],
                    isError=False,
                )
            return CallToolResult(
                content=[TextContent(type="text", text=f"Department '{dept}' not found")],
                isError=False,
            )

        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True,
        )

    async def list_prompts(self):
        from mcp.types import ListPromptsResult
        return ListPromptsResult(prompts=[])

    async def get_prompt(self, name, arguments=None):
        from mcp.types import GetPromptResult
        return GetPromptResult(messages=[])


# ---------------------------------------------------------------------------
# Helper: Create agent + runner with tool server
# ---------------------------------------------------------------------------


async def _make_agent_with_tools(
    name: str,
    instructions: str,
    server,
    *,
    max_turns: int = 10,
    log_to_session: bool = False,
):
    """Create a BaseAgent wired to a fake MCP server."""
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
    from orchestrator.agent.runner import AgentRunner
    from orchestrator.tools.executor import ToolExecutor
    from orchestrator.tools.util import MCPUtil

    await server.connect()
    tool_defs = await MCPUtil.get_function_tools(server)
    executor = ToolExecutor(tool_registry={server: None})
    await executor.initialize()

    agent = BaseAgent(
        name=name,
        instructions=instructions,
        tools=tool_defs,
        tool_executor=executor,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=log_to_session, max_turns=max_turns),
    )

    runner = AgentRunner(tool_executor=executor)
    return agent, runner


# ---------------------------------------------------------------------------
# Test: Agent uses calculator tool
# ---------------------------------------------------------------------------


class TestAgentToolCalling:
    """Test that agents call tools correctly and reason over results."""

    @_skip_on_api_error
    async def test_agent_calls_calculator_for_math(self):
        """Agent should use calculator tool for arithmetic questions."""
        _skip_if_no_api_key()

        server = CalculatorMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="math-agent",
            instructions="You are a math assistant. Use the calculate tool for ALL arithmetic. Report exact results.",
            server=server,
        )

        response = await runner.run(
            agent,
            "What is 137 multiplied by 29?",
            context=__import__("orchestrator.agent.types", fromlist=["RunContext"]).RunContext(run_id="e2e-calc"),
        )

        assert response.content is not None
        assert "3973" in response.content  # 137 * 29 = 3973
        assert len(server.call_log) >= 1
        assert response.status.value == "success"

    @_skip_on_api_error
    async def test_agent_handles_division_by_zero(self):
        """Agent should gracefully handle tool errors (division by zero)."""
        _skip_if_no_api_key()

        server = CalculatorMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="math-error-agent",
            instructions="You are a math assistant. Use the calculate tool. If a calculation fails, explain why.",
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "What is 100 divided by 0?",
            context=RunContext(run_id="e2e-div-zero"),
        )

        assert response.content is not None
        # Agent should explain the error, not crash
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["zero", "undefined", "cannot", "error", "impossible"])

    @_skip_on_api_error
    async def test_agent_chains_multiple_tool_calls(self):
        """Agent should chain multiple tool calls for complex problems."""
        _skip_if_no_api_key()

        server = CalculatorMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="chain-calc-agent",
            instructions=(
                "You are a math assistant. Use the calculate tool for each step. "
                "Show intermediate results. Be precise."
            ),
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "Calculate (15 + 7) * 3. Do step by step using the calculate tool.",
            context=RunContext(run_id="e2e-chain-calc"),
        )

        assert response.content is not None
        assert "66" in response.content  # (15 + 7) * 3 = 66
        # Agent should have made multiple tool calls
        assert len(server.call_log) >= 2
        assert response.turn_count >= 2  # Multiple turns for chained calls


# ---------------------------------------------------------------------------
# Test: Agent with multiple tools
# ---------------------------------------------------------------------------


class TestAgentMultiTool:
    """Test agent using multiple tools in a single conversation."""

    @_skip_on_api_error
    async def test_agent_selects_correct_tool(self):
        """Agent should pick the right tool for the question."""
        _skip_if_no_api_key()

        server = WeatherAndTimeMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="multi-tool-agent",
            instructions="You have access to weather and time tools. Use them when asked. Be concise.",
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "What's the weather like in London right now?",
            context=RunContext(run_id="e2e-multi-tool-weather"),
        )

        assert response.content is not None
        assert "58" in response.content or "cloudy" in response.content.lower()

    @_skip_on_api_error
    async def test_agent_uses_multiple_tools_in_one_query(self):
        """Agent should call multiple tools when the query requires both."""
        _skip_if_no_api_key()

        server = WeatherAndTimeMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="combo-tool-agent",
            instructions=(
                "You have weather and time tools. When asked about both, use both tools. "
                "Always use tools, never guess."
            ),
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "What's the weather and current time in New York (EST)?",
            context=RunContext(run_id="e2e-combo-tools"),
        )

        assert response.content is not None
        # Should mention weather info
        assert any(x in response.content for x in ["72", "sunny", "New York"])
        # Should have called at least 2 tools
        assert server.call_count >= 2


# ---------------------------------------------------------------------------
# Test: Agent handles tool failures gracefully
# ---------------------------------------------------------------------------


class TestAgentToolFailure:
    """Test agent behavior when tools fail."""

    @_skip_on_api_error
    async def test_agent_recovers_from_tool_error(self):
        """Agent should inform user when tool fails, not crash."""
        _skip_if_no_api_key()

        server = FailingToolMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="failure-agent",
            instructions=(
                "You have access to an API tool. If it fails, tell the user "
                "that the service is currently unavailable and suggest trying later."
            ),
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "Please call the API endpoint /users/status",
            context=RunContext(run_id="e2e-tool-fail"),
        )

        assert response.content is not None
        assert response.status.value == "success"  # Agent itself succeeds
        # Agent should communicate the failure to user
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["unavailable", "error", "fail", "unable", "sorry"])


# ---------------------------------------------------------------------------
# Test: Multi-step reasoning with data lookup
# ---------------------------------------------------------------------------


class TestAgentMultiStepReasoning:
    """Test agent performing multi-step data lookups and reasoning."""

    @_skip_on_api_error
    async def test_agent_cross_references_data(self):
        """Agent should look up employee, then their department, and synthesize."""
        _skip_if_no_api_key()

        server = DataLookupMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="data-agent",
            instructions=(
                "You are a data analyst. Use lookup_employee and lookup_department tools. "
                "Answer questions by looking up the data — never guess."
            ),
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "What department does Alice work in, and what's that department's budget?",
            context=RunContext(run_id="e2e-cross-ref"),
        )

        assert response.content is not None
        # Should contain Alice's department (Engineering) and its budget ($5M)
        assert "engineering" in response.content.lower() or "Engineering" in response.content
        assert "5000000" in response.content or "5,000,000" in response.content or "5 million" in response.content.lower()

    @_skip_on_api_error
    async def test_agent_handles_not_found_data(self):
        """Agent should handle gracefully when data isn't found."""
        _skip_if_no_api_key()

        server = DataLookupMCPServer()
        agent, runner = await _make_agent_with_tools(
            name="not-found-agent",
            instructions="You are a data analyst. Look up data using tools. If data isn't found, say so clearly.",
            server=server,
        )
        from orchestrator.agent.types import RunContext

        response = await runner.run(
            agent,
            "Look up the employee named 'Zara' and tell me their role.",
            context=RunContext(run_id="e2e-not-found"),
        )

        assert response.content is not None
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["not found", "no employee", "couldn't find", "does not exist", "no record"])
