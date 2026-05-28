"""
Unit tests for the tool-attention hallucination gate in ToolService.

Covers:
- Tool in promoted set → executes normally
- Tool not in promoted set → returns structured error without hitting MCP
- No promoted_tools in context → gate is inactive, executes normally
- context.metadata is None → gate is inactive
- Gate error response has correct structure
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agent.services.tool_service import ToolService
from orchestrator.llm.types import FunctionCall, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, call_id: str = "call_abc") -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments="{}"),
    )


def _make_context(promoted: set[str] | None = None, metadata: dict | None = None):
    ctx = MagicMock()
    if metadata is not None:
        ctx.metadata = metadata
    elif promoted is not None:
        ctx.metadata = {"promoted_tools": promoted}
    else:
        ctx.metadata = {}
    ctx.trace_id = "trace-123"
    ctx.run_id = "run-123"
    return ctx


def _make_agent(tool_executor=None):
    agent = MagicMock()
    agent.name = "test_agent"
    agent.tool_executor = tool_executor
    agent.policy_store = None
    agent.on_tool_call = None
    return agent


# ---------------------------------------------------------------------------
# Gate: tool in promoted set → passes through to executor
# ---------------------------------------------------------------------------


class TestGateAllowsPromotedTool:
    @pytest.mark.asyncio
    async def test_promoted_tool_reaches_executor(self):
        mock_executor = AsyncMock()
        mock_executor.tool_registry = {}
        mock_executor.execute_tool_calls = AsyncMock(
            return_value=[
                {"role": "tool", "tool_call_id": "call_abc", "content": "results here"},
            ]
        )

        agent = _make_agent(tool_executor=mock_executor)
        service = ToolService(tool_executor=None)

        tc = _make_tool_call("search_products")
        ctx = _make_context(promoted={"search_products", "add_to_cart"})

        result, meta = await service.execute_tool_call(agent, tc, ctx)

        mock_executor.execute_tool_calls.assert_called_once()
        assert result["role"] == "tool"
        assert meta["success"] is True

    @pytest.mark.asyncio
    async def test_all_tools_allowed_when_no_promoted_set(self):
        mock_executor = AsyncMock()
        mock_executor.tool_registry = {}
        mock_executor.execute_tool_calls = AsyncMock(
            return_value=[
                {"role": "tool", "tool_call_id": "call_abc", "content": "ok"},
            ]
        )

        agent = _make_agent(tool_executor=mock_executor)
        service = ToolService(tool_executor=None)

        # No promoted_tools key → gate inactive
        tc = _make_tool_call("any_tool")
        ctx = _make_context(metadata={})

        result, meta = await service.execute_tool_call(agent, tc, ctx)

        mock_executor.execute_tool_calls.assert_called_once()
        assert result["role"] == "tool"

    @pytest.mark.asyncio
    async def test_gate_inactive_when_metadata_is_none(self):
        mock_executor = AsyncMock()
        mock_executor.tool_registry = {}
        mock_executor.execute_tool_calls = AsyncMock(
            return_value=[
                {"role": "tool", "tool_call_id": "call_abc", "content": "ok"},
            ]
        )

        agent = _make_agent(tool_executor=mock_executor)
        service = ToolService(tool_executor=None)

        tc = _make_tool_call("any_tool")
        ctx = MagicMock()
        ctx.metadata = None
        ctx.trace_id = "t"
        ctx.run_id = "r"

        result, meta = await service.execute_tool_call(agent, tc, ctx)

        mock_executor.execute_tool_calls.assert_called_once()


# ---------------------------------------------------------------------------
# Gate: tool not in promoted set → returns error
# ---------------------------------------------------------------------------


class TestGateBlocksUnpromotedTool:
    @pytest.mark.asyncio
    async def test_unpromoted_tool_is_blocked(self):
        mock_executor = AsyncMock()
        mock_executor.tool_registry = {}

        agent = _make_agent(tool_executor=mock_executor)
        service = ToolService(tool_executor=None)

        tc = _make_tool_call("delete_account", call_id="call_xyz")
        ctx = _make_context(promoted={"search_products", "add_to_cart"})

        result, meta = await service.execute_tool_call(agent, tc, ctx)

        # MCP executor must NOT be called
        mock_executor.execute_tool_calls.assert_not_called()

        # Result is a tool error message
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_xyz"

        body = json.loads(result["content"])
        assert body["error"] == "tool_not_available"
        assert "delete_account" in body["message"]
        assert "search_products" in body["available"]
        assert "add_to_cart" in body["available"]

    @pytest.mark.asyncio
    async def test_gate_error_metadata_marks_failure(self):
        agent = _make_agent(tool_executor=MagicMock())
        service = ToolService(tool_executor=None)

        tc = _make_tool_call("hidden_tool")
        ctx = _make_context(promoted={"search_products"})

        _, meta = await service.execute_tool_call(agent, tc, ctx)

        assert meta["success"] is False
        assert meta["error"] == "tool_not_in_promoted_set"
        assert meta["tool_name"] == "hidden_tool"

    @pytest.mark.asyncio
    async def test_available_list_is_sorted(self):
        agent = _make_agent(tool_executor=MagicMock())
        service = ToolService(tool_executor=None)

        tc = _make_tool_call("unknown")
        ctx = _make_context(promoted={"z_tool", "a_tool", "m_tool"})

        result, _ = await service.execute_tool_call(agent, tc, ctx)
        body = json.loads(result["content"])

        assert body["available"] == sorted(["z_tool", "a_tool", "m_tool"])

    @pytest.mark.asyncio
    async def test_empty_promoted_set_blocks_all_tools(self):
        agent = _make_agent(tool_executor=MagicMock())
        service = ToolService(tool_executor=None)

        tc = _make_tool_call("search_products")
        ctx = _make_context(promoted=set())  # empty promoted set

        result, meta = await service.execute_tool_call(agent, tc, ctx)

        body = json.loads(result["content"])
        assert body["error"] == "tool_not_available"
        assert body["available"] == []
        assert meta["success"] is False

    @pytest.mark.asyncio
    async def test_gate_fires_before_global_executor_fallback(self):
        """Gate blocks the tool even when the agent has no executor but a global executor exists."""
        global_executor = AsyncMock()
        global_executor.tool_registry = {}

        # Agent has no executor of its own
        agent = _make_agent(tool_executor=None)
        # Global executor provided at ToolService level
        service = ToolService(tool_executor=global_executor)

        tc = _make_tool_call("forbidden_tool")
        ctx = _make_context(promoted={"allowed_tool"})

        result, meta = await service.execute_tool_call(agent, tc, ctx)

        # Gate must have fired — global executor must NOT be reached
        global_executor.execute_tool_calls.assert_not_called()

        body = json.loads(result["content"])
        assert body["error"] == "tool_not_available"
        assert meta["success"] is False
