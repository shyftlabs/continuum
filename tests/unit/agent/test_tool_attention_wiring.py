"""
Integration tests verifying tool-attention is wired into both execution paths.

Covers:
- Executor (non-streaming) calls apply_tool_attention and passes result to LLM
- Executor falls back to get_tools_for_llm() when apply_tool_attention returns None
- run_stream (streaming) calls apply_tool_attention and passes result to chat_stream
- run_stream falls back to get_tools_for_llm() when apply_tool_attention returns None
- Promoted set is overwritten each turn (not accumulated)
- AgentConfig.to_dict() includes tool_attention
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.config import AgentConfig
from orchestrator.agent.execution.executor import Executor
from orchestrator.agent.runner import AgentRunner
from orchestrator.agent.types import PrepareRunResult
from orchestrator.tools.tool_attention.config import ToolAttentionConfig
from orchestrator.tools.tool_attention.router import ToolAttentionRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dict_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": f"Does {name}", "parameters": {}},
    }


def _make_agent(tool_attention=None, num_tools: int = 5):
    agent = MagicMock()
    agent.name = "test_agent"
    agent.model = "gpt-4"
    agent.metadata = {}
    agent.config.reasoning_mode = False
    agent.config.react_mode = False
    agent.config.tool_attention = tool_attention
    agent.config.stage_priority = 5
    agent.enable_json_mode = False
    agent.output_schema = None
    agent.json_schema = None
    agent.is_handoff_tool_call.return_value = (False, None)
    agent.on_tool_call = None
    agent.get_tools_for_llm.return_value = [_make_dict_tool(f"tool_{i}") for i in range(num_tools)]
    return agent


def _make_llm_response(content: str = "Done") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = None
    resp.usage = None
    return resp


def _make_context(metadata: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.max_turns = 5
    ctx.session_id = None
    ctx.priority = 5
    ctx.metadata = metadata if metadata is not None else {}
    ctx.run_id = "run-1"
    ctx.trace_id = "trace-1"
    return ctx


def _make_run_state(messages: list) -> MagicMock:
    rs = MagicMock()
    rs.messages = list(messages)
    rs.turn_count = 0
    return rs


# ---------------------------------------------------------------------------
# Executor wiring
# ---------------------------------------------------------------------------


class TestExecutorToolAttentionWiring:
    @pytest.mark.asyncio
    async def test_executor_uses_filtered_tools_from_apply_tool_attention(self):
        """Executor reads _filtered_tools from context.metadata and passes them to LLM."""
        filtered = [_make_dict_tool("search_products")]
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_llm_response())

        executor = Executor(llm_client=mock_llm)
        agent = _make_agent(tool_attention=ToolAttentionConfig(k=2, min_tools=3))
        messages = [{"role": "user", "content": "find dog food"}]
        ctx = _make_context(metadata={"_filtered_tools": filtered})

        with patch("orchestrator.agent.execution.executor.LLMConfig") as MockConfig:
            MockConfig.from_agent_config.return_value = MagicMock()
            await executor.execute_loop(agent, messages, ctx, _make_run_state(messages))

        tools_sent = mock_llm.chat.call_args.kwargs["tools"]
        assert tools_sent == filtered
        assert len(tools_sent) == 1
        assert tools_sent[0]["function"]["name"] == "search_products"

    @pytest.mark.asyncio
    async def test_executor_falls_back_to_all_tools_when_apply_returns_none(self):
        """When apply_tool_attention returns None, all tools from get_tools_for_llm are used."""
        all_tools = [_make_dict_tool(f"tool_{i}") for i in range(5)]
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_llm_response())

        executor = Executor(llm_client=mock_llm)
        agent = _make_agent(tool_attention=ToolAttentionConfig(k=2, min_tools=3))
        agent.get_tools_for_llm.return_value = all_tools
        messages = [{"role": "user", "content": "find dog food"}]

        with (
            patch(
                "orchestrator.tools.tool_attention.router.apply_tool_attention",
                new_callable=AsyncMock,
                return_value=None,  # router decided not to filter
            ),
            patch("orchestrator.agent.execution.executor.LLMConfig") as MockConfig,
        ):
            MockConfig.from_agent_config.return_value = MagicMock()
            await executor.execute_loop(agent, messages, _make_context(), _make_run_state(messages))

        tools_sent = mock_llm.chat.call_args.kwargs["tools"]
        assert tools_sent == all_tools

    @pytest.mark.asyncio
    async def test_executor_skips_tool_attention_when_not_configured(self):
        """When tool_attention is None, apply_tool_attention returns None → all tools used."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_llm_response())

        executor = Executor(llm_client=mock_llm)
        agent = _make_agent(tool_attention=None)  # disabled
        all_tools = agent.get_tools_for_llm.return_value
        messages = [{"role": "user", "content": "hello"}]

        with (
            patch(
                "orchestrator.tools.tool_attention.router.apply_tool_attention",
                new_callable=AsyncMock,
                return_value=None,  # simulate disabled — returns None so fallback triggers
            ),
            patch("orchestrator.agent.execution.executor.LLMConfig") as MockConfig,
        ):
            MockConfig.from_agent_config.return_value = MagicMock()
            await executor.execute_loop(agent, messages, _make_context(), _make_run_state(messages))

        tools_sent = mock_llm.chat.call_args.kwargs["tools"]
        assert tools_sent == all_tools


# ---------------------------------------------------------------------------
# Multi-turn: promoted set is overwritten each turn
# ---------------------------------------------------------------------------


class TestMultiTurnPromotedSet:
    def _make_router(self, search_returns: list[str], min_tools: int = 3) -> ToolAttentionRouter:
        cfg = ToolAttentionConfig(k=5, min_tools=min_tools)
        router = ToolAttentionRouter(cfg)
        router._initialized = True
        router._registry = MagicMock()
        router._registry.ready = True
        router._registry.search.return_value = search_returns
        return router

    def _tools(self, names: list[str]) -> list[dict]:
        return [_make_dict_tool(n) for n in names]

    def _messages(self, query: str = "find dog food") -> list[dict]:
        return [{"role": "user", "content": query}]

    def test_promoted_set_is_overwritten_not_accumulated(self):
        """Each turn's route() call replaces promoted_tools, never adds to it."""
        all_tools = self._tools(["search", "add_to_cart", "get_cart", "checkout", "delete"])
        router = self._make_router(["search"])
        ctx = MagicMock()
        ctx.metadata = {}

        # Turn 1: search is promoted
        router.route(self._messages(), all_tools, ctx)
        assert "search" in ctx.metadata["promoted_tools"]
        assert "add_to_cart" not in ctx.metadata["promoted_tools"]

        # Turn 2: route returns different tool — promoted set is replaced
        router._registry.search.return_value = ["add_to_cart"]
        router.route(self._messages(), all_tools, ctx)
        assert "add_to_cart" in ctx.metadata["promoted_tools"]
        # search must NOT carry over from turn 1
        assert "search" not in ctx.metadata["promoted_tools"]

    def test_each_turn_writes_independent_promoted_set(self):
        """Two turns with different routing results produce independent sets in metadata."""
        all_tools = self._tools(["a", "b", "c", "d", "e"])
        router = self._make_router(["a", "b"])
        ctx = MagicMock()
        ctx.metadata = {}

        router.route(self._messages(), all_tools, ctx)
        turn1_set = set(ctx.metadata["promoted_tools"])  # snapshot

        router._registry.search.return_value = ["c", "d"]
        router.route(self._messages(), all_tools, ctx)
        turn2_set = ctx.metadata["promoted_tools"]

        # Turn 2 set is different from turn 1
        assert "c" in turn2_set
        assert "d" in turn2_set
        # Original turn 1 snapshot is unchanged (it was a different set object or value)
        assert "a" in turn1_set
        assert "c" not in turn1_set


# ---------------------------------------------------------------------------
# run_stream wiring
# ---------------------------------------------------------------------------


def _make_stream_agent(tool_attention=None, num_tools: int = 5):
    agent = MagicMock()
    agent.name = "stream_agent"
    agent.model = "gpt-4o-mini"
    agent.temperature = 0.7
    agent.max_tokens = 1024
    agent.gateway_mode = None
    agent.enable_json_mode = False
    agent.json_schema = None
    agent.json_strict = False
    agent.metadata = {}
    agent.config.tool_attention = tool_attention
    agent.on_end = None
    agent.is_handoff_tool_call.return_value = (False, None)
    agent.get_tools_for_llm.return_value = [_make_dict_tool(f"tool_{i}") for i in range(num_tools)]
    return agent


def _make_runner(mock_llm) -> AgentRunner:
    with patch("orchestrator.agent.runner.get_container") as mock_gc:
        mock_gc.return_value = MagicMock()
        runner = AgentRunner(llm_client=mock_llm)
    return runner


class TestRunStreamToolAttentionWiring:
    @pytest.mark.asyncio
    async def test_run_stream_uses_filtered_tools_from_apply_tool_attention(self):
        """run_stream reads _filtered_tools from context.metadata and passes them to chat_stream."""
        filtered = [_make_dict_tool("search_products")]
        captured: dict = {}

        async def fake_chat_stream(*, messages, tools, **kwargs):
            captured["tools"] = tools
            chunk = MagicMock()
            chunk.content = "Done"
            chunk.tool_calls = None
            yield chunk

        mock_llm = MagicMock()
        mock_llm.chat_stream = fake_chat_stream

        runner = _make_runner(mock_llm)
        agent = _make_stream_agent(tool_attention=ToolAttentionConfig(k=2, min_tools=3))
        messages = [{"role": "user", "content": "find dog food"}]

        mock_ctx = _make_context(metadata={"_filtered_tools": filtered})
        prepare_result = PrepareRunResult(
            success=True,
            context=mock_ctx,
            run_state=_make_run_state(messages),
        )
        runner._prepare_run = AsyncMock(return_value=prepare_result)
        runner._finalizer.finalize = AsyncMock()

        _ = [e async for e in runner.run_stream(agent, "find dog food")]

        assert captured["tools"] == filtered
        assert len(captured["tools"]) == 1
        assert captured["tools"][0]["function"]["name"] == "search_products"

    @pytest.mark.asyncio
    async def test_run_stream_falls_back_to_all_tools_when_apply_returns_none(self):
        """When apply_tool_attention returns None, chat_stream receives all tools."""
        all_tools = [_make_dict_tool(f"tool_{i}") for i in range(5)]
        captured: dict = {}

        async def fake_chat_stream(*, messages, tools, **kwargs):
            captured["tools"] = tools
            chunk = MagicMock()
            chunk.content = "Done"
            chunk.tool_calls = None
            yield chunk

        mock_llm = MagicMock()
        mock_llm.chat_stream = fake_chat_stream

        runner = _make_runner(mock_llm)
        agent = _make_stream_agent(tool_attention=ToolAttentionConfig(k=2, min_tools=3))
        agent.get_tools_for_llm.return_value = all_tools
        messages = [{"role": "user", "content": "find dog food"}]

        mock_ctx = _make_context()
        prepare_result = PrepareRunResult(
            success=True,
            context=mock_ctx,
            run_state=_make_run_state(messages),
        )
        runner._prepare_run = AsyncMock(return_value=prepare_result)
        runner._finalizer.finalize = AsyncMock()

        with patch(
            "orchestrator.tools.tool_attention.router.apply_tool_attention",
            new_callable=AsyncMock,
            return_value=None,
        ):
            _ = [e async for e in runner.run_stream(agent, "find dog food")]

        assert captured["tools"] == all_tools


# ---------------------------------------------------------------------------
# AgentConfig.to_dict() includes tool_attention
# ---------------------------------------------------------------------------


class TestAgentConfigToDict:
    def test_to_dict_includes_tool_attention_when_configured(self):
        """AgentConfig.to_dict() serialises tool_attention fields when set."""
        cfg = AgentConfig(tool_attention=ToolAttentionConfig(k=3, min_tools=5, threshold=0.1))
        d = cfg.to_dict()

        assert d["tool_attention"] is not None
        assert d["tool_attention"]["k"] == 3
        assert d["tool_attention"]["min_tools"] == 5
        assert d["tool_attention"]["threshold"] == 0.1

    def test_to_dict_has_none_tool_attention_when_not_configured(self):
        """AgentConfig.to_dict() returns None for tool_attention when disabled."""
        cfg = AgentConfig()
        d = cfg.to_dict()

        assert d["tool_attention"] is None

    def test_to_dict_includes_all_tool_attention_fields(self):
        """All ToolAttentionConfig fields appear in the serialised output."""
        ta = ToolAttentionConfig(
            k=7,
            min_tools=12,
            threshold=0.2,
            always_promote=["think"],
            collection_name="my_collection",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
        )
        cfg = AgentConfig(tool_attention=ta)
        d = cfg.to_dict()["tool_attention"]

        assert d["k"] == 7
        assert d["min_tools"] == 12
        assert d["threshold"] == 0.2
        assert d["always_promote"] == ["think"]
        assert d["collection_name"] == "my_collection"
        assert d["embedding_model"] == "all-MiniLM-L6-v2"
        assert d["embedding_dim"] == 384
