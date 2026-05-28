"""
Unit tests for tool-attention routing.

Covers:
- ToolAttentionConfig defaults and custom values
- _tool_summary() for dict and object tools
- ToolSummaryRegistry: initialize, search, failure handling
- _tool_name() and _extract_user_query() helpers
- ToolAttentionRouter.route(): filtering, always-promote, think, handoffs, k-cap
- apply_tool_attention(): lazy init, reuse, fallback to all tools
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.tools.tool_attention.config import ToolAttentionConfig
from orchestrator.tools.tool_attention.registry import ToolSummaryRegistry, _tool_summary
from orchestrator.tools.tool_attention.router import (
    ToolAttentionRouter,
    _extract_user_query,
    _tool_name,
    apply_tool_attention,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dict_tool(name: str, desc: str = "", props: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props or {}},
        },
    }


def _make_obj_tool(name: str, desc: str = "", props: dict | None = None):
    fn = MagicMock()
    fn.name = name
    fn.description = desc
    fn.parameters = {"type": "object", "properties": props or {}}
    tool = MagicMock()
    tool.function = fn
    return tool


def _make_context(metadata: dict | None = None):
    ctx = MagicMock()
    ctx.metadata = metadata if metadata is not None else {}
    return ctx


# ---------------------------------------------------------------------------
# ToolAttentionConfig
# ---------------------------------------------------------------------------


class TestToolAttentionConfig:
    def test_defaults(self):
        cfg = ToolAttentionConfig()
        assert cfg.k == 5
        assert cfg.min_tools == 10
        assert cfg.threshold == 0.0
        assert cfg.always_promote == []
        assert cfg.collection_name == "tool_attention_summaries"
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.embedding_dim == 384

    def test_custom_values(self):
        cfg = ToolAttentionConfig(k=3, min_tools=5, always_promote=["get_cart"])
        assert cfg.k == 3
        assert cfg.min_tools == 5
        assert cfg.always_promote == ["get_cart"]

    def test_always_promote_is_independent_per_instance(self):
        a = ToolAttentionConfig()
        b = ToolAttentionConfig()
        a.always_promote.append("tool_x")
        assert b.always_promote == []


# ---------------------------------------------------------------------------
# _tool_summary
# ---------------------------------------------------------------------------


class TestToolSummary:
    def test_dict_tool_with_inputs(self):
        tool = _make_dict_tool(
            "search_products",
            "Search catalogue",
            {"query": {"type": "string"}, "limit": {"type": "integer"}},
        )
        name, summary = _tool_summary(tool)
        assert name == "search_products"
        assert "search_products" in summary
        assert "Search catalogue" in summary
        assert "query" in summary
        assert "limit" in summary

    def test_dict_tool_no_inputs(self):
        tool = _make_dict_tool("get_status", "Get status")
        name, summary = _tool_summary(tool)
        assert name == "get_status"
        assert "Get status" in summary
        assert "Inputs" not in summary

    def test_object_tool(self):
        tool = _make_obj_tool("add_to_cart", "Add item", {"product_id": {"type": "string"}})
        name, summary = _tool_summary(tool)
        assert name == "add_to_cart"
        assert "add_to_cart" in summary
        assert "product_id" in summary

    def test_empty_description(self):
        tool = _make_dict_tool("noop", "")
        name, summary = _tool_summary(tool)
        assert name == "noop"
        assert "noop" in summary

    def test_dict_tool_with_none_parameters_does_not_crash(self):
        tool = {
            "type": "function",
            "function": {"name": "broken", "description": "x", "parameters": None},
        }
        name, summary = _tool_summary(tool)
        assert name == "broken"

    def test_dict_tool_with_missing_parameters_key(self):
        tool = {"type": "function", "function": {"name": "minimal", "description": "desc"}}
        name, summary = _tool_summary(tool)
        assert name == "minimal"
        assert "desc" in summary


# ---------------------------------------------------------------------------
# ToolSummaryRegistry
# ---------------------------------------------------------------------------


class TestToolSummaryRegistry:
    def _make_registry(self, **kwargs) -> ToolSummaryRegistry:
        cfg = ToolAttentionConfig(**kwargs)
        return ToolSummaryRegistry(cfg)

    def _mock_modules(self, mock_client=None, mock_encoder=None):
        """Inject fake pymilvus and sentence_transformers into sys.modules."""
        mock_pymilvus = MagicMock()
        mock_pymilvus.MilvusClient = MagicMock(return_value=mock_client or MagicMock())
        mock_pymilvus.DataType = MagicMock()

        mock_st = MagicMock()
        mock_st.SentenceTransformer = MagicMock(return_value=mock_encoder or MagicMock())

        return patch.dict(
            sys.modules,
            {
                "pymilvus": mock_pymilvus,
                "sentence_transformers": mock_st,
            },
        )

    @pytest.mark.asyncio
    async def test_initialize_creates_collection_and_upserts(self):
        registry = self._make_registry()
        tools = [_make_dict_tool("search", "Search"), _make_dict_tool("buy", "Buy")]

        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock()
        mock_encoder.encode.return_value.tolist.return_value = [[0.1] * 384, [0.2] * 384]

        with (
            self._mock_modules(mock_client, mock_encoder),
            patch("orchestrator.config.settings") as mock_settings,
        ):
            mock_settings.milvus_host = "localhost"
            mock_settings.milvus_port = 19530
            mock_settings.milvus_token = None
            await registry.initialize(tools)

        assert registry.ready
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_create_if_collection_exists(self):
        registry = self._make_registry()
        tools = [_make_dict_tool("search", "Search")]

        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock()
        mock_encoder.encode.return_value.tolist.return_value = [[0.1] * 384]

        with (
            self._mock_modules(mock_client, mock_encoder),
            patch("orchestrator.config.settings") as mock_settings,
        ):
            mock_settings.milvus_host = "localhost"
            mock_settings.milvus_port = 19530
            mock_settings.milvus_token = None
            await registry.initialize(tools)

        mock_client.create_collection.assert_not_called()
        assert registry.ready

    @pytest.mark.asyncio
    async def test_initialize_failure_leaves_registry_not_ready(self):
        registry = self._make_registry()

        # Simulate MilvusClient raising on construction
        mock_pymilvus = MagicMock()
        mock_pymilvus.MilvusClient = MagicMock(side_effect=Exception("conn refused"))
        mock_pymilvus.DataType = MagicMock()

        with (
            patch.dict(sys.modules, {"pymilvus": mock_pymilvus}),
            patch("orchestrator.config.settings") as mock_settings,
        ):
            mock_settings.milvus_host = "localhost"
            mock_settings.milvus_port = 19530
            mock_settings.milvus_token = None
            await registry.initialize([_make_dict_tool("x", "x")])

        assert not registry.ready

    def test_search_returns_empty_when_not_ready(self):
        registry = self._make_registry()
        assert registry.search("find dog food", k=3) == []

    def test_search_returns_tool_names(self):
        registry = self._make_registry()
        registry._ready = True
        mock_client = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock()
        mock_encoder.encode.return_value.tolist.return_value = [[0.1] * 384]
        mock_client.search.return_value = [
            [
                {"entity": {"tool_name": "search_products"}},
                {"entity": {"tool_name": "add_to_cart"}},
            ]
        ]
        registry._client = mock_client
        registry._encoder = mock_encoder

        result = registry.search("find dog food", k=2)
        assert result == ["search_products", "add_to_cart"]

    def test_search_returns_empty_on_milvus_error(self):
        registry = self._make_registry()
        registry._ready = True
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("milvus down")
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock()
        mock_encoder.encode.return_value.tolist.return_value = [[0.1] * 384]
        registry._client = mock_client
        registry._encoder = mock_encoder

        result = registry.search("find dog food", k=2)
        assert result == []

    def test_refresh_calls_upsert_when_ready(self):
        registry = self._make_registry()
        registry._ready = True
        mock_client = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock()
        mock_encoder.encode.return_value.tolist.return_value = [[0.1] * 384]
        registry._client = mock_client
        registry._encoder = mock_encoder

        registry.refresh([_make_dict_tool("search", "Search")])

        mock_client.upsert.assert_called_once()

    def test_refresh_is_noop_when_not_ready(self):
        registry = self._make_registry()
        # registry._ready is False by default
        mock_client = MagicMock()
        registry._client = mock_client

        registry.refresh([_make_dict_tool("search", "Search")])

        mock_client.upsert.assert_not_called()

    def test_refresh_is_noop_for_empty_list(self):
        registry = self._make_registry()
        registry._ready = True
        mock_client = MagicMock()
        registry._client = mock_client

        registry.refresh([])

        mock_client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# ToolAttentionRouter.initialize()
# ---------------------------------------------------------------------------


class TestToolAttentionRouterInitialize:
    @pytest.mark.asyncio
    async def test_initialize_calls_registry_with_tool_defs(self):
        cfg = ToolAttentionConfig(k=3, min_tools=3)
        router = ToolAttentionRouter(cfg)
        router._registry = MagicMock()
        router._registry.initialize = AsyncMock()

        tools = [_make_dict_tool("search", "Search"), _make_dict_tool("buy", "Buy")]
        await router.initialize(tools)

        router._registry.initialize.assert_called_once_with(tools)
        assert router._initialized is True

    @pytest.mark.asyncio
    async def test_initialized_flag_set_even_if_registry_fails(self):
        cfg = ToolAttentionConfig()
        router = ToolAttentionRouter(cfg)
        router._registry = MagicMock()
        # Registry.initialize raises — but router._initialized should still be set
        # because the failure is caught inside registry.initialize (fail-open)
        router._registry.initialize = AsyncMock()
        router._registry.ready = False  # registry didn't become ready

        await router.initialize([_make_dict_tool("x", "x")])

        assert router._initialized is True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestToolName:
    def test_dict_tool(self):
        assert _tool_name(_make_dict_tool("search_products", "")) == "search_products"

    def test_object_tool(self):
        tool = _make_obj_tool("add_to_cart", "")
        assert _tool_name(tool) == "add_to_cart"

    def test_empty_dict(self):
        assert _tool_name({}) == ""


class TestExtractUserQuery:
    def test_simple_user_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "find me dog food"},
        ]
        assert _extract_user_query(messages) == "find me dog food"

    def test_returns_most_recent_user_message(self):
        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second message"},
        ]
        assert _extract_user_query(messages) == "second message"

    def test_content_as_list(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "find dog food"}]},
        ]
        assert _extract_user_query(messages) == "find dog food"

    def test_no_user_message_returns_empty(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        assert _extract_user_query(messages) == ""

    def test_empty_messages(self):
        assert _extract_user_query([]) == ""


# ---------------------------------------------------------------------------
# ToolAttentionRouter.route()
# ---------------------------------------------------------------------------


def _make_router_with_mock_registry(
    search_returns: list[str],
    k: int = 5,
    min_tools: int = 3,
    always_promote: list[str] | None = None,
) -> ToolAttentionRouter:
    cfg = ToolAttentionConfig(k=k, min_tools=min_tools, always_promote=always_promote or [])
    router = ToolAttentionRouter(cfg)
    router._initialized = True
    router._registry = MagicMock()
    router._registry.ready = True
    router._registry.search.return_value = search_returns
    return router


class TestToolAttentionRouterRoute:
    def _messages(self, query: str = "find dog food") -> list[dict]:
        return [{"role": "user", "content": query}]

    def _tools(self, names: list[str]) -> list[dict]:
        return [_make_dict_tool(n, f"Does {n}") for n in names]

    def test_returns_none_when_not_initialized(self):
        cfg = ToolAttentionConfig(min_tools=3)
        router = ToolAttentionRouter(cfg)
        router._initialized = False
        result = router.route(self._messages(), self._tools(["a", "b", "c", "d"]), _make_context())
        assert result is None

    def test_returns_none_below_min_tools(self):
        router = _make_router_with_mock_registry(["a"], min_tools=10)
        tools = self._tools(["a", "b", "c"])  # only 3 tools, below min_tools=10
        result = router.route(self._messages(), tools, _make_context())
        assert result is None

    def test_returns_none_when_no_user_query(self):
        router = _make_router_with_mock_registry(["a", "b"], min_tools=3)
        messages = [{"role": "system", "content": "You are helpful."}]
        result = router.route(messages, self._tools(["a", "b", "c", "d"]), _make_context())
        assert result is None

    def test_filters_to_routed_tools(self):
        tools = self._tools(["search", "add_to_cart", "checkout", "get_cart", "delete"])
        router = _make_router_with_mock_registry(["search", "add_to_cart"], min_tools=3)
        result = router.route(self._messages(), tools, _make_context())
        names = [_tool_name(t) for t in result]
        assert "search" in names
        assert "add_to_cart" in names
        assert "checkout" not in names
        assert "delete" not in names

    def test_always_includes_think_tool(self):
        tools = self._tools(["search", "add_to_cart", "think", "checkout", "get_cart"])
        router = _make_router_with_mock_registry(["search"], min_tools=3)
        result = router.route(self._messages(), tools, _make_context())
        names = [_tool_name(t) for t in result]
        assert "think" in names

    def test_always_includes_handoff_tools(self):
        tools = self._tools(["search", "transfer_to_billing", "checkout", "get_cart", "delete"])
        router = _make_router_with_mock_registry(["search"], min_tools=3)
        result = router.route(self._messages(), tools, _make_context())
        names = [_tool_name(t) for t in result]
        assert "transfer_to_billing" in names

    def test_always_promote_config_respected(self):
        tools = self._tools(["search", "get_cart", "checkout", "delete", "update"])
        router = _make_router_with_mock_registry(
            ["search"], min_tools=3, always_promote=["get_cart"]
        )
        result = router.route(self._messages(), tools, _make_context())
        names = [_tool_name(t) for t in result]
        assert "get_cart" in names

    def test_stores_promoted_set_in_context_metadata(self):
        tools = self._tools(["search", "add_to_cart", "checkout", "get_cart", "delete"])
        router = _make_router_with_mock_registry(["search", "add_to_cart"], min_tools=3)
        ctx = _make_context({})
        router.route(self._messages(), tools, ctx)
        assert "promoted_tools" in ctx.metadata
        assert "search" in ctx.metadata["promoted_tools"]
        assert "add_to_cart" in ctx.metadata["promoted_tools"]

    def test_k_is_capped_at_total_tools(self):
        tools = self._tools(["a", "b", "c"])  # 3 tools
        router = _make_router_with_mock_registry([], min_tools=3, k=100)
        router.route(self._messages(), tools, _make_context())
        # search should have been called with k=3, not k=100
        call_args = router._registry.search.call_args
        assert call_args[0][1] == 3  # k argument


# ---------------------------------------------------------------------------
# apply_tool_attention()
# ---------------------------------------------------------------------------


class TestApplyToolAttention:
    def _make_agent(self, tool_attention=None, tools=None):
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.tool_attention = tool_attention
        agent.metadata = {}
        all_tools = tools or [_make_dict_tool(f"tool_{i}", f"Tool {i}") for i in range(12)]
        agent.get_tools_for_llm.return_value = all_tools
        return agent

    def _messages(self):
        return [{"role": "user", "content": "find dog food"}]

    @pytest.mark.asyncio
    async def test_returns_none_when_tool_attention_not_configured(self):
        agent = self._make_agent(tool_attention=None)
        ctx = _make_context()
        result = await apply_tool_attention(agent, self._messages(), ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_router_on_first_call(self):
        agent = self._make_agent(tool_attention=ToolAttentionConfig(min_tools=5))

        with patch("orchestrator.tools.tool_attention.router.ToolAttentionRouter") as MockRouter:
            mock_router = MagicMock()
            mock_router._initialized = True
            mock_router.initialize = AsyncMock()  # initialize is async, must be AsyncMock
            mock_router.route.return_value = [_make_dict_tool("search", "")]
            MockRouter.return_value = mock_router

            await apply_tool_attention(agent, self._messages(), _make_context())

        MockRouter.assert_called_once()
        assert "_tool_attention_router" in agent.metadata

    @pytest.mark.asyncio
    async def test_reuses_router_on_subsequent_calls(self):
        agent = self._make_agent(tool_attention=ToolAttentionConfig(min_tools=5))

        mock_router = MagicMock()
        mock_router._initialized = True
        mock_router.route.return_value = [_make_dict_tool("search", "")]
        agent.metadata["_tool_attention_router"] = mock_router

        await apply_tool_attention(agent, self._messages(), _make_context())
        await apply_tool_attention(agent, self._messages(), _make_context())

        # initialize should not be called since router already exists
        mock_router.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_on_routing_exception(self):
        agent = self._make_agent(tool_attention=ToolAttentionConfig(min_tools=5))

        mock_router = MagicMock()
        mock_router._initialized = True
        mock_router.route.side_effect = Exception("unexpected error")
        agent.metadata["_tool_attention_router"] = mock_router

        result = await apply_tool_attention(agent, self._messages(), _make_context())
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_filtered_tools_from_router(self):
        agent = self._make_agent(tool_attention=ToolAttentionConfig(min_tools=5))
        filtered = [_make_dict_tool("search", ""), _make_dict_tool("add_to_cart", "")]

        mock_router = MagicMock()
        mock_router._initialized = True
        mock_router.route.return_value = filtered
        agent.metadata["_tool_attention_router"] = mock_router

        result = await apply_tool_attention(agent, self._messages(), _make_context())
        assert result == filtered

    @pytest.mark.asyncio
    async def test_returns_none_when_router_route_returns_none(self):
        agent = self._make_agent(
            tool_attention=ToolAttentionConfig(min_tools=100),  # min_tools larger than actual
        )
        mock_router = MagicMock()
        mock_router._initialized = True
        mock_router.route.return_value = None
        agent.metadata["_tool_attention_router"] = mock_router

        result = await apply_tool_attention(agent, self._messages(), _make_context())
        assert result is None
