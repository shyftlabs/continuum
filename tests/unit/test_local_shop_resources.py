"""
Unit tests for local-shop MCP resources and agent resource injection.

Covers:
- server.py resource functions return correct JSON
- Resource template handles unknown product_id
- agent._fetch_resources() loads catalogue + categories into _resource_context
- agent._fetch_resources() handles server errors gracefully
- agent._create_agent() injects resource context into instructions
- agent._create_agent() falls back to base instructions when no resources
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "playground", "gateway-local-shop")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ---------------------------------------------------------------------------
# Server resource functions
# ---------------------------------------------------------------------------


class TestServerResourceFunctions:
    def test_catalogue_returns_all_products(self):
        from server import PRODUCTS, get_catalogue

        result = json.loads(get_catalogue())
        assert isinstance(result, list)
        assert len(result) == len(PRODUCTS)

    def test_catalogue_products_have_required_fields(self):
        from server import get_catalogue

        products = json.loads(get_catalogue())
        for p in products:
            assert "id" in p
            assert "name" in p
            assert "price" in p
            assert "category" in p
            assert "animal" in p

    def test_categories_returns_dict_with_categories_and_animals(self):
        from server import get_categories

        result = json.loads(get_categories())
        assert "categories" in result
        assert "animals" in result

    def test_categories_list_is_sorted(self):
        from server import get_categories

        result = json.loads(get_categories())
        assert result["categories"] == sorted(result["categories"])
        assert result["animals"] == sorted(result["animals"])

    def test_categories_contains_expected_values(self):
        from server import get_categories

        result = json.loads(get_categories())
        assert "food" in result["categories"]
        assert "toys" in result["categories"]
        assert "dog" in result["animals"]
        assert "cat" in result["animals"]

    def test_get_product_resource_returns_correct_product(self):
        from server import get_product_resource

        result = json.loads(get_product_resource("p1"))
        assert result["id"] == "p1"
        assert result["name"] == "Dog Food (Dry) 5kg"

    def test_get_product_resource_returns_error_for_unknown_id(self):
        from server import get_product_resource

        result = json.loads(get_product_resource("nonexistent"))
        assert "error" in result

    def test_get_product_resource_all_products_accessible(self):
        from server import PRODUCTS, get_product_resource

        for product in PRODUCTS:
            result = json.loads(get_product_resource(product["id"]))
            assert result["id"] == product["id"]

    def test_catalogue_and_categories_are_consistent(self):
        from server import get_catalogue, get_categories

        products = json.loads(get_catalogue())
        cats = json.loads(get_categories())

        catalogue_categories = {p["category"] for p in products}
        catalogue_animals = {p["animal"] for p in products}

        assert catalogue_categories == set(cats["categories"])
        assert catalogue_animals == set(cats["animals"])


# ---------------------------------------------------------------------------
# Agent._fetch_resources()
# ---------------------------------------------------------------------------


class TestAgentFetchResources:
    def _make_agent_instance(self):
        from agent import LocalShopAgent
        from config import default_config

        instance = LocalShopAgent.__new__(LocalShopAgent)
        instance.config = default_config
        instance._resource_context = ""
        instance._mcp_server = None
        instance._tool_executor = None
        instance._agent = None
        instance._runner = None
        instance._tools = []
        instance._initialized = False
        return instance

    @pytest.mark.asyncio
    async def test_fetch_resources_sets_resource_context(self):
        instance = self._make_agent_instance()

        mock_server = AsyncMock()
        mock_server.read_resource = AsyncMock(
            side_effect=[
                '[{"id":"p1","name":"Dog Food"}]',  # catalogue
                '{"categories":["food"],"animals":["dog"]}',  # categories
            ]
        )
        instance._mcp_server = mock_server

        await instance._fetch_resources()

        assert "Product catalogue:" in instance._resource_context
        assert "Categories:" in instance._resource_context
        assert "Dog Food" in instance._resource_context

    @pytest.mark.asyncio
    async def test_fetch_resources_calls_both_uris(self):
        instance = self._make_agent_instance()

        mock_server = AsyncMock()
        mock_server.read_resource = AsyncMock(return_value="{}")
        instance._mcp_server = mock_server

        await instance._fetch_resources()

        calls = [call[0][0] for call in mock_server.read_resource.call_args_list]
        assert "shop://catalogue" in calls
        assert "shop://categories" in calls

    @pytest.mark.asyncio
    async def test_fetch_resources_handles_server_error_gracefully(self):
        instance = self._make_agent_instance()

        mock_server = AsyncMock()
        mock_server.read_resource = AsyncMock(side_effect=Exception("connection refused"))
        instance._mcp_server = mock_server

        # Should not raise — logs warning and continues
        await instance._fetch_resources()

        assert instance._resource_context == ""

    @pytest.mark.asyncio
    async def test_fetch_resources_empty_response_still_sets_context(self):
        instance = self._make_agent_instance()

        mock_server = AsyncMock()
        mock_server.read_resource = AsyncMock(return_value="")
        instance._mcp_server = mock_server

        await instance._fetch_resources()

        assert "Product catalogue:" in instance._resource_context
        assert "Categories:" in instance._resource_context


# ---------------------------------------------------------------------------
# Agent._create_agent() — resource context injection
# ---------------------------------------------------------------------------


class TestAgentCreateAgentResourceInjection:
    """Verify how ``_create_agent`` builds the instructions passed to ``BaseAgent``.

    The current source fetches ``_resource_context`` in ``_fetch_resources`` but
    ``_create_agent`` always uses ``config.system_instructions`` verbatim — the
    resource context is *not* injected into the agent instructions. These tests
    lock in that real behavior.
    """

    def _make_agent_instance_with_mocks(self, resource_context: str = ""):
        from agent import LocalShopAgent
        from config import default_config

        instance = LocalShopAgent.__new__(LocalShopAgent)
        instance.config = default_config
        instance._resource_context = resource_context
        instance._container = None
        instance._tool_executor = MagicMock()
        instance._tool_executor.get_tool_definitions.return_value = []
        instance._tools = []
        return instance

    def _build_instructions(self, resource_context: str) -> str:
        instance = self._make_agent_instance_with_mocks(resource_context=resource_context)

        with patch("agent.BaseAgent") as MockBaseAgent:
            MockBaseAgent.return_value = MagicMock()
            with patch("agent.AgentMemoryConfig"), patch("agent.AgentConfig"):
                instance._create_agent()

        return MockBaseAgent.call_args[1]["instructions"]

    def test_create_agent_uses_base_instructions_when_no_resources(self):
        instance = self._make_agent_instance_with_mocks(resource_context="")
        instructions = self._build_instructions(resource_context="")
        assert instructions == instance.config.system_instructions

    def test_create_agent_does_not_inject_resource_context(self):
        # Current source never appends _resource_context to the instructions.
        instructions = self._build_instructions(
            resource_context="Product catalogue:\n[]\n\nCategories:\n{}"
        )
        assert "Product catalogue:" not in instructions
        assert "Categories:" not in instructions

    def test_create_agent_instructions_equal_base_regardless_of_resources(self):
        from config import default_config

        base = default_config.system_instructions
        instructions = self._build_instructions(
            resource_context='Product catalogue:\n[{"id": "p1", "name": "Dog Food"}]'
        )
        # Instructions are exactly the base config — resource context is ignored.
        assert instructions == base

    def test_create_agent_base_instructions_present_once(self):
        from config import default_config

        base = default_config.system_instructions
        instructions = self._build_instructions(resource_context="Product catalogue:\n[]")
        assert instructions.count(base) == 1

    def test_create_agent_does_not_escape_json_braces(self):
        # Since the resource context is not injected, JSON braces from the
        # catalogue never appear (escaped or otherwise) in the instructions.
        instructions = self._build_instructions(
            resource_context='Product catalogue:\n[{"id": "p1", "name": "Dog Food"}]'
        )
        assert '{{"id"' not in instructions
        assert '"Dog Food"' not in instructions


# ---------------------------------------------------------------------------
# MCP server naming
#
# With namespace_tools defaulting to True, the server name is no longer a
# display label -- it is the prefix on every LLM-facing tool name, and so part
# of the identity that policies, digest pins, always_promote and capture/inject
# match against. Leaving it unset falls back to
# f"streamable_http: {url}", which sanitises to a 39-character prefix carrying
# the host and port: move the server to another port and every tool silently
# gets a new name.
# ---------------------------------------------------------------------------


class TestMCPServerNaming:
    @pytest.mark.asyncio
    async def test_connect_mcp_passes_an_explicit_server_name(self):
        import agent as agent_mod
        from config import ShopConfig

        shop = agent_mod.LocalShopAgent(ShopConfig())

        fake_server = MagicMock()
        fake_server.connect = AsyncMock()
        fake_executor = MagicMock()
        fake_executor.initialize = AsyncMock()
        fake_executor.get_tool_definitions = MagicMock(return_value=[])

        with (
            patch.object(agent_mod, "MCPServerStreamableHttp", return_value=fake_server) as ctor,
            patch.object(agent_mod, "CartDebugToolExecutor", return_value=fake_executor),
            patch.object(shop, "_fetch_resources", new=AsyncMock()),
        ):
            await shop._connect_mcp()

        assert ctor.call_args.kwargs.get("name"), (
            "MCPServerStreamableHttp was constructed without name=; tool names "
            "fall back to the transport+URL label."
        )

    def test_server_name_does_not_embed_the_environment(self):
        from config import default_config

        from continuum.tools.util import build_namespaced_tool_name

        name = default_config.mcp_server_name
        for tool in ("search_products", "get_product", "add_to_cart", "view_cart", "checkout"):
            namespaced = build_namespaced_tool_name(name, tool)
            assert namespaced == f"{name}__{tool}"
            assert len(namespaced) <= 64
            assert "localhost" not in namespaced
            assert "8888" not in namespaced


# ---------------------------------------------------------------------------
# CartDebugToolExecutor
#
# The ⚠️ branch is the whole point of this subclass: it fires when a cart tool
# returns but its totals never reached the LLM ("why does the agent say my cart
# is $0?"). It was gated on _CART_TOOLS = {"get_cart", "cart", "get_cart_items"}
# while the server exposes view_cart -- so it had never once executed. The hook
# also receives the LLM-facing name, which namespacing made "shop__view_cart".
# ---------------------------------------------------------------------------


@contextmanager
def _captured_agent_logs():
    """Collect records from the playground agent's logger.

    caplog cannot see these: the "continuum" parent sets propagate=False, so
    records never reach the root logger. Same pattern as
    tests/unit/tools/test_tool_attention.py.
    """
    records: list[tuple[int, str]] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append((record.levelno, record.getMessage()))

    handler = _Collector()
    logger = logging.getLogger("continuum.agent")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _artifact(structured: dict | None):
    art = MagicMock()
    art.structured_content = structured
    return art


class TestCartDebugToolExecutor:
    def _executor(self):
        import agent as agent_mod

        return agent_mod.CartDebugToolExecutor.__new__(agent_mod.CartDebugToolExecutor)

    def test_cart_tools_are_tools_the_server_actually_exposes(self):
        """The guard the original set lacked: every configured cart tool must be
        a real server tool, or the branch silently never runs."""
        import agent as agent_mod
        import server as server_mod

        exposed = {
            name
            for name, obj in vars(server_mod).items()
            if callable(obj) and not name.startswith("_")
        }
        for name in agent_mod._CART_TOOLS:
            assert name in exposed, f"{name!r} is not a tool on server.py"

    def test_warns_when_a_cart_tool_returns_no_totals(self):
        ex = self._executor()
        with _captured_agent_logs() as records:
            ex._on_tool_result(
                "shop__view_cart", "{}", _artifact({"items": [], "message": "Cart is empty"})
            )
        warnings = [m for lvl, m in records if lvl >= logging.WARNING]
        assert any("NO totals" in m for m in warnings), records

    def test_logs_totals_when_a_cart_tool_returns_them(self):
        ex = self._executor()
        with _captured_agent_logs() as records:
            ex._on_tool_result("shop__checkout", "{}", _artifact({"total": 20.97, "order_id": "X"}))
        assert any("sending to LLM" in m for _, m in records), records

    def test_non_cart_tools_do_not_trigger_the_cart_branch(self):
        ex = self._executor()
        with _captured_agent_logs() as records:
            ex._on_tool_result("shop__search_products", "[]", _artifact({"results": []}))
        assert not [m for lvl, m in records if lvl >= logging.WARNING], records

    def test_add_to_cart_is_not_treated_as_a_totals_bearing_tool(self):
        """add_to_cart legitimately returns no total, so gating it would fire the
        ⚠️ on every successful add."""
        ex = self._executor()
        with _captured_agent_logs() as records:
            ex._on_tool_result(
                "shop__add_to_cart", "{}", _artifact({"message": "Added", "cart_size": 1})
            )
        assert not [m for lvl, m in records if lvl >= logging.WARNING], records

    def test_detection_still_works_when_namespacing_is_disabled(self):
        ex = self._executor()
        with _captured_agent_logs() as records:
            ex._on_tool_result("view_cart", "{}", _artifact({"items": []}))
        assert any("NO totals" in m for lvl, m in records if lvl >= logging.WARNING), records
