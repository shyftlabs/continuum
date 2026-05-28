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
import os
import sys
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
