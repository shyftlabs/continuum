"""
Unit tests for ToolExecutor internals.

Covers:
- Fix 2: json_path in ToolContextVariable is now resolved during context capture
"""

from __future__ import annotations

import pytest

from orchestrator.tools.executor import ToolExecutor
from orchestrator.tools.types import ToolContextConfig, ToolContextVariable


# ---------------------------------------------------------------------------
# _resolve_json_path
# ---------------------------------------------------------------------------


class TestResolveJsonPath:
    def test_flat_key(self):
        assert ToolExecutor._resolve_json_path({"a": 1}, "a") == 1

    def test_nested_two_levels(self):
        assert ToolExecutor._resolve_json_path(
            {"result": {"session_id": "abc"}}, "result.session_id"
        ) == "abc"

    def test_nested_three_levels(self):
        assert ToolExecutor._resolve_json_path(
            {"a": {"b": {"c": 42}}}, "a.b.c"
        ) == 42

    def test_missing_top_level_key_returns_none(self):
        assert ToolExecutor._resolve_json_path({"a": 1}, "b") is None

    def test_missing_nested_key_returns_none(self):
        assert ToolExecutor._resolve_json_path({"a": {}}, "a.b") is None

    def test_non_dict_node_returns_none(self):
        assert ToolExecutor._resolve_json_path({"a": "string"}, "a.b") is None

    def test_none_value_at_key_returns_none(self):
        assert ToolExecutor._resolve_json_path({"a": None}, "a") is None

    def test_empty_dict(self):
        assert ToolExecutor._resolve_json_path({}, "a") is None


# ---------------------------------------------------------------------------
# _capture_context_variables with json_path
# ---------------------------------------------------------------------------


def _make_executor_with_config(config: ToolContextConfig) -> ToolExecutor:
    executor = ToolExecutor()
    return executor


def _make_server(config: ToolContextConfig, name: str = "test-server"):
    from unittest.mock import MagicMock
    server = MagicMock()
    server.name = name
    server.context_config = config
    return server


class TestCaptureContextVariablesJsonPath:
    def test_captures_nested_value_via_json_path(self):
        """Variable with json_path extracts from nested response."""
        config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="session_id",
                    json_path="result.session_id",
                )
            ],
            auto_capture_common=False,
        )
        executor = ToolExecutor()
        server = _make_server(config)
        result_json = '{"result": {"session_id": "abc-123"}}'

        executor._capture_context_variables(server, "create_session", result_json)

        assert executor._context_state.get("test-server", "session_id") == "abc-123"

    def test_json_path_missing_in_response_captures_nothing(self):
        """If json_path points to a missing key, variable is not stored."""
        config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="session_id",
                    json_path="result.session_id",
                )
            ],
            auto_capture_common=False,
        )
        executor = ToolExecutor()
        server = _make_server(config)
        result_json = '{"result": {}}'

        executor._capture_context_variables(server, "create_session", result_json)

        assert executor._context_state.get("test-server", "session_id") is None

    def test_flat_capture_still_works_without_json_path(self):
        """Existing flat capture is unaffected when no json_path is set."""
        config = ToolContextConfig(auto_capture_common=True)
        executor = ToolExecutor()
        server = _make_server(config)
        result_json = '{"session_id": "flat-value"}'

        executor._capture_context_variables(server, "any_tool", result_json)

        assert executor._context_state.get("test-server", "session_id") == "flat-value"

    def test_json_path_variable_not_double_captured_by_flat_scan(self):
        """A variable captured via json_path is not also processed by the flat scan."""
        config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="session_id",
                    json_path="data.session_id",
                )
            ],
            auto_capture_common=True,
        )
        executor = ToolExecutor()
        server = _make_server(config)
        # Response has session_id at top level AND nested — json_path wins
        result_json = '{"session_id": "top_level", "data": {"session_id": "nested"}}'

        executor._capture_context_variables(server, "any_tool", result_json)

        # json_path takes precedence
        assert executor._context_state.get("test-server", "session_id") == "nested"

    def test_json_path_and_flat_variables_coexist(self):
        """json_path variable and a separate flat variable are both captured."""
        config = ToolContextConfig(
            variables=[
                ToolContextVariable(
                    name="token",
                    json_path="auth.token",
                    sensitive=True,
                )
            ],
            auto_capture_common=True,
        )
        executor = ToolExecutor()
        server = _make_server(config)
        result_json = '{"session_id": "sess-1", "auth": {"token": "tok-abc"}}'

        executor._capture_context_variables(server, "login", result_json)

        assert executor._context_state.get("test-server", "session_id") == "sess-1"
        assert executor._context_state.get("test-server", "token") == "tok-abc"
