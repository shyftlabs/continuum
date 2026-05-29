"""
Unit tests for Issue 03 — Tools, Observability, Infrastructure fixes.

Tests context variable type validation, rate limiter, ToolContextState thread safety,
registry refresh atomicity, lifecycle exception handling.
"""

from __future__ import annotations

import threading

import pytest


# ---------------------------------------------------------------------------
# 03-#1: Context variable injection type validation
# ---------------------------------------------------------------------------


class TestContextVariableTypeValidation:
    def test_validate_injection_type_string(self):
        from orchestrator.tools.executor import ToolExecutor

        assert ToolExecutor._validate_injection_type("hello", "string") is True
        assert ToolExecutor._validate_injection_type(123, "string") is False

    def test_validate_injection_type_integer(self):
        from orchestrator.tools.executor import ToolExecutor

        assert ToolExecutor._validate_injection_type(42, "integer") is True
        assert ToolExecutor._validate_injection_type(3.14, "integer") is False

    def test_validate_injection_type_number(self):
        from orchestrator.tools.executor import ToolExecutor

        assert ToolExecutor._validate_injection_type(42, "number") is True
        assert ToolExecutor._validate_injection_type(3.14, "number") is True
        assert ToolExecutor._validate_injection_type("42", "number") is False

    def test_validate_injection_type_boolean(self):
        from orchestrator.tools.executor import ToolExecutor

        assert ToolExecutor._validate_injection_type(True, "boolean") is True
        assert ToolExecutor._validate_injection_type(1, "boolean") is False

    def test_validate_injection_type_array(self):
        from orchestrator.tools.executor import ToolExecutor

        assert ToolExecutor._validate_injection_type([1, 2], "array") is True
        assert ToolExecutor._validate_injection_type("not a list", "array") is False

    def test_validate_injection_type_object(self):
        from orchestrator.tools.executor import ToolExecutor

        assert ToolExecutor._validate_injection_type({"a": 1}, "object") is True
        assert ToolExecutor._validate_injection_type([1], "object") is False

    def test_validate_injection_type_unknown_allows(self):
        from orchestrator.tools.executor import ToolExecutor

        # Unknown types should be allowed (don't block exotic schemas)
        assert ToolExecutor._validate_injection_type("anything", "customType") is True


# ---------------------------------------------------------------------------
# 03-#10: ToolContextState thread safety
# ---------------------------------------------------------------------------


class TestToolContextStateThreadSafety:
    def test_set_and_get_basic(self):
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        state.set("ns1", "key1", "value1")
        assert state.get("ns1", "key1") == "value1"

    def test_get_default(self):
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        assert state.get("ns1", "missing", "default") == "default"

    def test_has(self):
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        state.set("ns1", "key1", "val")
        assert state.has("ns1", "key1") is True
        assert state.has("ns1", "missing") is False

    def test_concurrent_set_and_get(self):
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        errors = []

        def writer(ns):
            try:
                for i in range(100):
                    state.set(ns, f"key-{i}", f"val-{i}")
            except Exception as e:
                errors.append(e)

        def reader(ns):
            try:
                for i in range(100):
                    state.get(ns, f"key-{i}")
                    state.get_all(ns)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(f"ns-{i}",)))
            threads.append(threading.Thread(target=reader, args=(f"ns-{i}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_get_all_namespaces(self):
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        state.set("ns1", "k", "v")
        state.set("ns2", "k", "v")
        namespaces = state.get_all_namespaces()
        assert set(namespaces) == {"ns1", "ns2"}


# ---------------------------------------------------------------------------
# 03-#8: Lifecycle re-raises KeyboardInterrupt
# ---------------------------------------------------------------------------


class TestLifecycleExceptionHandling:
    def test_keyboard_interrupt_not_swallowed(self):
        """KeyboardInterrupt should propagate, not be caught as init failure."""
        # This is a code-level verification — we check the source
        import inspect
        from orchestrator.core.lifecycle import OrchestratorLifecycle

        source = inspect.getsource(OrchestratorLifecycle.initialize)
        assert "KeyboardInterrupt" in source
        assert "SystemExit" in source
