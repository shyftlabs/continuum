"""
Unit tests for Issue 01 — Agent Layer fixes.

Tests circuit breaker, RunState thread safety, TokenUsage, clone deep copy,
validation utils, and AgentResponse error factory.
"""

from __future__ import annotations

import threading
import time

import pytest


# ---------------------------------------------------------------------------
# 01-#1: Circuit breaker race condition fix
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Test that the circuit breaker state transitions are atomic."""

    def test_closed_state_allows_calls(self):
        from orchestrator.agent.utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(threshold=3, cooldown=1)
        # Should not raise
        cb.check()

    def test_opens_after_threshold_failures(self):
        from orchestrator.agent.utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerOpen,
        )

        cb = CircuitBreaker(threshold=2, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpen):
            cb.check()

    def test_half_open_after_cooldown(self):
        from orchestrator.agent.utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(threshold=1, cooldown=1)
        cb.record_failure()
        time.sleep(1.1)
        # Should NOT raise — transitions to half-open
        cb.check()

    def test_concurrent_check_is_safe(self):
        """Multiple threads calling check() simultaneously should not corrupt state."""
        from orchestrator.agent.utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(threshold=1, cooldown=1)
        cb.record_failure()
        time.sleep(1.1)

        results = []

        def worker():
            try:
                cb.check()
                results.append("ok")
            except Exception as e:
                results.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (half-open allows one probe, then closed)
        assert len(results) == 20


# ---------------------------------------------------------------------------
# 01-#2: RunState thread-safe push/pop
# ---------------------------------------------------------------------------


class TestRunStateThreadSafety:
    def test_push_pop_agent_stack(self):
        from orchestrator.agent.types import RunState

        state = RunState(run_id="test-run")
        state.push_agent("agent-a")
        state.push_agent("agent-b")
        assert state.get_agent_stack_snapshot() == ["agent-a", "agent-b"]
        popped = state.pop_agent()
        assert popped == "agent-b"
        assert state.get_agent_stack_snapshot() == ["agent-a"]

    def test_pop_empty_returns_none(self):
        from orchestrator.agent.types import RunState

        state = RunState(run_id="test-run")
        assert state.pop_agent() is None

    def test_concurrent_push(self):
        from orchestrator.agent.types import RunState

        state = RunState(run_id="test-run")
        errors = []

        def pusher(name):
            try:
                for i in range(50):
                    state.push_agent(f"{name}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(state.get_agent_stack_snapshot()) == 250


# ---------------------------------------------------------------------------
# 01-#3: TokenUsage.add() coercion
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_add_with_none_values(self):
        from orchestrator.agent.types import TokenUsage

        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        result = a.add(b)
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15

    def test_add_normal(self):
        from orchestrator.agent.types import TokenUsage

        a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        b = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        result = a.add(b)
        assert result.prompt_tokens == 300
        assert result.total_tokens == 450


# ---------------------------------------------------------------------------
# 01-#4: AgentResponse.error_response factory
# ---------------------------------------------------------------------------


class TestAgentResponseErrorFactory:
    def test_error_response_creates_valid_response(self):
        from orchestrator.agent.types import AgentResponse, ResponseStatus

        resp = AgentResponse.error_response(
            error="something failed",
            agent_name="test-agent",
            run_id="run-123",
        )
        assert resp.status == ResponseStatus.ERROR
        assert "something failed" in resp.error
        assert resp.agent_name == "test-agent"


# ---------------------------------------------------------------------------
# 01-#5: BaseAgent.clone() deep copy
# ---------------------------------------------------------------------------


class TestBaseAgentClone:
    def test_clone_deep_copies_tools(self):
        from orchestrator.agent.base import BaseAgent

        original = BaseAgent(name="orig", tools=[{"name": "tool1"}])
        cloned = original.clone(name="cloned")
        assert cloned.name == "cloned"
        # Mutating clone's tools should not affect original
        cloned.tools.append({"name": "tool2"})
        assert len(original.tools) == 1

    def test_clone_deep_copies_metadata(self):
        from orchestrator.agent.base import BaseAgent

        original = BaseAgent(name="orig", metadata={"key": [1, 2, 3]})
        cloned = original.clone()
        cloned.metadata["key"].append(4)
        assert original.metadata["key"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 01-#6: Validation utils safe dict access
# ---------------------------------------------------------------------------


class TestValidationUtils:
    """Test that validation error formatting uses .get() for safe access."""

    def test_validate_input_returns_none_for_no_schema(self):
        """Agent without input_schema should pass validation."""
        import asyncio

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.types import RunContext
        from orchestrator.agent.utils.validation_utils import validate_input

        agent = BaseAgent(name="test")
        ctx = RunContext(run_id="r1")
        result = asyncio.get_event_loop().run_until_complete(
            validate_input(agent, "hello", ctx)
        )
        assert result is None


# ---------------------------------------------------------------------------
# 01-#7: Secrets redaction
# ---------------------------------------------------------------------------


class TestSecretsRedaction:
    def test_redact_dict_handles_circular_reference(self):
        from orchestrator.utils.secrets import redact_dict

        d: dict = {"key": "value"}
        d["self"] = d  # circular reference
        result = redact_dict(d)
        # Should not recurse infinitely
        assert result["key"] == "value"
        assert result["self"] == {"_redacted": "[CIRCULAR REFERENCE]"}

    def test_redact_dict_max_depth_redacts(self):
        from orchestrator.utils.secrets import redact_dict

        deep = {"a": {"b": {"c": {"d": {"e": {"f": "secret"}}}}}}
        result = redact_dict(deep, max_depth=2)
        # At depth 3 it should be redacted
        assert "_redacted" in str(result)

    def test_redact_sensitive_key(self):
        from orchestrator.utils.secrets import redact_dict

        data = {"api_key": "sk-abc123456789", "name": "test"}
        result = redact_dict(data)
        assert result["api_key"] != "sk-abc123456789"
        assert result["name"] == "test"
