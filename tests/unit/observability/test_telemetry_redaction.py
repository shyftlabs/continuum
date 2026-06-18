"""
Phase 4 — data-label TELEMETRY redaction.

Two protections, both at the trace input/output emission point:

1. Label-driven (mode: redact): when a run is tainted and a policy denies
   `telemetry` for those labels, the input/output is replaced with a redacted
   placeholder before it reaches Langfuse — the trace skeleton (timings, tokens,
   status) is kept, the content is not leaked.
2. Always-on secret masking: `redact_dict` (API keys / tokens / passwords) is
   applied even with zero labels — closing a real egress hole that existed
   because redaction was never wired into telemetry.

The gate reuses the shared resolver, so the ambient run policy works here too.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from continuum.agent.utils.context_utils import create_run_context
from continuum.observability.data_redaction import redact_for_telemetry
from continuum.security.policy import PolicyDecision


class TestRedactForTelemetry:
    def test_no_policy_still_masks_secrets(self):
        # Label-independent hole: secrets must be masked regardless of policy.
        out = redact_for_telemetry({"query": "hi", "api_key": "sk-secret123"})
        assert "sk-secret123" not in str(out)
        assert out["query"] == "hi"

    def test_denied_label_returns_placeholder(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(
            allowed=False, policy_name="no_pii_telemetry", reason="deny"
        )
        out = redact_for_telemetry(
            {"query": "my SSN is 123-45-6789"},
            policy_store=ps,
            subject="agent",
            labels={"pii"},
        )
        assert "123-45-6789" not in str(out)
        assert "_redacted" in out
        ps.check.assert_called_once_with(["agent", "pii"], "telemetry")

    def test_allowed_label_passes_through_but_masks_secrets(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        out = redact_for_telemetry(
            {"query": "hello", "token": "tok-abc123"},
            policy_store=ps,
            subject="agent",
            labels={"pii"},
        )
        assert out["query"] == "hello"  # content kept (allowed)
        assert "tok-abc123" not in str(out)  # but secrets still masked

    def test_uses_ambient_policy_when_not_passed(self):
        from continuum.security.policy_context import use_active_policy

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=False, policy_name="p", reason="deny")
        ctx = create_run_context(data_labels={"pii"})
        with use_active_policy(ps, "agent", ctx):
            out = redact_for_telemetry({"query": "secret stuff"})  # no policy args
        assert "_redacted" in out
        ps.check.assert_called_once_with(["agent", "pii"], "telemetry")

    def test_subject_only_when_no_labels(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        redact_for_telemetry({"q": "x"}, policy_store=ps, subject="agent")
        ps.check.assert_called_once_with("agent", "telemetry")


class TestStartTraceRedaction:
    async def test_start_trace_redacts_input_when_denied(self, monkeypatch):
        from continuum.agent.base import BaseAgent
        from continuum.agent.config import AgentConfig
        from continuum.agent.execution.run_lifecycle import RunLifecycle
        from continuum.security.policy import PolicyStore

        captured: dict = {}

        class FakeTrace:
            id = "t1"
            langfuse_trace = None

        class FakeTracingManager:
            def create_trace(self, **kwargs):
                captured["input"] = kwargs.get("input")
                return FakeTrace()

        import continuum.observability as obs

        monkeypatch.setattr(obs, "TracingManager", FakeTracingManager)

        agent = BaseAgent(name="a", instructions="t", config=AgentConfig())
        # A policy that denies telemetry for the "pii" label.
        store = PolicyStore()
        from continuum.security.policy import AccessPolicy

        store.add_policy(
            AccessPolicy(
                name="no_pii_telemetry",
                subjects=["pii"],
                resources=["telemetry"],
                effect="deny",
            )
        )
        agent.policy_store = store
        ctx = create_run_context(data_labels={"pii"})
        run_state = MagicMock()

        lifecycle = RunLifecycle()
        await lifecycle.start_trace(agent, ctx, run_state, input_preview="my SSN 123-45-6789")

        assert captured["input"] is not None
        assert "123-45-6789" not in str(captured["input"]), "tainted input reached telemetry"
        assert "_redacted" in str(captured["input"])


# ---------------------------------------------------------------------------
# @observe span redaction — the per-LLM-call prompts/completions (and other
# spans) must be redacted too, not just the trace-level preview.
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self):
        self.output = None

    def set_output(self, o):
        self.output = o

    def add_metadata(self, *a, **k):
        pass

    def set_error(self, *a, **k):
        pass


class _FakeSpanScope:
    last = None

    def __init__(self, name, input=None, metadata=None, level=None):  # noqa: A002
        self.input = input
        self.span = _FakeSpan()
        _FakeSpanScope.last = self

    async def __aenter__(self):
        return self.span

    async def __aexit__(self, *a):
        return False


class TestObserveSpanRedaction:
    async def test_observe_redacts_span_input_and_output_when_denied(self, monkeypatch):
        from continuum.observability import decorators
        from continuum.security.policy_context import use_active_policy

        monkeypatch.setattr(decorators, "SpanScope", _FakeSpanScope)

        @decorators.observe(name="llm_chat")
        async def call(messages):
            return {"completion": "patient SSN 123-45-6789"}

        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=False, policy_name="p", reason="deny")
        ctx = create_run_context(data_labels={"pii"})

        with use_active_policy(ps, "agent", ctx):
            result = await call({"messages": [{"role": "user", "content": "my SSN 123-45-6789"}]})

        # The function's real return value is untouched.
        assert result == {"completion": "patient SSN 123-45-6789"}

        # But the span captured a redacted input AND output — no raw content leaked.
        scope = _FakeSpanScope.last
        assert "123-45-6789" not in str(scope.input)
        assert "_redacted" in str(scope.input)
        assert "123-45-6789" not in str(scope.span.output)
        assert "_redacted" in str(scope.span.output)

    async def test_observe_passes_through_when_no_policy(self, monkeypatch):
        from continuum.observability import decorators

        monkeypatch.setattr(decorators, "SpanScope", _FakeSpanScope)

        @decorators.observe(name="llm_chat")
        async def call(messages):
            return {"completion": "hello"}

        result = await call({"messages": [{"role": "user", "content": "hi"}]})
        assert result == {"completion": "hello"}
        scope = _FakeSpanScope.last
        # No ambient policy → content passes through to the span (not redacted).
        assert "hello" in str(scope.span.output)
